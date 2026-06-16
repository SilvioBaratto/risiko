"""Azure OpenAI chat-completions client for action selection.

The LLM is asked to pick an INDEX into the ``legal_actions`` list. Output is
constrained to ``{"action_index": <int>}`` via Azure's structured-output
``response_format`` (json_schema, ``strict: true``) so the reply is tiny and
guaranteed parseable; the chosen action is always legal by construction.

Endpoint shape (deployment is embedded in the base URL):
    {AZURE_OPENAI_BASE_URL}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}

Auth uses the ``api-key`` header (Azure convention), not a bearer token.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import numpy as np

from src.agents.action_prompt import render_action_prompt
from src.utils.env import ensure_env_loaded
from src.utils.log import get_logger

_log = get_logger("llm")

ENV_BASE_URL = "AZURE_OPENAI_BASE_URL"
ENV_API_KEY = "AZURE_OPENAI_API_KEY"
ENV_API_VERSION = "AZURE_OPENAI_API_VERSION"

DEFAULT_API_VERSION = "2024-12-01-preview"

_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "action_choice",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"action_index": {"type": "integer"}},
            "required": ["action_index"],
            "additionalProperties": False,
        },
    },
}


class AzureConfigError(RuntimeError):
    """Raised when required Azure OpenAI env vars are missing."""


def call_azure_for_action_index(
    obs: dict[str, np.ndarray],
    legal_actions: list[dict[str, int]],
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
    timeout: float = 30.0,
    temperature: float = 0.1,
    top_p: float | None = None,
    strategy_hint: str | None = None,
) -> int | None:
    """Call Azure OpenAI and return an index into *legal_actions*, or None.

    ``model`` is sent in the request body but Azure routes on the deployment
    embedded in *base_url*. ``base_url`` / ``api_key`` / ``api_version`` fall back
    to the ``AZURE_OPENAI_BASE_URL`` / ``_API_KEY`` / ``_API_VERSION`` env vars.

    Returns:
        An index in ``[0, len(legal_actions))`` chosen by the model, or ``None``
        if the model declined / returned something unparseable / out of range.

    Raises:
        AzureConfigError: If base URL or API key cannot be resolved.
    """
    if not legal_actions:
        return None
    ensure_env_loaded()
    base = (base_url or os.environ.get(ENV_BASE_URL, "")).rstrip("/")
    key = api_key or os.environ.get(ENV_API_KEY, "")
    version = api_version or os.environ.get(ENV_API_VERSION) or DEFAULT_API_VERSION
    if not base or not key:
        raise AzureConfigError(
            f"Set {ENV_BASE_URL} and {ENV_API_KEY} (e.g. in a project .env) to use Azure OpenAI."
        )

    prompt = render_action_prompt(obs, legal_actions, strategy_hint)
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": _RESPONSE_FORMAT,
        "temperature": temperature,
        "max_tokens": 64,
    }
    if top_p is not None:
        body["top_p"] = top_p

    url = f"{base}/chat/completions?api-version={version}"
    headers = {"api-key": key, "Content-Type": "application/json"}
    _log.debug(
        "---PROMPT--- model=%s temp=%.2f n_legal=%d\n%s",
        model,
        temperature,
        len(legal_actions),
        prompt,
    )
    t0 = time.time()
    response = httpx.post(url, json=body, headers=headers, timeout=timeout)
    response.raise_for_status()
    elapsed = time.time() - t0
    data = response.json()
    return _parse_index(data, legal_actions, model, elapsed)


def _parse_index(
    data: dict[str, Any],
    legal_actions: list[dict[str, int]],
    model: str,
    elapsed: float,
) -> int | None:
    """Extract and range-check ``action_index`` from a chat-completions reply."""
    choices = data.get("choices") or []
    content = choices[0].get("message", {}).get("content", "") if choices else ""
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    if not content:
        _log.warning("Azure returned empty content (model=%s, %.2fs)", model, elapsed)
        return None
    try:
        idx = int(json.loads(content)["action_index"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        _log.warning("Azure reply not parseable as {action_index: int}: %s | raw=%r", e, content)
        return None
    if 0 <= idx < len(legal_actions):
        _log.info(
            "LLM model=%s idx=%d action=%s (%.2fs, %d tok)",
            model,
            idx,
            legal_actions[idx],
            elapsed,
            completion_tokens,
        )
        return idx
    _log.warning(
        "Azure returned out-of-range idx=%d (legal range 0..%d) — falling back",
        idx,
        len(legal_actions) - 1,
    )
    return None


__all__ = ["call_azure_for_action_index", "AzureConfigError"]
