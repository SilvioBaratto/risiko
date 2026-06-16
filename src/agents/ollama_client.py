"""Ollama chat-completions client for action selection (local or cloud).

The LLM is asked to pick an INDEX into the ``legal_actions`` list. Output is
constrained to ``{"action_index": <int>}`` via the OpenAI-compatible
``response_format`` (json_schema, ``strict: true``) so the reply is tiny and
guaranteed parseable; the chosen action is always legal by construction.

One client serves both deployment modes — they differ only in ``base_url``,
the API key, and the model name:

* **Local (free, no key):** ``OLLAMA_BASE_URL=http://localhost:11434/v1`` with
  a locally pulled model (e.g. ``gpt-oss:20b``). The key is ignored by Ollama
  but still sent as a bearer token, so any placeholder works.
* **Cloud (Ollama Turbo):** ``OLLAMA_BASE_URL=https://ollama.com/v1`` with
  ``OLLAMA_API_KEY=<key>`` and a hosted model (e.g. ``gpt-oss:120b``).

Endpoint (OpenAI-compatible; the base URL already ends in ``/v1``):
    {OLLAMA_BASE_URL}/chat/completions

Auth uses the standard ``Authorization: Bearer <key>`` header.

Note on reasoning models: ``gpt-oss`` reasons before emitting the final JSON,
which consumes completion tokens. ``max_tokens`` is therefore sized to leave
room for that hidden reasoning rather than the ~10-token answer alone.
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

ENV_BASE_URL = "OLLAMA_BASE_URL"
ENV_API_KEY = "OLLAMA_API_KEY"

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"  # placeholder; Ollama ignores it for local serving
DEFAULT_MAX_TOKENS = 512  # leaves headroom for gpt-oss reasoning before the JSON answer

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


def call_ollama_for_action_index(
    obs: dict[str, np.ndarray],
    legal_actions: list[dict[str, int]],
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
    temperature: float = 0.1,
    top_p: float | None = None,
    strategy_hint: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> int | None:
    """Call Ollama and return an index into *legal_actions*, or None.

    ``base_url`` / ``api_key`` fall back to the ``OLLAMA_BASE_URL`` /
    ``OLLAMA_API_KEY`` env vars, then to ``http://localhost:11434/v1`` /
    ``"ollama"``. Because local serving needs no real key, this never raises on
    a missing key — point ``base_url`` at ``https://ollama.com/v1`` and set
    ``OLLAMA_API_KEY`` to use hosted (cloud) models instead.

    Args:
        obs: Current observation dict (rendered into the prompt).
        legal_actions: Candidate actions; the model selects one by index.
        model: Model/tag name, e.g. ``gpt-oss:20b`` (local) or ``gpt-oss:120b`` (cloud).
        base_url: Override for ``OLLAMA_BASE_URL`` (must end in ``/v1``).
        api_key: Override for ``OLLAMA_API_KEY``.
        timeout: Per-call HTTP timeout in seconds.
        temperature: Sampling temperature.
        top_p: Optional nucleus-sampling parameter.
        strategy_hint: Optional directive injected into the prompt.
        max_tokens: Completion-token ceiling (sized for reasoning headroom).

    Returns:
        An index in ``[0, len(legal_actions))`` chosen by the model, or ``None``
        if the model declined / returned something unparseable / out of range.
    """
    if not legal_actions:
        return None
    ensure_env_loaded()
    base = (base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")
    key = api_key or os.environ.get(ENV_API_KEY) or DEFAULT_API_KEY

    prompt = render_action_prompt(obs, legal_actions, strategy_hint)
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": _RESPONSE_FORMAT,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        body["top_p"] = top_p

    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
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
        _log.warning("Ollama returned empty content (model=%s, %.2fs)", model, elapsed)
        return None
    try:
        idx = int(json.loads(content)["action_index"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        _log.warning("Ollama reply not parseable as {action_index: int}: %s | raw=%r", e, content)
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
        "Ollama returned out-of-range idx=%d (legal range 0..%d) — falling back",
        idx,
        len(legal_actions) - 1,
    )
    return None


__all__ = ["call_ollama_for_action_index", "ENV_BASE_URL", "ENV_API_KEY", "DEFAULT_BASE_URL"]
