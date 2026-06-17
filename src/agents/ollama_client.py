"""Ollama chat-completions client for action selection (local or cloud).

The LLM is asked to pick an INDEX into the ``legal_actions`` list. Output is
constrained to ``{"action_index": <int>}`` via Ollama's NATIVE ``/api/chat``
``format`` field (a raw JSON schema enforced with XGrammar constrained
decoding) so the reply is tiny and guaranteed parseable; the chosen action is
always legal by construction.

Why not the OpenAI-compatible ``/v1/chat/completions`` ``response_format``?
Ollama only honours that as loose "JSON mode" and silently ignores the
json_schema for many models (ollama#10937, #10001) — reasoning models such as
``glm`` then free-reason in prose and never emit the schema. The native
``format`` field is actually enforced, so we target ``/api/chat`` instead.

One client serves both deployment modes — they differ only in ``base_url``,
the API key, and the model name:

* **Local (free, no key):** ``OLLAMA_BASE_URL=http://localhost:11434/v1`` with
  a locally pulled model (e.g. ``gpt-oss:20b``). The key is ignored by Ollama
  but still sent as a bearer token, so any placeholder works.
* **Cloud (Ollama Turbo):** ``OLLAMA_BASE_URL=https://ollama.com/v1`` with
  ``OLLAMA_API_KEY=<key>`` and a hosted model (e.g. ``gpt-oss:120b``).

Endpoint: the base URL ends in ``/v1`` (for the model-list probe); the client
strips it and calls the native ``{root}/api/chat``. Auth uses the standard
``Authorization: Bearer <key>`` header.

Note on thinking: the answer is a single legal-action index that needs no
chain-of-thought, so thinking is DISABLED by default (``think: false``). On
reasoning models (``glm`` / ``gpt-oss`` / ``qwen3``) this skips the hidden
reasoning trace entirely and collapses per-call latency from ~10-113s to ~1s;
non-reasoning models ignore the flag. ``num_predict`` (``max_tokens``) still
leaves generous headroom in case ``think=True`` is passed explicitly.
"""

from __future__ import annotations

import json
import os
import re
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
DEFAULT_MAX_TOKENS = 16384  # reasoning models (gpt-oss, glm) think before the JSON; leave room

# Raw JSON schema for Ollama's NATIVE /api/chat `format` field. Unlike the
# OpenAI-compatible /v1 `response_format` (which Ollama only honours as loose
# "JSON mode" and silently ignores for many models — ollama#10937, #10001),
# the native `format` is enforced via XGrammar constrained decoding, so the
# reply is guaranteed to match this schema.
_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"action_index": {"type": "integer"}},
    "required": ["action_index"],
    "additionalProperties": False,
}

# System message enforcing JSON-only output. The native `format` mask already
# constrains decoding, but some models (e.g. gemma4) intermittently ignore it
# and free-reason in prose with no JSON — unparseable, forcing a random
# fallback. This steers them to emit only the object so the answer is always
# recoverable. Sent as a `system`-role message (portable across local & cloud;
# no Modelfile needed).
_SYSTEM_PROMPT = (
    "You are a Risiko move selector. You are given a numbered list of legal "
    'actions. Reply with ONLY a JSON object of the form {"action_index": N}, '
    "where N is the integer index of your chosen action. Do not write any "
    "prose, explanation, or reasoning — output the JSON object and nothing else."
)


def call_ollama_for_action_index(
    obs: dict[str, np.ndarray],
    legal_actions: list[dict[str, int]],
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    temperature: float = 0.1,
    top_p: float | None = None,
    strategy_hint: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    think: bool = False,
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
        think: Enable model chain-of-thought. Default False — the answer is a
            single legal-action index needing no reasoning, and disabling it
            cuts reasoning-model latency ~10-100x. Set True only if a model's
            index quality measurably improves with reasoning.

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
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "format": _ACTION_SCHEMA,  # enforced via XGrammar on the native endpoint
        "stream": False,
        # Thinking OFF — universal across models. For reasoning models (glm,
        # gpt-oss, qwen3) this skips the chain-of-thought entirely (the answer
        # is a single index, no reasoning needed) and collapses latency from
        # 10-113s to ~1s; for non-reasoning models (gemma, mistral) Ollama
        # silently ignores it. Absence of `think` defaults to True, so it MUST
        # be sent explicitly (ollama docs: capabilities/thinking). The native
        # `format` mask still applies on the immediate (non-thinking) output.
        "think": think,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if top_p is not None:
        body["options"]["top_p"] = top_p

    # Native /api/chat (NOT the OpenAI-compat /v1) — `base` ends in /v1, strip it.
    root = base[:-3] if base.endswith("/v1") else base
    url = f"{root}/api/chat"
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


def list_ollama_models(
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> set[str] | None:
    """Return the set of model ids the Ollama server advertises, or ``None``.

    Queries the OpenAI-compatible ``GET {base}/models`` endpoint. ``base_url`` /
    ``api_key`` fall back to ``OLLAMA_BASE_URL`` / ``OLLAMA_API_KEY`` then the
    local defaults — same resolution as :func:`call_ollama_for_action_index`.

    Returns ``None`` (rather than raising) on any connection / HTTP / parse
    error, so callers can treat "unknown" as "skip validation and proceed"
    instead of crashing a valid run on a flaky probe.
    """
    ensure_env_loaded()
    base = (base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")
    key = api_key or os.environ.get(ENV_API_KEY) or DEFAULT_API_KEY
    url = f"{base}/models"
    headers = {"Authorization": f"Bearer {key}"}
    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return {entry["id"] for entry in data.get("data", []) if "id" in entry}
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
        _log.warning("Could not list Ollama models from %s: %s", url, exc)
        return None


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
# Case-insensitive + separator-tolerant: models that ignore the enforced schema
# emit key variants — capital ({"Action_index": 18}), hyphen ({"action-index":
# 5}), or a space. Match them all; the underscore is the canonical form.
_INDEX_RE = re.compile(r'"?action[-_ ]?index"?\s*[:=]\s*(-?\d+)', re.IGNORECASE)


def _extract_action_index(content: str) -> int | None:
    """Pull an integer ``action_index`` out of a model reply, tolerantly.

    Output shape varies by backend: the native ``format`` schema yields
    ``{"action_index": <int>}`` locally, but cloud-proxied models often flatten
    the single-property object to a bare integer (e.g. ``"0"``); reasoning
    models may also wrap the answer in a ```json fence or add prose. Try strict
    JSON (object *or* bare int) first, then a fenced-JSON unwrap, then regex.
    """
    candidates = [content.strip()]
    fenced = _FENCE_RE.search(content)
    if fenced:
        candidates.append(fenced.group(1).strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        # Bare integer (cloud schema-flattening): "0" -> 0. Exclude bool.
        if isinstance(parsed, int) and not isinstance(parsed, bool):
            return parsed
        if isinstance(parsed, dict):
            # Case- and separator-insensitive key: models emit "Action_index",
            # "action-index", "action index" — normalise to the canonical form.
            for k, v in parsed.items():
                if isinstance(k, str) and k.lower().replace("-", "_").replace(" ", "_") == (
                    "action_index"
                ):
                    try:
                        return int(v)
                    except (ValueError, TypeError):
                        break
    match = _INDEX_RE.search(content)
    return int(match.group(1)) if match else None


def _parse_index(
    data: dict[str, Any],
    legal_actions: list[dict[str, int]],
    model: str,
    elapsed: float,
) -> int | None:
    """Extract and range-check ``action_index`` from a chat-completions reply."""
    # Native /api/chat shape: {"message": {"content", "thinking"}, "eval_count",
    # "done_reason"} — not the OpenAI-compat {"choices": [...], "usage": ...}.
    message = data.get("message") or {}
    content = message.get("content", "") or ""
    completion_tokens = data.get("eval_count", 0)
    done_reason = data.get("done_reason")
    if not content:
        # With `format` enforced, content should already be schema-valid JSON.
        # If it is empty (e.g. num_predict exhausted mid-thinking,
        # done_reason="length"), fall back to the thinking trace — the index
        # regex may still recover a number before giving up to RandomAgent.
        thinking = message.get("thinking") or message.get("reasoning") or ""
        if thinking:
            content = thinking
        else:
            _log.warning(
                "Ollama returned empty content (model=%s, %.2fs, done_reason=%s, tok=%d)",
                model,
                elapsed,
                done_reason,
                completion_tokens,
            )
            return None
    idx = _extract_action_index(content)
    if idx is None:
        _log.warning("Ollama reply has no usable action_index | raw=%r", content)
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


__all__ = [
    "call_ollama_for_action_index",
    "list_ollama_models",
    "ENV_BASE_URL",
    "ENV_API_KEY",
    "DEFAULT_BASE_URL",
]
