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
import random
import re
import time
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import numpy as np
from httpx import HTTPError as _HttpxHTTPError

from src.agents.action_prompt import render_action_prompt
from src.agents.diplomacy import DiplomacyNote
from src.utils.env import ensure_env_loaded
from src.utils.log import get_logger

_log = get_logger("llm")

# ── Optional call tracing (for visualization / full-game replay) ─────────────
# When a sink list is installed via start_call_trace(), every action and
# negotiation call appends a record with the exact prompt, the raw model
# response (including the model's `thinking` trace when it emits one), and the
# parsed result. Disabled by default (sink is None) → zero overhead and
# byte-identical behavior. Intended for single-game tracing (calls are recorded
# in the order they happen); not thread-safe across concurrent games.
_CALL_TRACE: list[dict[str, Any]] | None = None


def start_call_trace() -> list[dict[str, Any]]:
    """Begin recording LLM calls; returns the list that will collect records."""
    global _CALL_TRACE
    _CALL_TRACE = []
    return _CALL_TRACE


def stop_call_trace() -> list[dict[str, Any]] | None:
    """Stop recording and return the collected records (or None if not active)."""
    global _CALL_TRACE
    trace, _CALL_TRACE = _CALL_TRACE, None
    return trace


def _record_call(entry: dict[str, Any]) -> None:
    """Append a trace record if tracing is active (no-op otherwise)."""
    if _CALL_TRACE is not None:
        _CALL_TRACE.append(entry)


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

# Negotiation schema: only integer-array fields are constrained. The
# `messages_to` field (object with open-ended string keys) is intentionally
# omitted — XGrammar/GBNF cannot enforce dynamic key names, so constraining
# it yields no guarantee and may produce parse failures. The model may still
# emit `messages_to` as additional text; `_parse_negotiation` returns the
# full dict so callers can use it best-effort.
_NEGOTIATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "propose_alliance_with": {"type": "array", "items": {"type": "integer"}},
        "accept_alliance_with": {"type": "array", "items": {"type": "integer"}},
        "declare_war_on": {"type": "array", "items": {"type": "integer"}},
        "attack_priority": {"type": "array", "items": {"type": "integer"}},
    },
    "required": [
        "propose_alliance_with",
        "accept_alliance_with",
        "declare_war_on",
        "attack_priority",
    ],
}

_NEGOTIATION_SYSTEM = (
    "You are a Risiko diplomat. Based on the board state and current alliances, "
    "emit ONLY a JSON object with EXACTLY these four keys, each a list of "
    'player-index integers (use [] for none): "propose_alliance_with", '
    '"accept_alliance_with", "declare_war_on", "attack_priority". '
    "Do not invent other key names. Do not wrap the JSON in markdown code "
    "fences. Do not write any prose — output the raw JSON object and nothing else."
)

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


# Ollama cloud throttles with HTTP 429 — for per-minute/concurrency limits and,
# importantly, the 5-hour session and 7-day weekly usage quotas. The cloud does
# not reliably send Retry-After or quota headers (ollama/ollama#15663), so the
# only robust strategy is: on 429, wait and retry until the limit resets, rather
# than failing or playing a random move. Backoff is exponential with jitter,
# capped, and honours Retry-After when present.
_RATE_LIMIT_STATUS = 429
_BACKOFF_BASE_S = 5.0
_BACKOFF_CAP_S = 300.0  # 5 min between polls once backed off — cheap vs a 5h/7d wait


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a Retry-After header (delta-seconds or HTTP-date) into seconds, or None."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
        return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def _post_waiting_out_rate_limits(
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any],
    timeout: float,
    rate_limit_max_wait: float,
) -> httpx.Response:
    """POST, transparently waiting out HTTP 429 rate/usage limits.

    On 429 the call sleeps (Retry-After if given, else capped exponential backoff
    with jitter) and retries until it succeeds or cumulative wait exceeds
    *rate_limit_max_wait*. ``rate_limit_max_wait <= 0`` disables waiting (a 429
    raises immediately — legacy behaviour). Non-429 errors propagate unchanged.
    """
    start = time.time()
    attempt = 0
    while True:
        response = httpx.post(url, json=json_body, headers=headers, timeout=timeout)
        if response.status_code != _RATE_LIMIT_STATUS:
            response.raise_for_status()
            return response
        elapsed = time.time() - start
        if rate_limit_max_wait <= 0 or elapsed >= rate_limit_max_wait:
            response.raise_for_status()  # give up → propagate the 429
        wait = _retry_after_seconds(response)
        if wait is None:
            wait = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2**attempt))
            wait += random.uniform(0.0, wait * 0.25)  # jitter to de-sync callers
        _log.warning(
            "Ollama 429 rate-limited (session/weekly/RPM quota); waiting %.0fs then "
            "retrying — elapsed %.0fs of %.0fs budget",
            wait,
            elapsed,
            rate_limit_max_wait,
        )
        time.sleep(wait)
        attempt += 1


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
    diplomacy_note: DiplomacyNote | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    think: bool = False,
    rate_limit_max_wait: float = 0.0,
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
        diplomacy_note: Optional eval-only social context (allies, leader, grudges);
            ``None`` produces byte-identical output to the training-path call.
        max_tokens: Completion-token ceiling (sized for reasoning headroom).
        think: Enable model chain-of-thought. Default False — the answer is a
            single legal-action index needing no reasoning, and disabling it
            cuts reasoning-model latency ~10-100x. Set True only if a model's
            index quality measurably improves with reasoning.
        rate_limit_max_wait: Seconds to wait out HTTP 429 rate/usage limits
            (per call) before giving up; 0 disables waiting (legacy immediate
            fallback). Lets a run pause for the Ollama-cloud session/weekly
            reset instead of returning a random move.

    Returns:
        An index in ``[0, len(legal_actions))`` chosen by the model, or ``None``
        if the model declined / returned something unparseable / out of range.
    """
    if not legal_actions:
        return None
    ensure_env_loaded()
    base = (base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")
    key = api_key or os.environ.get(ENV_API_KEY) or DEFAULT_API_KEY

    prompt = render_action_prompt(obs, legal_actions, strategy_hint, diplomacy_note=diplomacy_note)
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
    response = _post_waiting_out_rate_limits(
        url,
        headers=headers,
        json_body=body,
        timeout=timeout,
        rate_limit_max_wait=rate_limit_max_wait,
    )
    elapsed = time.time() - t0
    data = response.json()
    idx = _parse_index(data, legal_actions, model, elapsed)
    if _CALL_TRACE is not None:
        message = data.get("message") or {}
        chosen = None
        if idx is not None and 0 <= idx < len(legal_actions):
            chosen = legal_actions[idx]
        _record_call(
            {
                "kind": "action",
                "model": model,
                "prompt": prompt,
                "response": message.get("content"),
                "thinking": message.get("thinking"),
                "parsed_index": idx,
                "chosen_action": chosen,
                "latency_s": round(elapsed, 3),
                "eval_count": data.get("eval_count"),
            }
        )
    return idx


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


def call_ollama_for_negotiation(
    *,
    root: str,
    key: str,
    model: str,
    prompt: str,
    max_message_tokens: int,
    temperature: float = 0.7,
    timeout: float = 60.0,
    think: bool = False,
    rate_limit_max_wait: float = 0.0,
) -> dict[str, Any] | None:
    """Call Ollama for a negotiation response; returns a parsed dict or None.

    Same fallback contract as :func:`call_ollama_for_action_index`: any HTTP,
    timeout, or parse error returns ``None`` without raising. HTTP 429
    rate/usage limits are waited out (see *rate_limit_max_wait*) rather than
    treated as a failure.

    Args:
        root: Ollama root URL, e.g. ``http://localhost:11434`` (no ``/v1``).
        key: API key (``Authorization: Bearer``).
        model: Model tag, e.g. ``gemma4:12b``.
        prompt: Pre-rendered negotiation prompt string.
        max_message_tokens: Hard ceiling for ``num_predict`` (caps response length).
        temperature: Sampling temperature; default 0.7 for varied social play.
        timeout: Per-call HTTP timeout in seconds.
        think: Enable model chain-of-thought. Default False.
        rate_limit_max_wait: Seconds to wait out HTTP 429 before giving up; 0
            disables waiting (legacy immediate fallback).
    """
    body = _build_negotiation_body(model, prompt, max_message_tokens, temperature, think)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    return _post_negotiation(
        f"{root}/api/chat", headers, body, timeout, rate_limit_max_wait, model, prompt
    )


def _build_negotiation_body(
    model: str,
    prompt: str,
    max_message_tokens: int,
    temperature: float,
    think: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": _NEGOTIATION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "format": _NEGOTIATION_SCHEMA,
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_predict": max_message_tokens},
    }


def _post_negotiation(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float,
    rate_limit_max_wait: float = 0.0,
    model: str = "",
    prompt: str = "",
) -> dict[str, Any] | None:
    try:
        resp = _post_waiting_out_rate_limits(
            url,
            headers=headers,
            json_body=body,
            timeout=timeout,
            rate_limit_max_wait=rate_limit_max_wait,
        )
        data = resp.json()
        parsed = _parse_negotiation(data)
        if _CALL_TRACE is not None:
            message = data.get("message") or {}
            _record_call(
                {
                    "kind": "negotiation",
                    "model": model,
                    "prompt": prompt,
                    "response": message.get("content"),
                    "thinking": message.get("thinking"),
                    "parsed": parsed,
                    "eval_count": data.get("eval_count"),
                }
            )
        return parsed
    except (_HttpxHTTPError, ValueError) as exc:
        _log.warning("call_ollama_for_negotiation error: %s", exc)
        return None


def _extract_json_object(content: str) -> str | None:
    """Isolate the outermost ``{...}`` span from a model reply.

    Cloud models often ignore the native ``format`` schema and wrap the object
    in markdown code fences (```json … ```) or surround it with prose. Slicing
    from the first ``{`` to the last ``}`` recovers the object in all those
    cases (fences and prose live outside the braces); returns ``None`` when no
    brace pair is present.
    """
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return content[start : end + 1]


def _parse_negotiation(data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract negotiation dict from raw Ollama reply; None on any failure."""
    message = data.get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        _log.warning("negotiation: empty content")
        return None
    if data.get("done_reason") == "length":
        _log.warning("negotiation: response truncated (done_reason=length) → None")
        return None
    candidate = _extract_json_object(content)
    if candidate is None:
        _log.warning("negotiation: no JSON object in %.80r", content)
        return None
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        _log.warning("negotiation: JSON parse error on %.80r", content)
        return None
    return parsed if isinstance(parsed, dict) else None


__all__ = [
    "call_ollama_for_action_index",
    "call_ollama_for_negotiation",
    "list_ollama_models",
    "_NEGOTIATION_SCHEMA",
    "ENV_BASE_URL",
    "ENV_API_KEY",
    "DEFAULT_BASE_URL",
]
