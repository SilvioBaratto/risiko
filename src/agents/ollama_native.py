"""Native Ollama HTTP client for action selection.

The LLM is asked to pick an INDEX into the legal_actions list rather than
invent an action. Output is a single integer enforced by Ollama's ``format``
JSON-schema parameter — small, fast, and guaranteed parseable. The chosen
action is always legal by construction.

Uses ``/api/chat`` with ``think: false`` to disable Qwen3 / Gemma 4 reasoning
mode, which would otherwise consume the token budget on hidden thinking.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import numpy as np

from src.utils.constants import CONTINENT_BONUSES, CONTINENTS, TERRITORY_NAMES
from src.utils.log import get_logger

_log = get_logger("llm")

_PHASE_NAMES = {
    0: "TRADE",
    1: "REINFORCE",
    2: "ATTACK",
    3: "CAPTURE_MOVE",
    4: "FORTIFY",
}

_PHASE_INSTRUCTIONS = {
    0: "Trade card sets for bonus armies (each set = 3 cards).",
    1: "Place reinforcement armies on your territories.",
    2: "Attack adjacent enemy territories, or skip to end attack phase.",
    3: "Move armies into the territory you just captured.",
    4: "Move armies between two of your connected territories, or skip.",
}

_ACTION_NAMES = {
    0: "TRADE", 1: "REINFORCE", 2: "ATTACK",
    3: "CAPTURE_MOVE", 4: "FORTIFY", 5: "SKIP",
}

_SYMBOLS = {0: "infantry", 1: "cavalry", 2: "artillery", 3: "wild"}


def call_ollama_for_action_index(
    obs: dict[str, np.ndarray],
    legal_actions: list[dict[str, int]],
    *,
    model: str,
    base_url: str | None = None,
    timeout: float = 30.0,
    temperature: float = 0.1,
    top_p: float | None = None,
    top_k: int | None = None,
    repeat_penalty: float | None = None,
    strategy_hint: str | None = None,
) -> int | None:
    """Call Ollama and return an index into *legal_actions*, or None on failure."""
    if not legal_actions:
        return None
    url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    url = url.removesuffix("/v1")
    prompt = _render_prompt(obs, legal_actions, strategy_hint)
    options: dict[str, Any] = {"temperature": temperature, "num_predict": 64}
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k
    if repeat_penalty is not None:
        options["repeat_penalty"] = repeat_penalty
    body = {
        "model": model,
        "think": False,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "format": {
            "type": "object",
            "properties": {
                "action_index": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": len(legal_actions) - 1,
                },
            },
            "required": ["action_index"],
        },
        "options": options,
    }
    _log.debug(
        "---PROMPT--- model=%s temp=%.2f n_legal=%d\n%s",
        model, temperature, len(legal_actions), prompt,
    )
    t0 = time.time()
    response = httpx.post(f"{url}/api/chat", json=body, timeout=timeout)
    response.raise_for_status()
    elapsed = time.time() - t0
    data = response.json()
    content = data.get("message", {}).get("content", "")
    eval_count = data.get("eval_count", 0)
    eval_duration_ms = data.get("eval_duration", 0) / 1_000_000
    if not content:
        _log.warning(
            "LLM returned empty content (model=%s, %.2fs, eval_duration=%.0fms) — "
            "thinking may still be enabled or model failed",
            model, elapsed, eval_duration_ms,
        )
        return None
    try:
        parsed = json.loads(content)
        idx = int(parsed["action_index"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        _log.warning("LLM reply not parseable as {action_index: int}: %s | raw=%r", e, content)
        return None
    if 0 <= idx < len(legal_actions):
        chosen = legal_actions[idx]
        _log.info(
            "LLM model=%s reply=%s → idx=%d action=%s (%.2fs, %d tok)",
            model, content, idx, chosen, elapsed, eval_count,
        )
        return idx
    _log.warning(
        "LLM returned out-of-range idx=%d (reply=%s, legal range 0..%d) — falling back",
        idx, content, len(legal_actions) - 1,
    )
    return None


def _render_prompt(
    obs: dict[str, np.ndarray],
    legal_actions: list[dict[str, int]],
    strategy_hint: str | None,
) -> str:
    """Render a structured, LLM-friendly prompt.

    Sections (in order):
      1. Header — player, phase, phase instruction
      2. Strategy hint (if provided)
      3. Status line — reinforcements / trade count / cards
      4. Board — grouped by continent, own territories marked with ★
      5. Legal actions — grouped by action type, semantic descriptions
      6. Reply instruction
    """
    me = int(obs["current_player"])
    phase = int(obs["phase"])
    phase_name = _PHASE_NAMES.get(phase, "UNKNOWN")
    phase_instr = _PHASE_INSTRUCTIONS.get(phase, "")

    parts: list[str] = [
        f"# RISIKO — Player {me}, phase {phase_name}",
        f"Task: {phase_instr}",
    ]
    if strategy_hint:
        parts.append(f"Strategy: {strategy_hint}")

    parts.append(_render_status(obs, phase))
    parts.append(_render_board(obs, me))
    parts.append(_render_legal_actions(legal_actions, obs))
    parts.append(
        f'Reply with JSON only: {{"action_index": <integer 0..{len(legal_actions) - 1}>}}'
    )
    return "\n\n".join(parts)


def _render_status(obs: dict[str, np.ndarray], phase: int) -> str:
    reinforcements = int(obs["reinforcements_remaining"])
    trade_count = int(obs["trade_count"])
    cards = _format_cards(obs)
    eliminated = [i for i, e in enumerate(obs["eliminated"].tolist()) if e]
    bits = [f"reinforcements_left={reinforcements}"] if phase == 1 else []
    bits.append(f"trade_count={trade_count}")
    bits.append(f"cards=[{cards}]")
    if eliminated:
        bits.append(f"eliminated_players={eliminated}")
    return "## Status\n" + " | ".join(bits)


def _render_board(obs: dict[str, np.ndarray], me: int) -> str:
    """Group territories by continent. Mark own territories with ★."""
    owner = obs["territory_owner"]
    armies = obs["armies"]
    lines = ["## Board (★ = your territory, P# = player number, a# = armies)"]
    for cont_name, tids in CONTINENTS.items():
        own_count = sum(1 for tid in tids if int(owner[tid]) == me)
        bonus = CONTINENT_BONUSES[cont_name]
        header = f"\n{cont_name} ({own_count}/{len(tids)} yours, +{bonus} bonus):"
        lines.append(header)
        for tid in tids:
            o = int(owner[tid])
            a = int(armies[tid])
            mark = "★" if o == me else " "
            lines.append(f"  {mark} [{tid:>2}] {TERRITORY_NAMES[tid]:<22} P{o} a{a}")
    return "\n".join(lines)


def _render_legal_actions(
    legal_actions: list[dict[str, int]],
    obs: dict[str, np.ndarray],
) -> str:
    """Group actions by type with phase-specific semantic descriptions."""
    grouped: dict[str, list[tuple[int, str]]] = {}
    for idx, a in enumerate(legal_actions):
        atype = int(a["action_type"])
        name = _ACTION_NAMES.get(atype, f"TYPE_{atype}")
        grouped.setdefault(name, []).append((idx, _describe_action(a, obs)))

    lines = ["## Legal Actions (pick one by INDEX)"]
    for name, items in grouped.items():
        lines.append(f"\n{name}:")
        for idx, desc in items:
            lines.append(f"  [{idx:>2}] {desc}")
    return "\n".join(lines)


def _describe_action(a: dict[str, int], obs: dict[str, np.ndarray]) -> str:
    """Produce a human-readable description of an action."""
    atype = int(a["action_type"])
    pa, pb, pc, pd = a["param_a"], a["param_b"], a["param_c"], a["param_d"]
    armies = obs["armies"]
    owner = obs["territory_owner"]
    if atype == 0:  # TRADE
        return f"trade cards [{pa},{pb},{pc}]"
    if atype == 1:  # REINFORCE
        return f"+{pb} on {TERRITORY_NAMES[pa]} (currently a{int(armies[pa])})"
    if atype == 2:  # ATTACK
        return (
            f"{TERRITORY_NAMES[pa]} (a{int(armies[pa])}) "
            f"→ {TERRITORY_NAMES[pb]} (P{int(owner[pb])}, a{int(armies[pb])}), "
            f"{pc} {'die' if pc == 1 else 'dice'}"
        )
    if atype == 3:  # CAPTURE_MOVE
        return f"move {pc} armies from {TERRITORY_NAMES[pa]} into {TERRITORY_NAMES[pb]}"
    if atype == 4:  # FORTIFY
        return f"move {pc} armies from {TERRITORY_NAMES[pa]} to {TERRITORY_NAMES[pb]}"
    if atype == 5:  # SKIP
        return "skip / end phase"
    return f"type={atype} params=({pa},{pb},{pc},{pd})"


def _format_cards(obs: dict[str, np.ndarray]) -> str:
    card_matrix = obs["cards"]
    cards: list[str] = []
    for i in range(card_matrix.shape[0]):
        if card_matrix[i].sum() > 0:
            sym = int(card_matrix[i].argmax())
            cards.append(_SYMBOLS.get(sym, "unknown"))
    return ", ".join(cards) if cards else "none"
