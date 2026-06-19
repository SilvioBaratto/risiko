"""Provider-agnostic prompt rendering for LLM action selection.

The LLM is asked to pick an INDEX into the ``legal_actions`` list rather than
invent an action. Output is therefore guaranteed legal by construction and tiny
(~10 tokens). This module owns only the *rendering* of the board/legal-action
prompt; the HTTP transport lives in the per-provider client modules.
"""

from __future__ import annotations

import numpy as np

from src.agents.diplomacy import DiplomacyNote
from src.utils.constants import CONTINENT_BONUSES, CONTINENTS, TERRITORY_NAMES

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
    0: "TRADE",
    1: "REINFORCE",
    2: "ATTACK",
    3: "CAPTURE_MOVE",
    4: "FORTIFY",
    5: "SKIP",
}

_SYMBOLS = {0: "infantry", 1: "cavalry", 2: "artillery", 3: "wild"}

__all__ = ["render_action_prompt"]


def render_action_prompt(
    obs: dict[str, np.ndarray],
    legal_actions: list[dict[str, int]],
    strategy_hint: str | None = None,
    *,
    diplomacy_note: DiplomacyNote | None = None,
) -> str:
    """Render a structured, LLM-friendly prompt.

    Sections (in order):
      1. Header — player, phase, phase instruction
      2. Strategy hint (if provided)
      3. Diplomacy note (if provided — eval-only social context)
      4. Status line — reinforcements / trade count / cards
      5. Board — grouped by continent, own territories marked with ★
      6. Legal actions — grouped by action type, semantic descriptions
      7. Reply instruction

    When ``diplomacy_note`` is ``None`` the output is byte-identical to the
    training-path render (disabled-equivalence guarantee).
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
    if diplomacy_note is not None:
        parts.append(_render_diplomacy(diplomacy_note))

    parts.append(_render_status(obs, phase))
    parts.append(_render_board(obs, me))
    parts.append(_render_legal_actions(legal_actions, obs))
    parts.append(f'Reply with JSON only: {{"action_index": <integer 0..{len(legal_actions) - 1}>}}')
    return "\n\n".join(parts)


def _render_diplomacy(note: DiplomacyNote) -> str:
    """Render the social-context section from a DiplomacyNote (eval-only)."""
    lines = ["## Diplomacy"]
    if note.allies:
        ally_str = ", ".join(f"P{a}" for a in sorted(note.allies))
        lines.append(f"Allies — do not attack: {ally_str}")
    if note.leader is not None:
        lines.append(f"Leader (prefer to target): P{note.leader}")
    if note.grudges:
        grudge_str = ", ".join(f"P{g}" for g in sorted(note.grudges))
        lines.append(f"Grudges: {grudge_str}")
    return "\n".join(lines)


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
    if atype == 3:  # CAPTURE_MOVE  (param_a = army count; territories live in state)
        return f"move {pa} armies into just-captured territory"
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
