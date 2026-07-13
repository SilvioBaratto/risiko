"""Prompt rendering for LLM-to-LLM negotiation (eval-only diplomacy layer).

Kept in a separate module from action_prompt.py so that the action-selection
path is untouched and issue #102 can extend action_prompt.py without conflict.

The board summary used to read ``board["territories"]`` — a key the environment's
observation has never had. It exposes ``territory_owner`` and ``armies`` as parallel
arrays instead, so the lookup silently returned ``{}`` and the ``## Territories``
section rendered empty in every negotiation, in every game. The models were choosing
who to ally with and who to betray while unable to see the board.
"""

from __future__ import annotations

from typing import Any

from src.utils.constants import CONTINENTS, NUM_TERRITORIES, TERRITORY_NAMES

__all__ = ["render_negotiation_prompt"]


def render_negotiation_prompt(
    player_id: int,
    board: dict[str, Any],
    allies: list[int],
    leader: int | None,
    grudges: dict[int, list[str]],
    max_chars: int,
) -> str:
    """Render a negotiation prompt; capped at *max_chars* characters.

    Args:
        player_id: The seat being asked to negotiate.
        board: An environment observation (``territory_owner`` / ``armies`` arrays).
        allies: Seats currently allied with *player_id*.
        leader: Seat holding the most territories, if known.
        grudges: Seat → list of grievances against it.
        max_chars: Hard cap on the rendered prompt.

    Returns:
        The prompt text.
    """
    header = f"# Risiko Negotiation — Player {player_id}"
    diplomacy = _render_diplomacy_context(allies, leader, grudges)

    # The budget is tight (max_message_tokens * 5 — 640 chars by default), and a naive
    # "join everything then slice" drops whatever sits last. That is how the first fix
    # to this file silently truncated the Diplomacy block mid-word, so the models could
    # see the board but not who they were allied with or who was leading.
    # Diplomacy is short and load-bearing: reserve it, and let the board take the rest,
    # shedding whole lines from the least important end.
    reserved = len(header) + len(diplomacy) + 4  # 4 = the two "\n\n" separators
    board_budget = max_chars - reserved
    board_summary = _fit_lines(_render_board_summary(board, player_id), board_budget)

    parts = [header, board_summary, diplomacy] if board_summary else [header, diplomacy]
    # Hard cap. Reserving diplomacy above means this slice normally has nothing left to
    # cut; it only bites when max_chars is smaller than the header itself.
    return "\n\n".join(parts)[:max_chars]


def _fit_lines(text: str, budget: int) -> str:
    """Trim *text* to *budget* characters, dropping whole lines from the end.

    Cutting mid-line would hand the model a half-rendered board row, which reads as a
    corrupt fact rather than a missing one.
    """
    if budget <= 0:
        return ""
    lines = text.split("\n")
    kept: list[str] = []
    used = 0
    for line in lines:
        cost = len(line) + 1
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept) if len(kept) > 1 else ""


def _render_board_summary(board: dict[str, Any], player_id: int) -> str:
    """Summarise the board per player, then per continent.

    Deliberately a summary and not the 42-line dump the action prompt uses: a
    negotiator needs to know who is strong, who is weak, and who is close to a
    continent bonus — not the army count of every single territory. Keeping it
    compact also keeps the negotiation call cheap, and it is called once per player
    per round.
    """
    owners = board.get("territory_owner")
    armies = board.get("armies")
    if owners is None or armies is None or len(owners) == 0:
        return "## Board\n  (unavailable)"

    n_players = int(board.get("n_players", max(int(o) for o in owners) + 1))
    held: dict[int, list[int]] = {p: [] for p in range(n_players)}
    for territory in range(min(NUM_TERRITORIES, len(owners))):
        held.setdefault(int(owners[territory]), []).append(territory)

    lines = ["## Board"]
    lines.append("  Standings (territories · armies):")
    ranked = sorted(held.items(), key=lambda kv: -len(kv[1]))
    for seat, territories in ranked:
        if not territories:
            continue
        army_total = sum(int(armies[t]) for t in territories)
        marker = " ← you" if seat == player_id else ""
        lines.append(f"    P{seat}: {len(territories)} territories · {army_total} armies{marker}")

    lines.append("  Continents (holder of most territories, and whether it is complete):")
    for continent, members in CONTINENTS.items():
        counts: dict[int, int] = {}
        for territory in members:
            counts[int(owners[territory])] = counts.get(int(owners[territory]), 0) + 1
        top_seat, top_count = max(counts.items(), key=lambda kv: kv[1])
        status = "COMPLETE" if top_count == len(members) else f"{top_count}/{len(members)}"
        lines.append(f"    {continent}: P{top_seat} {status}")

    yours = held.get(player_id, [])
    if yours:
        names = ", ".join(f"{TERRITORY_NAMES[t]}(a{int(armies[t])})" for t in yours)
        lines.append(f"  Your territories: {names}")
    return "\n".join(lines)


def _render_diplomacy_context(
    allies: list[int],
    leader: int | None,
    grudges: dict[int, list[str]],
) -> str:
    lines = ["## Diplomacy"]
    lines.append(f"  Allies: {allies if allies else 'none'}")
    if leader is not None:
        lines.append(f"  Leader (most territories): Player {leader}")
    for player, events in grudges.items():
        if events:
            lines.append(f"  Grudge vs P{player}: {events[0]}")
    return "\n".join(lines)
