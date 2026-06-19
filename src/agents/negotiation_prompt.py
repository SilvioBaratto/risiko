"""Prompt rendering for LLM-to-LLM negotiation (eval-only diplomacy layer).

Kept in a separate module from action_prompt.py so that the action-selection
path is untouched and issue #102 can extend action_prompt.py without conflict.
"""

from __future__ import annotations

from typing import Any

__all__ = ["render_negotiation_prompt"]


def render_negotiation_prompt(
    player_id: int,
    board: dict[str, Any],
    allies: list[int],
    leader: int | None,
    grudges: dict[int, list[str]],
    max_chars: int,
) -> str:
    """Render a negotiation prompt; capped at *max_chars* characters."""
    parts = [
        f"# Risiko Negotiation — Player {player_id}",
        _render_territory_summary(board),
        _render_diplomacy_context(allies, leader, grudges),
    ]
    return "\n\n".join(parts)[:max_chars]


def _render_territory_summary(board: dict[str, Any]) -> str:
    territories = board.get("territories", {})
    lines = ["## Territories"]
    lines.extend(
        f"  {name}: P{info.get('owner', '?')} a{info.get('armies', '?')}"
        for name, info in territories.items()
    )
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
