"""Deterministic social-state core for the eval-only diplomacy layer.

This module is pure Python — no LLM, no env I/O. It is safe to import in
any context (training or eval) without side effects.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Grudge:
    """Immutable record of a single attack event."""

    offender: int
    victim: int
    turn: int
    base_severity: float


@dataclass(frozen=True)
class Message:
    """Immutable snapshot of one negotiation message."""

    sender: int
    text: str


# ---------------------------------------------------------------------------
# Module-level pure helpers (keep the class body small)
# ---------------------------------------------------------------------------


def _pair(a: int, b: int) -> frozenset[int]:
    """Canonical unordered pair for alliance membership tests."""
    return frozenset({a, b})


def _decayed_severity(grudge: Grudge, now: int, decay: float) -> float:
    """Read-derived severity: base * decay^(now - turn)."""
    return grudge.base_severity * decay ** (now - grudge.turn)


_BASE_TRUST: float = 0.5
_ALLIANCE_BONUS: float = 0.5
_DEFAULT_MAX_LOG: int = 50


# ---------------------------------------------------------------------------
# DiplomacyState
# ---------------------------------------------------------------------------


class DiplomacyState:
    """Per-game social state: alliances, append-only grudge log, message history."""

    def __init__(self, decay: float = 0.9, max_log: int = _DEFAULT_MAX_LOG) -> None:
        """Initialise with a decay coefficient and bounded message log size."""
        self.alliances: set[frozenset[int]] = set()
        self.grudges: list[Grudge] = []
        self.message_log: deque[Message] = deque(maxlen=max_log)
        self.decay = decay

    def are_allied(self, a: int, b: int) -> bool:
        """Return True if a and b share an alliance (symmetric)."""
        return _pair(a, b) in self.alliances

    def form_alliance(self, a: int, b: int) -> None:
        """Add an alliance between a and b; idempotent."""
        self.alliances.add(_pair(a, b))

    def break_alliance(self, a: int, b: int) -> None:
        """Remove the alliance between a and b; no-op if absent."""
        self.alliances.discard(_pair(a, b))

    def declare_war_on(
        self, attacker: int, target: int, turn: int = 0, severity: float = 1.0
    ) -> None:
        """Break any alliance between attacker and target; record a grudge."""
        self.break_alliance(attacker, target)
        self.record_attack(offender=attacker, victim=target, turn=turn, base_severity=severity)

    def has_grudge(self, victim: int, aggressor: int) -> bool:
        """Return True if any grudge entry from aggressor against victim exists."""
        return any(g.offender == aggressor and g.victim == victim for g in self.grudges)

    def record_attack(self, offender: int, victim: int, turn: int, base_severity: float) -> None:
        """Append an immutable Grudge; never mutates existing entries."""
        self.grudges.append(
            Grudge(offender=offender, victim=victim, turn=turn, base_severity=base_severity)
        )

    def grudge_score(self, offender: int, victim: int, now: int) -> float:
        """Sum of decayed severities for matching grudges; read-only (log unchanged)."""
        return sum(
            _decayed_severity(g, now, self.decay)
            for g in self.grudges
            if g.offender == offender and g.victim == victim
        )

    def trust(self, a: int, b: int, now: int) -> float:
        """Bounded [0, 1] trust score; boosted by alliance, reduced by grudge."""
        bonus = _ALLIANCE_BONUS if self.are_allied(a, b) else 0.0
        raw = _BASE_TRUST + bonus - self.grudge_score(a, b, now)
        return max(0.0, min(1.0, raw))

    def leader(self, territory_owner: Sequence[int]) -> int:
        """Player with the most territories; lowest id breaks ties deterministically."""
        counts = np.bincount(territory_owner)
        return int(np.argmax(counts))


@dataclass
class DiplomacyNote:
    """Concise move-prompt context derived from DiplomacyState for one player/turn.

    Fields carry the ids of relevant players; callers pass a sorted list so
    the note is deterministic even if the source set has an arbitrary iteration
    order.  The rendering step (in action_prompt._render_diplomacy) also sorts,
    so unsorted inputs produce the same prompt.
    """

    allies: list[int] = field(default_factory=list)
    leader: int | None = None
    grudges: list[int] = field(default_factory=list)


def build_diplomacy_note(
    diplomacy_state: DiplomacyState,
    player: int,
    territory_owner: Sequence[int],
    n_players: int,
) -> DiplomacyNote:
    """Build a DiplomacyNote for *player* from current social state and board.

    Args:
        diplomacy_state: The live DiplomacyState for this game.
        player: The player whose move is being prompted.
        territory_owner: Flat array/list mapping territory id → owner id.
        n_players: Total number of player slots in the game.

    Returns:
        A DiplomacyNote with sorted ally and grudge lists.
    """
    others = [p for p in range(n_players) if p != player]
    board_leader = diplomacy_state.leader(territory_owner)
    return DiplomacyNote(
        allies=sorted(p for p in others if diplomacy_state.are_allied(player, p)),
        leader=board_leader if board_leader != player else None,
        grudges=sorted(p for p in others if diplomacy_state.has_grudge(player, p)),
    )


@dataclass(frozen=True)
class DiplomacyContext:
    """Context passed to LLM agents at each move step.

    Carries the live social state, the acting player's id, and the current
    board leader so agents can condition on alliance/grudge information without
    recomputing it.
    """

    state: DiplomacyState
    acting_player: int
    leader: int | None


__all__ = [
    "DiplomacyState",
    "DiplomacyNote",
    "DiplomacyContext",
    "Grudge",
    "Message",
    "build_diplomacy_note",
]
