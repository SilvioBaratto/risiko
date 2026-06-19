"""Cross-game reputation tracking for the eval-only diplomacy layer.

In-memory only; call reset() between eval sessions. Two ReputationBook
instances share no state — each is fully independent.
"""

from __future__ import annotations

from collections import defaultdict

_PENALTY_PER_BETRAYAL: float = 0.2


class ReputationBook:
    """Accumulates betrayal events across games within one eval session."""

    def __init__(self) -> None:
        """Initialise with an empty betrayal ledger."""
        self._counts: dict[int, int] = defaultdict(int)

    def record_betrayal(self, player: int) -> None:
        """Increment the betrayal count for player by one."""
        self._counts[player] += 1

    def betrayal_count(self, player: int) -> int:
        """Return total betrayals recorded for player in this session."""
        return self._counts[player]

    def trust_penalty(self, player: int) -> float:
        """Return a non-negative penalty in [0, 1]; monotone in betrayal count."""
        return min(1.0, self._counts[player] * _PENALTY_PER_BETRAYAL)

    def reset(self) -> None:
        """Clear all records; call between eval sessions."""
        self._counts.clear()

    def update(self, game_result: object = None) -> None:
        """Post-game hook called once per game by run_social_eval.

        Betrayals are recorded during the game via record_betrayal(), so this
        hook is a no-op; it exists so run_social_eval can signal game boundaries
        and callers can verify cross-game carry via the same instance.
        """


__all__ = ["ReputationBook"]
