"""GameState mutable container and phase constants for the Risiko environment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "GameState",
    "PHASE_TRADE",
    "PHASE_REINFORCE",
    "PHASE_ATTACK",
    "PHASE_CAPTURE_MOVE",
    "PHASE_FORTIFY",
    "MAX_CARDS",
]

PHASE_TRADE = 0
PHASE_REINFORCE = 1
PHASE_ATTACK = 2
PHASE_CAPTURE_MOVE = 3
PHASE_FORTIFY = 4

MAX_CARDS = 5


@dataclass
class GameState:
    """Mutable container for all episode state.

    Single source of randomness: every module draws from ``rng`` only —
    never from ``random``, ``np.random``, or ``torch`` directly.
    """

    territory_owner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    armies: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    cards: list[list[int]] = field(default_factory=list)
    deck: list[int] = field(default_factory=list)
    discard_pile: list[int] = field(default_factory=list)
    trade_count: int = 0
    current_player: int = 0
    phase: int = PHASE_REINFORCE
    reinforcements_remaining: int = 0
    turn_capture: int = 0
    eliminated: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    n_players: int = 0
    last_capture_dice: int = 0
    last_attacker: int = -1
    last_defender: int = -1
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
