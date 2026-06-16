"""Stateless board-setup functions for the Risiko environment.

Each public function mutates the passed ``GameState`` in-place (except
``new_state``).  All randomness flows exclusively through ``state.rng`` so
two calls with the same seed produce byte-identical boards.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.env_core.state import (
    PHASE_REINFORCE,
    PHASE_TRADE,
    GameState,
)
from src.utils.constants import (
    CONTINENT_BONUSES,
    CONTINENTS,
    NUM_TERRITORIES,
    STARTING_ARMIES,
)

__all__ = [
    "new_state",
    "distribute_territories",
    "place_initial_armies",
    "compute_reinforcements",
    "controls_continent",
    "set_initial_phase",
    "Phase",
    "STARTING_ARMIES",
]

_NUM_CARDS = 44


class Phase:
    """Integer phase constants re-exported under a class namespace."""

    TRADE = PHASE_TRADE
    REINFORCE = PHASE_REINFORCE


def new_state(n_players: int, seed: int | None = None) -> GameState:
    """Return a fresh ``GameState`` with zeroed arrays and a seeded RNG.

    Args:
        n_players: Number of players (2–6).
        seed:      RNG seed for full reproducibility.
    """
    return GameState(
        territory_owner=np.zeros(NUM_TERRITORIES, dtype=np.int32),
        armies=np.zeros(NUM_TERRITORIES, dtype=np.int32),
        cards=[[] for _ in range(n_players)],
        deck=list(range(_NUM_CARDS)),
        discard_pile=[],
        trade_count=0,
        current_player=0,
        phase=PHASE_REINFORCE,
        reinforcements_remaining=0,
        turn_capture=0,
        eliminated=np.zeros(n_players, dtype=np.int32),
        n_players=n_players,
        rng=np.random.default_rng(seed),
    )


def distribute_territories(state: GameState) -> None:
    """Shuffle the deck then assign each territory to a player with 1 army.

    The deck is shuffled first so per-seed card-draw order varies across seeds
    while remaining reproducible for the same seed.
    """
    state.rng.shuffle(state.deck)
    order = np.arange(NUM_TERRITORIES)
    state.rng.shuffle(order)
    for idx, terr in enumerate(order):
        state.territory_owner[terr] = idx % state.n_players
        state.armies[terr] = 1


def place_initial_armies(state: GameState) -> None:
    """Distribute surplus armies so each player totals ``STARTING_ARMIES[n_players]``.

    Every territory retains ≥ 1 army because ``distribute_territories`` already
    seeded each with 1 and only *additions* are placed here.
    """
    per_player = STARTING_ARMIES[state.n_players]
    placed = np.bincount(state.territory_owner, minlength=state.n_players)
    remaining = per_player - placed
    for player in range(state.n_players):
        owned = np.where(state.territory_owner == player)[0]
        if len(owned) == 0 or remaining[player] <= 0:
            continue
        additions = state.rng.choice(owned, size=int(remaining[player]), replace=True)
        for terr in additions:
            state.armies[terr] += 1


def compute_reinforcements(state: GameState) -> None:
    """Write ``max(3, owned // 3) + continent_bonus_sum`` to ``state.reinforcements_remaining``."""
    player = state.current_player
    n_owned = int(np.sum(state.territory_owner == player))
    territory_bonus = max(3, n_owned // 3)
    continent_bonus = sum(
        bonus
        for name, bonus in CONTINENT_BONUSES.items()
        if controls_continent(state, player, name)
    )
    state.reinforcements_remaining = territory_bonus + continent_bonus


def controls_continent(state: GameState, player: int, name: str) -> bool:
    """Return True only when *player* owns every territory in *name*."""
    return all(state.territory_owner[t] == player for t in CONTINENTS[name])


def set_initial_phase(
    state: GameState,
    has_tradeable: Callable[[GameState, int], bool],
) -> None:
    """Set ``state.phase`` to TRADE or REINFORCE via the injected ``has_tradeable`` predicate."""
    player = state.current_player
    state.phase = PHASE_TRADE if has_tradeable(state, player) else PHASE_REINFORCE
