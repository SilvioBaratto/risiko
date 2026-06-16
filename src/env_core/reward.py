"""Dense + sparse per-step reward calculator for the Risiko environment.

Public API
----------
snapshot(state)                                  -> Snapshot
compute_reward(state, player, prev_snapshot, cfg) -> float

The module is intentionally stateless.  All mutable game data flows in via
``state`` and ``prev_snapshot``; all coefficients flow in via ``cfg``.
``snapshot()`` works with both the real ``GameState`` (numpy arrays) and any
duck-typed state object that exposes ``.territory_owner``, ``.armies``, and
``.eliminated`` (used in tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.constants import CONTINENT_BONUSES, CONTINENTS
from src.utils.reward_config import RewardConfig

__all__ = ["Snapshot", "snapshot", "compute_reward", "is_eliminated"]


@dataclass
class Snapshot:
    """Immutable copies of the three state arrays needed for delta computation.

    Always stored as plain Python containers so downstream helpers can use
    uniform dict/set operations regardless of the source state type.
    """

    territory_owner: dict[int, int]
    armies: dict[int, int]
    eliminated: set[int]


# ── Public functions ──────────────────────────────────────────────────────────


def snapshot(state) -> Snapshot:
    """Return independent copies of territory_owner, armies, and eliminated."""
    to, ar, el = state.territory_owner, state.armies, state.eliminated
    if isinstance(to, np.ndarray):
        return Snapshot(
            territory_owner={i: int(to[i]) for i in range(len(to))},
            armies={i: int(ar[i]) for i in range(len(ar))},
            eliminated={i for i, v in enumerate(el) if v},
        )
    return Snapshot(
        territory_owner=dict(to),
        armies=dict(ar),
        eliminated=set(el),
    )


def is_eliminated(state, player: int) -> bool:
    """Return True when *player* is eliminated in *state*."""
    el = state.eliminated
    if isinstance(el, np.ndarray):
        return bool(el[player])
    return player in el


def compute_reward(
    state,
    player: int,
    prev_snapshot: Snapshot,
    cfg: RewardConfig,
) -> float:
    """Return the shaped per-step reward for *player*.

    Returns ``cfg.sparse_loss`` immediately when the player is eliminated;
    otherwise sums the four enabled dense components.
    """
    curr = snapshot(state)
    if player in curr.eliminated:
        return cfg.sparse_loss
    reward = _territory_delta(curr, prev_snapshot, player, cfg)
    reward += _continent_bonus_delta(curr, prev_snapshot, player, cfg)
    reward += _army_ratio(curr, player, cfg)
    reward += _elimination_bonus(curr, prev_snapshot, player, cfg)
    return reward


# ── Private helpers (each ≤ 10 lines) ────────────────────────────────────────


def _territory_delta(curr: Snapshot, prev: Snapshot, player: int, cfg: RewardConfig) -> float:
    curr_n = sum(1 for p in curr.territory_owner.values() if p == player)
    prev_n = sum(1 for p in prev.territory_owner.values() if p == player)
    return cfg.dense_territory_delta * (curr_n - prev_n)


def _controlled_continent_bonus(territory_owner: dict[int, int], player: int) -> int:
    return sum(
        bonus
        for name, bonus in CONTINENT_BONUSES.items()
        if all(territory_owner.get(t) == player for t in CONTINENTS[name])
    )


def _continent_bonus_delta(curr: Snapshot, prev: Snapshot, player: int, cfg: RewardConfig) -> float:
    delta = _controlled_continent_bonus(curr.territory_owner, player) - _controlled_continent_bonus(
        prev.territory_owner, player
    )
    return cfg.dense_continent_bonus_delta * delta


def _army_ratio(curr: Snapshot, player: int, cfg: RewardConfig) -> float:
    total = sum(curr.armies.values())
    if total == 0:
        return 0.0
    player_armies = sum(a for t, a in curr.armies.items() if curr.territory_owner.get(t) == player)
    return cfg.dense_army_ratio * (player_armies / total)


def _elimination_bonus(curr: Snapshot, prev: Snapshot, player: int, cfg: RewardConfig) -> float:
    newly_eliminated = curr.eliminated - prev.eliminated
    return cfg.dense_elimination_bonus * len(newly_eliminated)
