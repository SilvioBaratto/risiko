"""Reward-shaping configuration for the Risiko environment.

Dense rewards are individually ablatable via the dataclass fields,
allowing clean ablation studies without source changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Immutable reward-shaping coefficients.

    All dense coefficients default to 1.0 so that setting any
    field to 0.0 effectively disables that signal.
    """

    sparse_win: float = 100.0
    sparse_loss: float = -100.0
    dense_territory_delta: float = 1.0
    dense_continent_bonus_delta: float = 2.0
    dense_army_ratio: float = 0.5
    dense_elimination_bonus: float = 10.0
    invalid_action_penalty: float = -1.0
