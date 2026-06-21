"""Reward-shaping configuration for the Risiko environment.

Dense rewards are individually ablatable via the dataclass fields,
allowing clean ablation studies without source changes.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RewardConfig"]


@dataclass(frozen=True)
class RewardConfig:
    """Immutable reward-shaping coefficients.

    Sparse signals are normalised to ±1 so gradient magnitudes stay
    comparable across experiments. Dense coefficients are kept small
    (order 1e-2 or less) to avoid overwhelming the sparse signal.
    Setting any field to 0.0 effectively disables that component.
    """

    sparse_win: float = 1.0
    sparse_loss: float = -1.0
    dense_territory_delta: float = 0.01
    dense_continent_bonus_delta: float = 0.05
    dense_army_ratio: float = 0.005
    dense_elimination_bonus: float = 0.1
    invalid_action_penalty: float = -0.01
    # Per-step living penalty applied every non-terminal decision. Negative
    # values make stalling costly so the agent is pushed to *finish* games
    # (attack, eliminate, win) instead of turtling to a margin lead. Default 0.0
    # keeps the original behaviour; set a small negative (e.g. -1e-3) to break
    # the non-resolving turtle equilibrium seen in self-play.
    step_penalty: float = 0.0
    # Potential-based progress shaping (Ng et al.): each step adds
    # ``dense_progress_weight * (margin_now - margin_prev)`` where margin is the
    # territory lead over the strongest opponent in [-1, 1]. Rewards every move
    # that increases relative dominance (conquering / eliminating, esp. vs the
    # leader) and telescopes to the win, giving a dense gradient toward winning
    # without distorting the optimal policy. The main lever for breaking the
    # turtle equilibrium. 0.0 disables.
    dense_progress_weight: float = 0.0
    # Terminal signal when a game hits the turn cap without an elimination
    # (a draw). The learner's final transition receives
    # ``terminal_margin_weight * margin`` where ``margin`` is the territory
    # lead over the strongest opponent, normalised to [-1, 1]. Gives the value
    # head a graded win/loss target on games that never resolve, instead of a
    # bootstrap-across-games gap. Set to 0.0 to disable (pure sparse).
    terminal_margin_weight: float = 0.5
