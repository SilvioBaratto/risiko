"""Pretrained-policy validation gate: win-rate vs random-agent baseline.

Loads a BC checkpoint, plays it against an all-RandomAgent board over N games,
and returns a frozen result with the Wilson CI and a ``passed`` flag.
Gate passes when the Wilson CI **lower bound** exceeds the supplied threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

from risiko_rl.agent_loader import load_agent
from src.agents.random_agent import RandomAgent
from src.utils.log import get_logger
from training.evaluate import evaluate_agents

_log = get_logger("bc_eval")


@dataclass(frozen=True)
class PretrainedEvalResult:
    """Frozen evaluation result from ``evaluate_pretrained``."""

    win_rate: float
    ci: tuple[float, float]
    n_games: int
    baseline: float
    passed: bool


def evaluate_pretrained(
    checkpoint_path: str = "models/pretrained.pt",
    *,
    n_games: int,
    n_players: int = 6,
    seed: int,
    max_turns: int,
    gate: float = 0.16,
) -> PretrainedEvalResult:
    """Evaluate a pretrained checkpoint against random agents and apply the gate.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint to evaluate.
        n_games: Number of evaluation games.
        n_players: Total number of players (agent + random opponents).
        seed: Random seed for reproducibility.
        max_turns: Maximum turns per game.
        gate: Wilson CI lower-bound threshold; ``passed`` iff ``ci[0] > gate``.

    Returns:
        Frozen :class:`PretrainedEvalResult`; callers decide what to do with it.
    """
    agent = load_agent(checkpoint_path)
    opponent = RandomAgent()
    result = evaluate_agents(agent, opponent, n_games, n_players, seed, max_turns)
    ci = result.ci_a
    baseline = 1.0 / n_players
    passed = ci[0] > gate
    _log.info(
        "eval: win_rate=%.3f ci=(%.3f, %.3f) gate=%.3f passed=%s",
        result.win_rate_a,
        ci[0],
        ci[1],
        gate,
        passed,
    )
    return PretrainedEvalResult(
        win_rate=result.win_rate_a,
        ci=ci,
        n_games=n_games,
        baseline=baseline,
        passed=passed,
    )
