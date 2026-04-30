"""Monte Carlo baseline estimation with Wilson score confidence intervals."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.base import Agent
from src.agents.random_agent import RandomAgent
from src.env import RisikoEnv
from src.multi_agent import MultiAgentRunner


@dataclass(frozen=True)
class BaselineResult:
    """Statistics for a single baseline matchup."""

    matchup: str
    win_rate_left: float
    win_rate_right: float
    draw_rate: float
    avg_length: float
    ci_left: tuple[float, float]
    ci_right: tuple[float, float]

    @staticmethod
    def to_csv_all(results: dict[str, BaselineResult], path: Path) -> None:
        """Export all matchup results to a single CSV file."""
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "matchup",
                    "win_rate_left",
                    "win_rate_right",
                    "draw_rate",
                    "avg_length",
                    "ci_left_low",
                    "ci_left_high",
                    "ci_right_low",
                    "ci_right_high",
                ]
            )
            for r in results.values():
                writer.writerow(
                    [
                        r.matchup,
                        r.win_rate_left,
                        r.win_rate_right,
                        r.draw_rate,
                        r.avg_length,
                        r.ci_left[0],
                        r.ci_left[1],
                        r.ci_right[0],
                        r.ci_right[1],
                    ]
                )


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score confidence interval for a binomial proportion.

    Args:
        successes: Number of successful trials.
        n: Total number of trials.
        z: Z-score for the desired confidence level (default 1.96 for 95%).

    Returns:
        ``(lower_bound, upper_bound)`` in [0, 1].
    """
    if n == 0:
        raise ValueError("n must be > 0")
    if successes == 0:
        return 0.0, _wilson_upper(0.0, n, z)
    if successes == n:
        return _wilson_lower(1.0, n, z), 1.0
    p_hat = successes / n
    return _wilson_lower(p_hat, n, z), _wilson_upper(p_hat, n, z)


def _wilson_lower(p_hat: float, n: int, z: float) -> float:
    denom = 1 + z * z / n
    centre = p_hat + z * z / (2 * n)
    width = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return max(0.0, (centre - width) / denom)


def _wilson_upper(p_hat: float, n: int, z: float) -> float:
    denom = 1 + z * z / n
    centre = p_hat + z * z / (2 * n)
    width = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return min(1.0, (centre + width) / denom)


def run_baseline_tournament(
    matchups: Sequence[tuple[str, Agent, Agent]],
    n_games: int,
    n_players: int,
    seed: int = 42,
    max_turns: int = 10_000,
) -> dict[str, BaselineResult]:
    """Run a set of baseline matchups and return results with confidence intervals.

    Args:
        matchups: List of ``(name, agent_left, agent_right)`` tuples.
        n_games: Number of games per matchup.
        n_players: Total players per game (remaining seats filled with random agents).
        seed: Base RNG seed.
        max_turns: Turn cap per game.

    Returns:
        Dict mapping matchup name to ``BaselineResult``.
    """
    results: dict[str, BaselineResult] = {}
    for name, left, right in matchups:
        games = _run_matchup(left, right, n_games, n_players, seed, max_turns)
        results[name] = _build_baseline(name, games, n_games)
    return results


def _run_matchup(
    left: Agent,
    right: Agent,
    n_games: int,
    n_players: int,
    seed: int,
    max_turns: int,
) -> list[Any]:
    env = RisikoEnv(n_players=n_players)
    agents = _build_agent_list(left, right, n_players)
    runner = MultiAgentRunner(env, agents, max_turns=max_turns)
    return runner.run_games(n=n_games, seed=seed)


def _build_agent_list(left: Agent, right: Agent, n_players: int) -> list[Agent]:
    agents: list[Agent] = [left, right]
    for i in range(2, n_players):
        agents.append(RandomAgent(seed=200 + i))
    return agents


def _build_baseline(name: str, games: list[Any], n_games: int) -> BaselineResult:
    wins_left = sum(1 for g in games if g.winner == 0)
    wins_right = sum(1 for g in games if g.winner == 1)
    draws = n_games - wins_left - wins_right
    lengths = [g.n_turns for g in games]
    return BaselineResult(
        matchup=name,
        win_rate_left=wins_left / n_games,
        win_rate_right=wins_right / n_games,
        draw_rate=draws / n_games,
        avg_length=float(np.mean(lengths)) if lengths else 0.0,
        ci_left=wilson_ci(wins_left, n_games),
        ci_right=wilson_ci(wins_right, n_games),
    )
