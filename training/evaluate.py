"""Head-to-head evaluation harness for Risiko agents."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.base import Agent
from src.agents.random_agent import RandomAgent
from src.env import RisikoEnv
from src.multi_agent import MultiAgentRunner
from training.monte_carlo import wilson_ci


@dataclass(frozen=True)
class EvaluationResult:
    """Statistics from a head-to-head evaluation."""

    win_rate_a: float
    win_rate_b: float
    draw_rate: float
    avg_length: float
    territory_curves: list[np.ndarray]
    elimination_orders: list[list[int]]
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]

    def to_csv(self, path: Path) -> None:
        """Export a single-row summary to CSV."""
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "win_rate_a",
                    "win_rate_b",
                    "draw_rate",
                    "avg_length",
                    "ci_a_low",
                    "ci_a_high",
                    "ci_b_low",
                    "ci_b_high",
                ]
            )
            writer.writerow(
                [
                    self.win_rate_a,
                    self.win_rate_b,
                    self.draw_rate,
                    self.avg_length,
                    self.ci_a[0],
                    self.ci_a[1],
                    self.ci_b[0],
                    self.ci_b[1],
                ]
            )


def evaluate_agents(
    agent_a: Agent,
    agent_b: Agent,
    n_games: int,
    n_players: int,
    seed: int = 42,
    max_turns: int = 10_000,
) -> EvaluationResult:
    """Run *n_games* between *agent_a* (as player 0) and *agent_b* (as player 1).

    For *n_players* > 2 the remaining seats are filled with random agents.
    """
    results = _run_matchup(agent_a, agent_b, n_games, n_players, seed, max_turns)
    return _build_evaluation(results, n_games)


def _run_matchup(
    agent_a: Agent,
    agent_b: Agent,
    n_games: int,
    n_players: int,
    seed: int,
    max_turns: int,
) -> list[Any]:
    env = RisikoEnv(n_players=n_players)
    agents = _build_agent_list(agent_a, agent_b, n_players)
    runner = MultiAgentRunner(env, agents, max_turns=max_turns)
    return runner.run_games(n=n_games, seed=seed)


def _build_agent_list(agent_a: Agent, agent_b: Agent, n_players: int) -> list[Agent]:
    agents: list[Agent] = [agent_a, agent_b]
    for i in range(2, n_players):
        agents.append(RandomAgent(seed=100 + i))
    return agents


def _build_evaluation(results: list[Any], n_games: int) -> EvaluationResult:
    wins_a = sum(1 for r in results if r.winner == 0)
    wins_b = sum(1 for r in results if r.winner == 1)
    draws = n_games - wins_a - wins_b
    lengths = [r.n_turns for r in results]
    curves = [np.array(r.territory_history) for r in results]
    eliminations = [r.elimination_order for r in results]
    return EvaluationResult(
        win_rate_a=wins_a / n_games,
        win_rate_b=wins_b / n_games,
        draw_rate=draws / n_games,
        avg_length=float(np.mean(lengths)),
        territory_curves=curves,
        elimination_orders=eliminations,
        ci_a=wilson_ci(wins_a, n_games),
        ci_b=wilson_ci(wins_b, n_games),
    )
