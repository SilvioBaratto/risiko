"""Tests for head-to-head evaluation harness and Monte Carlo baselines."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.agents.random_agent import RandomAgent
from training.evaluate import EvaluationResult, evaluate_agents
from training.monte_carlo import BaselineResult, run_baseline_tournament, wilson_ci

# ------------------------------------------------------------------
# Wilson score confidence intervals
# ------------------------------------------------------------------


class TestWilsonCI:
    """Wilson score intervals for binomial proportions."""

    def test_perfect_win_rate(self) -> None:
        low, high = wilson_ci(10, 10)
        assert 0 < low < 1
        assert low < high
        assert high == 1.0

    def test_zero_wins(self) -> None:
        low, high = wilson_ci(0, 10)
        assert low == 0.0
        assert 0 < high < 1

    def test_50_percent(self) -> None:
        low, high = wilson_ci(50, 100)
        assert 0.4 < low < 0.5
        assert 0.5 < high < 0.6
        assert low < high

    def test_interval_width_decreases_with_n(self) -> None:
        low_10, high_10 = wilson_ci(5, 10)
        low_100, high_100 = wilson_ci(50, 100)
        assert (high_100 - low_100) < (high_10 - low_10)

    def test_n_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n"):
            wilson_ci(0, 0)


# ------------------------------------------------------------------
# EvaluationResult
# ------------------------------------------------------------------


class TestEvaluationResult:
    """EvaluationResult captures head-to-head statistics."""

    def test_fields_populated(self) -> None:
        result = EvaluationResult(
            win_rate_a=0.5,
            win_rate_b=0.3,
            draw_rate=0.2,
            avg_length=50.0,
            territory_curves=[np.array([[10, 10], [12, 8]])],
            elimination_orders=[[0], [1]],
            ci_a=(0.3, 0.7),
            ci_b=(0.1, 0.5),
        )
        assert result.win_rate_a == 0.5
        assert result.win_rate_b == 0.3
        assert result.draw_rate == 0.2
        assert result.avg_length == 50.0


# ------------------------------------------------------------------
# evaluate_agents
# ------------------------------------------------------------------


class TestEvaluateAgents:
    """Head-to-head evaluation between two agents."""

    def test_returns_evaluation_result(self) -> None:
        result = evaluate_agents(
            RandomAgent(seed=1),
            RandomAgent(seed=2),
            n_games=5,
            n_players=2,
            seed=42,
            max_turns=20,
        )
        assert isinstance(result, EvaluationResult)

    def test_win_rates_sum_to_one(self) -> None:
        result = evaluate_agents(
            RandomAgent(seed=1),
            RandomAgent(seed=2),
            n_games=10,
            n_players=2,
            seed=42,
            max_turns=20,
        )
        assert pytest.approx(result.win_rate_a + result.win_rate_b + result.draw_rate) == 1.0

    def test_ci_fields_present(self) -> None:
        result = evaluate_agents(
            RandomAgent(seed=1),
            RandomAgent(seed=2),
            n_games=5,
            n_players=2,
            seed=42,
            max_turns=20,
        )
        assert isinstance(result.ci_a, tuple)
        assert len(result.ci_a) == 2
        assert isinstance(result.ci_b, tuple)
        assert len(result.ci_b) == 2

    def test_territory_curves_shape(self) -> None:
        result = evaluate_agents(
            RandomAgent(seed=1),
            RandomAgent(seed=2),
            n_games=5,
            n_players=2,
            seed=42,
            max_turns=20,
        )
        assert len(result.territory_curves) == 5
        for curve in result.territory_curves:
            assert curve.ndim == 2  # (turns, n_players)
            assert curve.shape[1] == 2

    def test_elimination_orders_present(self) -> None:
        result = evaluate_agents(
            RandomAgent(seed=1),
            RandomAgent(seed=2),
            n_games=5,
            n_players=2,
            seed=42,
            max_turns=20,
        )
        assert len(result.elimination_orders) == 5

    def test_three_player_game(self) -> None:
        result = evaluate_agents(
            RandomAgent(seed=1),
            RandomAgent(seed=2),
            n_games=3,
            n_players=3,
            seed=42,
            max_turns=20,
        )
        assert result.win_rate_a + result.win_rate_b + result.draw_rate == pytest.approx(1.0)
        # Territory curves should have 3 columns for 3 players
        for curve in result.territory_curves:
            assert curve.shape[1] == 3

    def test_csv_export(self) -> None:
        result = evaluate_agents(
            RandomAgent(seed=1),
            RandomAgent(seed=2),
            n_games=3,
            n_players=2,
            seed=42,
            max_turns=20,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            result.to_csv(path)
            with path.open("r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert "win_rate_a" in rows[0]
            assert "win_rate_b" in rows[0]
            assert "draw_rate" in rows[0]
        finally:
            path.unlink()

    def test_avg_length_is_positive(self) -> None:
        result = evaluate_agents(
            RandomAgent(seed=1),
            RandomAgent(seed=2),
            n_games=3,
            n_players=2,
            seed=42,
            max_turns=20,
        )
        assert result.avg_length > 0


# ------------------------------------------------------------------
# Monte Carlo baseline
# ------------------------------------------------------------------


class TestRunBaselineTournament:
    """Monte Carlo baseline tournament with multiple matchups."""

    def test_returns_dict_of_results(self) -> None:
        matchups = [
            ("random_vs_random", RandomAgent(seed=1), RandomAgent(seed=2)),
        ]
        results = run_baseline_tournament(
            matchups,
            n_games=3,
            n_players=2,
            seed=42,
            max_turns=20,
        )
        assert isinstance(results, dict)
        assert "random_vs_random" in results
        assert isinstance(results["random_vs_random"], BaselineResult)

    def test_multiple_matchups(self) -> None:
        matchups = [
            ("a", RandomAgent(seed=1), RandomAgent(seed=2)),
            ("b", RandomAgent(seed=3), RandomAgent(seed=4)),
        ]
        results = run_baseline_tournament(
            matchups,
            n_games=3,
            n_players=2,
            seed=42,
            max_turns=20,
        )
        assert len(results) == 2
        assert "a" in results
        assert "b" in results

    def test_baseline_result_has_ci(self) -> None:
        matchups = [
            ("test", RandomAgent(seed=1), RandomAgent(seed=2)),
        ]
        results = run_baseline_tournament(
            matchups,
            n_games=5,
            n_players=2,
            seed=42,
            max_turns=20,
        )
        baseline = results["test"]
        assert isinstance(baseline.ci_left, tuple)
        assert isinstance(baseline.ci_right, tuple)
        assert baseline.ci_left[0] < baseline.ci_left[1]
        assert baseline.ci_right[0] < baseline.ci_right[1]

    def test_csv_export(self) -> None:
        matchups = [
            ("test", RandomAgent(seed=1), RandomAgent(seed=2)),
        ]
        results = run_baseline_tournament(
            matchups,
            n_games=3,
            n_players=2,
            seed=42,
            max_turns=20,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            BaselineResult.to_csv_all(results, path)
            with path.open("r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert "matchup" in rows[0]
            assert "win_rate_left" in rows[0]
        finally:
            path.unlink()
