"""TDD tests for strategy analysis from GameResult data."""

from __future__ import annotations

import numpy as np
import pytest

from src.multi_agent import GameResult
from training.strategy_analysis import StrategyAnalyzer

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_action(
    action_type: int,
    param_a: int = 0,
    param_b: int = 0,
    param_c: int = 0,
    param_d: int = 0,
    player: int = 0,
) -> dict[str, int]:
    return {
        "action_type": action_type,
        "param_a": param_a,
        "param_b": param_b,
        "param_c": param_c,
        "param_d": param_d,
        "player": player,
    }


@pytest.fixture
def simple_result() -> GameResult:
    """One game where learner (0) reinforces NA, attacks SA, fortifies."""
    return GameResult(
        winner=0,
        n_turns=10,
        territory_history=[np.array([21, 21]) for _ in range(10)],
        elimination_order=[1],
        card_trade_turns=[2, 7],
        action_log=[
            # Turn 1: reinforce Alaska (0 = North America)
            _make_action(1, param_a=0, param_b=3, player=0),
            _make_action(5, player=1),  # opponent skip
            # Turn 2: trade + reinforce
            _make_action(0, player=0),
            _make_action(1, param_a=1, param_b=2, player=0),  # Alberta = NA
            _make_action(5, player=1),
            # Turn 3: attack Brazil (10 = South America) from Venezuela (12 = SA)
            _make_action(2, param_a=12, param_b=10, param_c=3, player=0),
            _make_action(3, param_a=12, param_b=10, param_c=2, player=0),  # capture
            _make_action(5, player=1),
            # Turn 4: fortify Alaska -> Alberta (distance 1)
            _make_action(4, param_a=0, param_b=1, param_c=2, player=0),
            _make_action(5, player=1),
            # Turns 5-10: mostly skips
            *[a for _ in range(6) for a in (_make_action(5, player=0), _make_action(5, player=1))],
        ],
        trajectories=[],
        card_trade_hand_sizes=[3, 4],
    )


@pytest.fixture
def no_learner_actions_result() -> GameResult:
    """Game where learner takes no actions (all skips)."""
    return GameResult(
        winner=1,
        n_turns=5,
        territory_history=[np.array([0, 42]) for _ in range(5)],
        elimination_order=[0],
        card_trade_turns=[],
        action_log=[_make_action(5, player=p) for p in [0, 1] for _ in range(5)],
        trajectories=[],
        card_trade_hand_sizes=[],
    )


@pytest.fixture
def multi_game_results() -> list[GameResult]:
    """Two games for averaging tests."""
    g1 = GameResult(
        winner=0,
        n_turns=6,
        territory_history=[np.array([21, 21]) for _ in range(6)],
        elimination_order=[1],
        card_trade_turns=[1, 4],
        action_log=[
            _make_action(1, param_a=0, param_b=3, player=0),  # NA reinforce
            _make_action(5, player=1),
            _make_action(2, param_a=12, param_b=10, param_c=3, player=0),  # SA attack
            _make_action(5, player=1),
            _make_action(4, param_a=0, param_b=1, param_c=1, player=0),  # NA fortify
            _make_action(5, player=1),
            *[a for _ in range(6) for a in (_make_action(5, player=0), _make_action(5, player=1))],
        ],
        trajectories=[],
        card_trade_hand_sizes=[3, 5],
    )
    g2 = GameResult(
        winner=0,
        n_turns=6,
        territory_history=[np.array([21, 21]) for _ in range(6)],
        elimination_order=[1],
        card_trade_turns=[2],
        action_log=[
            _make_action(1, param_a=38, param_b=2, player=0),  # Australia reinforce
            _make_action(5, player=1),
            _make_action(2, param_a=38, param_b=39, param_c=3, player=0),  # Aus attack
            _make_action(3, param_a=38, param_b=39, param_c=2, player=0),  # capture
            _make_action(5, player=1),
            *[a for _ in range(8) for a in (_make_action(5, player=0), _make_action(5, player=1))],
        ],
        trajectories=[],
        card_trade_hand_sizes=[4],
    )
    return [g1, g2]


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


class TestStrategyAnalyzerInit:
    """Construction and init tests."""

    def test_accepts_game_results(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result], learner_id=0)
        assert len(analyzer._game_results) == 1

    def test_default_learner_id_is_zero(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        assert analyzer._learner_id == 0


# ------------------------------------------------------------------
# Continent priority
# ------------------------------------------------------------------


class TestContinentPriority:
    """Continent priority heuristics."""

    def test_returns_dict_with_continent_names(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        result = analyzer.continent_priority()
        assert all(
            name in result
            for name in [
                "North America",
                "South America",
                "Europe",
                "Africa",
                "Asia",
                "Australia",
            ]
        )

    def test_reinforcements_counted(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        result = analyzer.continent_priority()
        # Two reinforces: Alaska (NA) and Alberta (NA)
        assert result["North America"]["reinforce"] == 2

    def test_attacks_counted(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        result = analyzer.continent_priority()
        # One attack: Venezuela (12=SA) -> Brazil (10=SA)
        assert result["South America"]["attack"] == 1

    def test_percentages_relative_to_total_actions(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        result = analyzer.continent_priority()
        # 2 reinforces (both NA) + 1 attack (SA) = 3 continent-targeted actions
        assert result["North America"]["reinforce_pct"] == pytest.approx(2 / 3)
        assert result["South America"]["attack_pct"] == pytest.approx(1 / 3)

    def test_empty_when_no_actions(self, no_learner_actions_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([no_learner_actions_result])
        result = analyzer.continent_priority()
        assert all(v["reinforce"] == 0 and v["attack"] == 0 for v in result.values())


# ------------------------------------------------------------------
# Attack aggressiveness
# ------------------------------------------------------------------


class TestAttackAggressiveness:
    """Attack aggressiveness metrics."""

    def test_avg_attacks_per_turn(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        result = analyzer.attack_aggressiveness()
        # 1 attack in 10 turns
        assert result["avg_attacks_per_turn"] == pytest.approx(1 / 10)

    def test_capture_success_rate(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        result = analyzer.attack_aggressiveness()
        # 1 capture / 1 attack = 100%
        assert result["capture_success_rate"] == pytest.approx(1.0)

    def test_zero_when_no_attacks(self, no_learner_actions_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([no_learner_actions_result])
        result = analyzer.attack_aggressiveness()
        assert result["avg_attacks_per_turn"] == 0.0
        assert result["capture_success_rate"] == 0.0

    def test_multi_game_averaging(self, multi_game_results: list[GameResult]) -> None:
        analyzer = StrategyAnalyzer(multi_game_results)
        result = analyzer.attack_aggressiveness()
        # g1: 1 attack in 6 turns, g2: 1 attack in 6 turns
        assert result["avg_attacks_per_turn"] == pytest.approx(2 / 12)
        # g1: 0 captures / 1 attack, g2: 1 capture / 1 attack
        assert result["capture_success_rate"] == pytest.approx(0.5)


# ------------------------------------------------------------------
# Card trade timing
# ------------------------------------------------------------------


class TestCardTradeTiming:
    """Card trade timing patterns."""

    def test_avg_hand_size(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        result = analyzer.card_trade_timing()
        # hand sizes [3, 4]
        assert result["avg_hand_size_at_trade"] == pytest.approx(3.5)

    def test_trade_frequency_per_phase(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        result = analyzer.card_trade_timing()
        # 10 turns total: early=turns 0-3, mid=4-6, late=7-9
        # trades at turn 2 (early) and turn 7 (late)
        assert result["trades_per_turn_early"] == pytest.approx(1 / 4)
        assert result["trades_per_turn_mid"] == pytest.approx(0.0)
        assert result["trades_per_turn_late"] == pytest.approx(1 / 3)

    def test_empty_when_no_trades(self, no_learner_actions_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([no_learner_actions_result])
        result = analyzer.card_trade_timing()
        assert result["avg_hand_size_at_trade"] == 0.0
        assert result["trades_per_turn_early"] == 0.0


# ------------------------------------------------------------------
# Fortification patterns
# ------------------------------------------------------------------


class TestFortificationPatterns:
    """Fortification movement patterns."""

    def test_avg_armies_moved(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        result = analyzer.fortification_patterns()
        # 1 fortify with 2 armies
        assert result["avg_armies_moved"] == pytest.approx(2.0)

    def test_avg_distance(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        result = analyzer.fortification_patterns()
        # Alaska (0) -> Alberta (1) distance = 1
        assert result["avg_fortify_distance"] == pytest.approx(1.0)

    def test_zero_when_no_fortifies(self, no_learner_actions_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([no_learner_actions_result])
        result = analyzer.fortification_patterns()
        assert result["avg_armies_moved"] == 0.0
        assert result["avg_fortify_distance"] == 0.0


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------


class TestReport:
    """JSON report generation."""

    def test_returns_dict(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        report = analyzer.report()
        assert isinstance(report, dict)

    def test_contains_all_heuristics(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        report = analyzer.report()
        # Report is flat — check for representative keys from each heuristic
        assert any(k.startswith("north_america") for k in report)
        assert "avg_attacks_per_turn" in report
        assert "avg_hand_size_at_trade" in report
        assert "avg_armies_moved" in report

    def test_is_json_serializable(self, simple_result: GameResult) -> None:
        import json

        analyzer = StrategyAnalyzer([simple_result])
        report = analyzer.report()
        json_str = json.dumps(report)
        assert isinstance(json_str, str)

    def test_flat_structure(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        report = analyzer.report()
        # All top-level keys should have primitive values (no nested dicts)
        # continent_priority is nested, but we flatten it in the report
        for k, v in report.items():
            assert not isinstance(v, dict), f"{k} is a nested dict"

    def test_multi_game_report(self, multi_game_results: list[GameResult]) -> None:
        analyzer = StrategyAnalyzer(multi_game_results)
        report = analyzer.report()
        assert "avg_attacks_per_turn" in report
        assert report["avg_attacks_per_turn"] == pytest.approx(2 / 12)


# ------------------------------------------------------------------
# Attack distance distribution
# ------------------------------------------------------------------


class TestAttackDistanceDistribution:
    """BFS attack-distance distribution per game phase."""

    def test_when_adjacent_attack_distance_1_in_all_bucket(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        dist = analyzer.attack_distance_distribution()
        assert dist["all"].get("1") == 1

    def test_when_adjacent_attack_all_mean_is_1(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        dist = analyzer.attack_distance_distribution()
        assert dist["all"]["mean"] == pytest.approx(1.0)

    def test_when_adjacent_attack_all_total_is_1(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        dist = analyzer.attack_distance_distribution()
        assert dist["all"]["total"] == 1

    def test_when_multiple_attacks_all_mean_is_correct(
        self, multi_game_results: list[GameResult]
    ) -> None:
        analyzer = StrategyAnalyzer(multi_game_results)
        dist = analyzer.attack_distance_distribution()
        # g1: Venezuela(12)->Brazil(10) dist=1, g2: EAustralia(38)->Indonesia(39) dist=2
        assert dist["all"]["mean"] == pytest.approx(1.5)

    def test_when_multiple_attacks_all_total_is_2(
        self, multi_game_results: list[GameResult]
    ) -> None:
        analyzer = StrategyAnalyzer(multi_game_results)
        dist = analyzer.attack_distance_distribution()
        assert dist["all"]["total"] == 2

    def test_when_no_attacks_total_is_zero(self, no_learner_actions_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([no_learner_actions_result])
        dist = analyzer.attack_distance_distribution()
        assert dist["all"]["total"] == 0
        assert dist["all"]["mean"] == pytest.approx(0.0)

    def test_when_no_attacks_all_phase_keys_present(
        self, no_learner_actions_result: GameResult
    ) -> None:
        analyzer = StrategyAnalyzer([no_learner_actions_result])
        dist = analyzer.attack_distance_distribution()
        assert set(dist.keys()) == {"early", "mid", "late", "all"}

    def test_when_early_attack_falls_in_early_bucket(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        dist = analyzer.attack_distance_distribution()
        # Attack at action_log index 5 out of 22 entries → early phase
        assert dist["early"]["total"] == 1

    def test_when_report_includes_attack_dist_mean_keys(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        report = analyzer.report()
        assert "attack_dist_early_mean" in report
        assert "attack_dist_mid_mean" in report
        assert "attack_dist_late_mean" in report
        assert "attack_dist_all_mean" in report

    def test_when_report_attack_dist_all_mean_is_correct(self, simple_result: GameResult) -> None:
        analyzer = StrategyAnalyzer([simple_result])
        report = analyzer.report()
        assert report["attack_dist_all_mean"] == pytest.approx(1.0)
