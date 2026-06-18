"""Tests for multi-agent runner, Transition, and GameResult dataclasses."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from src.agents.base import Agent
from src.agents.random_agent import RandomAgent
from src.env import RisikoEnv
from src.multi_agent import GameResult, MultiAgentRunner, Transition

# ------------------------------------------------------------------
# Transition
# ------------------------------------------------------------------


class TestTransitionFields:
    """Transition is a frozen dataclass with exact field contract."""

    def _make_transition(self, **overrides):
        defaults = {
            "obs": {"territory_owner": np.zeros(42, dtype=np.int32)},
            "action": {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0},
            "reward": 1.0,
        }
        return Transition(**{**defaults, **overrides})

    def test_when_transition_constructed_then_obs_stored(self):
        obs = {"territory_owner": np.ones(42, dtype=np.int32)}
        t = self._make_transition(obs=obs)
        assert t.obs is obs

    def test_when_transition_constructed_then_action_stored(self):
        action = {"action_type": 2, "param_a": 1, "param_b": 2, "param_c": 3, "param_d": 0}
        t = self._make_transition(action=action)
        assert t.action == action

    def test_when_transition_constructed_then_reward_stored(self):
        t = self._make_transition(reward=-0.5)
        assert t.reward == -0.5

    def test_when_log_prob_omitted_then_default_is_nan(self):
        t = self._make_transition()
        assert math.isnan(t.log_prob)

    def test_when_value_omitted_then_default_is_nan(self):
        t = self._make_transition()
        assert math.isnan(t.value)

    def test_when_legal_actions_omitted_then_default_is_empty_list(self):
        t = self._make_transition()
        assert t.legal_actions == []

    def test_when_legal_actions_omitted_then_instances_do_not_share_list(self):
        t1 = self._make_transition()
        t2 = self._make_transition()
        assert t1.legal_actions is not t2.legal_actions

    def test_when_log_prob_provided_then_value_stored(self):
        t = self._make_transition(log_prob=-1.23)
        assert t.log_prob == pytest.approx(-1.23)

    def test_when_value_provided_then_value_stored(self):
        t = self._make_transition(value=0.75)
        assert t.value == pytest.approx(0.75)


class TestTransitionIter:
    """Transition.__iter__ yields (obs, action, reward) for 3-tuple unpacking."""

    def _base(self):
        obs = {"territory_owner": np.zeros(42, dtype=np.int32)}
        action = {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}
        return obs, action

    def test_when_unpacked_then_yields_three_items(self):
        obs, action = self._base()
        items = list(Transition(obs=obs, action=action, reward=2.0))
        assert len(items) == 3

    def test_when_unpacked_then_first_item_is_obs(self):
        obs, action = self._base()
        unpacked_obs, _, _ = Transition(obs=obs, action=action, reward=2.0)
        assert unpacked_obs is obs

    def test_when_unpacked_then_second_item_is_action(self):
        obs, action = self._base()
        _, unpacked_action, _ = Transition(obs=obs, action=action, reward=2.0)
        assert unpacked_action is action

    def test_when_unpacked_then_third_item_is_reward(self):
        obs, action = self._base()
        _, _, reward = Transition(obs=obs, action=action, reward=-0.1)
        assert reward == pytest.approx(-0.1)

    def test_when_iterated_in_for_loop_then_unpacks_correctly(self):
        obs = {"territory_owner": np.zeros(42, dtype=np.int32)}
        action = {"action_type": 1, "param_a": 3, "param_b": 2, "param_c": 0, "param_d": 0}
        trajectories = [Transition(obs=obs, action=action, reward=0.5)]
        for o, a, r in trajectories:
            assert o is obs
            assert a is action
            assert r == pytest.approx(0.5)


class TestTransitionFrozen:
    """Transition must be frozen — mutation must raise an error."""

    def test_when_reward_mutated_then_error_raised(self):
        t = Transition(
            obs={},
            action={"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0},
            reward=1.0,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            t.reward = 99.0


class TestTransitionImport:
    """Both dataclasses importable from src.multi_agent."""

    def test_when_transition_imported_then_class_is_available(self):
        from src.multi_agent import Transition

        assert Transition is not None

    def test_when_game_result_imported_then_class_is_available(self):
        from src.multi_agent import GameResult

        assert GameResult is not None


class TestGameResultContract:
    """GameResult frozen dataclass field contract."""

    def _make_result(self, **overrides):
        defaults = {
            "winner": 0,
            "n_turns": 10,
            "territory_history": [np.zeros(3, dtype=np.int32)],
            "elimination_order": [],
            "card_trade_turns": [],
            "action_log": [],
            "trajectories": [],
        }
        return GameResult(**{**defaults, **overrides})

    def test_when_card_trade_hand_sizes_omitted_then_default_is_empty_list(self):
        assert self._make_result().card_trade_hand_sizes == []

    def test_when_card_trade_hand_sizes_omitted_then_instances_do_not_share_list(self):
        r1 = self._make_result()
        r2 = self._make_result()
        assert r1.card_trade_hand_sizes is not r2.card_trade_hand_sizes

    def test_when_winner_is_none_then_stored_as_none(self):
        assert self._make_result(winner=None).winner is None

    def test_when_trajectories_provided_then_stored(self):
        obs = {"territory_owner": np.zeros(42, dtype=np.int32)}
        action = {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}
        t = Transition(obs=obs, action=action, reward=0.0)
        r = self._make_result(trajectories=[t])
        assert len(r.trajectories) == 1
        assert r.trajectories[0] is t

    def test_when_game_result_mutated_then_error_raised(self):
        r = self._make_result()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            r.winner = 99


# ------------------------------------------------------------------
# GameResult
# ------------------------------------------------------------------


class TestGameResult:
    """GameResult captures structured outcome data."""

    def test_creation(self) -> None:
        result = GameResult(
            winner=0,
            n_turns=50,
            territory_history=[np.array([14, 14, 14])],
            elimination_order=[],
            card_trade_turns=[],
            action_log=[],
            trajectories=[],
        )
        assert result.winner == 0
        assert result.n_turns == 50
        assert len(result.territory_history) == 1

    def test_draw_winner_is_none(self) -> None:
        result = GameResult(
            winner=None,
            n_turns=100,
            territory_history=[],
            elimination_order=[],
            card_trade_turns=[],
            action_log=[],
            trajectories=[],
        )
        assert result.winner is None

    def test_territory_history_shape(self) -> None:
        hist = [np.array([10, 16, 16]), np.array([8, 18, 16])]
        result = GameResult(
            winner=1,
            n_turns=2,
            territory_history=hist,
            elimination_order=[2, 0],
            card_trade_turns=[1],
            action_log=[{"action_type": 5}, {"action_type": 2}],
            trajectories=[],
        )
        assert len(result.territory_history) == result.n_turns
        assert result.elimination_order == [2, 0]
        assert result.winner == 1


# ------------------------------------------------------------------
# MultiAgentRunner — initialization
# ------------------------------------------------------------------


class TestMultiAgentRunnerInit:
    """Runner construction validates inputs."""

    def test_requires_agents_match_players(self) -> None:
        env = RisikoEnv(n_players=3)
        agents = [RandomAgent(seed=1), RandomAgent(seed=2)]
        with pytest.raises(ValueError, match="3 agents"):
            MultiAgentRunner(env, agents)

    def test_requires_at_least_two_players(self) -> None:
        env = RisikoEnv(n_players=2)
        agents = [RandomAgent(seed=1), RandomAgent(seed=2)]
        runner = MultiAgentRunner(env, agents)
        assert runner._n_players == 2

    def test_max_turns_default(self) -> None:
        env = RisikoEnv(n_players=2)
        agents = [RandomAgent(seed=1), RandomAgent(seed=2)]
        runner = MultiAgentRunner(env, agents)
        assert runner._max_turns == 1000


# ------------------------------------------------------------------
# MultiAgentRunner — single game execution
# ------------------------------------------------------------------


class TestMultiAgentRunnerRunGame:
    """Runner plays complete games and returns structured results."""

    @pytest.fixture
    def two_player_agents(self) -> list[Agent]:
        return [RandomAgent(seed=1), RandomAgent(seed=2)]

    def test_game_completes_without_crash(self, two_player_agents: list[Agent]) -> None:
        env = RisikoEnv(n_players=2)
        runner = MultiAgentRunner(env, two_player_agents, max_turns=30)
        result = runner.run_game(seed=42)
        # Random agents may not eliminate each other within max_turns
        assert result.winner is None or isinstance(result.winner, int)
        assert result.n_turns > 0

    def test_territory_history_tracks_control(self, two_player_agents: list[Agent]) -> None:
        env = RisikoEnv(n_players=2)
        runner = MultiAgentRunner(env, two_player_agents, max_turns=30)
        result = runner.run_game(seed=42)
        # Each territory_history entry should sum to NUM_TERRITORIES
        for counts in result.territory_history:
            assert counts.sum() == 42

    def test_elimination_order_is_list(self, two_player_agents: list[Agent]) -> None:
        env = RisikoEnv(n_players=3)
        agents = [RandomAgent(seed=1), RandomAgent(seed=2), RandomAgent(seed=3)]
        runner = MultiAgentRunner(env, agents, max_turns=30)
        result = runner.run_game(seed=42)
        assert isinstance(result.elimination_order, list)
        if result.winner is not None:
            assert result.winner not in result.elimination_order

    def test_action_log_records_actions(self, two_player_agents: list[Agent]) -> None:
        env = RisikoEnv(n_players=2)
        runner = MultiAgentRunner(env, two_player_agents, max_turns=30)
        result = runner.run_game(seed=42)
        assert len(result.action_log) == result.n_turns
        for entry in result.action_log:
            assert "action_type" in entry
            assert "player" in entry

    def test_trajectories_present(self, two_player_agents: list[Agent]) -> None:
        env = RisikoEnv(n_players=2)
        runner = MultiAgentRunner(env, two_player_agents, max_turns=30)
        result = runner.run_game(seed=42)
        assert len(result.trajectories) == result.n_turns
        for obs, action, reward in result.trajectories:
            assert isinstance(obs, dict)
            assert isinstance(action, dict)
            assert isinstance(reward, float)

    def test_determinism_same_seed(self, two_player_agents: list[Agent]) -> None:
        env_a = RisikoEnv(n_players=2)
        env_b = RisikoEnv(n_players=2)
        agents_a = [RandomAgent(seed=1), RandomAgent(seed=2)]
        agents_b = [RandomAgent(seed=1), RandomAgent(seed=2)]
        runner_a = MultiAgentRunner(env_a, agents_a, max_turns=50)
        runner_b = MultiAgentRunner(env_b, agents_b, max_turns=50)
        result_a = runner_a.run_game(seed=99)
        result_b = runner_b.run_game(seed=99)
        assert result_a.winner == result_b.winner
        assert result_a.n_turns == result_b.n_turns
        assert result_a.elimination_order == result_b.elimination_order
        for arr_a, arr_b in zip(
            result_a.territory_history, result_b.territory_history, strict=False
        ):
            np.testing.assert_array_equal(arr_a, arr_b)

    def test_different_seeds_differ(self, two_player_agents: list[Agent]) -> None:
        env_a = RisikoEnv(n_players=2)
        env_b = RisikoEnv(n_players=2)
        agents_a = [RandomAgent(seed=1), RandomAgent(seed=2)]
        agents_b = [RandomAgent(seed=1), RandomAgent(seed=2)]
        runner_a = MultiAgentRunner(env_a, agents_a, max_turns=50)
        runner_b = MultiAgentRunner(env_b, agents_b, max_turns=50)
        result_a = runner_a.run_game(seed=1)
        result_b = runner_b.run_game(seed=2)
        # Different env seeds should produce different territory distributions
        any_differs = any(
            not np.array_equal(a, b)
            for a, b in zip(result_a.territory_history, result_b.territory_history, strict=False)
        )
        assert any_differs

    def test_max_turns_truncation(self, two_player_agents: list[Agent]) -> None:
        env = RisikoEnv(n_players=2)
        runner = MultiAgentRunner(env, two_player_agents, max_turns=5)
        result = runner.run_game(seed=42)
        # max_turns caps PLAYER-TURNS, not env steps: the game stops once the
        # env has completed 5 turns. n_turns counts env steps (transitions), so
        # it spans several steps per turn and exceeds the player-turn cap.
        assert env.state.turns_elapsed == 5
        assert result.n_turns >= 5
        # Turn-cap truncation is a draw — no single survivor → winner=None.
        assert result.winner is None


# ------------------------------------------------------------------
# MultiAgentRunner — agent error fallback
# ------------------------------------------------------------------


class CrashingAgent(Agent):
    """Agent that always raises on act()."""

    def act(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> dict[str, int]:
        raise RuntimeError("boom")

    def act_with_meta(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
        action = self.act(obs, legal_actions, deterministic=deterministic)
        return action, torch.tensor(float("nan")), torch.tensor(float("nan"))


class TestMultiAgentRunnerFallback:
    """Runner falls back to random when an agent crashes."""

    def test_fallback_on_crash(self) -> None:
        env = RisikoEnv(n_players=2)
        agents = [CrashingAgent(), RandomAgent(seed=1)]
        runner = MultiAgentRunner(env, agents, max_turns=30)
        # Should not raise; CrashingAgent falls back to random
        result = runner.run_game(seed=42)
        assert result.winner is None or isinstance(result.winner, int)

    def test_fallback_agent_is_random(self) -> None:
        env = RisikoEnv(n_players=2)
        agents = [CrashingAgent(), RandomAgent(seed=1)]
        runner = MultiAgentRunner(env, agents, max_turns=30)
        result = runner.run_game(seed=42)
        assert result.n_turns > 0


# ------------------------------------------------------------------
# _GameRecorder — elimination detection
# ------------------------------------------------------------------


class TestGameRecorderEliminationDetection:
    """_GameRecorder must detect and record elimination order from obs transitions."""

    def _make_obs(self, eliminated: list[int]) -> dict[str, np.ndarray]:
        return {
            "territory_owner": np.zeros(42, dtype=np.int32),
            "armies": np.zeros(42, dtype=np.int32),
            "phase": np.array(0, dtype=np.int32),
            "current_player": np.array(0, dtype=np.int32),
            "cards": np.zeros((5, 4), dtype=np.int32),
            "continent_control": np.zeros(6, dtype=np.int32),
            "trade_count": np.array(0, dtype=np.int32),
            "reinforcements_remaining": np.array(0, dtype=np.int32),
            "turn_capture": np.array(0, dtype=np.int32),
            "n_players": np.array(3, dtype=np.int32),
            "eliminated": np.array(eliminated, dtype=np.int32),
        }

    def test_detects_single_elimination(self) -> None:
        from src.multi_agent import _GameRecorder

        env = RisikoEnv(n_players=3)
        recorder = _GameRecorder(env, 3)
        recorder.record_step(self._make_obs([0, 0, 0, 0, 0, 0]), {"action_type": 5}, 0.0)
        recorder.record_step(self._make_obs([0, 0, 1, 0, 0, 0]), {"action_type": 5}, 0.0)
        assert recorder._elimination_order == [2]

    def test_detects_multiple_eliminations_same_step(self) -> None:
        from src.multi_agent import _GameRecorder

        env = RisikoEnv(n_players=4)
        recorder = _GameRecorder(env, 4)
        recorder.record_step(self._make_obs([0, 0, 0, 0, 0, 0]), {"action_type": 5}, 0.0)
        recorder.record_step(self._make_obs([0, 0, 1, 1, 0, 0]), {"action_type": 5}, 0.0)
        assert recorder._elimination_order == [2, 3]

    def test_no_false_positives_on_repeat_obs(self) -> None:
        from src.multi_agent import _GameRecorder

        env = RisikoEnv(n_players=3)
        recorder = _GameRecorder(env, 3)
        recorder.record_step(self._make_obs([0, 0, 1, 0, 0, 0]), {"action_type": 5}, 0.0)
        recorder.record_step(self._make_obs([0, 0, 1, 0, 0, 0]), {"action_type": 5}, 0.0)
        assert recorder._elimination_order == [2]

    def test_elimination_order_in_game_result(self) -> None:
        from src.multi_agent import _GameRecorder

        env = RisikoEnv(n_players=3)
        recorder = _GameRecorder(env, 3)
        recorder.record_step(self._make_obs([0, 0, 0, 0, 0, 0]), {"action_type": 5}, 0.0)
        recorder.record_step(self._make_obs([0, 0, 1, 0, 0, 0]), {"action_type": 5}, 0.0)
        recorder.record_step(self._make_obs([0, 1, 1, 0, 0, 0]), {"action_type": 5}, 0.0)
        result = recorder.build_result()
        assert result.elimination_order == [2, 1]


class TestMultiAgentRunnerRunGames:
    """Batch execution returns multiple results."""

    def test_returns_correct_count(self) -> None:
        env = RisikoEnv(n_players=2)
        agents = [RandomAgent(seed=1), RandomAgent(seed=2)]
        runner = MultiAgentRunner(env, agents, max_turns=30)
        results = runner.run_games(n=3, seed=42)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, GameResult)

    def test_sequential_seeds(self) -> None:
        env = RisikoEnv(n_players=2)
        agents = [RandomAgent(seed=1), RandomAgent(seed=2)]
        runner = MultiAgentRunner(env, agents, max_turns=30)
        results = runner.run_games(n=3, seed=100)
        # Each game should get a different seed
        winners = [r.winner for r in results]
        assert all(w in {0, 1, None} for w in winners)
