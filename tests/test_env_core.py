"""Core tests for the Risiko Gymnasium environment."""

from __future__ import annotations

import numpy as np
import pytest

from src.env import (
    PHASE_ATTACK,
    PHASE_CAPTURE_MOVE,
    PHASE_FORTIFY,
    PHASE_REINFORCE,
    PHASE_TRADE,
    RisikoEnv,
)
from src.utils.constants import ADJACENCY


def _skip() -> dict[str, int]:
    return {
        "action_type": 5,
        "param_a": 0,
        "param_b": 0,
        "param_c": 0,
        "param_d": 0,
    }


def _reinforce(terr: int, rem: int) -> dict[str, int]:
    return {"action_type": 1, "param_a": terr, "param_b": rem, "param_c": 0, "param_d": 0}


class TestEnvInterface:
    """Gymnasium interface contract."""

    def test_is_gymnasium_env(self):
        """RisikoEnv must inherit from gymnasium.Env."""
        import gymnasium as gym

        env = RisikoEnv(n_players=3)
        assert isinstance(env, gym.Env)

    def test_observation_space_is_dict(self):
        """Observation space must be a Dict with required keys."""
        env = RisikoEnv(n_players=3)
        obs = env.observation_space
        import gymnasium as gym

        assert isinstance(obs, gym.spaces.Dict)
        for key in (
            "territory_owner",
            "armies",
            "phase",
            "current_player",
            "cards",
            "continent_control",
            "trade_count",
            "reinforcements_remaining",
            "turn_capture",
            "n_players",
            "eliminated",
        ):
            assert key in obs.spaces, f"missing observation key: {key}"

    def test_action_space_is_dict(self):
        """Action space must be a Dict with required keys."""
        env = RisikoEnv(n_players=3)
        act = env.action_space
        import gymnasium as gym

        assert isinstance(act, gym.spaces.Dict)
        for key in ("action_type", "param_a", "param_b", "param_c", "param_d"):
            assert key in act.spaces, f"missing action key: {key}"

    def test_reset_returns_obs_and_info(self):
        """Reset must return (observation, info)."""
        env = RisikoEnv(n_players=3)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert "legal_actions" in info

    def test_reset_is_deterministic(self):
        """reset(seed=42) must produce identical observations."""
        env1 = RisikoEnv(n_players=3)
        obs1, _ = env1.reset(seed=42)
        env2 = RisikoEnv(n_players=3)
        obs2, _ = env2.reset(seed=42)
        for key in obs1:
            assert np.array_equal(obs1[key], obs2[key])

    def test_step_returns_correct_tuple(self):
        """Step must return (obs, reward, terminated, truncated, info)."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        action = _skip()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)


class TestBoardSetup:
    """Initial board setup properties."""

    def test_all_territories_have_owner(self):
        """After reset, every territory must have an owner."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        owners = obs["territory_owner"]
        assert len(owners) == 42
        assert all(owners >= 0)
        assert all(owners < 3)

    def test_every_territory_has_at_least_one_army(self):
        """After reset, every territory must have at least 1 army."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        armies = obs["armies"]
        assert all(armies >= 1)

    def test_total_armies_match_starting_count(self):
        """Total armies on board must equal starting armies × players."""
        from src.utils.constants import STARTING_ARMIES

        for n in range(2, 7):
            env = RisikoEnv(n_players=n)
            obs, _ = env.reset(seed=42)
            total = int(obs["armies"].sum())
            expected = STARTING_ARMIES[n] * n
            assert total == expected

    def test_no_eliminated_players_at_start(self):
        """No player should be eliminated at the start."""
        env = RisikoEnv(n_players=4)
        obs, _ = env.reset(seed=42)
        assert all(obs["eliminated"][:4] == 0)

    def test_phase_is_reinforce_or_trade_after_reset(self):
        """After reset, the current phase must be REINFORCE or TRADE."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        assert obs["phase"] in (PHASE_REINFORCE, PHASE_TRADE)


class TestReinforcements:
    """Reinforcement computation and placement."""

    def test_reinforcements_positive(self):
        """Reinforcements remaining must be positive at turn start."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        assert obs["reinforcements_remaining"] > 0

    def test_placing_armies_reduces_reinforcements(self):
        """Placing armies must reduce reinforcements remaining."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        while obs["phase"] == PHASE_TRADE:
            env.step(_skip())
            obs = env._get_obs()
        start = int(obs["reinforcements_remaining"])
        player = int(obs["current_player"])
        terr = int(np.where(obs["territory_owner"] == player)[0][0])
        action = {
            "action_type": 1,
            "param_a": terr,
            "param_b": 1,
            "param_c": 0,
            "param_d": 0,
        }
        obs, _, _, _, _ = env.step(action)
        assert obs["reinforcements_remaining"] == start - 1

    def test_placing_zero_armies_is_invalid(self):
        """Placing 0 armies must be an invalid action."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        while obs["phase"] == PHASE_TRADE:
            env.step(_skip())
            obs = env._get_obs()
        player = int(obs["current_player"])
        terr = int(np.where(obs["territory_owner"] == player)[0][0])
        action = {
            "action_type": 1,
            "param_a": terr,
            "param_b": 0,
            "param_c": 0,
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0


class TestAttackPhase:
    """Attack validation and dice resolution."""

    def test_can_end_attack_phase(self):
        """Ending attack phase must advance to fortify."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        # Skip to attack phase
        while obs["phase"] != PHASE_ATTACK:
            if obs["phase"] == PHASE_TRADE:
                env.step(_skip())
            elif obs["phase"] == PHASE_REINFORCE:
                player = int(obs["current_player"])
                terr = int(np.where(obs["territory_owner"] == player)[0][0])
                rem = int(obs["reinforcements_remaining"])
                env.step(_reinforce(terr, rem))
            obs = env._get_obs()
        assert obs["phase"] == PHASE_ATTACK
        env.step({"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0})
        obs = env._get_obs()
        assert obs["phase"] == PHASE_FORTIFY

    def test_cannot_attack_non_adjacent(self):
        """Attacking a non-adjacent territory must be invalid."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        while obs["phase"] != PHASE_ATTACK:
            if obs["phase"] == PHASE_TRADE:
                env.step(_skip())
            elif obs["phase"] == PHASE_REINFORCE:
                player = int(obs["current_player"])
                terr = int(np.where(obs["territory_owner"] == player)[0][0])
                rem = int(obs["reinforcements_remaining"])
                env.step(_reinforce(terr, rem))
            obs = env._get_obs()
        player = int(obs["current_player"])
        attacker = int(np.where((obs["territory_owner"] == player) & (obs["armies"] >= 2))[0][0])
        non_adjacent = [t for t in range(42) if t not in ADJACENCY[attacker]]
        target = non_adjacent[0]
        action = {
            "action_type": 2,
            "param_a": attacker,
            "param_b": target,
            "param_c": 1,
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0

    def test_cannot_attack_with_one_army(self):
        """Attacking from a territory with 1 army must be invalid."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        while obs["phase"] != PHASE_ATTACK:
            if obs["phase"] == PHASE_TRADE:
                env.step(_skip())
            elif obs["phase"] == PHASE_REINFORCE:
                player = int(obs["current_player"])
                terr = int(np.where(obs["territory_owner"] == player)[0][0])
                rem = int(obs["reinforcements_remaining"])
                env.step(_reinforce(terr, rem))
            obs = env._get_obs()
        player = int(obs["current_player"])
        single_army = int(np.where((obs["territory_owner"] == player) & (obs["armies"] == 1))[0][0])
        enemies = [t for t in ADJACENCY[single_army] if obs["territory_owner"][t] != player]
        target = enemies[0] if enemies else 0
        action = {
            "action_type": 2,
            "param_a": single_army,
            "param_b": target,
            "param_c": 1,
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0

    def test_attack_changes_armies(self):
        """A valid attack must change army counts on both sides."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        while obs["phase"] != PHASE_ATTACK:
            if obs["phase"] == PHASE_TRADE:
                env.step(_skip())
            elif obs["phase"] == PHASE_REINFORCE:
                player = int(obs["current_player"])
                terr = int(np.where(obs["territory_owner"] == player)[0][0])
                rem = int(obs["reinforcements_remaining"])
                env.step(_reinforce(terr, rem))
            obs = env._get_obs()
        player = int(obs["current_player"])
        candidates = np.where((obs["territory_owner"] == player) & (obs["armies"] >= 3))[0]
        for attacker in candidates:
            enemies = [t for t in ADJACENCY[attacker] if obs["territory_owner"][t] != player]
            if enemies:
                target = enemies[0]
                break
        else:
            pytest.skip("No valid attack found")
        pre_attacker = int(obs["armies"][attacker])
        pre_target = int(obs["armies"][target])
        action = {
            "action_type": 2,
            "param_a": int(attacker),
            "param_b": int(target),
            "param_c": 1,
            "param_d": 0,
        }
        obs, _, _, _, _ = env.step(action)
        post_attacker = int(obs["armies"][attacker])
        post_target = int(obs["armies"][target])
        assert post_attacker != pre_attacker or post_target != pre_target


class TestCards:
    """Card system properties."""

    def test_no_player_has_more_than_five_cards(self):
        """No player should ever hold more than 5 cards."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        cards = obs["cards"]
        non_zero = np.sum(np.any(cards != 0, axis=1))
        assert non_zero <= 5

    def test_trade_advances_trade_count(self):
        """Trading cards must advance the trade counter."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        if obs["phase"] == PHASE_TRADE:
            env.state.cards[env.state.current_player] = [0, 1, 2]
            action = {
                "action_type": 0,
                "param_a": 0,
                "param_b": 1,
                "param_c": 2,
                "param_d": 0,
            }
            obs, _, _, _, _ = env.step(action)
            assert obs["trade_count"] >= 1


class TestFortify:
    """Fortification phase properties."""

    def test_can_skip_fortify(self):
        """Skipping fortification must advance to next player's turn."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        start_player = int(obs["current_player"])
        # Get to fortify phase by ending attack
        while obs["phase"] != PHASE_FORTIFY:
            if obs["phase"] == PHASE_TRADE:
                env.step(_skip())
            elif obs["phase"] == PHASE_REINFORCE:
                player = int(obs["current_player"])
                terr = int(np.where(obs["territory_owner"] == player)[0][0])
                rem = int(obs["reinforcements_remaining"])
                env.step(_reinforce(terr, rem))
            elif obs["phase"] == PHASE_ATTACK:
                env.step(_skip())
            elif obs["phase"] == PHASE_CAPTURE_MOVE:
                env.step(
                    {
                        "action_type": 3,
                        "param_a": env.state.last_capture_dice,
                        "param_b": 0,
                        "param_c": 0,
                        "param_d": 0,
                    }
                )
            obs = env._get_obs()
        assert obs["phase"] == PHASE_FORTIFY
        env.step({"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0})
        obs = env._get_obs()
        assert obs["current_player"] != start_player
        assert obs["phase"] in (PHASE_TRADE, PHASE_REINFORCE)


class TestWinDetection:
    """Win and elimination detection."""

    def test_win_terminated_is_false_at_start(self):
        """Terminated must be False after reset."""
        env = RisikoEnv(n_players=3)
        _, _ = env.reset(seed=42)
        action = {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}
        _, _, terminated, _, _ = env.step(action)
        assert not terminated

    def test_terminated_true_when_one_player_left(self):
        """When all opponents are eliminated, terminated must be True."""
        env = RisikoEnv(n_players=2)
        obs, _ = env.reset(seed=42)
        # Give player 0 all territories and eliminate player 1
        env.state.territory_owner[:] = 0
        env.state.armies[:] = 10
        env.state.eliminated[1] = 1
        # Advance through current player's turn with valid actions
        while obs["phase"] == PHASE_TRADE:
            env.step(_skip())
            obs = env._get_obs()
        # Place reinforcements
        player = int(obs["current_player"])
        terr = int(np.where(obs["territory_owner"] == player)[0][0])
        rem = int(obs["reinforcements_remaining"])
        env.step({"action_type": 1, "param_a": terr, "param_b": rem, "param_c": 0, "param_d": 0})
        # End attack
        env.step({"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0})
        # End fortify
        _, _, terminated, _, _ = env.step(_skip())
        assert terminated


class TestInvalidActions:
    """Invalid action handling."""

    def test_invalid_action_returns_penalty(self):
        """An invalid action must return a negative reward."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        # Get to attack phase
        while obs["phase"] != PHASE_ATTACK:
            if obs["phase"] == PHASE_TRADE:
                env.step(_skip())
            elif obs["phase"] == PHASE_REINFORCE:
                player = int(obs["current_player"])
                terr = int(np.where(obs["territory_owner"] == player)[0][0])
                rem = int(obs["reinforcements_remaining"])
                env.step(_reinforce(terr, rem))
            obs = env._get_obs()
        player = int(obs["current_player"])
        attacker = int(np.where((obs["territory_owner"] == player) & (obs["armies"] >= 2))[0][0])
        enemies = [t for t in ADJACENCY[attacker] if obs["territory_owner"][t] != player]
        target = enemies[0] if enemies else 0
        action = {
            "action_type": 2,
            "param_a": attacker,
            "param_b": target,
            "param_c": 0,  # 0 dice — invalid
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0


class TestLegalActions:
    """Legal action computation."""

    def test_info_contains_legal_actions(self):
        """Info dict must contain legal_actions after every step."""
        env = RisikoEnv(n_players=3)
        _, info = env.reset(seed=42)
        assert "legal_actions" in info
        assert isinstance(info["legal_actions"], list)

    def test_legal_actions_non_empty(self):
        """Legal actions must never be empty."""
        env = RisikoEnv(n_players=3)
        obs, info = env.reset(seed=42)
        assert len(info["legal_actions"]) > 0
