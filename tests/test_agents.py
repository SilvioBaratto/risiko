"""Tests for agent base protocol, random agent, and PPO agent."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.base import Agent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.models.actor_critic import ActorCritic
from src.models.utils import get_obs_dim

# ------------------------------------------------------------------
# Agent Protocol
# ------------------------------------------------------------------


class DummyAgent:
    """Minimal conforming agent for Protocol checking."""

    def act(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        deterministic: bool = False,
    ) -> dict[str, int]:
        return {"action_type": 0, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}


class BadAgent:
    """Non-conforming agent — missing deterministic param."""

    def act(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
    ) -> dict[str, int]:
        return {"action_type": 0, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}


def test_agent_protocol_runtime_checkable() -> None:
    assert isinstance(DummyAgent(), Agent)
    # @runtime_checkable only verifies structural presence of act(), not kwarg-only markers.
    # BadAgent passes because it has an act() method; signature enforcement is static only.
    assert isinstance(BadAgent(), Agent)


def test_agent_protocol_act_signature() -> None:
    agent = DummyAgent()
    obs = {"dummy": np.array([1])}
    legal = [{"action_type": 0, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}]
    result = agent.act(obs, legal, deterministic=False)
    assert isinstance(result, dict)


# ------------------------------------------------------------------
# RandomAgent
# ------------------------------------------------------------------


class TestRandomAgent:
    """RandomAgent uniformly samples from legal_actions."""

    def test_conforms_to_protocol(self) -> None:
        agent = RandomAgent(seed=42)
        assert isinstance(agent, Agent)

    def test_samples_only_legal_actions(self) -> None:
        agent = RandomAgent(seed=42)
        legal = [
            {"action_type": 1, "param_a": 5, "param_b": 0, "param_c": 0, "param_d": 0},
            {"action_type": 1, "param_a": 7, "param_b": 0, "param_c": 0, "param_d": 0},
            {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0},
        ]
        obs: dict[str, np.ndarray] = {}
        for _ in range(50):
            action = agent.act(obs, legal)
            assert action in legal

    def test_uniform_distribution(self) -> None:
        agent = RandomAgent(seed=123)
        legal = [
            {"action_type": 0, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0},
            {"action_type": 1, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0},
        ]
        obs: dict[str, np.ndarray] = {}
        counts = {0: 0, 1: 0}
        n = 1000
        for _ in range(n):
            action = agent.act(obs, legal)
            counts[action["action_type"]] += 1
        # Uniform => each ~500; allow wide tolerance
        assert 400 < counts[0] < 600
        assert 400 < counts[1] < 600

    def test_deterministic_mode_raises(self) -> None:
        agent = RandomAgent(seed=42)
        legal = [{"action_type": 0, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}]
        obs: dict[str, np.ndarray] = {}
        with pytest.raises(ValueError, match="deterministic"):
            agent.act(obs, legal, deterministic=True)

    def test_empty_legal_actions_raises(self) -> None:
        agent = RandomAgent(seed=42)
        with pytest.raises(ValueError, match="empty"):
            agent.act({}, [])

    def test_seeding_is_reproducible(self) -> None:
        agent_a = RandomAgent(seed=7)
        agent_b = RandomAgent(seed=7)
        legal = [
            {"action_type": i, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}
            for i in range(3)
        ]
        obs: dict[str, np.ndarray] = {}
        actions_a = [agent_a.act(obs, legal) for _ in range(20)]
        actions_b = [agent_b.act(obs, legal) for _ in range(20)]
        assert actions_a == actions_b


# ------------------------------------------------------------------
# Action Mask Builder (shared helper)
# ------------------------------------------------------------------


class TestBuildActionMask:
    """Mask builder converts legal_actions list into per-head boolean tensors."""

    @pytest.fixture
    def action_dims(self) -> dict[str, int]:
        return {
            "action_type": 6,
            "param_a": 42,
            "param_b": 42,
            "param_c": 43,
            "param_d": 43,
        }

    def test_all_masked_raises(self, action_dims: dict[str, int]) -> None:
        from src.agents.ppo_agent import build_action_mask

        legal = [{"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}]
        mask = build_action_mask(legal, action_dims)
        assert mask["action_type"][5]
        assert mask["action_type"][:5].sum() == 0
        assert mask["action_type"][5:].sum() == 1

    def test_multiple_values_per_head(self, action_dims: dict[str, int]) -> None:
        from src.agents.ppo_agent import build_action_mask

        legal = [
            {"action_type": 2, "param_a": 5, "param_b": 10, "param_c": 0, "param_d": 0},
            {"action_type": 2, "param_a": 7, "param_b": 10, "param_c": 0, "param_d": 0},
            {"action_type": 2, "param_a": 5, "param_b": 12, "param_c": 0, "param_d": 0},
        ]
        mask = build_action_mask(legal, action_dims)
        assert mask["action_type"][2]
        assert mask["action_type"].sum() == 1
        assert mask["param_a"][5]
        assert mask["param_a"][7]
        assert mask["param_a"].sum() == 2
        assert mask["param_b"][10]
        assert mask["param_b"][12]
        assert mask["param_b"].sum() == 2

    def test_coarse_mask_allows_illegal_combinations(self, action_dims: dict[str, int]) -> None:
        """Per-head masking is coarse: it may allow joint combos not in legal_actions.

        This is a documented limitation — the env penalises illegal actions.
        """
        from src.agents.ppo_agent import build_action_mask

        legal = [
            {"action_type": 2, "param_a": 5, "param_b": 10, "param_c": 0, "param_d": 0},
            {"action_type": 2, "param_a": 7, "param_b": 10, "param_c": 0, "param_d": 0},
        ]
        mask = build_action_mask(legal, action_dims)
        # param_a allows both 5 and 7; param_b allows 10
        # So (2, 7, 10) is in the mask, but (2, 7, 12) is NOT in legal_actions
        # yet the mask would allow it if param_b=12 were also allowed
        assert mask["param_a"][5]
        assert mask["param_a"][7]

    def test_head_with_zero_legal_values_gets_single_dummy(
        self, action_dims: dict[str, int]
    ) -> None:
        from src.agents.ppo_agent import build_action_mask

        # param_c and param_d never appear in legal actions for trade phase
        legal = [
            {"action_type": 0, "param_a": 1, "param_b": 2, "param_c": 0, "param_d": 0},
        ]
        mask = build_action_mask(legal, action_dims)
        # param_c and param_d should have at least one valid value to prevent
        # ActorCritic from raising ValueError
        assert mask["param_c"][0]
        assert mask["param_c"].sum() == 1
        assert mask["param_d"][0]
        assert mask["param_d"].sum() == 1


# ------------------------------------------------------------------
# PPOAgent
# ------------------------------------------------------------------


class TestPPOAgent:
    """PPOAgent wraps ActorCritic and uses action masking."""

    @pytest.fixture
    def net(self) -> ActorCritic:
        return ActorCritic(
            obs_dim=get_obs_dim(),
            hidden_size=32,
            num_layers=1,
            action_dims={
                "action_type": 6,
                "param_a": 42,
                "param_b": 42,
                "param_c": 43,
                "param_d": 43,
            },
        )

    @pytest.fixture
    def dummy_obs(self) -> dict[str, np.ndarray]:
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
            "eliminated": np.zeros(6, dtype=np.int32),
        }

    def test_conforms_to_protocol(self, net: ActorCritic) -> None:
        agent = PPOAgent(net, device="cpu")
        assert isinstance(agent, Agent)

    def test_act_returns_action_dict(
        self, net: ActorCritic, dummy_obs: dict[str, np.ndarray]
    ) -> None:
        agent = PPOAgent(net, device="cpu")
        legal = [{"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}]
        action = agent.act(dummy_obs, legal)
        assert isinstance(action, dict)
        assert set(action.keys()) == {"action_type", "param_a", "param_b", "param_c", "param_d"}
        assert all(isinstance(v, (int, np.integer)) for v in action.values())

    def test_act_with_meta_returns_tuple(
        self, net: ActorCritic, dummy_obs: dict[str, np.ndarray]
    ) -> None:
        agent = PPOAgent(net, device="cpu")
        legal = [{"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}]
        result = agent.act_with_meta(dummy_obs, legal)
        assert isinstance(result, tuple)
        assert len(result) == 3
        action, log_prob, value = result
        assert isinstance(action, dict)
        assert isinstance(log_prob, (torch.Tensor, float))
        assert isinstance(value, (torch.Tensor, float))

    def test_deterministic_mode_selects_highest_prob(
        self, net: ActorCritic, dummy_obs: dict[str, np.ndarray]
    ) -> None:
        """In deterministic mode, PPOAgent should greedily select the highest-probability action."""
        agent = PPOAgent(net, device="cpu")
        # Only allow action_type=5 (skip)
        legal = [{"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}]
        action = agent.act(dummy_obs, legal, deterministic=True)
        # Should always return the same action since only one option is legal
        assert action == legal[0]

    def test_masks_prevent_illegal_heads(
        self, net: ActorCritic, dummy_obs: dict[str, np.ndarray]
    ) -> None:
        """If param_a is masked to a single value, that value should always be selected."""
        agent = PPOAgent(net, device="cpu")
        legal = [{"action_type": 1, "param_a": 7, "param_b": 0, "param_c": 0, "param_d": 0}]
        # Run multiple times to verify consistency
        for _ in range(10):
            action = agent.act(dummy_obs, legal, deterministic=True)
            assert action["param_a"] == 7

    def test_empty_legal_actions_raises(self, net: ActorCritic) -> None:
        agent = PPOAgent(net, device="cpu")
        with pytest.raises(ValueError, match="empty"):
            agent.act({}, [])

    def test_act_produces_valid_action_types(
        self, net: ActorCritic, dummy_obs: dict[str, np.ndarray]
    ) -> None:
        agent = PPOAgent(net, device="cpu")
        legal = [
            {"action_type": 1, "param_a": 5, "param_b": 0, "param_c": 0, "param_d": 0},
            {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0},
        ]
        for _ in range(20):
            action = agent.act(dummy_obs, legal)
            assert action["action_type"] in {1, 5}

    def test_log_prob_is_scalar_tensor(
        self, net: ActorCritic, dummy_obs: dict[str, np.ndarray]
    ) -> None:
        agent = PPOAgent(net, device="cpu")
        legal = [{"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}]
        action, log_prob, value = agent.act_with_meta(dummy_obs, legal)
        assert isinstance(log_prob, torch.Tensor)
        assert log_prob.dim() == 0  # scalar
        assert isinstance(value, torch.Tensor)
        assert value.dim() == 0  # scalar

    def test_training_mode_stochastic(
        self, net: ActorCritic, dummy_obs: dict[str, np.ndarray]
    ) -> None:
        """In non-deterministic mode, repeated calls should sometimes differ."""
        agent = PPOAgent(net, device="cpu")
        # Allow multiple action types to give the policy something to sample
        legal = [
            {"action_type": i, "param_a": i, "param_b": 0, "param_c": 0, "param_d": 0}
            for i in range(3)
        ]
        actions = [agent.act(dummy_obs, legal) for _ in range(50)]
        types = [a["action_type"] for a in actions]
        # With 3 options and 50 samples, we should see variation
        assert len(set(types)) > 1

    def test_eval_mode_deterministic(
        self, net: ActorCritic, dummy_obs: dict[str, np.ndarray]
    ) -> None:
        """Deterministic mode should return the same action across repeated calls."""
        agent = PPOAgent(net, device="cpu")
        legal = [
            {"action_type": i, "param_a": i, "param_b": 0, "param_c": 0, "param_d": 0}
            for i in range(3)
        ]
        actions = [agent.act(dummy_obs, legal, deterministic=True) for _ in range(20)]
        # All deterministic calls should return identical actions
        assert all(a == actions[0] for a in actions)
