"""Tests for the Actor-Critic shared-trunk network."""

from __future__ import annotations

import pytest
import torch

from src.models.actor_critic import ActorCritic
from src.models.utils import get_obs_dim

_ACTION_DIMS = {
    "action_type": 6,
    "param_a": 42,
    "param_b": 42,
    "param_c": 43,
    "param_d": 43,
}


def _make_net(**kwargs) -> ActorCritic:
    """Create an ActorCritic with default or overridden kwargs."""
    defaults = {
        "obs_dim": get_obs_dim(),
        "hidden_size": 64,
        "num_layers": 2,
        "action_dims": _ACTION_DIMS,
    }
    defaults.update(kwargs)
    return ActorCritic(**defaults)


def _random_flat(batch_size: int = 1) -> torch.Tensor:
    """Generate random flattened observations."""
    return torch.randn(batch_size, get_obs_dim(), dtype=torch.float32)


class TestInit:
    """Construction and parameter sanity."""

    def test_is_nn_module(self):
        """ActorCritic must be a torch.nn.Module."""
        net = _make_net()
        assert isinstance(net, torch.nn.Module)

    def test_has_trunk_parameters(self):
        """Trunk must have learnable parameters."""
        net = _make_net(hidden_size=32, num_layers=2)
        trunk_params = list(net.trunk.parameters())
        assert len(trunk_params) > 0

    def test_has_policy_parameters(self):
        """Policy heads must have learnable parameters."""
        net = _make_net()
        for key in _ACTION_DIMS:
            assert key in net.policy.heads
            head = net.policy.heads[key]
            assert len(list(head.parameters())) > 0

    def test_has_value_parameters(self):
        """Value head must have learnable parameters."""
        net = _make_net()
        assert len(list(net.value.parameters())) > 0


class TestForward:
    """Forward pass shape and type contracts."""

    def test_returns_tuple(self):
        """forward() returns a 2-tuple."""
        net = _make_net()
        logits, value = net(_random_flat(1))
        assert isinstance(logits, dict)
        assert isinstance(value, torch.Tensor)

    def test_single_logits_shapes(self):
        """Single obs produces correct policy head shapes."""
        net = _make_net()
        logits, _ = net(_random_flat(1))
        for key, dim in _ACTION_DIMS.items():
            assert logits[key].shape == (1, dim), f"{key} shape mismatch"

    def test_batch_logits_shapes(self):
        """Batch obs produces correct batched policy head shapes."""
        net = _make_net()
        logits, _ = net(_random_flat(8))
        for key, dim in _ACTION_DIMS.items():
            assert logits[key].shape == (8, dim), f"{key} shape mismatch"

    def test_single_value_shape(self):
        """Single obs produces value shape (1, 1)."""
        net = _make_net()
        _, value = net(_random_flat(1))
        assert value.shape == (1, 1)

    def test_batch_value_shape(self):
        """Batch obs produces value shape (batch, 1)."""
        net = _make_net()
        _, value = net(_random_flat(16))
        assert value.shape == (16, 1)

    def test_logits_are_unbounded(self):
        """Raw logits are not probabilities (vary across dimensions)."""
        net = _make_net()
        logits, _ = net(_random_flat(1))
        for v in logits.values():
            # With random init, logits should vary across the action dimension
            assert v.std() > 0.001, "logits are constant — network may be broken"


class TestGetActionAndValue:
    """Sampling and evaluation contract."""

    def test_returns_four_tensors(self):
        """get_action_and_value returns 4 tensors when sampling."""
        net = _make_net()
        action, log_prob, entropy, value = net.get_action_and_value(_random_flat(1))
        assert isinstance(action, dict)
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(entropy, torch.Tensor)
        assert isinstance(value, torch.Tensor)

    def test_action_keys_match(self):
        """Sampled action has the same keys as action_dims."""
        net = _make_net()
        action, _, _, _ = net.get_action_and_value(_random_flat(1))
        assert set(action.keys()) == set(_ACTION_DIMS.keys())

    def test_action_values_are_valid_indices(self):
        """Each sampled action value is a valid index for its space."""
        net = _make_net()
        action, _, _, _ = net.get_action_and_value(_random_flat(1))
        for key, dim in _ACTION_DIMS.items():
            assert 0 <= action[key].item() < dim

    def test_batch_action_shapes(self):
        """Batch sampling returns batched action dicts."""
        net = _make_net()
        action, _, _, _ = net.get_action_and_value(_random_flat(8))
        for key in _ACTION_DIMS:
            assert action[key].shape == (8,)

    def test_log_prob_is_scalar_per_sample(self):
        """Log prob is a scalar per batch element."""
        net = _make_net()
        _, log_prob, _, _ = net.get_action_and_value(_random_flat(4))
        assert log_prob.shape == (4,)

    def test_entropy_is_scalar_per_sample(self):
        """Entropy is a scalar per batch element."""
        net = _make_net()
        _, _, entropy, _ = net.get_action_and_value(_random_flat(4))
        assert entropy.shape == (4,)

    def test_evaluate_provided_action(self):
        """get_action_and_value can evaluate a provided action."""
        net = _make_net()
        action = {k: torch.tensor([0]) for k in _ACTION_DIMS}
        _, log_prob, entropy, value = net.get_action_and_value(_random_flat(1), action=action)
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert value.shape == (1, 1)

    def test_evaluate_matches_sampling_log_prob_shape(self):
        """Evaluating a fixed action returns the same shapes as sampling."""
        net = _make_net()
        obs = _random_flat(4)
        sampled_action, sampled_lp, sampled_ent, _ = net.get_action_and_value(obs)
        _, eval_lp, eval_ent, _ = net.get_action_and_value(obs, action=sampled_action)
        assert eval_lp.shape == sampled_lp.shape
        assert eval_ent.shape == sampled_ent.shape

    def test_log_prob_is_negative(self):
        """Log prob of a valid action is negative (probability < 1)."""
        net = _make_net()
        _, log_prob, _, _ = net.get_action_and_value(_random_flat(1))
        assert log_prob.item() < 0.0


class TestActionMasks:
    """Action masking behavior."""

    def test_masked_logits_are_zeroed(self):
        """Masking zeros out forbidden action logits."""
        net = _make_net()
        obs = _random_flat(1)
        mask = {
            "action_type": torch.tensor([[True, True, False, False, False, False]]),
        }
        action, _, _, _ = net.get_action_and_value(obs, action_masks=mask)
        assert action["action_type"].item() in {0, 1}

    def test_all_false_mask_raises(self):
        """An all-False mask raises ValueError."""
        net = _make_net()
        obs = _random_flat(1)
        mask = {
            "action_type": torch.tensor([[False, False, False, False, False, False]]),
        }
        with pytest.raises(ValueError):
            net.get_action_and_value(obs, action_masks=mask)

    def test_mask_does_not_affect_unmasked_keys(self):
        """Masking one key does not affect others."""
        net = _make_net()
        obs = _random_flat(1)
        mask = {
            "action_type": torch.tensor([[True, True, False, False, False, False]]),
        }
        action, _, _, _ = net.get_action_and_value(obs, action_masks=mask)
        # param_a should still sample freely from all 42 options
        assert 0 <= action["param_a"].item() < 42


class TestGradientFlow:
    """Backpropagation contracts."""

    def test_policy_loss_updates_trunk(self):
        """Backward on policy loss updates trunk weights."""
        net = _make_net()
        obs = _random_flat(4)
        action, log_prob, _, _ = net.get_action_and_value(obs)
        loss = -log_prob.mean()
        loss.backward()
        for p in net.trunk.parameters():
            assert p.grad is not None
        # We don't actually step the optimizer; just verify grads exist

    def test_value_loss_updates_trunk(self):
        """Backward on value loss updates trunk weights."""
        net = _make_net()
        obs = _random_flat(4)
        _, _, _, value = net.get_action_and_value(obs)
        loss = value.pow(2).mean()
        loss.backward()
        for p in net.trunk.parameters():
            assert p.grad is not None

    def test_combined_loss_updates_all(self):
        """Combined policy + value loss updates trunk, policy, and value heads."""
        net = _make_net()
        obs = _random_flat(4)
        action, log_prob, entropy, value = net.get_action_and_value(obs)
        loss = -log_prob.mean() + 0.5 * value.pow(2).mean() - 0.01 * entropy.mean()
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None

    def test_trunk_weights_change_after_optimizer_step(self):
        """After optimizer step, trunk weights differ from before."""
        net = _make_net()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        trunk_before = list(net.trunk.parameters())[0].clone()
        obs = _random_flat(4)
        _, log_prob, _, value = net.get_action_and_value(obs)
        loss = -log_prob.mean() + 0.5 * value.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trunk_after = list(net.trunk.parameters())[0]
        assert not torch.allclose(trunk_before, trunk_after)
