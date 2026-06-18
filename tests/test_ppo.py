"""Tests for the PPO Trainer."""

from __future__ import annotations

import torch

from src.config import PPOConfig
from src.models.actor_critic import ActorCritic
from src.models.ppo import PPOTrainer
from src.models.replay_buffer import RolloutBuffer
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


def _fill_buffer(
    buf: RolloutBuffer,
    net: ActorCritic,
    n_steps: int,
) -> None:
    """Fill a rollout buffer with synthetic transitions."""
    for _ in range(n_steps):
        obs = torch.randn(1, get_obs_dim(), dtype=torch.float32)
        action, log_prob, entropy, value = net.get_action_and_value(obs)
        buf.add(
            state={"phase": torch.ones(1, dtype=torch.int32).numpy()},
            action=action,
            reward=1.0,
            done=False,
            value=value.item(),
            log_prob=log_prob.item(),
        )


def _fill_buffer_with_obs(
    buf: RolloutBuffer,
    net: ActorCritic,
    n_steps: int,
) -> None:
    """Fill a rollout buffer with realistic observation dicts."""
    import numpy as np

    for _ in range(n_steps):
        obs = torch.randn(1, get_obs_dim(), dtype=torch.float32)
        action, log_prob, entropy, value = net.get_action_and_value(obs)
        state = {
            "territory_owner": np.random.randint(0, 6, 42, dtype=np.int32),
            "armies": np.random.randint(1, 10, 42, dtype=np.int32),
            "phase": np.int32(1),
            "current_player": np.int32(2),
            "cards": np.random.randint(0, 2, (5, 4), dtype=np.int32),
            "continent_control": np.random.randint(0, 2, 6, dtype=np.int32),
            "trade_count": np.int32(4),
            "reinforcements_remaining": np.int32(5),
            "turn_capture": np.int32(1),
            "n_players": np.int32(3),
            "eliminated": np.zeros(6, dtype=np.int32),
        }
        # Build a small legal_actions set that includes the sampled action
        # plus a few alternatives. Singleton legal_actions would collapse the
        # masked distribution to a dirac → zero entropy → tests for
        # non-zero entropy_loss would fail trivially.
        sampled = {k: int(v.item()) for k, v in action.items()}
        legal = [sampled]
        for offset in (1, 2):
            alt = {k: (sampled[k] + offset) % net.action_dims[k] for k in net.action_dims}
            legal.append(alt)
        buf.add(
            state=state,
            action=action,
            reward=1.0,
            done=False,
            value=value.item(),
            log_prob=log_prob.item(),
            legal_actions=legal,
        )


class TestInit:
    """Construction contracts."""

    def test_stores_config(self):
        """PPOTrainer stores the passed config."""
        net = _make_net()
        cfg = PPOConfig()
        trainer = PPOTrainer(net, cfg)
        assert trainer.config == cfg

    def test_creates_optimizer(self):
        """PPOTrainer creates an Adam optimizer."""
        net = _make_net()
        cfg = PPOConfig()
        trainer = PPOTrainer(net, cfg)
        assert isinstance(trainer.optimizer, torch.optim.Adam)


class TestUpdate:
    """PPO update contracts."""

    def test_update_changes_weights(self):
        """Running update changes at least one network parameter."""
        net = _make_net()
        cfg = PPOConfig(n_epochs=2, batch_size=4)
        trainer = PPOTrainer(net, cfg)
        buf = RolloutBuffer(capacity=8, device="cpu")
        _fill_buffer_with_obs(buf, net, 8)
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        before = list(net.parameters())[0].clone()
        trainer.update(buf)
        after = list(net.parameters())[0]
        assert not torch.allclose(before, after)

    def test_returns_metrics_dict(self):
        """update() returns a dict with expected keys."""
        net = _make_net()
        cfg = PPOConfig(n_epochs=1, batch_size=4)
        trainer = PPOTrainer(net, cfg)
        buf = RolloutBuffer(capacity=8, device="cpu")
        _fill_buffer_with_obs(buf, net, 8)
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        metrics = trainer.update(buf)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy_loss" in metrics
        assert "approx_kl" in metrics
        assert "clip_fraction" in metrics
        assert "explained_variance" in metrics

    def test_policy_loss_is_negative(self):
        """Policy loss is negative (we maximize clipped surrogate).

        Uses raw advantages: with per-batch advantage normalization (zero-mean)
        the surrogate is centered near 0 and its sign is no longer an invariant.
        """
        net = _make_net()
        cfg = PPOConfig(n_epochs=1, batch_size=4, normalize_advantage=False)
        trainer = PPOTrainer(net, cfg)
        buf = RolloutBuffer(capacity=8, device="cpu")
        _fill_buffer_with_obs(buf, net, 8)
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        metrics = trainer.update(buf)
        assert metrics["policy_loss"] < 0

    def test_value_loss_is_non_negative(self):
        """Value loss (MSE) is non-negative."""
        net = _make_net()
        cfg = PPOConfig(n_epochs=1, batch_size=4)
        trainer = PPOTrainer(net, cfg)
        buf = RolloutBuffer(capacity=8, device="cpu")
        _fill_buffer_with_obs(buf, net, 8)
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        metrics = trainer.update(buf)
        assert metrics["value_loss"] >= 0

    def test_entropy_loss_is_negative(self):
        """Entropy loss is negative (we maximize entropy)."""
        net = _make_net()
        cfg = PPOConfig(n_epochs=1, batch_size=4)
        trainer = PPOTrainer(net, cfg)
        buf = RolloutBuffer(capacity=8, device="cpu")
        _fill_buffer_with_obs(buf, net, 8)
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        metrics = trainer.update(buf)
        assert metrics["entropy_loss"] < 0

    def test_clip_fraction_is_non_negative(self):
        """Clip fraction is between 0 and 1."""
        net = _make_net()
        cfg = PPOConfig(n_epochs=1, batch_size=4)
        trainer = PPOTrainer(net, cfg)
        buf = RolloutBuffer(capacity=8, device="cpu")
        _fill_buffer_with_obs(buf, net, 8)
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        metrics = trainer.update(buf)
        assert 0.0 <= metrics["clip_fraction"] <= 1.0

    def test_approx_kl_is_non_negative(self):
        """Approximate KL divergence is non-negative."""
        net = _make_net()
        cfg = PPOConfig(n_epochs=1, batch_size=4)
        trainer = PPOTrainer(net, cfg)
        buf = RolloutBuffer(capacity=8, device="cpu")
        _fill_buffer_with_obs(buf, net, 8)
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        metrics = trainer.update(buf)
        assert metrics["approx_kl"] >= 0.0

    def test_explained_variance_between_minus_one_and_one(self):
        """Explained variance is in [-1, 1]."""
        net = _make_net()
        cfg = PPOConfig(n_epochs=1, batch_size=4)
        trainer = PPOTrainer(net, cfg)
        buf = RolloutBuffer(capacity=8, device="cpu")
        _fill_buffer_with_obs(buf, net, 8)
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        metrics = trainer.update(buf)
        assert -1.0 <= metrics["explained_variance"] <= 1.0


class TestGradientClipping:
    """Gradient clipping enforcement."""

    def test_gradient_norm_is_clipped(self):
        """After update, gradient norm does not exceed max_grad_norm."""
        net = _make_net()
        cfg = PPOConfig(
            n_epochs=1,
            batch_size=4,
            max_grad_norm=0.01,
        )
        trainer = PPOTrainer(net, cfg)
        buf = RolloutBuffer(capacity=8, device="cpu")
        _fill_buffer_with_obs(buf, net, 8)
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        trainer.update(buf)
        # After clipping, all parameter gradients should have small norms
        for p in net.parameters():
            if p.grad is not None:
                assert p.grad.norm() <= 0.01 * 1.01  # small tolerance

    def test_clipping_is_active(self):
        """When ratio exceeds 1 ± eps, clip_fraction > 0."""
        net = _make_net()
        cfg = PPOConfig(
            n_epochs=5,
            batch_size=4,
            clip_epsilon=0.1,
        )
        trainer = PPOTrainer(net, cfg)
        buf = RolloutBuffer(capacity=16, device="cpu")
        _fill_buffer_with_obs(buf, net, 16)
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        metrics = trainer.update(buf)
        # With multiple epochs, policy changes enough to trigger clipping
        assert metrics["clip_fraction"] > 0.0


class TestMultipleEpochs:
    """Multiple epoch behavior."""

    def test_multiple_epochs_run(self):
        """update() runs all epochs without error."""
        net = _make_net()
        cfg = PPOConfig(n_epochs=3, batch_size=4)
        trainer = PPOTrainer(net, cfg)
        buf = RolloutBuffer(capacity=8, device="cpu")
        _fill_buffer_with_obs(buf, net, 8)
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        metrics = trainer.update(buf)
        assert "policy_loss" in metrics
