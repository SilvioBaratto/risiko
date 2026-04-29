"""End-to-end integration test for Cycle 2 model components."""

from __future__ import annotations

import numpy as np
import torch

from src.config import PPOConfig
from src.models.actor_critic import ActorCritic
from src.models.gae import compute_gae
from src.models.ppo import PPOTrainer
from src.models.replay_buffer import RolloutBuffer
from src.models.utils import flatten_obs, get_obs_dim, stack_obs

_ACTION_DIMS = {
    "action_type": 6,
    "param_a": 42,
    "param_b": 42,
    "param_c": 43,
    "param_d": 43,
}


def _make_net() -> ActorCritic:
    """Create a small ActorCritic for integration tests."""
    return ActorCritic(
        obs_dim=get_obs_dim(),
        hidden_size=32,
        num_layers=2,
        action_dims=_ACTION_DIMS,
    )


def _sample_obs() -> dict[str, np.ndarray]:
    """Create a single observation dict."""
    return {
        "territory_owner": np.random.randint(0, 6, 42, dtype=np.int32),
        "armies": np.random.randint(1, 10, 42, dtype=np.int32),
        "phase": np.array(1, dtype=np.int32),
        "current_player": np.array(2, dtype=np.int32),
        "cards": np.random.randint(0, 2, (5, 4), dtype=np.int32),
        "continent_control": np.random.randint(0, 2, 6, dtype=np.int32),
        "trade_count": np.array(4, dtype=np.int32),
        "reinforcements_remaining": np.array(5, dtype=np.int32),
        "turn_capture": np.array(1, dtype=np.int32),
        "n_players": np.array(3, dtype=np.int32),
        "eliminated": np.zeros(6, dtype=np.int32),
    }


class TestEndToEnd:
    """Full pipeline: env obs → flatten → actor_critic → sample → buffer → GAE → PPO update."""

    def test_single_update_changes_weights(self):
        """A single PPO update changes model parameters."""
        net = _make_net()
        trainer = PPOTrainer(net, PPOConfig(n_epochs=2, batch_size=4, lr=0.01))
        buf = RolloutBuffer(capacity=10, device="cpu")

        for _ in range(10):
            obs = _sample_obs()
            stacked = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
            flat = flatten_obs(stacked)
            action, log_prob, entropy, value = net.get_action_and_value(flat)
            buf.add(
                state=obs,
                action={k: v.squeeze(0) for k, v in action.items()},
                reward=1.0,
                done=False,
                value=value.item(),
                log_prob=log_prob.item(),
            )

        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )

        before = list(net.parameters())[0].clone()
        metrics = trainer.update(buf)
        after = list(net.parameters())[0]

        assert not torch.allclose(before, after)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy_loss" in metrics
        assert "approx_kl" in metrics
        assert "clip_fraction" in metrics
        assert "explained_variance" in metrics

    def test_pipeline_with_stack_obs(self):
        """Using stack_obs correctly batches observations before flattening."""
        net = _make_net()
        obs_list = [_sample_obs() for _ in range(4)]
        stacked = stack_obs(obs_list, "cpu")
        flat = flatten_obs(stacked)
        action, log_prob, entropy, value = net.get_action_and_value(flat)

        assert flat.shape == (4, 137)
        assert log_prob.shape == (4,)
        assert value.shape == (4, 1)
        for key in _ACTION_DIMS:
            assert action[key].shape == (4,)

    def test_gae_computation_from_buffer(self):
        """GAE computed from buffer matches standalone compute_gae."""
        buf = RolloutBuffer(capacity=5, device="cpu")
        for _ in range(5):
            buf.add(
                state=_sample_obs(),
                action={k: torch.tensor(0) for k in _ACTION_DIMS},
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
            )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
            gamma=0.99,
            gae_lambda=0.95,
        )

        rewards = torch.ones(5)
        values = torch.ones(6) * 0.5
        dones = torch.zeros(6)
        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

        assert buf.advantages is not None
        assert buf.returns is not None
        assert torch.allclose(buf.advantages, advantages, atol=1e-5)
        assert torch.allclose(buf.returns, returns, atol=1e-5)

    def test_mini_batch_contains_all_required_keys(self):
        """A mini-batch from the buffer contains all keys PPO needs."""
        net = _make_net()
        buf = RolloutBuffer(capacity=5, device="cpu")
        for _ in range(5):
            obs = _sample_obs()
            stacked = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
            flat = flatten_obs(stacked)
            action, log_prob, entropy, value = net.get_action_and_value(flat)
            buf.add(
                state=obs,
                action={k: v.squeeze(0) for k, v in action.items()},
                reward=1.0,
                done=False,
                value=value.item(),
                log_prob=log_prob.item(),
            )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )

        batch = next(buf.get(batch_size=3))
        assert "obs" in batch
        assert "actions" in batch
        assert "advantages" in batch
        assert "returns" in batch
        assert "log_probs" in batch
        assert "values" in batch

        flat_obs = flatten_obs(batch["obs"])
        assert flat_obs.shape == (3, 137)

    def test_gradient_flows_from_ppo_to_actor_critic(self):
        """A PPO update computes gradients for all ActorCritic parameters."""
        net = _make_net()
        trainer = PPOTrainer(net, PPOConfig(n_epochs=1, batch_size=4))
        buf = RolloutBuffer(capacity=8, device="cpu")
        for _ in range(8):
            obs = _sample_obs()
            stacked = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
            flat = flatten_obs(stacked)
            action, log_prob, entropy, value = net.get_action_and_value(flat)
            buf.add(
                state=obs,
                action={k: v.squeeze(0) for k, v in action.items()},
                reward=1.0,
                done=False,
                value=value.item(),
                log_prob=log_prob.item(),
            )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )

        trainer.update(buf)

        for p in net.parameters():
            assert p.grad is not None
