"""Tests for checkpoint save/resume infrastructure."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.checkpoint import CheckpointManager
from src.config import PPOConfig, TrainingConfig
from src.env import RisikoEnv
from src.models.actor_critic import ActorCritic
from src.models.ppo import PPOTrainer
from src.utils.seed import set_global_seeds


@pytest.fixture
def simple_trainer():
    """Create a minimal PPOTrainer for checkpoint tests."""
    net = ActorCritic(
        obs_dim=10,
        hidden_size=8,
        num_layers=1,
        action_dims={"action_type": 6, "param_a": 42},
    )
    config = TrainingConfig(total_timesteps=1000, ppo=PPOConfig(lr=0.001))
    trainer = PPOTrainer(net, config.ppo)
    return trainer


@pytest.fixture
def rng_state():
    """Create a representative RNG state dict."""
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    env = RisikoEnv(n_players=2)
    env.reset(seed=7)
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
        "env": env.state.rng.bit_generator.state,
    }


@pytest.fixture
def temp_checkpoint(tmp_path):
    """Return a temporary path for checkpoint files."""
    return tmp_path / "checkpoint.pt"


class TestCheckpointManagerSave:
    """Saving checkpoints writes a .pt file with all required state."""

    def test_save_creates_file(self, simple_trainer, rng_state, temp_checkpoint):
        trainer = simple_trainer
        env = RisikoEnv(n_players=2)
        config = TrainingConfig(total_timesteps=1000)
        CheckpointManager.save(temp_checkpoint, trainer, env, rng_state, episode=42, config=config)
        assert temp_checkpoint.exists()

    def test_save_overwrites_existing(self, simple_trainer, rng_state, temp_checkpoint):
        trainer = simple_trainer
        env = RisikoEnv(n_players=2)
        config = TrainingConfig(total_timesteps=1000)
        temp_checkpoint.write_text("old")
        CheckpointManager.save(temp_checkpoint, trainer, env, rng_state, episode=1, config=config)
        assert temp_checkpoint.read_bytes() != b"old"


class TestCheckpointManagerLoad:
    """Loading checkpoints restores training state deterministically."""

    def test_load_returns_resume_dict(self, simple_trainer, rng_state, temp_checkpoint):
        trainer = simple_trainer
        env = RisikoEnv(n_players=2)
        config = TrainingConfig(total_timesteps=1000)
        CheckpointManager.save(temp_checkpoint, trainer, env, rng_state, episode=42, config=config)
        resume = CheckpointManager.load(temp_checkpoint)
        assert isinstance(resume, dict)
        assert "trainer_state" in resume
        assert "rng_state" in resume
        assert "episode" in resume
        assert "config" in resume

    def test_load_restores_trainer_weights(self, simple_trainer, rng_state, temp_checkpoint):
        trainer = simple_trainer
        env = RisikoEnv(n_players=2)
        config = TrainingConfig(total_timesteps=1000)
        # Mutate weights so we can verify restoration
        with torch.no_grad():
            trainer.net.trunk[0].weight.fill_(1.23)
        original_weight = trainer.net.trunk[0].weight.clone()
        CheckpointManager.save(temp_checkpoint, trainer, env, rng_state, episode=1, config=config)
        # Corrupt weights
        with torch.no_grad():
            trainer.net.trunk[0].weight.fill_(0.0)
        resume = CheckpointManager.load(temp_checkpoint)
        trainer.load_checkpoint(resume["trainer_state"])
        assert torch.allclose(trainer.net.trunk[0].weight, original_weight)

    def test_load_restores_optimizer_state(self, simple_trainer, rng_state, temp_checkpoint):
        trainer = simple_trainer
        env = RisikoEnv(n_players=2)
        config = TrainingConfig(total_timesteps=1000)
        # Take an optimizer step to populate state
        dummy_loss = trainer.net.trunk[0].weight.sum()
        trainer.optimizer.zero_grad()
        dummy_loss.backward()
        trainer.optimizer.step()
        original_state = {
            k: {sk: sv.clone() if isinstance(sv, torch.Tensor) else sv for sk, sv in v.items()}
            for k, v in trainer.optimizer.state_dict()["state"].items()
        }
        CheckpointManager.save(temp_checkpoint, trainer, env, rng_state, episode=1, config=config)
        # Reset optimizer state
        trainer.optimizer.state_dict()["state"].clear()
        resume = CheckpointManager.load(temp_checkpoint)
        trainer.load_checkpoint(resume["trainer_state"])
        restored_state = trainer.optimizer.state_dict()["state"]
        assert len(restored_state) == len(original_state)

    def test_load_restores_episode_counter(self, simple_trainer, rng_state, temp_checkpoint):
        trainer = simple_trainer
        env = RisikoEnv(n_players=2)
        config = TrainingConfig(total_timesteps=1000)
        CheckpointManager.save(temp_checkpoint, trainer, env, rng_state, episode=99, config=config)
        resume = CheckpointManager.load(temp_checkpoint)
        assert resume["episode"] == 99

    def test_load_restores_config(self, simple_trainer, rng_state, temp_checkpoint):
        trainer = simple_trainer
        env = RisikoEnv(n_players=2)
        config = TrainingConfig(total_timesteps=123456, seed=77)
        CheckpointManager.save(temp_checkpoint, trainer, env, rng_state, episode=1, config=config)
        resume = CheckpointManager.load(temp_checkpoint)
        restored = resume["config"]
        assert restored.total_timesteps == 123456
        assert restored.seed == 77

    def test_load_restores_rng_state(self, simple_trainer, rng_state, temp_checkpoint):
        trainer = simple_trainer
        env = RisikoEnv(n_players=2)
        config = TrainingConfig(total_timesteps=1000)
        CheckpointManager.save(temp_checkpoint, trainer, env, rng_state, episode=1, config=config)
        resume = CheckpointManager.load(temp_checkpoint)
        assert "torch" in resume["rng_state"]
        assert "numpy" in resume["rng_state"]
        assert "random" in resume["rng_state"]
        assert "env" in resume["rng_state"]


class TestPPOTrainerCheckpointMethods:
    """PPOTrainer exposes save/load helpers used by CheckpointManager."""

    def test_save_checkpoint_returns_dict(self, simple_trainer):
        state = simple_trainer.save_checkpoint()
        assert isinstance(state, dict)
        assert "model" in state
        assert "optimizer" in state

    def test_load_checkpoint_restores_model(self, simple_trainer):
        import copy

        original_weight = simple_trainer.net.trunk[0].weight.clone()
        state = copy.deepcopy(simple_trainer.save_checkpoint())
        with torch.no_grad():
            simple_trainer.net.trunk[0].weight.fill_(0.0)
        simple_trainer.load_checkpoint(state)
        assert torch.allclose(simple_trainer.net.trunk[0].weight, original_weight)

    def test_load_checkpoint_restores_optimizer(self, simple_trainer):
        dummy_loss = simple_trainer.net.trunk[0].weight.sum()
        simple_trainer.optimizer.zero_grad()
        dummy_loss.backward()
        simple_trainer.optimizer.step()
        original_state = simple_trainer.optimizer.state_dict()
        state = simple_trainer.save_checkpoint()
        # Corrupt optimizer by re-creating it
        simple_trainer.optimizer = torch.optim.Adam(simple_trainer.net.parameters(), lr=0.1)
        simple_trainer.load_checkpoint(state)
        restored_state = simple_trainer.optimizer.state_dict()
        assert len(restored_state["state"]) == len(original_state["state"])


class TestDeterministicResume:
    """Resuming from a checkpoint yields identical trajectories."""

    def test_resume_reproduces_same_env_sequence(self, temp_checkpoint):
        """After restoring RNG state, env.reset + steps produce same results."""
        seed = 42
        set_global_seeds(seed)
        env = RisikoEnv(n_players=2)
        obs1, info1 = env.reset(seed=seed)

        # Capture RNG state after reset
        rng_state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
            "env": env.state.rng.bit_generator.state,
        }

        # Take a few steps
        actions = []
        for _ in range(5):
            legal = info1["legal_actions"]
            action = (
                legal[0]
                if legal
                else {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}
            )
            obs1, reward, terminated, truncated, info1 = env.step(action)
            actions.append(action)
            if terminated or truncated:
                break

        # Save checkpoint with captured RNG state
        trainer = PPOTrainer(
            ActorCritic(obs_dim=10, hidden_size=4, num_layers=1, action_dims={"action_type": 6}),
            TrainingConfig().ppo,
        )
        CheckpointManager.save(
            temp_checkpoint, trainer, env, rng_state, episode=0, config=TrainingConfig()
        )

        # Now restore RNGs and replay
        resume = CheckpointManager.load(temp_checkpoint)
        restored_rng = resume["rng_state"]
        torch.set_rng_state(restored_rng["torch"])
        np.random.set_state(restored_rng["numpy"])
        random.setstate(restored_rng["random"])
        env2 = RisikoEnv(n_players=2)
        env2.reset(seed=seed)
        env2.state.rng = np.random.default_rng()
        env2.state.rng.bit_generator.state = restored_rng["env"]

        # Replay same actions
        for action in actions:
            obs2, reward2, terminated2, truncated2, info2 = env2.step(action)

        # Compare final states
        np.testing.assert_array_equal(env.state.territory_owner, env2.state.territory_owner)
        np.testing.assert_array_equal(env.state.armies, env2.state.armies)
