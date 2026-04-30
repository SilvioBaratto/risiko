"""Tests for the self-play training loop."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents.random_agent import RandomAgent
from src.config import PPOConfig, SelfPlayConfig, TrainingConfig
from src.env import RisikoEnv
from training.self_play import SelfPlayTrainer

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def tiny_cfg() -> TrainingConfig:
    return TrainingConfig(
        total_timesteps=10,
        n_envs=1,
        save_freq=0,
        eval_freq=0,
        seed=42,
        device="cpu",
        ppo=PPOConfig(n_steps=32, n_epochs=1, batch_size=8, lr=1e-3),
        self_play=SelfPlayConfig(opponent_update_freq=100, promote_threshold=1.0),
    )


@pytest.fixture
def tiny_action_dims() -> dict[str, int]:
    return {
        "action_type": 6,
        "param_a": 42,
        "param_b": 42,
        "param_c": 43,
        "param_d": 43,
    }


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


class TestSelfPlayTrainerInit:
    """SelfPlayTrainer initialises all components."""

    def test_trainer_and_agent_created(self, tiny_cfg: TrainingConfig) -> None:
        trainer = SelfPlayTrainer(tiny_cfg)
        assert trainer._trainer is not None
        assert trainer._agent is not None

    def test_opponent_net_created(self, tiny_cfg: TrainingConfig) -> None:
        trainer = SelfPlayTrainer(tiny_cfg)
        assert trainer._opponent_net is not None

    def test_opponent_is_frozen(self, tiny_cfg: TrainingConfig) -> None:
        trainer = SelfPlayTrainer(tiny_cfg)
        for p in trainer._opponent_net.parameters():
            assert not p.requires_grad

    def test_checkpoint_dir_default(self, tiny_cfg: TrainingConfig) -> None:
        trainer = SelfPlayTrainer(tiny_cfg)
        assert trainer._checkpoint_dir == Path("models")

    def test_episode_starts_at_zero(self, tiny_cfg: TrainingConfig) -> None:
        trainer = SelfPlayTrainer(tiny_cfg)
        assert trainer._episode == 0

    def test_env_created(self, tiny_cfg: TrainingConfig) -> None:
        trainer = SelfPlayTrainer(tiny_cfg)
        assert isinstance(trainer._env, RisikoEnv)


# ------------------------------------------------------------------
# Episode collection
# ------------------------------------------------------------------


class TestSelfPlayTrainerRunEpisode:
    """Episode collection populates the rollout buffer."""

    def test_run_episode_returns_game_result(self, tiny_cfg: TrainingConfig) -> None:
        trainer = SelfPlayTrainer(tiny_cfg)
        buffer = trainer._make_buffer()
        result = trainer._run_episode(buffer)
        assert result.winner is None or isinstance(result.winner, int)
        assert result.n_turns > 0

    def test_learner_transitions_buffered(self, tiny_cfg: TrainingConfig) -> None:
        trainer = SelfPlayTrainer(tiny_cfg)
        buffer = trainer._make_buffer()
        trainer._run_episode(buffer)
        # At least some learner transitions should be collected
        assert len(buffer) > 0

    def test_buffer_contains_correct_keys(self, tiny_cfg: TrainingConfig) -> None:
        trainer = SelfPlayTrainer(tiny_cfg)
        buffer = trainer._make_buffer()
        trainer._run_episode(buffer)
        if len(buffer) == 0:
            pytest.skip("No learner transitions in this episode")
        assert len(buffer.observations) == len(buffer)
        assert len(buffer.actions) == len(buffer)
        assert len(buffer.rewards) == len(buffer)
        assert len(buffer.values) == len(buffer)
        assert len(buffer.log_probs) == len(buffer)

    def test_log_prob_is_real(self, tiny_cfg: TrainingConfig) -> None:
        """Log probs should be real (not NaN) and non-positive (prob <= 1)."""
        trainer = SelfPlayTrainer(tiny_cfg)
        buffer = trainer._make_buffer()
        trainer._run_episode(buffer)
        if len(buffer) == 0:
            pytest.skip("No learner transitions in this episode")
        assert all(not np.isnan(lp) for lp in buffer.log_probs)
        assert all(lp <= 0.0 or lp == pytest.approx(0.0) for lp in buffer.log_probs)


# ------------------------------------------------------------------
# PPO update
# ------------------------------------------------------------------


class TestSelfPlayTrainerUpdate:
    """PPO update changes network weights."""

    def test_update_changes_weights(self, tiny_cfg: TrainingConfig) -> None:
        trainer = SelfPlayTrainer(tiny_cfg)
        buffer = trainer._make_buffer()
        trainer._run_episode(buffer)
        if len(buffer) == 0:
            pytest.skip("No learner transitions")
        trainer._compute_advantages(buffer)
        before = {k: v.clone() for k, v in trainer._trainer.net.state_dict().items()}
        trainer._trainer.update(buffer)
        after = trainer._trainer.net.state_dict()
        changed = any(not torch.equal(before[k], after[k]) for k in before)
        assert changed

    def test_update_returns_metrics(self, tiny_cfg: TrainingConfig) -> None:
        trainer = SelfPlayTrainer(tiny_cfg)
        buffer = trainer._make_buffer()
        trainer._run_episode(buffer)
        if len(buffer) == 0:
            pytest.skip("No learner transitions")
        trainer._compute_advantages(buffer)
        metrics = trainer._trainer.update(buffer)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics


# ------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------


class TestSelfPlayTrainerCheckpoint:
    """Save and restore training state."""

    def test_save_creates_file(self, tiny_cfg: TrainingConfig, tmp_path: Path) -> None:
        trainer = SelfPlayTrainer(tiny_cfg, checkpoint_dir=tmp_path)
        trainer._episode = 5
        path = tmp_path / "test.pt"
        trainer.save_checkpoint(path)
        assert path.exists()

    def test_load_restores_episode(self, tiny_cfg: TrainingConfig, tmp_path: Path) -> None:
        trainer = SelfPlayTrainer(tiny_cfg, checkpoint_dir=tmp_path)
        trainer._episode = 7
        path = tmp_path / "test.pt"
        trainer.save_checkpoint(path)

        new_trainer = SelfPlayTrainer(tiny_cfg, checkpoint_dir=tmp_path)
        new_trainer.load_checkpoint(path)
        assert new_trainer._episode == 7

    def test_load_restores_opponent_state(self, tiny_cfg: TrainingConfig, tmp_path: Path) -> None:
        trainer = SelfPlayTrainer(tiny_cfg, checkpoint_dir=tmp_path)
        trainer.save_checkpoint(tmp_path / "test.pt")

        # Mutate opponent before loading
        new_trainer = SelfPlayTrainer(tiny_cfg, checkpoint_dir=tmp_path)
        for p in new_trainer._opponent_net.parameters():
            p.data.zero_()

        new_trainer.load_checkpoint(tmp_path / "test.pt")
        # Opponent should now match the saved state (non-zero)
        assert any(p.abs().sum().item() > 0 for p in new_trainer._opponent_net.parameters())

    def test_load_restores_network_weights(self, tiny_cfg: TrainingConfig, tmp_path: Path) -> None:
        trainer = SelfPlayTrainer(tiny_cfg, checkpoint_dir=tmp_path)
        trainer._episode = 3
        trainer.save_checkpoint(tmp_path / "test.pt")

        new_trainer = SelfPlayTrainer(tiny_cfg, checkpoint_dir=tmp_path)
        new_trainer.load_checkpoint(tmp_path / "test.pt")
        old_state = trainer._trainer.net.state_dict()
        new_state = new_trainer._trainer.net.state_dict()
        assert all(torch.equal(old_state[k], new_state[k]) for k in old_state)

    def test_best_metric_value_restored(self, tiny_cfg: TrainingConfig, tmp_path: Path) -> None:
        trainer = SelfPlayTrainer(tiny_cfg, checkpoint_dir=tmp_path)
        trainer._best_metric_value = 0.75
        trainer.save_checkpoint(tmp_path / "test.pt")

        new_trainer = SelfPlayTrainer(tiny_cfg, checkpoint_dir=tmp_path)
        new_trainer.load_checkpoint(tmp_path / "test.pt")
        assert new_trainer._best_metric_value == pytest.approx(0.75)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------


class TestSelfPlayTrainerTrain:
    """Full training loop runs without crash."""

    def test_runs_without_crash(self, tiny_cfg: TrainingConfig, tmp_path: Path) -> None:
        trainer = SelfPlayTrainer(tiny_cfg, checkpoint_dir=tmp_path, log_dir=tmp_path / "logs")
        trainer.train()
        assert trainer._episode >= tiny_cfg.total_timesteps

    def test_episodes_progress(self, tiny_cfg: TrainingConfig, tmp_path: Path) -> None:
        trainer = SelfPlayTrainer(tiny_cfg, checkpoint_dir=tmp_path, log_dir=tmp_path / "logs")
        trainer.train()
        assert trainer._episode == tiny_cfg.total_timesteps


# ------------------------------------------------------------------
# Opponent promotion
# ------------------------------------------------------------------


class TestSelfPlayTrainerPromotion:
    """Opponent promotion copies current weights."""

    def test_promote_copies_weights(self, tiny_cfg: TrainingConfig) -> None:
        trainer = SelfPlayTrainer(tiny_cfg)
        before = {k: v.clone() for k, v in trainer._opponent_net.state_dict().items()}
        # Update current net to make it differ from opponent
        for p in trainer._agent._net.parameters():
            p.data += 1.0
        trainer._promote_current_to_opponent()
        after = trainer._opponent_net.state_dict()
        assert all(not torch.equal(before[k], after[k]) for k in before)

    def test_evaluate_returns_win_rate(self, tiny_cfg: TrainingConfig) -> None:
        trainer = SelfPlayTrainer(tiny_cfg)
        # Play against a random agent to get a non-trivial win rate

        opponent = RandomAgent(seed=99)
        from training.evaluate import evaluate_agents

        result = evaluate_agents(
            trainer._agent,
            opponent,
            n_games=5,
            n_players=2,
            seed=42,
            max_turns=20,
        )
        assert 0.0 <= result.win_rate_a <= 1.0


# ------------------------------------------------------------------
# Resume from checkpoint
# ------------------------------------------------------------------


class TestSelfPlayTrainerResume:
    """Resume training from a checkpoint."""

    def test_maybe_resume_from_latest(self, tiny_cfg: TrainingConfig, tmp_path: Path) -> None:
        # Create a checkpoint
        trainer = SelfPlayTrainer(tiny_cfg, checkpoint_dir=tmp_path)
        trainer._episode = 3
        trainer.save_checkpoint(tmp_path / "latest.pt")

        # New trainer should resume
        new_trainer = SelfPlayTrainer(tiny_cfg, checkpoint_dir=tmp_path)
        new_trainer._maybe_resume()
        assert new_trainer._episode == 3

    def test_no_resume_if_no_checkpoint(self, tiny_cfg: TrainingConfig, tmp_path: Path) -> None:
        new_trainer = SelfPlayTrainer(tiny_cfg, checkpoint_dir=tmp_path)
        new_trainer._maybe_resume()
        assert new_trainer._episode == 0


# ------------------------------------------------------------------
# Integration with config fields
# ------------------------------------------------------------------


class TestSelfPlayConfigFields:
    """New config fields are wired correctly."""

    def test_self_play_config_in_training_config(self) -> None:
        cfg = TrainingConfig()
        assert cfg.self_play.opponent_update_freq == 50
        assert cfg.self_play.promote_threshold == 0.55
        assert cfg.self_play.eval_games == 10

    def test_reward_config_in_training_config(self) -> None:
        cfg = TrainingConfig()
        assert cfg.reward.sparse_win == 100.0
        assert cfg.reward.sparse_loss == -100.0

    def test_training_config_is_frozen(self) -> None:
        cfg = TrainingConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.seed = 123  # type: ignore[misc]
