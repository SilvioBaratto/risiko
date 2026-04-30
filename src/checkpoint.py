"""Checkpoint save/resume infrastructure for training runs."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

from src.config import PPOConfig, TrainingConfig
from src.env import RisikoEnv
from src.models.ppo import PPOTrainer


def _config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    """Serialize a TrainingConfig to a plain dict."""
    return dataclasses.asdict(config)


def _config_from_dict(raw: dict[str, Any]) -> TrainingConfig:
    """Deserialize a TrainingConfig from a plain dict."""
    from src.config import NetworkConfig, SelfPlayConfig
    from src.utils.reward_config import RewardConfig

    ppo_raw = raw.get("ppo", {})
    network_raw = raw.get("network", {})
    reward_raw = raw.get("reward", {})
    self_play_raw = raw.get("self_play", {})
    return TrainingConfig(
        total_timesteps=raw.get("total_timesteps", 1_000_000),
        n_envs=raw.get("n_envs", 1),
        save_freq=raw.get("save_freq", 100),
        eval_freq=raw.get("eval_freq", 50),
        seed=raw.get("seed", 42),
        device=raw.get("device", "auto"),
        ppo=PPOConfig(**ppo_raw),
        network=NetworkConfig(**network_raw),
        reward=RewardConfig(**reward_raw),
        self_play=SelfPlayConfig(**self_play_raw),
    )


class CheckpointManager:
    """Save and load training checkpoints deterministically."""

    @staticmethod
    def save(
        path: Path,
        trainer: PPOTrainer,
        env: RisikoEnv,
        rng_state: dict[str, Any],
        episode: int,
        config: TrainingConfig,
    ) -> None:
        """Save a checkpoint to disk.

        Args:
            path: Destination file path.
            trainer: PPOTrainer to checkpoint.
            env: Environment instance to capture state from.
            rng_state: Dict with 'torch', 'numpy', 'random', 'env' keys.
            episode: Current episode counter.
            config: Training configuration.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trainer_state": trainer.save_checkpoint(),
            "rng_state": rng_state,
            "episode": episode,
            "config": _config_to_dict(config),
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        """Load a checkpoint and return a resume dict.

        Args:
            path: Checkpoint file path.

        Returns:
            Dict with keys: trainer_state, rng_state, episode, config.
        """
        payload = torch.load(path, weights_only=False)
        payload["config"] = _config_from_dict(payload["config"])
        return payload
