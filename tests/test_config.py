"""Tests for the hyperparameter configuration system."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    NetworkConfig,
    PPOConfig,
    TrainingConfig,
    load_config,
    merge_cli_overrides,
    resolve_device,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPPOConfigDefaults:
    """PPO hyperparameter defaults."""

    def test_lr_default(self):
        """Default learning rate is 3e-4."""
        cfg = PPOConfig()
        assert cfg.lr == 3e-4

    def test_gamma_default(self):
        """Default discount factor is 0.99."""
        cfg = PPOConfig()
        assert cfg.gamma == 0.99

    def test_gae_lambda_default(self):
        """Default GAE lambda is 0.95."""
        cfg = PPOConfig()
        assert cfg.gae_lambda == 0.95

    def test_clip_epsilon_default(self):
        """Default clip epsilon is 0.2."""
        cfg = PPOConfig()
        assert cfg.clip_epsilon == 0.2

    def test_entropy_coef_default(self):
        """Default entropy coefficient is 0.01."""
        cfg = PPOConfig()
        assert cfg.entropy_coef == 0.01

    def test_value_loss_coef_default(self):
        """Default value loss coefficient is 0.5."""
        cfg = PPOConfig()
        assert cfg.value_loss_coef == 0.5

    def test_max_grad_norm_default(self):
        """Default max gradient norm is 0.5."""
        cfg = PPOConfig()
        assert cfg.max_grad_norm == 0.5

    def test_n_steps_default(self):
        """Default rollout steps is 2048."""
        cfg = PPOConfig()
        assert cfg.n_steps == 2048

    def test_n_epochs_default(self):
        """Default update epochs is 10."""
        cfg = PPOConfig()
        assert cfg.n_epochs == 10

    def test_batch_size_default(self):
        """Default batch size is 64."""
        cfg = PPOConfig()
        assert cfg.batch_size == 64

    def test_is_frozen(self):
        """PPOConfig is immutable."""
        cfg = PPOConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.lr = 1e-3  # type: ignore[assignment]


class TestNetworkConfigDefaults:
    """Network architecture defaults."""

    def test_hidden_sizes_default(self):
        """Default hidden layers are (256, 256)."""
        cfg = NetworkConfig()
        assert cfg.hidden_sizes == (256, 256)

    def test_activation_default(self):
        """Default activation is tanh."""
        cfg = NetworkConfig()
        assert cfg.activation == "tanh"

    def test_is_frozen(self):
        """NetworkConfig is immutable."""
        cfg = NetworkConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.activation = "relu"  # type: ignore[assignment]


class TestTrainingConfigDefaults:
    """Top-level training config defaults."""

    def test_total_timesteps_default(self):
        """Default total timesteps is 1_000_000."""
        cfg = TrainingConfig()
        assert cfg.total_timesteps == 1_000_000

    def test_n_envs_default(self):
        """Default parallel envs is 1."""
        cfg = TrainingConfig()
        assert cfg.n_envs == 1

    def test_save_freq_default(self):
        """Default save frequency is 100 episodes."""
        cfg = TrainingConfig()
        assert cfg.save_freq == 100

    def test_eval_freq_default(self):
        """Default eval frequency is 50 episodes."""
        cfg = TrainingConfig()
        assert cfg.eval_freq == 50

    def test_seed_default(self):
        """Default seed is 42."""
        cfg = TrainingConfig()
        assert cfg.seed == 42

    def test_device_default(self):
        """Default device is auto."""
        cfg = TrainingConfig()
        assert cfg.device == "auto"

    def test_contains_ppo_config(self):
        """TrainingConfig nests a PPOConfig."""
        cfg = TrainingConfig()
        assert isinstance(cfg.ppo, PPOConfig)

    def test_contains_network_config(self):
        """TrainingConfig nests a NetworkConfig."""
        cfg = TrainingConfig()
        assert isinstance(cfg.network, NetworkConfig)

    def test_is_frozen(self):
        """TrainingConfig is immutable."""
        cfg = TrainingConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.seed = 123  # type: ignore[assignment]


class TestLoadConfig:
    """YAML config loading."""

    def test_load_default_yaml(self):
        """Loads config/default.yaml without error."""
        path = REPO_ROOT / "config" / "default.yaml"
        cfg = load_config(path)
        assert isinstance(cfg, TrainingConfig)
        assert cfg.ppo.lr == 3e-4

    def test_missing_file_raises(self):
        """Nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(REPO_ROOT / "config" / "missing.yaml")


class TestResolveDevice:
    """Device resolution."""

    def test_auto_returns_cuda_when_available(self):
        """Auto resolves to cuda when torch reports availability."""
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_device("auto") == "cuda"

    def test_auto_returns_cpu_when_cuda_unavailable(self):
        """Auto resolves to cpu when cuda is absent."""
        with patch("torch.cuda.is_available", return_value=False):
            assert resolve_device("auto") == "cpu"

    def test_explicit_device_passthrough(self):
        """Explicit device strings pass through unchanged."""
        assert resolve_device("cuda:0") == "cuda:0"
        assert resolve_device("cpu") == "cpu"


class TestMergeCliOverrides:
    """CLI override merging."""

    def test_override_top_level_int(self):
        """Overriding seed casts to int."""
        cfg = TrainingConfig()
        new_cfg = merge_cli_overrides(cfg, {"seed": "123"})
        assert new_cfg.seed == 123
        assert isinstance(new_cfg.seed, int)

    def test_override_top_level_float(self):
        """Overriding total_timesteps casts to int."""
        cfg = TrainingConfig()
        new_cfg = merge_cli_overrides(cfg, {"total_timesteps": "500000"})
        assert new_cfg.total_timesteps == 500_000

    def test_override_nested_ppo_float(self):
        """Overriding ppo.lr casts to float."""
        cfg = TrainingConfig()
        new_cfg = merge_cli_overrides(cfg, {"ppo.lr": "0.001"})
        assert new_cfg.ppo.lr == 1e-3
        assert isinstance(new_cfg.ppo.lr, float)

    def test_override_nested_ppo_int(self):
        """Overriding ppo.n_steps casts to int."""
        cfg = TrainingConfig()
        new_cfg = merge_cli_overrides(cfg, {"ppo.n_steps": "1024"})
        assert new_cfg.ppo.n_steps == 1024

    def test_override_returns_frozen(self):
        """Merged config remains frozen."""
        cfg = TrainingConfig()
        new_cfg = merge_cli_overrides(cfg, {"seed": "7"})
        with pytest.raises(FrozenInstanceError):
            new_cfg.seed = 99  # type: ignore[assignment]

    def test_override_preserves_untouched_fields(self):
        """Fields not in overrides keep their original values."""
        cfg = TrainingConfig(seed=7, total_timesteps=500_000)
        new_cfg = merge_cli_overrides(cfg, {"device": "cpu"})
        assert new_cfg.seed == 7
        assert new_cfg.total_timesteps == 500_000
        assert new_cfg.device == "cpu"

    def test_override_list_field(self):
        """Overriding network.hidden_sizes parses list."""
        cfg = TrainingConfig()
        new_cfg = merge_cli_overrides(cfg, {"network.hidden_sizes": "[128, 128]"})
        assert new_cfg.network.hidden_sizes == (128, 128)

    def test_override_bool_field(self):
        """Overriding a bool field casts correctly."""
        # We'll add a bool field to TrainingConfig for this test, or test a different approach
        # For now, test via a custom dataclass with bool
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _BoolCfg:
            flag: bool = False

        bool_cfg = _BoolCfg()
        new_cfg = merge_cli_overrides(bool_cfg, {"flag": "true"})
        assert new_cfg.flag is True


class TestCliFlags:
    """CLI flag acceptance via the train subcommand."""

    def test_train_help_shows_config_flag(self):
        """--config appears in help text."""
        result = subprocess.run(
            [sys.executable, "-m", "risiko_rl.cli", "train", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--config" in result.stdout

    def test_train_help_shows_lr_flag(self):
        """--lr appears in help text."""
        result = subprocess.run(
            [sys.executable, "-m", "risiko_rl.cli", "train", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--lr" in result.stdout

    def test_train_help_shows_seed_flag(self):
        """--seed appears in help text."""
        result = subprocess.run(
            [sys.executable, "-m", "risiko_rl.cli", "train", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--seed" in result.stdout

    def test_train_help_shows_device_flag(self):
        """--device appears in help text."""
        result = subprocess.run(
            [sys.executable, "-m", "risiko_rl.cli", "train", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--device" in result.stdout

    def test_train_help_shows_timesteps_flag(self):
        """--timesteps appears in help text."""
        result = subprocess.run(
            [sys.executable, "-m", "risiko_rl.cli", "train", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--timesteps" in result.stdout

    def test_train_help_shows_override_flag(self):
        """--override appears in help text."""
        result = subprocess.run(
            [sys.executable, "-m", "risiko_rl.cli", "train", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--override" in result.stdout or "-o" in result.stdout
