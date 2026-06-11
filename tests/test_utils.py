"""Tests for reward configuration and seeding utilities."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from src.utils import reward_config as reward_config_module
from src.utils import seed as seed_module
from src.utils.reward_config import RewardConfig
from src.utils.seed import set_global_seeds


class TestRewardConfig:
    """RewardConfig dataclass properties."""

    def test_is_frozen_dataclass(self):
        """RewardConfig must be a frozen dataclass."""
        cfg = RewardConfig()
        assert cfg.__dataclass_params__.frozen  # type: ignore[union-attr]

    def test_default_coefficients_present(self):
        """All required coefficients must have sensible defaults."""
        cfg = RewardConfig()
        assert hasattr(cfg, "sparse_win")
        assert hasattr(cfg, "sparse_loss")
        assert hasattr(cfg, "dense_territory_delta")
        assert hasattr(cfg, "dense_continent_bonus_delta")
        assert hasattr(cfg, "dense_army_ratio")
        assert hasattr(cfg, "dense_elimination_bonus")
        assert hasattr(cfg, "invalid_action_penalty")

    def test_default_values(self):
        """Defaults must follow the design spec."""
        cfg = RewardConfig()
        assert cfg.sparse_win == 1.0
        assert cfg.sparse_loss == -1.0
        assert cfg.dense_territory_delta == 0.01
        assert cfg.dense_continent_bonus_delta == 0.05
        assert cfg.dense_army_ratio == 0.005
        assert cfg.dense_elimination_bonus == 0.1
        assert cfg.invalid_action_penalty == -0.01

    def test_cannot_mutate_after_creation(self):
        """Frozen dataclass must reject attribute mutation."""
        cfg = RewardConfig()
        try:
            cfg.sparse_win = 50.0  # type: ignore[assignment]
        except Exception as exc:
            assert "frozen" in str(exc).lower() or "cannot" in str(exc).lower()
        else:
            raise AssertionError("Expected mutation to fail on frozen dataclass")

    def test_custom_coefficients(self):
        """Custom values must be accepted at creation."""
        cfg = RewardConfig(
            sparse_win=200.0,
            dense_territory_delta=0.0,
            invalid_action_penalty=-5.0,
        )
        assert cfg.sparse_win == 200.0
        assert cfg.dense_territory_delta == 0.0
        assert cfg.invalid_action_penalty == -5.0
        assert cfg.sparse_loss == -1.0  # default preserved

    def test_total_coefficients_count(self):
        """There must be exactly 7 coefficient fields."""
        cfg = RewardConfig()
        fields = [f for f in cfg.__dataclass_fields__ if not f.startswith("_")]
        assert len(fields) == 7

    def test_when_field_mutated_then_frozen_instance_error_raised(self):
        """Mutation must raise FrozenInstanceError specifically."""
        cfg = RewardConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.dense_army_ratio = 9.9  # type: ignore[misc]

    def test_when_dense_coefficients_inspected_then_each_within_1e_1(self):
        """Every dense coefficient must stay at order 1e-1 or smaller."""
        cfg = RewardConfig()
        dense = [
            cfg.dense_territory_delta,
            cfg.dense_continent_bonus_delta,
            cfg.dense_army_ratio,
            cfg.dense_elimination_bonus,
        ]
        for coef in dense:
            assert abs(coef) <= 0.1

    def test_when_dense_field_zeroed_then_others_keep_defaults(self):
        """Zeroing one dense coefficient must not touch the others (ablation)."""
        cfg = RewardConfig(dense_continent_bonus_delta=0.0)
        assert cfg.dense_continent_bonus_delta == 0.0
        assert cfg.dense_territory_delta == 0.01
        assert cfg.dense_army_ratio == 0.005
        assert cfg.dense_elimination_bonus == 0.1

    def test_when_module_imported_then_all_exports_reward_config(self):
        """The module must declare __all__ exporting RewardConfig."""
        assert reward_config_module.__all__ == ["RewardConfig"]


class TestSeeding:
    """Global seeding utility properties."""

    def test_returns_generator(self):
        """set_global_seeds must return a numpy Generator."""
        rng = set_global_seeds(42)
        assert isinstance(rng, np.random.Generator)

    def test_deterministic_numpy(self):
        """Numpy output must be identical after resetting to same seed."""
        set_global_seeds(42)
        a = np.random.rand(10)
        set_global_seeds(42)
        b = np.random.rand(10)
        assert np.array_equal(a, b)

    def test_deterministic_torch(self):
        """Torch output must be identical after resetting to same seed."""
        set_global_seeds(42)
        a = torch.rand(10)
        set_global_seeds(42)
        b = torch.rand(10)
        assert torch.allclose(a, b)

    def test_different_seeds_produce_different_output(self):
        """Different seeds must produce different numpy samples."""
        set_global_seeds(42)
        a = np.random.rand(10)
        set_global_seeds(123)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)

    def test_returned_generator_is_independent(self):
        """The returned Generator must not share state with the global numpy RNG."""
        rng = set_global_seeds(42)
        # Sample from the returned generator
        local_sample = rng.random(10)
        # Sample from global np.random — should be different
        global_sample = np.random.rand(10)
        assert not np.array_equal(local_sample, global_sample)

    def test_generator_produces_deterministic_sequence(self):
        """The returned Generator must produce identical sequences for the same seed."""
        rng1 = set_global_seeds(42)
        seq1 = rng1.random(100)
        rng2 = set_global_seeds(42)
        seq2 = rng2.random(100)
        assert np.array_equal(seq1, seq2)

    def test_when_seeds_set_then_cudnn_determinism_flags_enabled(self):
        """Determinism flags for cuDNN must be enabled for reproducible torch ops."""
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        set_global_seeds(42)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_when_module_imported_then_all_exports_set_global_seeds(self):
        """The seed module must declare __all__ exporting set_global_seeds."""
        assert seed_module.__all__ == ["set_global_seeds"]
