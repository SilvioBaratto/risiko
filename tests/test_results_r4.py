"""Tests for global criterion: every training run is tagged with its seed.

Issue #50 — global [UNIT] criterion.
"""

from __future__ import annotations


class TestTrainingConfigSeed:
    """The seed must be a first-class field on TrainingConfig so runs can be tagged."""

    def test_when_training_config_created_with_seed_then_seed_is_accessible(self):
        """Criterion: every run tagged with its seed."""
        from src.config import TrainingConfig

        cfg = TrainingConfig(seed=99)
        assert cfg.seed == 99

    def test_when_two_configs_differ_in_seed_then_seeds_are_distinct(self):
        """Different seeds must be distinguishable — basis for unique run tagging."""
        from src.config import TrainingConfig

        a = TrainingConfig(seed=1)
        b = TrainingConfig(seed=2)
        assert a.seed != b.seed
