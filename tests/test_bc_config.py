"""Example tests for issue #75 — BCConfig + HeuristicConfig frozen dataclasses.

All tests are derived purely from the acceptance criteria; no implementation
source was read. Tests are written in the Red phase and will fail until the
implementation is complete.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import BCConfig, HeuristicConfig, TrainingConfig, load_config, merge_cli_overrides

# ---------------------------------------------------------------------------
# TrainingConfig — default-safe zero-arg construction
# ---------------------------------------------------------------------------


def test_when_training_config_is_constructed_with_no_args_then_no_error_is_raised():
    TrainingConfig()


def test_when_training_config_is_constructed_with_no_args_then_bc_field_is_bcconfig_instance():
    cfg = TrainingConfig()
    assert isinstance(cfg.bc, BCConfig)


def test_when_training_config_is_constructed_then_bc_heuristic_is_heuristicconfig_instance():
    assert isinstance(TrainingConfig().bc.heuristic, HeuristicConfig)


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------


def test_when_src_config_is_imported_then_bcconfig_is_exported_in_all():
    import src.config as _mod

    assert "BCConfig" in _mod.__all__


def test_when_src_config_is_imported_then_heuristicconfig_is_exported_in_all():
    import src.config as _mod

    assert "HeuristicConfig" in _mod.__all__


# ---------------------------------------------------------------------------
# BCConfig — required fields
# ---------------------------------------------------------------------------

_BC_CORE_FIELDS = frozenset(
    {
        "n_games",
        "n_players",
        "max_turns",
        "seed",
        "dataset_dir",
        "shard_size",
        "demonstrator",
        "epochs",
        "batch_size",
        "lr",
        "value_loss_coef",
        "val_split",
        "early_stop_patience",
        "output_path",
    }
)
_BC_EXTRA_FIELDS = frozenset({"explore_eps", "label_smoothing", "entropy_coef"})
_BC_ALL_FIELDS = _BC_CORE_FIELDS | _BC_EXTRA_FIELDS


def test_when_bcconfig_is_constructed_then_all_required_core_fields_exist():
    cfg = BCConfig()
    missing = [name for name in _BC_CORE_FIELDS if not hasattr(cfg, name)]
    assert not missing, f"BCConfig is missing fields: {missing}"


def test_when_bcconfig_is_constructed_then_explore_eps_field_exists():
    assert hasattr(BCConfig(), "explore_eps")


def test_when_bcconfig_is_constructed_then_label_smoothing_field_exists():
    assert hasattr(BCConfig(), "label_smoothing")


def test_when_bcconfig_is_constructed_then_entropy_coef_field_exists():
    assert hasattr(BCConfig(), "entropy_coef")


# ---------------------------------------------------------------------------
# BCConfig — default values
# ---------------------------------------------------------------------------


def test_when_bcconfig_is_constructed_then_dataset_dir_default_is_data_bc():
    assert BCConfig().dataset_dir == "data/bc"


def test_when_bcconfig_is_constructed_then_demonstrator_default_is_heuristic():
    assert BCConfig().demonstrator == "heuristic"


def test_when_bcconfig_is_constructed_then_output_path_default_is_models_pretrained_pt():
    assert BCConfig().output_path == "models/pretrained.pt"


def test_when_bcconfig_is_constructed_then_explore_eps_is_approximately_0_1():
    # Criterion states "default ~0.1" — allow ±0.05 tolerance.
    assert abs(BCConfig().explore_eps - 0.1) <= 0.05


def test_when_bcconfig_is_constructed_then_label_smoothing_is_approximately_0_05():
    # Criterion states "default ~0.05" — allow ±0.03 tolerance.
    assert abs(BCConfig().label_smoothing - 0.05) <= 0.03


def test_when_bcconfig_is_constructed_then_entropy_coef_is_approximately_0_01():
    # Criterion states "default ~0.01" — allow ±0.005 tolerance.
    assert abs(BCConfig().entropy_coef - 0.01) <= 0.005


# ---------------------------------------------------------------------------
# HeuristicConfig — required fields and nesting
# ---------------------------------------------------------------------------

_HEURISTIC_REQUIRED_FIELDS = frozenset(
    {
        "reinforce_border_weight",
        "reinforce_continent_weight",
        "reinforce_threat_weight",
        "attack_min_odds",
        "attack_odds_stop",
        "attack_continent_weight",
        "attack_weakest_enemy_weight",
        "attack_dice_policy",
        "fortify_threat_weight",
        "fortify_min_interior_armies",
        "capture_move_fraction",
        "card_target_weakest_weight",
    }
)


def test_when_heuristicconfig_is_constructed_then_all_required_fields_exist():
    cfg = HeuristicConfig()
    missing = [name for name in _HEURISTIC_REQUIRED_FIELDS if not hasattr(cfg, name)]
    assert not missing, f"HeuristicConfig is missing fields: {missing}"


def test_when_bcconfig_is_constructed_then_heuristic_field_is_heuristicconfig_instance():
    assert isinstance(BCConfig().heuristic, HeuristicConfig)


# ---------------------------------------------------------------------------
# Frozen — mutation must raise FrozenInstanceError
# ---------------------------------------------------------------------------


def test_when_bcconfig_lr_is_mutated_then_frozen_instance_error_is_raised():
    cfg = BCConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.lr = 9.9  # type: ignore[misc]


def test_when_bcconfig_dataset_dir_is_mutated_then_frozen_instance_error_is_raised():
    cfg = BCConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.dataset_dir = "/tmp/evil"  # type: ignore[misc]


def test_when_heuristicconfig_attack_min_odds_is_mutated_then_frozen_instance_error_is_raised():
    cfg = HeuristicConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.attack_min_odds = 99.0  # type: ignore[misc]


def test_when_heuristicconfig_attack_dice_policy_is_mutated_then_frozen_instance_error_is_raised():
    cfg = HeuristicConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.attack_dice_policy = "min"  # type: ignore[misc]


# Property: every declared BCConfig field is immutable.
@given(st.sampled_from(sorted(_BC_ALL_FIELDS)))
@settings(max_examples=len(_BC_ALL_FIELDS))
def test_when_any_bcconfig_field_is_assigned_then_frozen_instance_error_is_raised(field_name):
    cfg = BCConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(cfg, field_name, None)


# Property: every declared HeuristicConfig field is immutable.
@given(st.sampled_from(sorted(_HEURISTIC_REQUIRED_FIELDS)))
@settings(max_examples=len(_HEURISTIC_REQUIRED_FIELDS))
def test_when_any_heuristicconfig_field_is_assigned_then_frozen_instance_error_is_raised(
    field_name,
):
    cfg = HeuristicConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(cfg, field_name, None)


# ---------------------------------------------------------------------------
# safe_torch round-trip — BCConfig + HeuristicConfig survive weights_only load
# ---------------------------------------------------------------------------


def test_when_checkpoint_embedding_bcconfig_is_saved_then_safe_torch_load_returns_training_config():
    from src.utils.safe_torch import safe_torch_load

    cfg = TrainingConfig(bc=BCConfig(lr=0.001))
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = Path(f.name)
    try:
        torch.save(cfg, path)
        loaded = safe_torch_load(path)
        assert isinstance(loaded, TrainingConfig)
    finally:
        path.unlink(missing_ok=True)


def test_when_checkpoint_embedding_bcconfig_is_loaded_then_bc_field_is_bcconfig():
    from src.utils.safe_torch import safe_torch_load

    cfg = TrainingConfig(bc=BCConfig(lr=0.002))
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = Path(f.name)
    try:
        torch.save(cfg, path)
        loaded = safe_torch_load(path)
        assert isinstance(loaded.bc, BCConfig)
    finally:
        path.unlink(missing_ok=True)


def test_when_checkpoint_embedding_bcconfig_is_loaded_then_heuristic_field_is_heuristicconfig():
    from src.utils.safe_torch import safe_torch_load

    cfg = TrainingConfig(bc=BCConfig())
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = Path(f.name)
    try:
        torch.save(cfg, path)
        loaded = safe_torch_load(path)
        assert isinstance(loaded.bc.heuristic, HeuristicConfig)
    finally:
        path.unlink(missing_ok=True)


def test_when_checkpoint_embedding_bcconfig_is_round_tripped_then_lr_value_is_preserved():
    from src.utils.safe_torch import safe_torch_load

    original_lr = 0.00314
    cfg = TrainingConfig(bc=BCConfig(lr=original_lr))
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = Path(f.name)
    try:
        torch.save(cfg, path)
        loaded = safe_torch_load(path)
        assert loaded.bc.lr == pytest.approx(original_lr)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# YAML round-trip — load_config reconstructs correct frozen types
# ---------------------------------------------------------------------------

_YAML_WITH_BC_AND_HEURISTIC = """\
bc:
  lr: 0.005
  n_games: 50
  demonstrator: mixed
  heuristic:
    attack_min_odds: 2.5
    capture_move_fraction: 0.7
"""


def test_when_yaml_with_bc_block_is_loaded_then_result_is_training_config():
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(_YAML_WITH_BC_AND_HEURISTIC)
        path = Path(f.name)
    try:
        cfg = load_config(path)
        assert isinstance(cfg, TrainingConfig)
    finally:
        path.unlink(missing_ok=True)


def test_when_yaml_with_bc_block_is_loaded_then_bc_is_bcconfig():
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(_YAML_WITH_BC_AND_HEURISTIC)
        path = Path(f.name)
    try:
        cfg = load_config(path)
        assert isinstance(cfg.bc, BCConfig)
    finally:
        path.unlink(missing_ok=True)


def test_when_yaml_with_bc_block_is_loaded_then_bc_lr_value_is_correct():
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(_YAML_WITH_BC_AND_HEURISTIC)
        path = Path(f.name)
    try:
        cfg = load_config(path)
        assert cfg.bc.lr == pytest.approx(0.005)
    finally:
        path.unlink(missing_ok=True)


def test_when_yaml_with_bc_block_is_loaded_then_bc_demonstrator_value_is_correct():
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(_YAML_WITH_BC_AND_HEURISTIC)
        path = Path(f.name)
    try:
        cfg = load_config(path)
        assert cfg.bc.demonstrator == "mixed"
    finally:
        path.unlink(missing_ok=True)


def test_when_yaml_with_heuristic_block_is_loaded_then_heuristic_is_heuristicconfig():
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(_YAML_WITH_BC_AND_HEURISTIC)
        path = Path(f.name)
    try:
        cfg = load_config(path)
        assert isinstance(cfg.bc.heuristic, HeuristicConfig)
    finally:
        path.unlink(missing_ok=True)


def test_when_yaml_with_heuristic_block_is_loaded_then_attack_min_odds_value_is_correct():
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(_YAML_WITH_BC_AND_HEURISTIC)
        path = Path(f.name)
    try:
        cfg = load_config(path)
        assert cfg.bc.heuristic.attack_min_odds == pytest.approx(2.5)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# merge_cli_overrides — dot-path casting and unknown-path validation
# ---------------------------------------------------------------------------


def test_when_bc_lr_override_is_applied_then_lr_becomes_float():
    cfg = TrainingConfig()
    updated = merge_cli_overrides(cfg, {"bc.lr": "0.001"})
    assert updated.bc.lr == pytest.approx(0.001)
    assert isinstance(updated.bc.lr, float)


def test_when_bc_demonstrator_override_is_applied_then_demonstrator_becomes_str_mixed():
    cfg = TrainingConfig()
    updated = merge_cli_overrides(cfg, {"bc.demonstrator": "mixed"})
    assert updated.bc.demonstrator == "mixed"
    assert isinstance(updated.bc.demonstrator, str)


def test_when_bc_heuristic_attack_min_odds_override_is_applied_then_value_becomes_float():
    cfg = TrainingConfig()
    updated = merge_cli_overrides(cfg, {"bc.heuristic.attack_min_odds": "2.0"})
    assert updated.bc.heuristic.attack_min_odds == pytest.approx(2.0)
    assert isinstance(updated.bc.heuristic.attack_min_odds, float)


def test_when_unknown_bc_path_is_overridden_then_value_error_is_raised():
    cfg = TrainingConfig()
    with pytest.raises(ValueError):
        merge_cli_overrides(cfg, {"bc.nope": "x"})


def test_when_cli_override_is_applied_then_original_config_is_unchanged():
    """merge_cli_overrides must be non-destructive (returns a new frozen object)."""
    cfg = TrainingConfig()
    original_lr = cfg.bc.lr
    merge_cli_overrides(cfg, {"bc.lr": "0.999"})
    assert cfg.bc.lr == pytest.approx(original_lr)
