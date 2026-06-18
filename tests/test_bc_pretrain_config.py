"""Source-blind example tests for issue #86.

Derived entirely from the acceptance criteria text. No implementation source
was read; every assertion targets observable black-box behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import TrainingConfig, load_config, merge_cli_overrides

_BC_PRETRAIN_YAML = Path("config/bc_pretrain.yaml")

# ---------------------------------------------------------------------------
# Criterion: config/bc_pretrain.yaml exists and supplies required blocks
# ---------------------------------------------------------------------------


def test_when_bc_pretrain_yaml_is_checked_then_file_exists():
    assert _BC_PRETRAIN_YAML.exists(), "config/bc_pretrain.yaml must exist"


def test_when_bc_pretrain_yaml_is_parsed_then_bc_block_is_present():
    raw = yaml.safe_load(_BC_PRETRAIN_YAML.read_text())
    assert "bc" in raw, "YAML must contain a top-level 'bc:' block"


def test_when_bc_pretrain_yaml_is_parsed_then_bc_block_contains_all_bcconfig_fields():
    """Every BCConfig field must appear explicitly in the YAML bc: block."""
    raw = yaml.safe_load(_BC_PRETRAIN_YAML.read_text())
    bc = raw.get("bc", {})
    required_fields = [
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
        "explore_eps",
        "label_smoothing",
        "entropy_coef",
    ]
    for field in required_fields:
        assert field in bc, f"bc: block is missing required field '{field}'"


def test_when_bc_pretrain_yaml_is_parsed_then_ppo_block_with_gamma_is_present():
    raw = yaml.safe_load(_BC_PRETRAIN_YAML.read_text())
    assert "ppo" in raw, "YAML must contain a top-level 'ppo:' block"
    assert "gamma" in raw["ppo"], "ppo: block must contain 'gamma'"


def test_when_bc_pretrain_yaml_is_parsed_then_network_hidden_sizes_is_present():
    raw = yaml.safe_load(_BC_PRETRAIN_YAML.read_text())
    assert "network" in raw, "YAML must contain a top-level 'network:' block"
    assert "hidden_sizes" in raw["network"], "network: block must contain 'hidden_sizes'"


def test_when_bc_pretrain_yaml_is_parsed_then_reward_block_is_present():
    raw = yaml.safe_load(_BC_PRETRAIN_YAML.read_text())
    assert "reward" in raw, "YAML must contain a top-level 'reward:' block"


# ---------------------------------------------------------------------------
# Criterion: load_config returns TrainingConfig with correct bc field values
# ---------------------------------------------------------------------------


def test_when_load_config_called_with_bc_pretrain_yaml_then_training_config_is_returned():
    cfg = load_config(_BC_PRETRAIN_YAML)
    assert isinstance(cfg, TrainingConfig)


def test_when_load_config_called_with_bc_pretrain_yaml_then_bc_dataset_dir_is_data_bc():
    cfg = load_config(_BC_PRETRAIN_YAML)
    assert cfg.bc.dataset_dir == "data/bc"


def test_when_load_config_called_with_bc_pretrain_yaml_then_bc_output_path_is_pretrained_pt():
    cfg = load_config(_BC_PRETRAIN_YAML)
    assert cfg.bc.output_path == "models/pretrained.pt"


def test_when_load_config_called_with_bc_pretrain_yaml_then_bc_n_players_is_6():
    cfg = load_config(_BC_PRETRAIN_YAML)
    assert cfg.bc.n_players == 6


def test_when_load_config_called_with_bc_pretrain_yaml_then_bc_values_match_yaml():
    """cfg.bc values must reflect what the YAML declares, not silently fall back to defaults.

    Spot-checks every explicitly declared field in the YAML bc: block against cfg.bc.
    This catches the case where the YAML is present but the loader ignores the bc: block.
    """
    raw = yaml.safe_load(_BC_PRETRAIN_YAML.read_text())
    cfg = load_config(_BC_PRETRAIN_YAML)
    bc_raw = raw.get("bc", {})
    if "n_games" in bc_raw:
        assert cfg.bc.n_games == bc_raw["n_games"]
    if "seed" in bc_raw:
        assert cfg.bc.seed == bc_raw["seed"]
    if "shard_size" in bc_raw:
        assert cfg.bc.shard_size == bc_raw["shard_size"]
    if "epochs" in bc_raw:
        assert cfg.bc.epochs == bc_raw["epochs"]
    if "batch_size" in bc_raw:
        assert cfg.bc.batch_size == bc_raw["batch_size"]
    if "lr" in bc_raw:
        assert cfg.bc.lr == pytest.approx(bc_raw["lr"])
    if "explore_eps" in bc_raw:
        assert cfg.bc.explore_eps == pytest.approx(bc_raw["explore_eps"])
    if "label_smoothing" in bc_raw:
        assert cfg.bc.label_smoothing == pytest.approx(bc_raw["label_smoothing"])
    if "entropy_coef" in bc_raw:
        assert cfg.bc.entropy_coef == pytest.approx(bc_raw["entropy_coef"])


# ---------------------------------------------------------------------------
# Criterion: cfg.ppo.gamma and cfg.network.hidden_sizes present after load
# ---------------------------------------------------------------------------


def test_when_load_config_called_then_ppo_gamma_is_present_and_is_float():
    cfg = load_config(_BC_PRETRAIN_YAML)
    assert hasattr(cfg.ppo, "gamma"), "cfg.ppo.gamma must exist after load"
    assert isinstance(cfg.ppo.gamma, float)


def test_when_load_config_called_then_network_hidden_sizes_is_present_and_non_empty():
    cfg = load_config(_BC_PRETRAIN_YAML)
    assert hasattr(cfg.network, "hidden_sizes"), "cfg.network.hidden_sizes must exist after load"
    assert len(cfg.network.hidden_sizes) > 0


def test_when_load_config_called_then_ppo_gamma_matches_yaml():
    raw = yaml.safe_load(_BC_PRETRAIN_YAML.read_text())
    cfg = load_config(_BC_PRETRAIN_YAML)
    assert cfg.ppo.gamma == pytest.approx(raw["ppo"]["gamma"])


def test_when_load_config_called_then_network_hidden_sizes_matches_yaml():
    raw = yaml.safe_load(_BC_PRETRAIN_YAML.read_text())
    cfg = load_config(_BC_PRETRAIN_YAML)
    assert list(cfg.network.hidden_sizes) == raw["network"]["hidden_sizes"]


# ---------------------------------------------------------------------------
# Criterion: --override bc.* works via merge_cli_overrides
# ---------------------------------------------------------------------------


def test_when_bc_n_games_override_applied_then_cfg_bc_n_games_is_updated():
    cfg = load_config(_BC_PRETRAIN_YAML)
    updated = merge_cli_overrides(cfg, {"bc.n_games": "20"})
    assert updated.bc.n_games == 20


def test_when_bc_epochs_override_applied_then_cfg_bc_epochs_is_updated():
    cfg = load_config(_BC_PRETRAIN_YAML)
    updated = merge_cli_overrides(cfg, {"bc.epochs": "1"})
    assert updated.bc.epochs == 1


def test_when_bc_lr_override_applied_then_cfg_bc_lr_is_updated():
    cfg = load_config(_BC_PRETRAIN_YAML)
    updated = merge_cli_overrides(cfg, {"bc.lr": "0.001"})
    assert updated.bc.lr == pytest.approx(0.001)


def test_when_bc_dataset_dir_override_applied_then_cfg_bc_dataset_dir_is_updated():
    cfg = load_config(_BC_PRETRAIN_YAML)
    updated = merge_cli_overrides(cfg, {"bc.dataset_dir": "data/custom_bc"})
    assert updated.bc.dataset_dir == "data/custom_bc"


def test_when_unknown_bc_field_used_in_override_then_error_is_raised():
    """An unknown bc.* override path must raise, not silently succeed."""
    cfg = load_config(_BC_PRETRAIN_YAML)
    with pytest.raises((ValueError, KeyError)):
        merge_cli_overrides(cfg, {"bc.nonexistent_field_xyz": "99"})


def test_when_unknown_top_level_section_used_in_override_then_error_is_raised():
    cfg = load_config(_BC_PRETRAIN_YAML)
    with pytest.raises((ValueError, KeyError)):
        merge_cli_overrides(cfg, {"completely_unknown_section.field": "1"})


def test_when_original_config_is_used_after_override_then_it_is_unchanged():
    """merge_cli_overrides must return a new config; the original must be immutable."""
    cfg = load_config(_BC_PRETRAIN_YAML)
    original_n_games = cfg.bc.n_games
    merge_cli_overrides(cfg, {"bc.n_games": "9999"})
    assert cfg.bc.n_games == original_n_games


# ---------------------------------------------------------------------------
# Criterion: data/bc/ in .gitignore; checkpoints under models/ remain tracked
# ---------------------------------------------------------------------------


def test_when_gitignore_is_read_then_data_bc_directory_is_excluded():
    gitignore_path = Path(".gitignore")
    assert gitignore_path.exists(), ".gitignore file must exist"
    lines = [ln.strip() for ln in gitignore_path.read_text().splitlines()]
    assert any(ln in {"data/bc", "data/bc/"} for ln in lines), (
        "'data/bc' or 'data/bc/' must appear in .gitignore"
    )


def test_when_gitignore_is_read_then_models_directory_is_not_globally_ignored():
    """Checkpoints under models/ must remain tracked in git."""
    active_lines = [
        ln.strip()
        for ln in Path(".gitignore").read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert "models/" not in active_lines, "'models/' must not appear as a gitignore pattern"
    assert "models" not in active_lines, "'models' must not appear as a gitignore pattern"


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# Invariant: bc.* integer overrides round-trip through merge_cli_overrides
# ---------------------------------------------------------------------------


@given(st.integers(min_value=1, max_value=100_000))
@settings(max_examples=25)
def test_when_bc_n_games_overridden_with_any_positive_integer_then_value_is_preserved(n_games):
    """bc.n_games override round-trips for any positive integer."""
    cfg = load_config(_BC_PRETRAIN_YAML)
    updated = merge_cli_overrides(cfg, {"bc.n_games": str(n_games)})
    assert updated.bc.n_games == n_games


@given(st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=25)
def test_when_bc_seed_overridden_with_any_non_negative_integer_then_value_is_preserved(seed):
    """bc.seed override round-trips for any non-negative integer."""
    cfg = load_config(_BC_PRETRAIN_YAML)
    updated = merge_cli_overrides(cfg, {"bc.seed": str(seed)})
    assert updated.bc.seed == seed


@given(st.integers(min_value=1, max_value=1_000))
@settings(max_examples=25)
def test_when_bc_epochs_overridden_with_any_positive_integer_then_value_is_preserved(epochs):
    """bc.epochs override round-trips for any positive integer."""
    cfg = load_config(_BC_PRETRAIN_YAML)
    updated = merge_cli_overrides(cfg, {"bc.epochs": str(epochs)})
    assert updated.bc.epochs == epochs
