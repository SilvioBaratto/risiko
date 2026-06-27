"""
Shape tests for config/tournament.yaml — Issue #115.

Derived solely from the acceptance criteria; no implementation source was read.
All tests are expected to FAIL (Red) until the implementation is complete.
"""

import pathlib

import pytest
import yaml

CONFIG_PATH = pathlib.Path("config/tournament.yaml")

REQUIRED_ROSTER = frozenset(
    {
        # Light roster (Low/Medium usage-level) — see config/tournament.yaml.
        "gpt-oss:20b-cloud",
        "gpt-oss:120b-cloud",
        "gemma4:cloud",
        "qwen3.5:cloud",
        "nemotron-3-super:cloud",
        "minimax-m2.1:cloud",
    }
)

CATALOG_MODULE_REF = "src.agents.strategies"


@pytest.fixture(scope="module")
def cfg():
    """Load config/tournament.yaml once per test module."""
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


# ── YAML loading ──────────────────────────────────────────────────────────────


def test_when_config_file_loaded_then_parses_as_dict(cfg):
    """config/tournament.yaml must load via yaml.safe_load and return a mapping."""
    assert isinstance(cfg, dict)


# ── 6-model roster ────────────────────────────────────────────────────────────


def test_when_models_key_read_then_exactly_six_entries_present(cfg):
    """Models must have exactly 6 entries."""
    assert len(cfg["models"]) == 6


def test_when_models_key_read_then_equals_required_roster(cfg):
    """Models set must match the required 6-model roster exactly."""
    assert set(cfg["models"]) == REQUIRED_ROSTER


# ── Diplomacy rounds ──────────────────────────────────────────────────────────


def test_when_n_rounds_read_then_at_least_one(cfg):
    """n_rounds must be >= 1. Reduced from 2 to 1 for throughput; combined with
    negotiation_cadence > 1 this cuts the dominant per-turn diplomacy call cost.
    """
    assert cfg["n_rounds"] >= 1


def test_when_negotiation_cadence_read_then_at_least_one(cfg):
    """negotiation_cadence must be >= 1 (negotiate every Nth player-turn)."""
    assert cfg["negotiation_cadence"] >= 1


# ── Scale / concurrency ───────────────────────────────────────────────────────


def test_when_n_games_read_then_is_positive_integer(cfg):
    """Target game count (n_games) must be a positive int."""
    n = cfg["n_games"]
    assert isinstance(n, int), f"expected int, got {type(n).__name__}"
    assert n > 0


def test_when_max_concurrency_read_then_is_positive_integer(cfg):
    """max_concurrency must be a positive int."""
    c = cfg["max_concurrency"]
    assert isinstance(c, int), f"expected int, got {type(c).__name__}"
    assert c > 0


# ── Reproducibility seed ──────────────────────────────────────────────────────


def test_when_seed_read_then_is_integer(cfg):
    """Seed must be an integer."""
    assert isinstance(cfg["seed"], int), f"expected int, got {type(cfg['seed']).__name__}"


# ── Strategy catalog reference ────────────────────────────────────────────────


def test_when_catalog_ref_read_then_points_at_src_agents_strategies(cfg):
    """strategy_catalog must reference 'src.agents.strategies'."""
    assert cfg["strategy_catalog"] == CATALOG_MODULE_REF


def test_when_strategies_list_read_then_matches_strategy_catalog_slugs(cfg):
    """Strategies must equal {s.name for s in STRATEGY_CATALOG}."""
    from src.agents.strategies import STRATEGY_CATALOG  # Red: ImportError until implemented

    expected = {s.name for s in STRATEGY_CATALOG}
    assert set(cfg["strategies"]) == expected, (
        f"config strategies {set(cfg['strategies'])} != catalog slugs {expected}"
    )


# ── Sampling parameters ───────────────────────────────────────────────────────


def test_when_temperature_declared_then_in_closed_unit_interval(cfg):
    """If temperature is declared, it must be in [0.0, 1.0]."""
    temp = cfg.get("temperature")
    if temp is not None:
        assert 0.0 <= float(temp) <= 1.0, f"temperature {temp} outside [0, 1]"


def test_when_top_p_declared_then_in_closed_unit_interval(cfg):
    """If top_p is declared, it must be in [0.0, 1.0]."""
    top_p = cfg.get("top_p")
    if top_p is not None:
        assert 0.0 <= float(top_p) <= 1.0, f"top_p {top_p} outside [0, 1]"


# ── Negative guard ────────────────────────────────────────────────────────────


def test_when_tournament_yaml_checked_then_absent_from_training_yamls():
    """config/tournament.yaml must NOT appear in TRAINING_YAMLS (never fed to load_config)."""
    from src.config import TRAINING_YAMLS  # Red: AttributeError until the list is defined

    training_paths = [str(p) for p in TRAINING_YAMLS]
    assert "config/tournament.yaml" not in training_paths, (
        "tournament.yaml must not be in TRAINING_YAMLS"
    )
    assert not any("tournament.yaml" in p for p in training_paths), (
        "no TRAINING_YAMLS path may contain 'tournament.yaml'"
    )
