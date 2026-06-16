"""Spec-derived tests for issue #58: PlayerConfig, DEFAULT_6P_PROFILES, YAML pool.

Criteria covered:
  [UNIT] PlayerConfig is a frozen dataclass with player_id, temperature,
         top_p, strategy_hint, model (default "gpt-4.1")
  [UNIT] DEFAULT_6P_PROFILES matches the six-profile table exactly
  [UNIT] load_profiles_from_yaml() round-trips config/default_6p.yaml;
         raises ValueError on out-of-range (player_id not in 0–5) or
         duplicate player_id
  [UNIT] config/default_6p.yaml and config/llm_6p.yaml parse cleanly and
         contain only Azure-relevant keys (no Ollama/BAML residue)
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import tempfile

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.player_config import DEFAULT_6P_PROFILES, PlayerConfig, load_profiles_from_yaml

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _write_profiles_yaml(profiles: list[PlayerConfig], path: str | pathlib.Path) -> None:
    """Serialise a list of PlayerConfig objects to a YAML file for round-trip tests."""
    data = [
        {
            "player_id": p.player_id,
            "temperature": float(p.temperature),
            "top_p": float(p.top_p),
            "strategy_hint": p.strategy_hint,
            "model": p.model,
        }
        for p in profiles
    ]
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh)


# ────────────────────────────────────────────────────────────────────────────
# PlayerConfig — frozen dataclass contract
# ────────────────────────────────────────────────────────────────────────────


class TestPlayerConfigDataclass:
    """Frozen dataclass contract for PlayerConfig."""

    def test_when_model_omitted_then_default_is_gpt_41(self):
        """Criterion: model field defaults to 'gpt-4.1'."""
        pc = PlayerConfig(player_id=0, temperature=0.5, top_p=0.9, strategy_hint="hint")
        assert pc.model == "gpt-4.1"

    def test_when_all_fields_supplied_then_fields_are_stored_correctly(self):
        """Criterion: PlayerConfig has player_id, temperature, top_p, strategy_hint, model."""
        pc = PlayerConfig(
            player_id=3, temperature=0.3, top_p=0.95, strategy_hint="defend", model="gpt-4.1"
        )
        assert pc.player_id == 3
        assert pc.temperature == pytest.approx(0.3)
        assert pc.top_p == pytest.approx(0.95)
        assert pc.strategy_hint == "defend"
        assert pc.model == "gpt-4.1"

    def test_when_attribute_mutated_then_frozen_error_is_raised(self):
        """Criterion: PlayerConfig is frozen (immutable after construction)."""
        pc = PlayerConfig(player_id=0, temperature=0.5, top_p=0.9, strategy_hint="hint")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            pc.temperature = 0.99  # type: ignore[misc]

    def test_when_class_introspected_then_it_is_a_dataclass(self):
        """Criterion: PlayerConfig is a dataclass (not a plain class or namedtuple)."""
        assert dataclasses.is_dataclass(PlayerConfig)

    def test_when_two_identical_instances_compared_then_they_are_equal(self):
        """Frozen dataclasses with the same fields must compare equal (value semantics)."""
        pc1 = PlayerConfig(player_id=1, temperature=0.4, top_p=0.9, strategy_hint="same")
        pc2 = PlayerConfig(player_id=1, temperature=0.4, top_p=0.9, strategy_hint="same")
        assert pc1 == pc2

    def test_when_instances_with_different_player_id_compared_then_they_are_not_equal(self):
        pc1 = PlayerConfig(player_id=0, temperature=0.5, top_p=0.9, strategy_hint="hint")
        pc2 = PlayerConfig(player_id=1, temperature=0.5, top_p=0.9, strategy_hint="hint")
        assert pc1 != pc2


# ────────────────────────────────────────────────────────────────────────────
# DEFAULT_6P_PROFILES — spec table must match exactly
# ────────────────────────────────────────────────────────────────────────────

# Source of truth: requirements.md § 4 "Default six-player LLM profile set"
_SPEC_TABLE = [
    (0, 0.1, 0.90, "Play greedily: maximise territory gain each turn."),
    (1, 0.4, 0.90, "Focus on securing full continents before expanding."),
    (2, 0.7, 0.85, "Eliminate the weakest player early; control cards."),
    (3, 0.3, 0.95, "Fortify borders and expand only when safe."),
    (4, 0.9, 0.80, "Play unpredictably; avoid predictable attack patterns."),
    (5, 0.5, 0.90, "Balance attack and defence; trade cards conservatively."),
]


class TestDefault6PProfiles:
    """DEFAULT_6P_PROFILES must match the spec table exactly."""

    def test_when_profiles_counted_then_exactly_six_profiles_exist(self):
        """Criterion: six-profile pool for a 6-player game."""
        assert len(DEFAULT_6P_PROFILES) == 6

    def test_when_player_ids_collected_then_all_ids_zero_through_five_are_present(self):
        """Criterion: profiles cover all player slots 0–5 with no gaps."""
        ids = {p.player_id for p in DEFAULT_6P_PROFILES}
        assert ids == {0, 1, 2, 3, 4, 5}

    @pytest.mark.parametrize("player_id,temperature,top_p,strategy_hint", _SPEC_TABLE)
    def test_when_profile_for_player_retrieved_then_values_match_spec_table(
        self, player_id: int, temperature: float, top_p: float, strategy_hint: str
    ):
        """Criterion: DEFAULT_6P_PROFILES matches the six-profile table exactly."""
        profile = next(p for p in DEFAULT_6P_PROFILES if p.player_id == player_id)
        assert profile.temperature == pytest.approx(temperature), (
            f"player {player_id}: temperature expected {temperature}, got {profile.temperature}"
        )
        assert profile.top_p == pytest.approx(top_p), (
            f"player {player_id}: top_p expected {top_p}, got {profile.top_p}"
        )
        assert profile.strategy_hint == strategy_hint, f"player {player_id}: strategy_hint mismatch"

    def test_when_all_default_models_checked_then_every_profile_uses_gpt_41(self):
        """Criterion: default model for all profiles is 'gpt-4.1'."""
        for profile in DEFAULT_6P_PROFILES:
            assert profile.model == "gpt-4.1", (
                f"player {profile.player_id} has model={profile.model!r}, expected 'gpt-4.1'"
            )

    def test_when_default_profiles_checked_then_no_duplicate_player_ids_exist(self):
        ids = [p.player_id for p in DEFAULT_6P_PROFILES]
        assert len(ids) == len(set(ids)), "Duplicate player_id found in DEFAULT_6P_PROFILES"


# ────────────────────────────────────────────────────────────────────────────
# load_profiles_from_yaml — valid input
# ────────────────────────────────────────────────────────────────────────────


class TestLoadProfilesFromYaml:
    """load_profiles_from_yaml validation, error handling, and round-trip."""

    def test_when_default_6p_yaml_loaded_then_profiles_match_spec_table(self):
        """Criterion: load_profiles_from_yaml() round-trips config/default_6p.yaml."""
        yaml_path = PROJECT_ROOT / "config" / "default_6p.yaml"
        loaded = load_profiles_from_yaml(str(yaml_path))
        by_id = {p.player_id: p for p in loaded}
        for pid, temp, tp, hint in _SPEC_TABLE:
            assert pid in by_id, f"player_id {pid} missing from loaded profiles"
            assert by_id[pid].temperature == pytest.approx(temp)
            assert by_id[pid].top_p == pytest.approx(tp)
            assert by_id[pid].strategy_hint == hint

    def test_when_single_valid_profile_yaml_loaded_then_one_profile_returned(
        self, tmp_path: pathlib.Path
    ):
        data = [
            {
                "player_id": 2,
                "temperature": 0.7,
                "top_p": 0.85,
                "strategy_hint": "attack early",
                "model": "gpt-4.1",
            }
        ]
        path = tmp_path / "profiles.yaml"
        path.write_text(yaml.safe_dump(data))
        loaded = load_profiles_from_yaml(str(path))
        assert len(loaded) == 1
        assert loaded[0].player_id == 2
        assert loaded[0].temperature == pytest.approx(0.7)

    # ── Error cases ──────────────────────────────────────────────────────────

    def test_when_player_id_is_6_then_value_error_is_raised(self, tmp_path: pathlib.Path):
        """Criterion: raises ValueError when player_id not in 0–5 (upper bound)."""
        data = [{"player_id": 6, "temperature": 0.5, "top_p": 0.9, "strategy_hint": "x"}]
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError):
            load_profiles_from_yaml(str(path))

    def test_when_player_id_is_negative_then_value_error_is_raised(self, tmp_path: pathlib.Path):
        """Criterion: raises ValueError when player_id not in 0–5 (lower bound)."""
        data = [{"player_id": -1, "temperature": 0.5, "top_p": 0.9, "strategy_hint": "x"}]
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError):
            load_profiles_from_yaml(str(path))

    def test_when_two_profiles_share_player_id_then_value_error_is_raised(
        self, tmp_path: pathlib.Path
    ):
        """Criterion: raises ValueError on duplicate player_id."""
        data = [
            {"player_id": 0, "temperature": 0.5, "top_p": 0.9, "strategy_hint": "a"},
            {"player_id": 0, "temperature": 0.3, "top_p": 0.8, "strategy_hint": "b"},
        ]
        path = tmp_path / "dup.yaml"
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError):
            load_profiles_from_yaml(str(path))

    def test_when_boundary_player_id_0_loaded_then_no_error_is_raised(self, tmp_path: pathlib.Path):
        """player_id = 0 is the minimum valid value."""
        data = [{"player_id": 0, "temperature": 0.1, "top_p": 0.9, "strategy_hint": "greedy"}]
        path = tmp_path / "valid.yaml"
        path.write_text(yaml.safe_dump(data))
        loaded = load_profiles_from_yaml(str(path))
        assert loaded[0].player_id == 0

    def test_when_boundary_player_id_5_loaded_then_no_error_is_raised(self, tmp_path: pathlib.Path):
        """player_id = 5 is the maximum valid value."""
        data = [{"player_id": 5, "temperature": 0.5, "top_p": 0.9, "strategy_hint": "balance"}]
        path = tmp_path / "valid.yaml"
        path.write_text(yaml.safe_dump(data))
        loaded = load_profiles_from_yaml(str(path))
        assert loaded[0].player_id == 5


# ────────────────────────────────────────────────────────────────────────────
# Round-trip property: write profiles → YAML → load → original data preserved
#
# Criterion: load_profiles_from_yaml() round-trips (implied by "round-trips"
#            in the acceptance criteria text).
# This is an explicit round-trip invariant → @given property.
# ────────────────────────────────────────────────────────────────────────────


@given(
    st.lists(
        st.integers(min_value=0, max_value=5),
        min_size=1,
        max_size=6,
        unique=True,
    )
)
@settings(max_examples=40)
def test_when_profiles_written_to_yaml_and_loaded_back_then_field_values_are_preserved(
    player_ids: list[int],
) -> None:
    """Round-trip invariant: all profile fields survive a YAML serialise/load cycle."""
    profiles = [
        PlayerConfig(
            player_id=pid,
            temperature=round(0.05 * (pid + 1), 4),
            top_p=round(0.85 + 0.01 * pid, 4),
            strategy_hint=f"strategy for player {pid}",
        )
        for pid in sorted(player_ids)
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        tmp_path = fh.name

    try:
        _write_profiles_yaml(profiles, tmp_path)
        loaded = load_profiles_from_yaml(tmp_path)
        loaded_by_id = {p.player_id: p for p in loaded}

        for original in profiles:
            assert original.player_id in loaded_by_id, (
                f"player_id {original.player_id} missing after round-trip"
            )
            lp = loaded_by_id[original.player_id]
            assert lp.temperature == pytest.approx(original.temperature, abs=1e-6)
            assert lp.top_p == pytest.approx(original.top_p, abs=1e-6)
            assert lp.strategy_hint == original.strategy_hint
            assert lp.model == original.model
    finally:
        os.unlink(tmp_path)


# ────────────────────────────────────────────────────────────────────────────
# YAML config files: must parse and contain no Ollama / BAML residue
#
# Criterion: config/default_6p.yaml and config/llm_6p.yaml parse cleanly
#            and contain only Azure-relevant keys.
# ────────────────────────────────────────────────────────────────────────────

_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {"ollama", "baml", "baml_src", "baml_client", "baml_options", "ollama_model"}
)


def _collect_all_keys(obj: object) -> set[str]:
    """Recursively collect all lowercase string keys from a nested YAML value."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k).lower())
            keys |= _collect_all_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _collect_all_keys(item)
    return keys


class TestYamlConfigFilesAzureOnly:
    """YAML config files must parse cleanly and contain only Azure-relevant keys."""

    def test_when_default_6p_yaml_opened_then_it_parses_without_error(self):
        """Criterion: config/default_6p.yaml parses cleanly."""
        yaml_path = PROJECT_ROOT / "config" / "default_6p.yaml"
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh)
        assert data is not None, "default_6p.yaml must not be empty"

    def test_when_default_6p_yaml_keys_inspected_then_no_ollama_or_baml_key_present(self):
        """Criterion: no Ollama/BAML residue in default_6p.yaml."""
        yaml_path = PROJECT_ROOT / "config" / "default_6p.yaml"
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh)
        found = _collect_all_keys(data) & _FORBIDDEN_KEYS
        assert not found, f"Forbidden (Ollama/BAML) keys found in default_6p.yaml: {found}"

    def test_when_llm_6p_yaml_exists_then_it_is_at_expected_path(self):
        """Criterion: config/llm_6p.yaml exists (Azure pool config)."""
        assert (PROJECT_ROOT / "config" / "llm_6p.yaml").exists(), "config/llm_6p.yaml must exist"

    def test_when_llm_6p_yaml_opened_then_it_parses_without_error(self):
        """Criterion: config/llm_6p.yaml parses cleanly."""
        yaml_path = PROJECT_ROOT / "config" / "llm_6p.yaml"
        assert yaml_path.exists(), "config/llm_6p.yaml must exist"
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh)
        assert data is not None, "llm_6p.yaml must not be empty"

    def test_when_llm_6p_yaml_keys_inspected_then_no_ollama_or_baml_key_present(self):
        """Criterion: no Ollama/BAML residue in llm_6p.yaml."""
        yaml_path = PROJECT_ROOT / "config" / "llm_6p.yaml"
        assert yaml_path.exists(), "config/llm_6p.yaml must exist"
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh)
        found = _collect_all_keys(data) & _FORBIDDEN_KEYS
        assert not found, f"Forbidden (Ollama/BAML) keys found in llm_6p.yaml: {found}"
