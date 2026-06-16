"""Tests for PlayerConfig dataclass, DEFAULT_6P_PROFILES, and load_profiles_from_yaml."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from src.agents.player_config import (
    DEFAULT_6P_PROFILES,
    DEFAULT_MODEL,
    PlayerConfig,
    load_profiles_from_yaml,
)


class TestPlayerConfigFrozen:
    """Frozen dataclass guarantees and field defaults."""

    def test_when_mutating_field_raises_frozen_error(self) -> None:
        cfg = PlayerConfig(
            player_id=0,
            temperature=0.1,
            top_p=0.9,
            strategy_hint="hint",
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.temperature = 0.5  # type: ignore[misc]

    def test_when_model_omitted_defaults_to_gpt_oss_120b(self) -> None:
        cfg = PlayerConfig(
            player_id=0,
            temperature=0.1,
            top_p=0.9,
            strategy_hint="hint",
        )
        assert cfg.model == "gpt-oss:120b"
        assert cfg.model == DEFAULT_MODEL

    def test_when_model_provided_stores_value(self) -> None:
        cfg = PlayerConfig(
            player_id=0,
            temperature=0.1,
            top_p=0.9,
            strategy_hint="hint",
            model="gpt-4o",
        )
        assert cfg.model == "gpt-4o"


class TestDefault6PProfiles:
    """Default six-player profile constant validation."""

    def test_when_loading_defaults_exactly_six_profiles_are_returned(self) -> None:
        assert len(DEFAULT_6P_PROFILES) == 6

    def test_when_checking_type_defaults_is_tuple(self) -> None:
        assert isinstance(DEFAULT_6P_PROFILES, tuple)

    def test_when_checking_player_ids_are_zero_through_five(self) -> None:
        ids = [p.player_id for p in DEFAULT_6P_PROFILES]
        assert ids == list(range(6))

    def test_when_checking_player_0_temperature_is_0_1(self) -> None:
        assert DEFAULT_6P_PROFILES[0].temperature == pytest.approx(0.1)

    def test_when_checking_player_0_top_p_is_0_9(self) -> None:
        assert DEFAULT_6P_PROFILES[0].top_p == pytest.approx(0.9)

    def test_when_checking_player_0_strategy_hint(self) -> None:
        assert DEFAULT_6P_PROFILES[0].strategy_hint == (
            "Play greedily: maximise territory gain each turn."
        )

    def test_when_checking_player_1_temperature_is_0_4(self) -> None:
        assert DEFAULT_6P_PROFILES[1].temperature == pytest.approx(0.4)

    def test_when_checking_player_1_strategy_hint(self) -> None:
        assert DEFAULT_6P_PROFILES[1].strategy_hint == (
            "Focus on securing full continents before expanding."
        )

    def test_when_checking_player_2_temperature_is_0_7(self) -> None:
        assert DEFAULT_6P_PROFILES[2].temperature == pytest.approx(0.7)

    def test_when_checking_player_2_top_p_is_0_85(self) -> None:
        assert DEFAULT_6P_PROFILES[2].top_p == pytest.approx(0.85)

    def test_when_checking_player_2_strategy_hint(self) -> None:
        assert DEFAULT_6P_PROFILES[2].strategy_hint == (
            "Eliminate the weakest player early; control cards."
        )

    def test_when_checking_player_3_temperature_is_0_3(self) -> None:
        assert DEFAULT_6P_PROFILES[3].temperature == pytest.approx(0.3)

    def test_when_checking_player_3_top_p_is_0_95(self) -> None:
        assert DEFAULT_6P_PROFILES[3].top_p == pytest.approx(0.95)

    def test_when_checking_player_3_strategy_hint(self) -> None:
        assert DEFAULT_6P_PROFILES[3].strategy_hint == (
            "Fortify borders and expand only when safe."
        )

    def test_when_checking_player_4_temperature_is_0_9(self) -> None:
        assert DEFAULT_6P_PROFILES[4].temperature == pytest.approx(0.9)

    def test_when_checking_player_4_top_p_is_0_8(self) -> None:
        assert DEFAULT_6P_PROFILES[4].top_p == pytest.approx(0.8)

    def test_when_checking_player_4_strategy_hint(self) -> None:
        assert DEFAULT_6P_PROFILES[4].strategy_hint == (
            "Play unpredictably; avoid predictable attack patterns."
        )

    def test_when_checking_player_5_temperature_is_0_5(self) -> None:
        assert DEFAULT_6P_PROFILES[5].temperature == pytest.approx(0.5)

    def test_when_checking_player_5_strategy_hint(self) -> None:
        assert DEFAULT_6P_PROFILES[5].strategy_hint == (
            "Balance attack and defence; trade cards conservatively."
        )

    def test_when_checking_all_player_ids_are_distinct(self) -> None:
        ids = [p.player_id for p in DEFAULT_6P_PROFILES]
        assert len(set(ids)) == len(ids)

    def test_when_checking_all_models_default_to_gpt_oss_120b(self) -> None:
        assert all(p.model == "gpt-oss:120b" for p in DEFAULT_6P_PROFILES)


class TestLoadProfilesFromYaml:
    """YAML loader validation and error handling."""

    def test_when_loading_default_yaml_round_trips_to_six_profiles(self) -> None:
        yaml_path = Path(__file__).resolve().parents[1] / "config" / "default_6p.yaml"
        profiles = load_profiles_from_yaml(yaml_path)
        assert profiles == list(DEFAULT_6P_PROFILES)

    def test_when_loading_valid_yaml_returns_player_config_objects(self, tmp_path: Path) -> None:
        data = [
            {
                "player_id": 0,
                "temperature": 0.5,
                "top_p": 0.9,
                "strategy_hint": "test hint",
            }
        ]
        path = tmp_path / "profiles.yaml"
        path.write_text(yaml.dump(data))
        profiles = load_profiles_from_yaml(path)
        assert len(profiles) == 1
        assert isinstance(profiles[0], PlayerConfig)
        assert profiles[0].player_id == 0
        assert profiles[0].temperature == pytest.approx(0.5)

    def test_when_yaml_omits_model_field_defaults_to_gpt_oss_120b(self, tmp_path: Path) -> None:
        data = [
            {
                "player_id": 0,
                "temperature": 0.1,
                "top_p": 0.9,
                "strategy_hint": "hint",
            }
        ]
        path = tmp_path / "profiles.yaml"
        path.write_text(yaml.dump(data))
        profiles = load_profiles_from_yaml(path)
        assert profiles[0].model == "gpt-oss:120b"

    def test_when_yaml_has_duplicate_player_ids_raises_value_error(self, tmp_path: Path) -> None:
        data = [
            {"player_id": 0, "temperature": 0.1, "top_p": 0.9, "strategy_hint": "hint"},
            {"player_id": 0, "temperature": 0.4, "top_p": 0.9, "strategy_hint": "hint 2"},
        ]
        path = tmp_path / "profiles.yaml"
        path.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match="duplicate"):
            load_profiles_from_yaml(path)

    def test_when_yaml_has_player_id_above_5_raises_value_error(self, tmp_path: Path) -> None:
        data = [{"player_id": 6, "temperature": 0.1, "top_p": 0.9, "strategy_hint": "hint"}]
        path = tmp_path / "profiles.yaml"
        path.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match="out.of.range"):
            load_profiles_from_yaml(path)

    def test_when_yaml_has_negative_player_id_raises_value_error(self, tmp_path: Path) -> None:
        data = [{"player_id": -1, "temperature": 0.1, "top_p": 0.9, "strategy_hint": "hint"}]
        path = tmp_path / "profiles.yaml"
        path.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match="out.of.range"):
            load_profiles_from_yaml(path)
