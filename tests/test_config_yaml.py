"""Round-trip and variant-value tests for the shipped YAML configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import TrainingConfig, load_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
TRAINING_YAMLS = ["default.yaml", "llm_6p.yaml", "random_6p_pretrain.yaml"]


def _raw(name: str) -> dict:
    """Load a YAML file as a plain dict (no dataclass coercion)."""
    return yaml.safe_load((CONFIG_DIR / name).read_text())


class TestTrainingYamlRoundTrip:
    """Every TrainingConfig YAML loads into a valid TrainingConfig."""

    @pytest.mark.parametrize("name", TRAINING_YAMLS)
    def test_when_loaded_then_returns_training_config(self, name: str) -> None:
        """load_config returns a TrainingConfig for each shipped YAML."""
        assert isinstance(load_config(CONFIG_DIR / name), TrainingConfig)


class TestDefaultYaml:
    """default.yaml 2-player baseline."""

    def test_when_default_then_n_players_is_2(self) -> None:
        """Baseline is a 2-player game."""
        assert load_config(CONFIG_DIR / "default.yaml").self_play.n_players == 2

    def test_when_default_then_max_turns_present_explicitly(self) -> None:
        """max_turns: 2000 must be written explicitly in the file."""
        raw = _raw("default.yaml")
        assert "max_turns" in raw
        assert raw["max_turns"] == 2000


class TestLlm6pYaml:
    """llm_6p.yaml — RL agent vs 5 LLM opponents."""

    def test_when_llm_6p_then_six_players(self) -> None:
        """LLM curriculum runs with 6 players."""
        assert load_config(CONFIG_DIR / "llm_6p.yaml").self_play.n_players == 6

    def test_when_llm_6p_then_profiles_path_points_to_default_6p(self) -> None:
        """The LLM profile pool path is wired in."""
        cfg = load_config(CONFIG_DIR / "llm_6p.yaml")
        assert cfg.self_play.llm_profiles_path == "config/default_6p.yaml"


class TestRandom6pPretrainYaml:
    """random_6p_pretrain.yaml — Phase-1 pretrain, no LLM."""

    def test_when_pretrain_then_short_horizon_six_players(self) -> None:
        """Pretrain uses 100k timesteps, 6 players, 1500-turn cap."""
        cfg = load_config(CONFIG_DIR / "random_6p_pretrain.yaml")
        assert cfg.total_timesteps == 100000
        assert cfg.self_play.n_players == 6
        assert cfg.max_turns == 1500

    def test_when_pretrain_then_llm_profiles_path_omitted(self) -> None:
        """No LLM profile path — random fillers only."""
        assert "llm_profiles_path" not in _raw("random_6p_pretrain.yaml")["self_play"]
        cfg = load_config(CONFIG_DIR / "random_6p_pretrain.yaml")
        assert cfg.self_play.llm_profiles_path is None


class TestDefault6pProfiles:
    """default_6p.yaml — the per-player LLM sampling pool (not a TrainingConfig)."""

    PROFILE_KEYS = {"temperature", "top_p", "top_k", "repeat_penalty", "strategy_hint"}

    def test_when_loaded_then_exactly_six_profiles(self) -> None:
        """The pool holds six player profiles."""
        assert len(_raw("default_6p.yaml")) == 6

    def test_when_loaded_then_each_profile_has_required_keys(self) -> None:
        """Every profile carries the five sampling fields."""
        for profile in _raw("default_6p.yaml"):
            assert set(profile) >= self.PROFILE_KEYS

    def test_when_loaded_then_temperatures_distinct_and_span_range(self) -> None:
        """Temperatures are distinct and span 0.1–0.9."""
        temps = [p["temperature"] for p in _raw("default_6p.yaml")]
        assert len(set(temps)) == 6
        assert min(temps) == 0.1
        assert max(temps) == 0.9
