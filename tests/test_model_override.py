"""Tests for the ``--model`` train override and Ollama model pre-flight check.

Covers the wiring added so ``risiko-rl train --model <name>`` swaps the model
for every LLM opponent uniformly, plus the startup validation that fails fast
when a requested model is not available on the Ollama server.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from risiko_rl.cli import _preflight_validate_model
from src.config import SelfPlayConfig, TrainingConfig, merge_cli_overrides

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_6P = str(PROJECT_ROOT / "config" / "default_6p.yaml")


# ──────────────────────────────────────────────────────────────────────────────
# Config: self_play.llm_model round-trip
# ──────────────────────────────────────────────────────────────────────────────


class TestLlmModelOverrideConfig:
    def test_when_field_omitted_then_defaults_to_none(self) -> None:
        assert SelfPlayConfig().llm_model is None

    def test_when_overridden_then_round_trips_through_cli_merge(self) -> None:
        cfg = TrainingConfig(self_play=SelfPlayConfig(llm_profiles_path="x"))
        merged = merge_cli_overrides(cfg, {"self_play.llm_model": "kimi-k2.6:cloud"})
        assert merged.self_play.llm_model == "kimi-k2.6:cloud"


# ──────────────────────────────────────────────────────────────────────────────
# _build_llm_opponents: uniform model override, per-slot sampling preserved
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildLlmOpponentsModelOverride:
    def test_when_llm_model_set_then_all_opponents_use_it(self) -> None:
        from training.self_play import _build_llm_opponents

        opponents = _build_llm_opponents(6, DEFAULT_6P, llm_model="custom:model")
        assert len(opponents) == 5
        assert all(op is not None and op._model == "custom:model" for op in opponents)

    def test_when_llm_model_set_then_per_slot_sampling_is_preserved(self) -> None:
        from training.self_play import _build_llm_opponents

        opponents = _build_llm_opponents(6, DEFAULT_6P, llm_model="custom:model")
        # First five profiles (player_id 0..4) have these temperatures.
        assert [round(op._temperature, 1) for op in opponents] == [0.1, 0.4, 0.7, 0.3, 0.9]

    def test_when_llm_model_omitted_then_profile_models_are_kept(self) -> None:
        from training.self_play import _build_llm_opponents

        opponents = _build_llm_opponents(6, DEFAULT_6P)
        assert all(op._model == "gemma4:12b-mlx" for op in opponents)


# ──────────────────────────────────────────────────────────────────────────────
# list_ollama_models: parse the /v1/models payload, swallow errors as None
# ──────────────────────────────────────────────────────────────────────────────


def _fake_models_response(ids: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"data": [{"id": i} for i in ids]}
    return resp


class TestListOllamaModels:
    def test_when_server_responds_then_returns_id_set(self) -> None:
        with (
            patch("src.agents.ollama_client.ensure_env_loaded", return_value=None),
            patch(
                "src.agents.ollama_client.httpx.get",
                return_value=_fake_models_response(["gemma4:12b-mlx", "glm-5.1:cloud"]),
            ),
        ):
            from src.agents.ollama_client import list_ollama_models

            assert list_ollama_models() == {"gemma4:12b-mlx", "glm-5.1:cloud"}

    def test_when_http_error_then_returns_none(self) -> None:
        import httpx

        with (
            patch("src.agents.ollama_client.ensure_env_loaded", return_value=None),
            patch(
                "src.agents.ollama_client.httpx.get",
                side_effect=httpx.ConnectError("refused"),
            ),
        ):
            from src.agents.ollama_client import list_ollama_models

            assert list_ollama_models() is None


# ──────────────────────────────────────────────────────────────────────────────
# CLI pre-flight: fail fast on missing model, proceed on unknown/available
# ──────────────────────────────────────────────────────────────────────────────


def _llm_cfg(model: str | None) -> TrainingConfig:
    return TrainingConfig(self_play=SelfPlayConfig(llm_profiles_path=DEFAULT_6P, llm_model=model))


class TestPreflightValidateModel:
    def test_when_model_missing_then_raises_bad_parameter(self) -> None:
        with (
            patch(
                "src.agents.ollama_client.list_ollama_models",
                return_value={"gemma4:12b-mlx"},
            ),
            pytest.raises(typer.BadParameter),
        ):
            _preflight_validate_model(_llm_cfg("not-a-real-model"))

    def test_when_model_available_then_passes(self) -> None:
        with patch(
            "src.agents.ollama_client.list_ollama_models",
            return_value={"glm-5.1:cloud", "gemma4:12b-mlx"},
        ):
            _preflight_validate_model(_llm_cfg("glm-5.1:cloud"))  # no raise

    def test_when_server_unreachable_then_proceeds(self) -> None:
        with patch("src.agents.ollama_client.list_ollama_models", return_value=None):
            _preflight_validate_model(_llm_cfg("anything:latest"))  # no raise

    def test_when_no_profiles_path_then_skips_validation(self) -> None:
        # self-play mode: no LLM opponents → never queries the server.
        with patch(
            "src.agents.ollama_client.list_ollama_models",
            side_effect=AssertionError("should not be called"),
        ):
            _preflight_validate_model(TrainingConfig(self_play=SelfPlayConfig()))
