"""Tests for the LLM opponent with mocked Ollama client."""

from __future__ import annotations

import pickle
from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import patch

import httpx
import numpy as np
import pytest
import torch

from src.agents.base import Agent
from src.agents.llm_opponent import LLMOpponent
from src.agents.player_config import PlayerConfig
from src.env import RisikoEnv

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def agent() -> LLMOpponent:
    return LLMOpponent(timeout=0.5)


@pytest.fixture
def env() -> RisikoEnv:
    return RisikoEnv(n_players=3)


@pytest.fixture
def dummy_obs(env: RisikoEnv) -> dict[str, np.ndarray]:
    obs, _ = env.reset(seed=42)
    return obs


@pytest.fixture
def dummy_legal(env: RisikoEnv) -> list[dict[str, int]]:
    _, info = env.reset(seed=42)
    return info["legal_actions"]


# ------------------------------------------------------------------
# Protocol conformance
# ------------------------------------------------------------------


class TestProtocol:
    """LLMOpponent conforms to the Agent protocol."""

    def test_is_agent(self, agent: LLMOpponent) -> None:
        assert isinstance(agent, Agent)

    def test_act_signature(self, agent: LLMOpponent, dummy_obs, dummy_legal) -> None:
        with patch.object(agent, "_call_with_timeout", return_value=0):
            result = agent.act(dummy_obs, dummy_legal)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"action_type", "param_a", "param_b", "param_c", "param_d"}

    def test_act_with_meta_returns_tensors(
        self, agent: LLMOpponent, dummy_obs, dummy_legal
    ) -> None:
        with patch.object(agent, "_call_with_timeout", return_value=0):
            action, log_prob, value = agent.act_with_meta(dummy_obs, dummy_legal)
        assert isinstance(action, dict)
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(value, torch.Tensor)
        assert torch.isnan(log_prob)
        assert torch.isnan(value)

    def test_empty_legal_actions_raises(self, agent: LLMOpponent, dummy_obs) -> None:
        with pytest.raises(ValueError, match="empty"):
            agent.act(dummy_obs, [])


# ------------------------------------------------------------------
# Fallback scenarios
# ------------------------------------------------------------------


class TestFallback:
    """Fallback to RandomAgent on various failure modes."""

    def test_timeout_fallback(self, agent: LLMOpponent, dummy_obs, dummy_legal) -> None:
        """On ThreadPoolExecutor timeout, fallback to random."""
        with patch.object(agent, "_call_with_timeout", side_effect=FuturesTimeoutError):
            result = agent.act(dummy_obs, dummy_legal)
        assert isinstance(result, dict)
        assert result in dummy_legal

    def test_http_error_fallback(self, agent: LLMOpponent, dummy_obs, dummy_legal) -> None:
        """On HTTP error, fallback to random."""
        with patch.object(agent, "_call_with_timeout", side_effect=httpx.ConnectError("refused")):
            result = agent.act(dummy_obs, dummy_legal)
        assert isinstance(result, dict)
        assert result in dummy_legal

    def test_generic_exception_fallback(self, agent: LLMOpponent, dummy_obs, dummy_legal) -> None:
        """On unexpected exception, fallback to random."""
        with patch.object(agent, "_call_with_timeout", side_effect=RuntimeError("boom")):
            result = agent.act(dummy_obs, dummy_legal)
        assert isinstance(result, dict)
        assert result in dummy_legal

    def test_none_index_fallback(self, agent: LLMOpponent, dummy_obs, dummy_legal) -> None:
        """If LLM returns None (empty/invalid response), fallback to random."""
        with patch.object(agent, "_call_with_timeout", return_value=None):
            result = agent.act(dummy_obs, dummy_legal)
        assert isinstance(result, dict)
        assert result in dummy_legal


# ------------------------------------------------------------------
# Valid action path
# ------------------------------------------------------------------


class TestValidAction:
    """Happy path: LLM returns a valid index into legal_actions."""

    def test_first_legal_action(self, agent: LLMOpponent, dummy_obs, dummy_legal) -> None:
        """Mock LLM returns index 0, which is the first legal action."""
        with patch.object(agent, "_call_with_timeout", return_value=0):
            result = agent.act(dummy_obs, dummy_legal)
        assert result == dummy_legal[0]

    def test_last_legal_action(self, agent: LLMOpponent, dummy_obs, dummy_legal) -> None:
        """Mock LLM returns the last index."""
        last = len(dummy_legal) - 1
        with patch.object(agent, "_call_with_timeout", return_value=last):
            result = agent.act(dummy_obs, dummy_legal)
        assert result == dummy_legal[last]


# ------------------------------------------------------------------
# Constructor / configuration
# ------------------------------------------------------------------


class TestConfiguration:
    """LLMOpponent accepts runtime configuration."""

    def test_default_values(self) -> None:
        a = LLMOpponent()
        assert a._model == "gemma4:12b-mlx"
        assert a._timeout == 120.0
        assert a._temperature == 0.1

    def test_custom_values(self) -> None:
        a = LLMOpponent(model="gpt-oss:20b", timeout=2.0, temperature=0.5)
        assert a._model == "gpt-oss:20b"
        assert a._timeout == 2.0
        assert a._temperature == 0.5

    def test_custom_ollama_overrides(self) -> None:
        a = LLMOpponent(
            base_url="http://localhost:11434/v1",
            api_key="my-ollama-key",
        )
        assert a._base_url == "http://localhost:11434/v1"
        assert a._api_key == "my-ollama-key"

    def test_when_player_config_set_strategy_hint_is_stored(self) -> None:
        from src.agents.player_config import PlayerConfig

        config = PlayerConfig(
            player_id=1,
            temperature=0.4,
            top_p=0.9,
            strategy_hint="Secure continents.",
        )
        a = LLMOpponent(player_config=config)
        assert a._strategy_hint == "Secure continents."

    def test_when_player_config_set_overrides_temperature(self) -> None:
        """player_config.temperature takes precedence over the scalar kwarg."""
        from src.agents.player_config import PlayerConfig

        config = PlayerConfig(
            player_id=0,
            temperature=0.9,
            top_p=0.9,
            strategy_hint="aggressive",
        )
        a = LLMOpponent(temperature=0.1, player_config=config)
        assert a._temperature == pytest.approx(0.9)

    def test_when_player_config_set_uses_config_model(self) -> None:
        from src.agents.player_config import PlayerConfig

        config = PlayerConfig(
            player_id=3,
            temperature=0.3,
            top_p=0.95,
            strategy_hint="Fortify borders.",
        )
        a = LLMOpponent(player_config=config)
        assert a._player_config is config
        assert a._temperature == pytest.approx(0.3)
        assert a._model == config.model


# ------------------------------------------------------------------
# Ollama call wiring
# ------------------------------------------------------------------


class TestOllamaWiring:
    """_call_with_timeout forwards config to the Ollama client."""

    def test_when_act_then_ollama_client_receives_overrides(self, dummy_obs, dummy_legal) -> None:
        from src.agents.player_config import PlayerConfig

        config = PlayerConfig(player_id=0, temperature=0.7, top_p=0.85, strategy_hint="hint")
        a = LLMOpponent(
            player_config=config,
            base_url="http://localhost:11434/v1",
            api_key="test-key",
        )
        with patch(
            "src.agents.llm_opponent.call_ollama_for_action_index", return_value=0
        ) as mock_call:
            a.act(dummy_obs, dummy_legal)
        mock_call.assert_called_once()
        kwargs = mock_call.call_args.kwargs
        assert kwargs["model"] == "gemma4:12b-mlx"
        assert kwargs["base_url"] == "http://localhost:11434/v1"
        assert kwargs["api_key"] == "test-key"
        assert kwargs["temperature"] == pytest.approx(0.7)
        assert kwargs["top_p"] == pytest.approx(0.85)
        assert kwargs["strategy_hint"] == "hint"


# ------------------------------------------------------------------
# Pickle serialization
# ------------------------------------------------------------------


class TestPickle:
    """LLMOpponent is pickle-safe (needed for self-play checkpoints)."""

    def test_pickle_preserves_model(self) -> None:
        a = LLMOpponent(model="gpt-oss:20b", timeout=5.0)
        restored = pickle.loads(pickle.dumps(a))
        assert restored._model == "gpt-oss:20b"

    def test_pickle_preserves_temperature(self) -> None:
        a = LLMOpponent(temperature=0.77)
        restored = pickle.loads(pickle.dumps(a))
        assert restored._temperature == pytest.approx(0.77)

    def test_pickle_preserves_strategy_hint(self) -> None:
        cfg = PlayerConfig(player_id=2, temperature=0.4, top_p=0.85, strategy_hint="Control cards.")
        a = LLMOpponent(player_config=cfg)
        restored = pickle.loads(pickle.dumps(a))
        assert restored._strategy_hint == "Control cards."

    def test_pickle_restored_satisfies_agent_protocol(self) -> None:
        a = LLMOpponent()
        restored = pickle.loads(pickle.dumps(a))
        assert isinstance(restored, Agent)

    def test_pickle_roundtrip_preserves_top_p(self) -> None:
        cfg = PlayerConfig(player_id=0, temperature=0.5, top_p=0.77, strategy_hint="x")
        a = LLMOpponent(player_config=cfg)
        restored = pickle.loads(pickle.dumps(a))
        assert restored._top_p == pytest.approx(0.77)
