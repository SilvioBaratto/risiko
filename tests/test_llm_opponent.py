"""Tests for the LLM opponent with mocked native Ollama client."""

from __future__ import annotations

import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import patch

import httpx
import numpy as np
import pytest
import torch

from src.agents.base import Agent
from src.agents.llm_opponent import LLMOpponent
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
        assert a._model == "risiko"
        assert a._timeout == 30.0
        assert a._temperature == 0.1

    def test_custom_values(self) -> None:
        a = LLMOpponent(model="gemma4:latest", timeout=2.0, temperature=0.5)
        assert a._model == "gemma4:latest"
        assert a._timeout == 2.0
        assert a._temperature == 0.5

    def test_custom_ollama_url(self) -> None:
        a = LLMOpponent(ollama_url="http://custom:11434/v1")
        assert a._ollama_url == "http://custom:11434/v1"

    def test_when_player_config_set_strategy_hint_is_stored(self) -> None:
        from src.agents.player_config import PlayerConfig

        config = PlayerConfig(
            player_id=1,
            temperature=0.4,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
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
            top_k=40,
            repeat_penalty=1.1,
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
            top_k=30,
            repeat_penalty=1.2,
            strategy_hint="Fortify borders.",
        )
        a = LLMOpponent(player_config=config)
        assert a._player_config is config
        assert a._temperature == pytest.approx(0.3)
        assert a._model == config.model


# ------------------------------------------------------------------
# Eviction
# ------------------------------------------------------------------


class TestEviction:
    """evict is opt-in via evict_after_call=True (default off for training)."""

    def test_default_no_evict_on_success(self, dummy_obs, dummy_legal) -> None:
        a = LLMOpponent(timeout=0.5)
        with (
            patch.object(a, "_call_with_timeout", return_value=0),
            patch("src.agents.llm_opponent.evict") as mock_evict,
        ):
            a.act(dummy_obs, dummy_legal)
        mock_evict.assert_not_called()

    def test_when_evict_enabled_evict_is_called_on_success(self, dummy_obs, dummy_legal) -> None:
        a = LLMOpponent(timeout=0.5, evict_after_call=True)
        with (
            patch.object(a, "_call_with_timeout", return_value=0),
            patch("src.agents.llm_opponent.evict") as mock_evict,
        ):
            a.act(dummy_obs, dummy_legal)
        mock_evict.assert_called_once()

    def test_when_evict_enabled_evict_is_called_on_timeout(self, dummy_obs, dummy_legal) -> None:
        a = LLMOpponent(timeout=0.5, evict_after_call=True)
        with (
            patch.object(a, "_call_with_timeout", side_effect=FuturesTimeoutError),
            patch("src.agents.llm_opponent.evict") as mock_evict,
        ):
            a.act(dummy_obs, dummy_legal)
        mock_evict.assert_called_once()

    def test_when_evict_enabled_evict_is_called_on_http_error(self, dummy_obs, dummy_legal) -> None:
        a = LLMOpponent(timeout=0.5, evict_after_call=True)
        with (
            patch.object(a, "_call_with_timeout", side_effect=httpx.ConnectError("boom")),
            patch("src.agents.llm_opponent.evict") as mock_evict,
        ):
            a.act(dummy_obs, dummy_legal)
        mock_evict.assert_called_once()


# ------------------------------------------------------------------
# Serialisation
# ------------------------------------------------------------------


class TestSerialisation:
    """OLLAMA_LOCK serialises concurrent calls across LLMOpponent instances."""

    def test_when_two_opponents_act_concurrently_calls_are_serialised(
        self, dummy_obs, dummy_legal
    ) -> None:
        order: list[str] = []

        def slow_call(*_args, **_kwargs) -> int:
            order.append("start")
            time.sleep(0.05)
            order.append("end")
            return 0

        agent1 = LLMOpponent()
        agent2 = LLMOpponent()

        with (
            patch("src.agents.llm_opponent.call_ollama_for_action_index", side_effect=slow_call),
            patch("src.agents.llm_opponent.evict"),
        ):
            t1 = threading.Thread(target=agent1.act, args=(dummy_obs, dummy_legal))
            t2 = threading.Thread(target=agent2.act, args=(dummy_obs, dummy_legal))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

        assert order == ["start", "end", "start", "end"], f"Expected serialised calls, got: {order}"
