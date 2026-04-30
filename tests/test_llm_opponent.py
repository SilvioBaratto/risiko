"""Tests for the LLM opponent with mocked BAML client."""

from __future__ import annotations

from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from baml_client import types as baml_types
from src.agents.baml_bridge import ActionType
from src.agents.base import Agent
from src.agents.llm_opponent import LLMOpponent
from src.env import RisikoEnv

_STR_MAP = {
    ActionType.TRADE: "trade",
    ActionType.REINFORCE: "reinforce",
    ActionType.ATTACK: "attack",
    ActionType.CAPTURE_MOVE: "capture_move",
    ActionType.FORTIFY: "fortify",
    ActionType.SKIP: "skip",
}


def _action_type_to_str(action_type: int) -> str:
    return _STR_MAP.get(action_type, "skip")


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
        with patch.object(agent, "_baml", new_callable=MagicMock) as mock_baml:
            mock_baml.GenerateRisikoAction.return_value = baml_types.RisikoAction(
                action_type="skip",
                param_a=0,
                param_b=0,
                param_c=0,
                param_d=0,
            )
            result = agent.act(dummy_obs, dummy_legal)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"action_type", "param_a", "param_b", "param_c", "param_d"}

    def test_act_with_meta_returns_tensors(
        self, agent: LLMOpponent, dummy_obs, dummy_legal
    ) -> None:
        with patch.object(agent, "_baml", new_callable=MagicMock) as mock_baml:
            mock_baml.GenerateRisikoAction.return_value = baml_types.RisikoAction(
                action_type="skip",
                param_a=0,
                param_b=0,
                param_c=0,
                param_d=0,
            )
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

    def test_baml_error_fallback(self, agent: LLMOpponent, dummy_obs, dummy_legal) -> None:
        """On BamlClientError, fallback to random."""
        from baml_py.baml_py import BamlClientError

        with patch.object(agent, "_call_with_timeout", side_effect=BamlClientError("fail")):
            result = agent.act(dummy_obs, dummy_legal)
        assert isinstance(result, dict)
        assert result in dummy_legal

    def test_generic_exception_fallback(self, agent: LLMOpponent, dummy_obs, dummy_legal) -> None:
        """On unexpected exception, fallback to random."""
        with patch.object(agent, "_call_with_timeout", side_effect=RuntimeError("boom")):
            result = agent.act(dummy_obs, dummy_legal)
        assert isinstance(result, dict)
        assert result in dummy_legal

    def test_illegal_action_fallback(self, agent: LLMOpponent, dummy_obs, dummy_legal) -> None:
        """If LLM returns an action not in legal_actions, fallback to random."""
        illegal = baml_types.RisikoAction(
            action_type="attack",
            param_a=99,
            param_b=99,
            param_c=99,
            param_d=99,
        )
        with patch.object(agent, "_call_with_timeout", return_value=illegal):
            result = agent.act(dummy_obs, dummy_legal)
        assert isinstance(result, dict)
        assert result in dummy_legal


# ------------------------------------------------------------------
# Valid action path
# ------------------------------------------------------------------


class TestValidAction:
    """Happy path: LLM returns a legal action."""

    def test_first_legal_action(self, agent: LLMOpponent, dummy_obs, dummy_legal) -> None:
        """Mock LLM returns the first legal action exactly."""
        first = dummy_legal[0]
        type_str = _action_type_to_str(first["action_type"])
        baml_action = baml_types.RisikoAction(
            action_type=type_str,
            param_a=first["param_a"],
            param_b=first["param_b"],
            param_c=first["param_c"],
            param_d=first["param_d"],
        )
        with patch.object(agent, "_call_with_timeout", return_value=baml_action):
            result = agent.act(dummy_obs, dummy_legal)
        assert result == first

    def test_reinforce_action(self, agent: LLMOpponent, dummy_obs, dummy_legal) -> None:
        reinforce = next(
            (a for a in dummy_legal if a["action_type"] == ActionType.REINFORCE),
            None,
        )
        if reinforce is None:
            pytest.skip("No reinforce action available in this state")
        baml_action = baml_types.RisikoAction(
            action_type="reinforce",
            param_a=reinforce["param_a"],
            param_b=reinforce["param_b"],
            param_c=reinforce["param_c"],
            param_d=reinforce["param_d"],
        )
        with patch.object(agent, "_call_with_timeout", return_value=baml_action):
            result = agent.act(dummy_obs, dummy_legal)
        assert result == reinforce


# ------------------------------------------------------------------
# Constructor / configuration
# ------------------------------------------------------------------


class TestConfiguration:
    """LLMOpponent accepts runtime configuration."""

    def test_default_values(self) -> None:
        agent = LLMOpponent()
        assert agent._model == "qwen3.5:4b"
        assert agent._timeout == 5.0
        assert agent._temperature == 0.1

    def test_custom_values(self) -> None:
        agent = LLMOpponent(model="qwen:7b", timeout=2.0, temperature=0.5)
        assert agent._model == "qwen:7b"
        assert agent._timeout == 2.0
        assert agent._temperature == 0.5

    def test_custom_ollama_url(self) -> None:
        agent = LLMOpponent(ollama_url="http://custom:11434/v1")
        assert agent._ollama_url == "http://custom:11434/v1"

    def test_lazy_client_not_created_on_init(self) -> None:
        agent = LLMOpponent()
        assert agent._baml is None

    def test_client_created_on_first_use(self, agent: LLMOpponent) -> None:
        """Accessing the _baml property creates the client lazily."""
        assert agent._baml is None
        _ = agent._baml_client  # trigger property access
        assert agent._baml is not None

    def test_pickle_resets_baml_client(self, agent: LLMOpponent) -> None:
        """Pickling strips the BAML client; unpickling resets it to None."""
        assert agent._baml is None
        state = agent.__getstate__()
        assert state["_baml"] is None

        agent2 = LLMOpponent.__new__(LLMOpponent)
        agent2.__setstate__(state)
        assert agent2._baml is None
        assert agent2._model == agent._model
        assert agent2._timeout == agent._timeout
