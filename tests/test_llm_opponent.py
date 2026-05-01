"""Tests for the LLM opponent with mocked BAML client."""

from __future__ import annotations

import threading
import time
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

    def test_when_player_config_set_top_p_top_k_repeat_penalty_are_injected_into_registry(
        self,
    ) -> None:
        """PlayerConfig sampling params are forwarded to the ClientRegistry."""
        from src.agents.player_config import PlayerConfig

        config = PlayerConfig(
            player_id=2,
            temperature=0.7,
            top_p=0.85,
            top_k=50,
            repeat_penalty=1.15,
            strategy_hint="Eliminate the weakest.",
        )
        with patch("src.agents.llm_opponent.ClientRegistry") as mock_registry_cls:
            mock_reg_instance = mock_registry_cls.return_value
            with patch("src.agents.llm_opponent.baml_client"):
                agent = LLMOpponent(player_config=config)
                _ = agent._baml_client  # trigger lazy init
        captured_options = mock_reg_instance.add_llm_client.call_args.kwargs["options"]
        assert captured_options["top_p"] == pytest.approx(0.85)
        assert captured_options["top_k"] == 50
        assert captured_options["repeat_penalty"] == pytest.approx(1.15)
        assert captured_options["temperature"] == pytest.approx(0.7)

    def test_when_player_config_none_scalar_construction_is_unchanged(self) -> None:
        """Constructing without player_config keeps existing scalar behaviour."""
        with patch("src.agents.llm_opponent.ClientRegistry") as mock_registry_cls:
            mock_reg_instance = mock_registry_cls.return_value
            with patch("src.agents.llm_opponent.baml_client"):
                agent = LLMOpponent(temperature=0.3)
                _ = agent._baml_client
        captured_options = mock_reg_instance.add_llm_client.call_args.kwargs["options"]
        assert captured_options["temperature"] == pytest.approx(0.3)
        assert "top_p" not in captured_options
        assert "top_k" not in captured_options
        assert "repeat_penalty" not in captured_options

    def test_when_both_player_config_and_scalars_player_config_overrides_temperature(
        self,
    ) -> None:
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
        agent = LLMOpponent(temperature=0.1, player_config=config)
        assert agent._temperature == pytest.approx(0.9)

    def test_when_player_config_set_strategy_hint_is_stored(self) -> None:
        """strategy_hint from player_config is accessible on the agent."""
        from src.agents.player_config import PlayerConfig

        config = PlayerConfig(
            player_id=1,
            temperature=0.4,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            strategy_hint="Secure continents.",
        )
        agent = LLMOpponent(player_config=config)
        assert agent._strategy_hint == "Secure continents."

    def test_when_player_config_set_accepts_player_config_parameter(self) -> None:
        """LLMOpponent can be constructed with only player_config, no scalars."""
        from src.agents.player_config import PlayerConfig

        config = PlayerConfig(
            player_id=3,
            temperature=0.3,
            top_p=0.95,
            top_k=30,
            repeat_penalty=1.2,
            strategy_hint="Fortify borders.",
        )
        agent = LLMOpponent(player_config=config)
        assert agent._player_config is config
        assert agent._temperature == pytest.approx(0.3)
        assert agent._model == config.model


# ------------------------------------------------------------------
# Strategy hint propagation
# ------------------------------------------------------------------


class TestStrategyHint:
    """strategy_hint is forwarded through build_snapshot."""

    def test_when_player_config_with_hint_build_snapshot_receives_hint(
        self, dummy_obs, dummy_legal
    ) -> None:
        import contextlib
        from unittest.mock import MagicMock, patch

        from src.agents.player_config import PlayerConfig

        config = PlayerConfig(
            player_id=0,
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            strategy_hint="Play greedily.",
        )
        agent = LLMOpponent(player_config=config)
        with patch("src.agents.llm_opponent.build_snapshot") as mock_bs:
            mock_snapshot = MagicMock()
            mock_bs.return_value = mock_snapshot
            with (
                patch.object(agent, "_call_with_timeout", side_effect=RuntimeError("stop")),
                contextlib.suppress(Exception),
            ):
                agent.act(dummy_obs, dummy_legal)
        mock_bs.assert_called_once_with(dummy_obs, dummy_legal, strategy_hint="Play greedily.")


# ------------------------------------------------------------------
# Eviction
# ------------------------------------------------------------------


class TestEviction:
    """evict is called on every LLM action path."""

    def test_when_act_succeeds_evict_is_called(
        self, agent: LLMOpponent, dummy_obs, dummy_legal
    ) -> None:
        first = dummy_legal[0]
        baml_action = baml_types.RisikoAction(
            action_type=_action_type_to_str(first["action_type"]),
            param_a=first["param_a"],
            param_b=first["param_b"],
            param_c=first["param_c"],
            param_d=first["param_d"],
        )
        with (
            patch.object(agent, "_call_with_timeout", return_value=baml_action),
            patch("src.agents.llm_opponent.evict") as mock_evict,
        ):
            agent.act(dummy_obs, dummy_legal)
        mock_evict.assert_called_once()

    def test_when_timeout_evict_is_called(self, agent: LLMOpponent, dummy_obs, dummy_legal) -> None:
        with (
            patch.object(agent, "_call_with_timeout", side_effect=FuturesTimeoutError),
            patch("src.agents.llm_opponent.evict") as mock_evict,
        ):
            agent.act(dummy_obs, dummy_legal)
        mock_evict.assert_called_once()

    def test_when_baml_error_evict_is_called(
        self, agent: LLMOpponent, dummy_obs, dummy_legal
    ) -> None:
        from baml_py.baml_py import BamlClientError

        with (
            patch.object(agent, "_call_with_timeout", side_effect=BamlClientError("fail")),
            patch("src.agents.llm_opponent.evict") as mock_evict,
        ):
            agent.act(dummy_obs, dummy_legal)
        mock_evict.assert_called_once()

    def test_when_generic_exception_evict_is_called(
        self, agent: LLMOpponent, dummy_obs, dummy_legal
    ) -> None:
        with (
            patch.object(agent, "_call_with_timeout", side_effect=RuntimeError("boom")),
            patch("src.agents.llm_opponent.evict") as mock_evict,
        ):
            agent.act(dummy_obs, dummy_legal)
        mock_evict.assert_called_once()

    def test_when_illegal_action_evict_is_called(
        self, agent: LLMOpponent, dummy_obs, dummy_legal
    ) -> None:
        illegal = baml_types.RisikoAction(
            action_type="attack", param_a=99, param_b=99, param_c=99, param_d=99
        )
        with (
            patch.object(agent, "_call_with_timeout", return_value=illegal),
            patch("src.agents.llm_opponent.evict") as mock_evict,
        ):
            agent.act(dummy_obs, dummy_legal)
        mock_evict.assert_called_once()


# ------------------------------------------------------------------
# Serialisation
# ------------------------------------------------------------------


class TestSerialisation:
    """OLLAMA_LOCK serialises concurrent BAML calls across LLMOpponent instances."""

    def test_when_two_opponents_act_concurrently_baml_calls_are_serialised(
        self, dummy_obs, dummy_legal
    ) -> None:
        order: list[str] = []

        def slow_generate(snapshot: object) -> baml_types.RisikoAction:
            order.append("start")
            time.sleep(0.05)
            order.append("end")
            return baml_types.RisikoAction(
                action_type="skip", param_a=0, param_b=0, param_c=0, param_d=0
            )

        agent1 = LLMOpponent()
        agent2 = LLMOpponent()
        mock_baml1 = MagicMock()
        mock_baml1.GenerateRisikoAction = slow_generate
        mock_baml2 = MagicMock()
        mock_baml2.GenerateRisikoAction = slow_generate

        with (
            patch.object(agent1, "_baml", mock_baml1),
            patch.object(agent2, "_baml", mock_baml2),
            patch("src.agents.llm_opponent.evict"),
        ):
            t1 = threading.Thread(target=agent1.act, args=(dummy_obs, dummy_legal))
            t2 = threading.Thread(target=agent2.act, args=(dummy_obs, dummy_legal))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

        assert order == ["start", "end", "start", "end"], f"Expected serialised calls, got: {order}"
