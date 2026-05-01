"""LLM-based opponent using BAML with timeout and robust error handling."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

import numpy as np
import torch
from baml_py import ClientRegistry
from baml_py.baml_py import BamlClientError, BamlError

from baml_client import b as baml_client
from baml_client import types as baml_types
from src.agents.baml_bridge import baml_action_to_dict, build_snapshot, is_legal
from src.agents.base import Agent
from src.agents.ollama_eviction import OLLAMA_LOCK, evict
from src.agents.player_config import PlayerConfig
from src.agents.random_agent import RandomAgent

logger = logging.getLogger(__name__)


class LLMOpponent(Agent):
    """Agent that queries a local LLM via BAML with timeout and fallback."""

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        timeout: float = 5.0,
        temperature: float = 0.1,
        ollama_url: str | None = None,
        player_config: PlayerConfig | None = None,
    ) -> None:
        """Create an LLM opponent with configurable model and timeout."""
        self._player_config = player_config
        self._model = player_config.model if player_config else model
        self._timeout = timeout
        self._temperature = player_config.temperature if player_config else temperature
        self._ollama_url = ollama_url
        self._strategy_hint = player_config.strategy_hint if player_config else None
        self._fallback = RandomAgent()
        self._baml: Any | None = None

    def act(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> dict[str, int]:
        """Select an action via LLM with timeout, or fall back to random."""
        self._validate(legal_actions)
        action, reason = self._try_llm_action(obs, legal_actions)
        if action is not None:
            return action
        logger.info("LLM opponent fallback: %s", reason)
        return self._fallback.act(obs, legal_actions)

    def act_with_meta(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
        """Select an action and return value/logprob tensors."""
        action = self.act(obs, legal_actions, deterministic=deterministic)
        value = torch.tensor(float("nan"))
        logprob = torch.tensor(float("nan"))
        return action, value, logprob

    def _try_llm_action(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
    ) -> tuple[dict[str, int] | None, str]:
        """Attempt to get a valid action from the LLM."""
        try:
            snapshot = build_snapshot(obs, legal_actions, strategy_hint=self._strategy_hint)
            baml_action = self._call_with_timeout(snapshot)
            action = baml_action_to_dict(baml_action)
            if is_legal(action, legal_actions):
                return action, ""
            logger.warning("LLM returned illegal action %s; falling back", action)
            return None, "illegal_action"
        except FuturesTimeoutError:
            logger.warning("LLM call timed out after %.1fs; falling back", self._timeout)
            return None, "timeout"
        except (BamlClientError, BamlError) as e:
            logger.warning("BAML client error (%s): %s; falling back", type(e).__name__, e)
            return None, "baml_error"
        except Exception as e:
            logger.warning("LLM call failed (%s: %s); falling back", type(e).__name__, e)
            return None, "parse_error"
        finally:
            evict(self._model, self._base_url_for_eviction())

    def _call_with_timeout(
        self,
        snapshot: baml_types.GameStateSnapshot,
    ) -> baml_types.RisikoAction:
        """Call the LLM with a hard timeout using ThreadPoolExecutor."""
        with OLLAMA_LOCK, ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._baml_client.GenerateRisikoAction, snapshot)
            return future.result(timeout=self._timeout)

    def _base_url_for_eviction(self) -> str:
        """Return the native Ollama base URL (strip /v1 if present)."""
        url = self._ollama_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return url.removesuffix("/v1")

    @property
    def _baml_client(self) -> Any:
        """Lazy-initialised BAML client with custom registry."""
        if self._baml is None:
            pc = self._player_config
            self._baml = _make_client(
                self._model,
                self._temperature,
                self._ollama_url,
                top_p=pc.top_p if pc else None,
                top_k=pc.top_k if pc else None,
                repeat_penalty=pc.repeat_penalty if pc else None,
            )
        return self._baml

    def __getstate__(self) -> dict[str, Any]:
        """Exclude unpicklable BAML client from state."""
        state = self.__dict__.copy()
        state["_baml"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state; BAML client is recreated lazily on first use."""
        self.__dict__.update(state)
        self._baml = None

    @staticmethod
    def _validate(legal_actions: list[dict[str, int]]) -> None:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")


def _make_client(
    model: str,
    temperature: float,
    ollama_url: str | None,
    *,
    top_p: float | None = None,
    top_k: int | None = None,
    repeat_penalty: float | None = None,
) -> Any:
    """Create a configured BAML client for Ollama."""
    url = ollama_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    options: dict[str, Any] = {
        "base_url": url,
        "model": model,
        "temperature": temperature,
        "max_tokens": 1024,
    }
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k
    if repeat_penalty is not None:
        options["repeat_penalty"] = repeat_penalty
    registry = ClientRegistry()
    registry.add_llm_client(name="Model", provider="openai-generic", options=options)
    registry.set_primary("Model")
    return baml_client.with_options(client_registry=registry)
