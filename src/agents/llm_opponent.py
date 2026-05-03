"""LLM-based opponent using Ollama's native /api/chat with timeout and fallback."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

import httpx
import numpy as np
import torch

from src.agents.base import Agent
from src.agents.ollama_eviction import OLLAMA_LOCK, evict
from src.agents.ollama_native import call_ollama_for_action_index
from src.agents.player_config import PlayerConfig
from src.agents.random_agent import RandomAgent
from src.utils.log import get_logger

logger = get_logger("llm_opponent")


class LLMOpponent(Agent):
    """Agent that queries a local LLM via Ollama with timeout and fallback."""

    def __init__(
        self,
        model: str = "risiko",
        timeout: float = 30.0,
        temperature: float = 0.1,
        ollama_url: str | None = None,
        player_config: PlayerConfig | None = None,
        evict_after_call: bool = False,
    ) -> None:
        """Create an LLM opponent with configurable model and timeout.

        Args:
            model: Ollama model name (overridden by ``player_config.model``).
            timeout: Hard timeout per LLM call in seconds.
            temperature: Sampling temperature (overridden by ``player_config``).
            ollama_url: Optional override for ``OLLAMA_BASE_URL``.
            player_config: Per-player sampling profile; takes precedence over scalars.
            evict_after_call: If True, send keep_alive=0 after each call (frees
                GPU memory but causes ~10s cold-start on the next call). Default
                False for training, where keeping the model warm is critical.
        """
        self._player_config = player_config
        self._model = player_config.model if player_config else model
        self._timeout = timeout
        self._temperature = player_config.temperature if player_config else temperature
        self._ollama_url = ollama_url
        self._strategy_hint = player_config.strategy_hint if player_config else None
        self._evict_after_call = evict_after_call
        self._fallback = RandomAgent()

    def act(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> dict[str, int]:
        """Select an action via LLM with timeout, or fall back to random."""
        del deterministic
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
        """Attempt to get a legal action from the LLM by index selection."""
        try:
            idx = self._call_with_timeout(obs, legal_actions)
            if idx is None:
                return None, "empty_or_invalid_response"
            return legal_actions[idx], ""
        except FuturesTimeoutError:
            logger.warning("LLM call timed out after %.1fs; falling back", self._timeout)
            return None, "timeout"
        except httpx.HTTPError as e:
            logger.warning("HTTP error (%s): %s; falling back", type(e).__name__, e)
            return None, "http_error"
        except Exception as e:
            logger.warning("LLM call failed (%s: %s); falling back", type(e).__name__, e)
            return None, "parse_error"
        finally:
            if self._evict_after_call:
                evict(self._model, self._base_url_for_eviction())

    def _call_with_timeout(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
    ) -> int | None:
        """Call Ollama with a hard timeout via ThreadPoolExecutor."""
        pc = self._player_config
        with OLLAMA_LOCK, ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                call_ollama_for_action_index,
                obs,
                legal_actions,
                model=self._model,
                base_url=self._ollama_url,
                timeout=self._timeout,
                temperature=self._temperature,
                top_p=pc.top_p if pc else None,
                top_k=pc.top_k if pc else None,
                repeat_penalty=pc.repeat_penalty if pc else None,
                strategy_hint=self._strategy_hint,
            )
            return future.result(timeout=self._timeout)

    def _base_url_for_eviction(self) -> str:
        """Return the native Ollama base URL (strip /v1 if present)."""
        url = self._ollama_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return url.removesuffix("/v1")

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state."""
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state."""
        self.__dict__.update(state)

    @staticmethod
    def _validate(legal_actions: list[dict[str, int]]) -> None:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
