"""LLM-based opponent backed by Ollama (local or cloud), with timeout and random fallback."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

import httpx
import numpy as np
import torch

from src.agents.base import Agent
from src.agents.diplomacy import DiplomacyNote
from src.agents.ollama_client import call_ollama_for_action_index
from src.agents.player_config import DEFAULT_MODEL, PlayerConfig
from src.agents.random_agent import RandomAgent
from src.utils.log import get_logger

logger = get_logger("llm_opponent")


class LLMOpponent(Agent):
    """Agent that queries Ollama for an action, with timeout and fallback."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        temperature: float = 0.1,
        *,
        top_p: float = 0.9,
        strategy_hint: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        player_config: PlayerConfig | None = None,
    ) -> None:
        """Create an LLM opponent backed by Ollama.

        Args:
            model: Model/tag name (overridden by ``player_config.model``), e.g.
                ``gpt-oss:20b`` (local) or ``gpt-oss:120b`` (cloud).
            timeout: Hard timeout per LLM call in seconds.
            temperature: Sampling temperature (overridden by ``player_config``).
            top_p: Nucleus sampling parameter (overridden by ``player_config``).
            strategy_hint: Prompt directive (overridden by ``player_config``).
            base_url: Optional override for ``OLLAMA_BASE_URL``.
            api_key: Optional override for ``OLLAMA_API_KEY``.
            player_config: Per-player sampling profile; takes precedence over scalars.
        """
        self._player_config = player_config
        self._model = player_config.model if player_config else model
        self._timeout = timeout
        self._temperature = player_config.temperature if player_config else temperature
        self._top_p = player_config.top_p if player_config else top_p
        self._strategy_hint = player_config.strategy_hint if player_config else strategy_hint
        self._base_url = base_url
        self._api_key = api_key
        self._fallback = RandomAgent()

    def act(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
        diplomacy: DiplomacyNote | None = None,
    ) -> dict[str, int]:
        """Select an action via LLM with timeout, or fall back to random.

        When ``diplomacy`` is ``None`` the call is byte-identical to the
        training-path (disabled-equivalence guarantee).
        """
        del deterministic
        self._validate(legal_actions)
        action, reason = self._try_llm_action(obs, legal_actions, diplomacy=diplomacy)
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
        diplomacy: DiplomacyNote | None = None,
        context=None,
    ) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
        """Select an action and return value/logprob tensors.

        The eval runner (#103) passes ``context`` (a DiplomacyContext); training
        paths leave both ``context`` and ``diplomacy`` as ``None`` so the call
        is byte-identical to the pre-diplomacy path.
        """
        note = diplomacy
        if note is None and context is not None:
            from src.agents.diplomacy import build_diplomacy_note

            owners = obs.get("territory_owner", [])
            note = build_diplomacy_note(
                context.state, context.acting_player, owners, self._n_players_from_obs(obs)
            )
        action = self.act(obs, legal_actions, deterministic=deterministic, diplomacy=note)
        value = torch.tensor(float("nan"))
        logprob = torch.tensor(float("nan"))
        return action, value, logprob

    @staticmethod
    def _n_players_from_obs(obs: dict[str, Any]) -> int:
        """Infer player count from the eliminated mask in the observation."""
        eliminated = obs.get("eliminated")
        if eliminated is not None:
            return int(len(eliminated))
        return 6  # Risiko default

    def _try_llm_action(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        diplomacy: DiplomacyNote | None = None,
    ) -> tuple[dict[str, int] | None, str]:
        """Attempt to get a legal action from the LLM by index selection."""
        try:
            idx = self._call_with_timeout(obs, legal_actions, diplomacy=diplomacy)
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

    def _call_with_timeout(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        diplomacy: DiplomacyNote | None = None,
    ) -> int | None:
        """Call Ollama with a hard timeout via ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                call_ollama_for_action_index,
                obs,
                legal_actions,
                **self._build_ollama_kwargs(diplomacy),
            )
            return future.result(timeout=self._timeout)

    def _build_ollama_kwargs(self, diplomacy: DiplomacyNote | None) -> dict[str, Any]:
        """Build keyword args for call_ollama_for_action_index.

        ``diplomacy_note`` is added only when ``diplomacy`` is not ``None`` so
        that the no-diplomacy call is byte-identical to the training-path
        (disabled-equivalence guarantee asserted in test_102_move_prompt_diplomacy).
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "base_url": self._base_url,
            "api_key": self._api_key,
            "timeout": self._timeout,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "strategy_hint": self._strategy_hint,
        }
        if diplomacy is not None:
            kwargs["diplomacy_note"] = diplomacy
        return kwargs

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


__all__ = ["LLMOpponent", "DEFAULT_MODEL"]
