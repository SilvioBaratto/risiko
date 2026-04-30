"""LLM-based opponent using BAML with robust error handling."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from baml_py import ClientRegistry

from baml_client import b as baml_client
from baml_client import types as baml_types
from src.agents.base import Agent
from src.agents.random_agent import RandomAgent
from src.utils.constants import TERRITORY_NAMES, territory_to_continent

# Configure BAML client at runtime to handle Docker networking
_ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
_client_registry = ClientRegistry()
_client_registry.add_llm_client(
    name="Model",
    provider="openai-generic",
    options={
        "base_url": _ollama_url,
        "model": "qwen3.5:4b",
        "temperature": 0.1,
        "max_tokens": 1024,
    },
)
_client_registry.set_primary("Model")

# Create configured BAML client
b = baml_client.with_options(client_registry=_client_registry)

logger = logging.getLogger(__name__)

_PHASE_MAP = {
    0: baml_types.Phase.TRADE,
    1: baml_types.Phase.REINFORCE,
    2: baml_types.Phase.ATTACK,
    3: baml_types.Phase.CAPTURE_MOVE,
    4: baml_types.Phase.FORTIFY,
}

_SYMBOL_NAMES = {
    0: "infantry",
    1: "cavalry",
    2: "artillery",
    3: "wild",
}

_ACTION_TYPE_STR_TO_INT = {
    "trade": 0,
    "reinforce": 1,
    "attack": 2,
    "capture_move": 3,
    "fortify": 4,
    "skip": 5,
}


class LLMOpponent(Agent):
    """Agent that queries a local LLM via BAML, falling back to random."""

    def __init__(self) -> None:
        """Create an LLM opponent with a random fallback."""
        self._fallback = RandomAgent()

    def act(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> dict[str, int]:
        """Select an action via LLM or fall back to random."""
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        try:
            snapshot = _obs_to_snapshot(obs, legal_actions)
            baml_action = b.GenerateRisikoAction(snapshot)
            action = _baml_action_to_dict(baml_action)
            # Validate against legal_actions (approximate match on action_type)
            if _is_legal(action, legal_actions):
                return action
        except Exception:
            logger.exception("LLM call failed, falling back to random")
        return self._fallback.act(obs, legal_actions)

    def act_with_meta(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> tuple[dict[str, int], Any, Any]:
        """Select an action and return dummy meta values."""
        action = self.act(obs, legal_actions, deterministic=deterministic)
        return action, 0.0, 0.0


def _obs_to_snapshot(
    obs: dict[str, np.ndarray],
    legal_actions: list[dict[str, int]],
) -> baml_types.GameStateSnapshot:
    """Convert an environment observation to a BAML GameStateSnapshot."""
    phase = _PHASE_MAP.get(int(obs["phase"]), baml_types.Phase.TRADE)

    territories = [
        baml_types.TerritorySnapshot(
            id=i,
            name=TERRITORY_NAMES[i],
            owner=int(obs["territory_owner"][i]),
            armies=int(obs["armies"][i]),
            continent=territory_to_continent(i) or "",
        )
        for i in range(42)
    ]

    cards: list[baml_types.CardInfo] = []
    card_matrix = obs["cards"]
    for i in range(card_matrix.shape[0]):
        sym = int(card_matrix[i].argmax())
        if card_matrix[i].sum() > 0:
            cards.append(baml_types.CardInfo(index=i, symbol=_SYMBOL_NAMES.get(sym, "unknown")))

    baml_legal = []
    for a in legal_actions:
        pa, pb, pc, pd = a["param_a"], a["param_b"], a["param_c"], a["param_d"]
        desc = f"type={a['action_type']} params=({pa},{pb},{pc},{pd})"
        baml_legal.append(
            baml_types.LegalAction(
                action_type="skip" if a["action_type"] == 5 else "action",
                description=desc,
            )
        )

    return baml_types.GameStateSnapshot(
        current_player=int(obs["current_player"]),
        phase=phase,
        territories=territories,
        cards=cards,
        reinforcements_remaining=int(obs["reinforcements_remaining"]),
        trade_count=int(obs["trade_count"]),
        turn_capture=bool(obs["turn_capture"]),
        n_players=int(obs["n_players"]),
        eliminated=obs["eliminated"].tolist(),
        continent_control=obs["continent_control"].tolist(),
        legal_actions=baml_legal,
    )


def _baml_action_to_dict(baml_action: baml_types.RisikoAction) -> dict[str, int]:
    """Convert a BAML RisikoAction to an env action dict."""
    action_type_str = baml_action.action_type.lower().replace(" ", "_")
    action_type = _ACTION_TYPE_STR_TO_INT.get(action_type_str, 5)
    return {
        "action_type": action_type,
        "param_a": baml_action.param_a,
        "param_b": baml_action.param_b,
        "param_c": baml_action.param_c,
        "param_d": baml_action.param_d,
    }


def _is_legal(action: dict[str, int], legal_actions: list[dict[str, int]]) -> bool:
    """Check whether *action* is in *legal_actions* (exact match)."""
    return any(all(action[k] == legal[k] for k in action) for legal in legal_actions)
