"""Bridge between env observation dicts and BAML GameStateSnapshot types."""

from __future__ import annotations

import numpy as np

from baml_client import types as baml_types
from src.utils.constants import TERRITORY_NAMES, territory_to_continent

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


class ActionType:
    """Integer action type constants matching the environment."""

    TRADE = 0
    REINFORCE = 1
    ATTACK = 2
    CAPTURE_MOVE = 3
    FORTIFY = 4
    SKIP = 5


class ActionTypeStr:
    """String action type constants for LLM output parsing."""

    TRADE = "trade"
    REINFORCE = "reinforce"
    ATTACK = "attack"
    CAPTURE_MOVE = "capture_move"
    FORTIFY = "fortify"
    SKIP = "skip"


_STR_TO_INT = {
    ActionTypeStr.TRADE: ActionType.TRADE,
    ActionTypeStr.REINFORCE: ActionType.REINFORCE,
    ActionTypeStr.ATTACK: ActionType.ATTACK,
    ActionTypeStr.CAPTURE_MOVE: ActionType.CAPTURE_MOVE,
    ActionTypeStr.FORTIFY: ActionType.FORTIFY,
    ActionTypeStr.SKIP: ActionType.SKIP,
}

_ACTION_TYPE_TO_STR = {
    ActionType.TRADE: ActionTypeStr.TRADE,
    ActionType.REINFORCE: ActionTypeStr.REINFORCE,
    ActionType.ATTACK: ActionTypeStr.ATTACK,
    ActionType.CAPTURE_MOVE: ActionTypeStr.CAPTURE_MOVE,
    ActionType.FORTIFY: ActionTypeStr.FORTIFY,
    ActionType.SKIP: ActionTypeStr.SKIP,
}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def build_snapshot(
    obs: dict[str, np.ndarray],
    legal_actions: list[dict[str, int]],
    *,
    strategy_hint: str | None = None,
) -> baml_types.GameStateSnapshot:
    """Convert an env observation into a BAML GameStateSnapshot."""
    phase = _PHASE_MAP.get(int(obs["phase"]), baml_types.Phase.TRADE)
    return baml_types.GameStateSnapshot(
        current_player=int(obs["current_player"]),
        phase=phase,
        territories=_build_territories(obs),
        cards=_build_cards(obs),
        reinforcements_remaining=int(obs["reinforcements_remaining"]),
        trade_count=int(obs["trade_count"]),
        turn_capture=bool(obs["turn_capture"]),
        n_players=int(obs["n_players"]),
        eliminated=obs["eliminated"].tolist(),
        continent_control=obs["continent_control"].tolist(),
        legal_actions=_build_legal_actions(legal_actions),
        strategy_hint=strategy_hint,
    )


def baml_action_to_dict(baml_action: baml_types.RisikoAction) -> dict[str, int]:
    """Convert a BAML RisikoAction to an env action dict."""
    action_type_str = baml_action.action_type.lower().replace(" ", "_")
    action_type = _STR_TO_INT.get(action_type_str, ActionType.SKIP)
    return {
        "action_type": action_type,
        "param_a": baml_action.param_a,
        "param_b": baml_action.param_b,
        "param_c": baml_action.param_c,
        "param_d": baml_action.param_d,
    }


def is_legal(action: dict[str, int], legal_actions: list[dict[str, int]]) -> bool:
    """Check whether *action* exactly matches any entry in *legal_actions*."""
    return any(all(action[k] == legal[k] for k in action) for legal in legal_actions)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_territories(obs: dict[str, np.ndarray]) -> list[baml_types.TerritorySnapshot]:
    return [
        baml_types.TerritorySnapshot(
            id=i,
            name=TERRITORY_NAMES[i],
            owner=int(obs["territory_owner"][i]),
            armies=int(obs["armies"][i]),
            continent=territory_to_continent(i) or "",
        )
        for i in range(42)
    ]


def _build_cards(obs: dict[str, np.ndarray]) -> list[baml_types.CardInfo]:
    cards: list[baml_types.CardInfo] = []
    card_matrix = obs["cards"]
    for i in range(card_matrix.shape[0]):
        sym = int(card_matrix[i].argmax())
        if card_matrix[i].sum() > 0:
            cards.append(
                baml_types.CardInfo(
                    index=i,
                    symbol=_SYMBOL_NAMES.get(sym, "unknown"),
                )
            )
    return cards


def _build_legal_actions(
    legal_actions: list[dict[str, int]],
) -> list[baml_types.LegalAction]:
    return [
        baml_types.LegalAction(
            action_type=_ACTION_TYPE_TO_STR.get(a["action_type"], ActionTypeStr.SKIP),
            description=_describe_action(a),
        )
        for a in legal_actions
    ]


def _describe_action(a: dict[str, int]) -> str:
    pa, pb, pc, pd = a["param_a"], a["param_b"], a["param_c"], a["param_d"]
    return f"type={a['action_type']} params=({pa},{pb},{pc},{pd})"
