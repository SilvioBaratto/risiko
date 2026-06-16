"""Legal-action enumeration per phase for the Risiko environment.

Action encoding (all actions are dicts with five int fields):

  action_type | meaning
  ------------|-----------------------------------------------------------
  0 (TRADE)   | param_a/b/c = hand indices of the three traded cards
  1 (REINFORCE)| param_a = territory, param_b = army count (1..remaining)
  2 (ATTACK)  | param_a = attacker, param_b = defender, param_c = dice
  3 (CAPTURE) | param_a = armies to move into the captured territory
  4 (FORTIFY) | param_a = source, param_b = dest, param_c = army count
  5 (SKIP)    | all params = 0  — ends the current phase

Import contract:
  adjacency   : getattr(state, 'adjacency', ADJACENCY)  — real states use the
                global board; test fixtures may carry their own mini-board.
  trade combos: forms_valid_set from src.env_core.cards
  army cap    : MAX_ARMIES_PARAM from src.utils.constants
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from src.env_core.cards import forms_valid_set
from src.env_core.state import (
    PHASE_ATTACK,
    PHASE_CAPTURE_MOVE,
    PHASE_FORTIFY,
    PHASE_REINFORCE,
    PHASE_TRADE,
    GameState,
)
from src.utils.constants import ADJACENCY, MAX_ARMIES_PARAM

__all__ = ["get_legal_actions"]

Action = dict[str, int]

_SKIP: Action = {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}


def _act(t: int, a: int = 0, b: int = 0, c: int = 0) -> Action:
    return {"action_type": t, "param_a": a, "param_b": b, "param_c": c, "param_d": 0}


def _adj(state: GameState) -> dict[int, list[int]]:
    """Return the adjacency graph; accepts a test-fixture 'adjacency' override."""
    return getattr(state, "adjacency", ADJACENCY)


# ---------------------------------------------------------------------------
# Per-phase enumerators
# ---------------------------------------------------------------------------


def _trade_actions(state: GameState) -> list[Action]:
    hand: list[int] = state.cards[state.current_player]
    actions: list[Action] = [dict(_SKIP)]
    for combo in combinations(range(len(hand)), 3):
        if forms_valid_set([hand[i] for i in combo]):
            a, b, c = combo
            actions.append(_act(0, a, b, c))
    return actions


def _reinforce_actions(state: GameState) -> list[Action]:
    if state.reinforcements_remaining == 0:
        return [dict(_SKIP)]
    player = state.current_player
    cap = min(state.reinforcements_remaining, MAX_ARMIES_PARAM)
    owned = np.where(state.territory_owner == player)[0]
    return [_act(1, int(t), n) for t in owned for n in range(1, cap + 1)]


def _attack_actions(state: GameState, adj: dict[int, list[int]]) -> list[Action]:
    player = state.current_player
    srcs = np.where((state.territory_owner == player) & (state.armies >= 2))[0]
    actions: list[Action] = [dict(_SKIP)]
    for src in srcs:
        max_dice = min(3, int(state.armies[src]) - 1)
        for dst in adj.get(int(src), []):
            if state.territory_owner[dst] != player:
                for d in range(1, max_dice + 1):
                    actions.append(_act(2, int(src), int(dst), d))
    return actions


def _capture_move_actions(state: GameState) -> list[Action]:
    lo = state.last_capture_dice
    hi = min(int(state.armies[state.last_attacker]) - 1, MAX_ARMIES_PARAM)
    return [_act(3, n) for n in range(lo, hi + 1)]


def _fortify_actions(state: GameState, adj: dict[int, list[int]]) -> list[Action]:
    player = state.current_player
    srcs = np.where((state.territory_owner == player) & (state.armies >= 2))[0]
    dsts = np.where(state.territory_owner == player)[0]
    actions: list[Action] = [dict(_SKIP)]
    owner = state.territory_owner
    for src in srcs:
        cap = min(int(state.armies[src]) - 1, MAX_ARMIES_PARAM)
        for dst in dsts:
            if int(dst) == int(src):
                continue
            if _is_connected(int(src), int(dst), player, owner, adj):
                for n in range(1, cap + 1):
                    actions.append(_act(4, int(src), int(dst), n))
    return actions


def _is_connected(
    src: int,
    dst: int,
    player: int,
    territory_owner: np.ndarray,
    adj: dict[int, list[int]],
) -> bool:
    """Iterative DFS over territories owned by *player* to test connectivity."""
    visited: set[int] = set()
    stack = [src]
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        if node in visited:
            continue
        visited.add(node)
        for nb in adj.get(node, []):
            if territory_owner[nb] == player and nb not in visited:
                stack.append(nb)
    return False


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_DISPATCH = {
    PHASE_TRADE: lambda s, a: _trade_actions(s),
    PHASE_REINFORCE: lambda s, a: _reinforce_actions(s),
    PHASE_ATTACK: _attack_actions,
    PHASE_CAPTURE_MOVE: lambda s, a: _capture_move_actions(s),
    PHASE_FORTIFY: _fortify_actions,
}


def get_legal_actions(state: GameState) -> list[Action]:
    """Return every legal action for the current phase; never returns an empty list."""
    adj = _adj(state)
    handler = _DISPATCH.get(state.phase)
    return handler(state, adj) if handler is not None else [dict(_SKIP)]
