"""Dice combat resolution, territory capture, and post-capture army move."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.env_core.cards import draw_card, transfer_cards
from src.env_core.state import MAX_CARDS, PHASE_ATTACK, PHASE_CAPTURE_MOVE, PHASE_FORTIFY
from src.utils.constants import ADJACENCY

if TYPE_CHECKING:
    from src.env_core.state import GameState

__all__ = [
    "validate_attack",
    "resolve_attack",
    "validate_capture_move",
    "apply_capture_move",
]


def validate_attack(state: GameState, attacker: int, defender: int, dice: int) -> bool:
    """Return True iff all six attack preconditions are satisfied."""
    if state.territory_owner[attacker] != state.current_player:
        return False
    if state.territory_owner[defender] == state.current_player:
        return False
    if defender not in ADJACENCY[attacker]:
        return False
    if state.armies[attacker] < 2:
        return False
    if not (1 <= dice <= 3):
        return False
    return not dice >= state.armies[attacker]


def _roll_dice(rng: np.random.Generator, n: int) -> np.ndarray:
    """Roll *n* dice (1–6) and return them sorted descending."""
    return np.sort(rng.integers(1, 7, size=n))[::-1]


def _resolve_pairs(atk_dice: np.ndarray, def_dice: np.ndarray) -> tuple[int, int]:
    """Compare descending pairs; ties go to defender. Return (atk_losses, def_losses)."""
    pairs = min(len(atk_dice), len(def_dice))
    atk_losses = sum(1 for a, d in zip(atk_dice[:pairs], def_dice[:pairs], strict=False) if a <= d)
    def_losses = pairs - atk_losses
    return atk_losses, def_losses


def _eliminate_if_needed(state: GameState, player: int) -> None:
    """Eliminate *player* and transfer their cards when they hold no territories."""
    if np.any(state.territory_owner == player):
        return
    state.eliminated[player] = 1
    transfer_cards(state, player, state.current_player)


def _record_capture_metadata(state: GameState, attacker: int, defender: int, dice: int) -> None:
    """Write ownership flip and all capture bookkeeping fields onto state."""
    state.territory_owner[defender] = state.current_player
    state.armies[defender] = 0
    state.turn_capture = 1
    state.last_capture_dice = dice
    state.last_attacker = attacker
    state.last_defender = defender


def _apply_capture(state: GameState, attacker: int, defender: int, dice: int) -> None:
    """Orchestrate capture: metadata, elimination, card draw, phase transition."""
    defender_player = int(state.territory_owner[defender])
    _record_capture_metadata(state, attacker, defender, dice)
    _eliminate_if_needed(state, defender_player)
    if len(state.cards[state.current_player]) < MAX_CARDS:
        draw_card(state, state.current_player)
    state.phase = PHASE_CAPTURE_MOVE if state.armies[attacker] >= 2 else PHASE_FORTIFY


def resolve_attack(state: GameState, attacker: int, defender: int, dice: int) -> None:
    """Roll both sides, decrement armies, and apply capture when defender reaches 0."""
    atk_dice = _roll_dice(state.rng, dice)
    def_dice = _roll_dice(state.rng, min(2, int(state.armies[defender])))
    atk_losses, def_losses = _resolve_pairs(atk_dice, def_dice)
    state.armies[attacker] -= atk_losses
    state.armies[defender] -= def_losses
    if state.armies[defender] == 0:
        _apply_capture(state, attacker, defender, dice)
    elif state.armies[attacker] < 2:
        state.phase = PHASE_FORTIFY


def validate_capture_move(state: GameState, move: int) -> bool:
    """True iff last_capture_dice <= move <= armies[last_attacker] - 1."""
    return bool(state.last_capture_dice <= move <= state.armies[state.last_attacker] - 1)


def apply_capture_move(state: GameState, move: int) -> None:
    """Move armies from last_attacker to last_defender and advance the phase."""
    attacker = state.last_attacker
    defender = state.last_defender
    state.armies[defender] += move
    state.armies[attacker] -= move
    state.phase = PHASE_ATTACK if state.armies[attacker] >= 2 else PHASE_FORTIFY
