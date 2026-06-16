"""Source-blind contract tests for issue #46: GameState mutable container and phase constants.

Tests are derived exclusively from the acceptance-criteria text and the oracle
classification. No implementation source was read during authoring (R4 phase).

Oracle-classified [UNIT] criteria covered here:
  - GameState mutable @dataclass with the 16 specified fields and types
  - Phase constants PHASE_TRADE=0 … PHASE_FORTIFY=4; MAX_CARDS=5
  - __all__ exports GameState, five phase constants, and MAX_CARDS
  - GameState() constructs with sensible defaults; rng defaults to a valid np.random.Generator
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import src.env_core.state as _state_module
from src.env_core.state import (
    MAX_CARDS,
    PHASE_ATTACK,
    PHASE_CAPTURE_MOVE,
    PHASE_FORTIFY,
    PHASE_REINFORCE,
    PHASE_TRADE,
    GameState,
)

# ---------------------------------------------------------------------------
# Phase constant values
# ---------------------------------------------------------------------------


class TestPhaseConstants:
    """Phase constants have the correct integer values and are all distinct."""

    def test_when_phase_trade_is_imported_then_value_is_zero(self):
        assert PHASE_TRADE == 0

    def test_when_phase_reinforce_is_imported_then_value_is_one(self):
        assert PHASE_REINFORCE == 1

    def test_when_phase_attack_is_imported_then_value_is_two(self):
        assert PHASE_ATTACK == 2

    def test_when_phase_capture_move_is_imported_then_value_is_three(self):
        assert PHASE_CAPTURE_MOVE == 3

    def test_when_phase_fortify_is_imported_then_value_is_four(self):
        assert PHASE_FORTIFY == 4

    def test_when_max_cards_is_imported_then_value_is_five(self):
        assert MAX_CARDS == 5

    def test_when_all_phase_constants_are_listed_then_they_are_distinct_integers(self):
        constants = [
            PHASE_TRADE,
            PHASE_REINFORCE,
            PHASE_ATTACK,
            PHASE_CAPTURE_MOVE,
            PHASE_FORTIFY,
        ]
        assert all(isinstance(c, int) for c in constants)
        assert len(set(constants)) == 5


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------


class TestDunderAll:
    """__all__ must export GameState, all five phase constants, and MAX_CARDS."""

    def test_when_module_all_is_inspected_then_gamestate_is_present(self):
        assert "GameState" in _state_module.__all__

    def test_when_module_all_is_inspected_then_phase_trade_is_present(self):
        assert "PHASE_TRADE" in _state_module.__all__

    def test_when_module_all_is_inspected_then_phase_reinforce_is_present(self):
        assert "PHASE_REINFORCE" in _state_module.__all__

    def test_when_module_all_is_inspected_then_phase_attack_is_present(self):
        assert "PHASE_ATTACK" in _state_module.__all__

    def test_when_module_all_is_inspected_then_phase_capture_move_is_present(self):
        assert "PHASE_CAPTURE_MOVE" in _state_module.__all__

    def test_when_module_all_is_inspected_then_phase_fortify_is_present(self):
        assert "PHASE_FORTIFY" in _state_module.__all__

    def test_when_module_all_is_inspected_then_max_cards_is_present(self):
        assert "MAX_CARDS" in _state_module.__all__


# ---------------------------------------------------------------------------
# GameState construction: field presence and zero/empty defaults
# ---------------------------------------------------------------------------


class TestGameStateConstruction:
    """GameState() constructs with the specified field types and zero/empty defaults."""

    def test_when_gamestate_is_default_constructed_then_territory_owner_is_int32_ndarray(self):
        state = GameState()
        assert isinstance(state.territory_owner, np.ndarray)
        assert state.territory_owner.dtype == np.int32

    def test_when_gamestate_is_default_constructed_then_armies_is_int32_ndarray(self):
        state = GameState()
        assert isinstance(state.armies, np.ndarray)
        assert state.armies.dtype == np.int32

    def test_when_gamestate_is_default_constructed_then_cards_is_list(self):
        state = GameState()
        assert isinstance(state.cards, list)

    def test_when_gamestate_is_default_constructed_then_deck_is_list(self):
        state = GameState()
        assert isinstance(state.deck, list)

    def test_when_gamestate_is_default_constructed_then_discard_pile_is_list(self):
        state = GameState()
        assert isinstance(state.discard_pile, list)

    def test_when_gamestate_is_default_constructed_then_trade_count_is_int(self):
        state = GameState()
        assert isinstance(state.trade_count, int)

    def test_when_gamestate_is_default_constructed_then_current_player_is_int(self):
        state = GameState()
        assert isinstance(state.current_player, int)

    def test_when_gamestate_is_default_constructed_then_phase_is_int(self):
        state = GameState()
        assert isinstance(state.phase, int)

    def test_when_gamestate_is_default_constructed_then_reinforcements_remaining_is_int(self):
        state = GameState()
        assert isinstance(state.reinforcements_remaining, int)

    def test_when_gamestate_is_default_constructed_then_turn_capture_is_int(self):
        state = GameState()
        assert isinstance(state.turn_capture, int)

    def test_when_gamestate_is_default_constructed_then_eliminated_is_int32_ndarray(self):
        state = GameState()
        assert isinstance(state.eliminated, np.ndarray)
        assert state.eliminated.dtype == np.int32

    def test_when_gamestate_is_default_constructed_then_n_players_is_int(self):
        state = GameState()
        assert isinstance(state.n_players, int)

    def test_when_gamestate_is_default_constructed_then_last_capture_dice_is_int(self):
        state = GameState()
        assert isinstance(state.last_capture_dice, int)

    def test_when_gamestate_is_default_constructed_then_last_attacker_is_int(self):
        state = GameState()
        assert isinstance(state.last_attacker, int)

    def test_when_gamestate_is_default_constructed_then_last_defender_is_int(self):
        state = GameState()
        assert isinstance(state.last_defender, int)

    def test_when_gamestate_is_default_constructed_then_rng_is_numpy_generator(self):
        state = GameState()
        assert isinstance(state.rng, np.random.Generator)


# ---------------------------------------------------------------------------
# Mutability — the dataclass must NOT be frozen
# ---------------------------------------------------------------------------


class TestGameStateMutability:
    """GameState must NOT be frozen — direct field assignment must never raise."""

    def test_when_phase_is_assigned_a_new_value_then_updated_value_is_visible(self):
        state = GameState()
        state.phase = PHASE_ATTACK
        assert state.phase == PHASE_ATTACK

    def test_when_armies_is_replaced_with_a_new_array_then_assignment_succeeds(self):
        state = GameState()
        new_armies = np.zeros(42, dtype=np.int32)
        state.armies = new_armies
        assert state.armies is new_armies

    def test_when_current_player_is_mutated_then_new_value_is_visible(self):
        state = GameState()
        state.current_player = 5
        assert state.current_player == 5

    def test_when_reinforcements_remaining_is_mutated_then_new_value_is_visible(self):
        state = GameState()
        state.reinforcements_remaining = 10
        assert state.reinforcements_remaining == 10

    def test_when_n_players_is_mutated_then_new_value_is_visible(self):
        state = GameState()
        state.n_players = 6
        assert state.n_players == 6

    def test_when_cards_is_replaced_then_new_list_is_visible(self):
        state = GameState()
        state.cards = [[], [], []]
        assert state.cards == [[], [], []]


# ---------------------------------------------------------------------------
# Property: default rng is a functional np.random.Generator
# Invariant: rng.integers(0, n) must not raise and must return value in [0, n)
# for any valid upper bound n >= 1  (never-raises-for-valid-input + ordering)
# ---------------------------------------------------------------------------


@given(st.integers(min_value=1, max_value=10_000))
def test_when_default_rng_samples_integers_then_result_is_within_stated_bounds(upper):
    state = GameState()
    result = state.rng.integers(0, upper)
    assert 0 <= result < upper


# ---------------------------------------------------------------------------
# Property: mutating phase to any integer never raises (mutable dataclass invariant)
# A frozen dataclass would raise FrozenInstanceError on every attempt.
# ---------------------------------------------------------------------------


@given(st.integers())
def test_when_phase_is_assigned_any_integer_then_no_exception_is_raised(value):
    state = GameState()
    state.phase = value
    assert state.phase == value
