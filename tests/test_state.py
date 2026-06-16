"""Tests for src/env_core/state.py — GameState container and phase constants."""

from __future__ import annotations

import numpy as np

from src.env_core.state import (
    MAX_CARDS,
    PHASE_ATTACK,
    PHASE_CAPTURE_MOVE,
    PHASE_FORTIFY,
    PHASE_REINFORCE,
    PHASE_TRADE,
    GameState,
)


class TestPhaseConstants:
    """Phase constants have the correct values and are distinct."""

    def test_when_phase_trade_imported_then_value_is_zero(self):
        assert PHASE_TRADE == 0

    def test_when_phase_reinforce_imported_then_value_is_one(self):
        assert PHASE_REINFORCE == 1

    def test_when_phase_attack_imported_then_value_is_two(self):
        assert PHASE_ATTACK == 2

    def test_when_phase_capture_move_imported_then_value_is_three(self):
        assert PHASE_CAPTURE_MOVE == 3

    def test_when_phase_fortify_imported_then_value_is_four(self):
        assert PHASE_FORTIFY == 4

    def test_when_all_phase_constants_collected_then_all_are_distinct(self):
        phases = [PHASE_TRADE, PHASE_REINFORCE, PHASE_ATTACK, PHASE_CAPTURE_MOVE, PHASE_FORTIFY]
        assert len(set(phases)) == 5

    def test_when_max_cards_imported_then_value_is_five(self):
        assert MAX_CARDS == 5


class TestModuleExports:
    """__all__ must include every public symbol."""

    def test_when_module_all_checked_then_game_state_is_exported(self):
        import src.env_core.state as m

        assert "GameState" in m.__all__

    def test_when_module_all_checked_then_all_phase_constants_are_exported(self):
        import src.env_core.state as m

        for name in (
            "PHASE_TRADE",
            "PHASE_REINFORCE",
            "PHASE_ATTACK",
            "PHASE_CAPTURE_MOVE",
            "PHASE_FORTIFY",
        ):
            assert name in m.__all__, f"{name} missing from __all__"

    def test_when_module_all_checked_then_max_cards_is_exported(self):
        import src.env_core.state as m

        assert "MAX_CARDS" in m.__all__


class TestGameStateConstruction:
    """GameState() must construct with sensible zero/empty defaults."""

    def test_when_game_state_constructed_then_territory_owner_is_int32_array(self):
        state = GameState()
        assert isinstance(state.territory_owner, np.ndarray)
        assert state.territory_owner.dtype == np.int32

    def test_when_game_state_constructed_then_armies_is_int32_array(self):
        state = GameState()
        assert isinstance(state.armies, np.ndarray)
        assert state.armies.dtype == np.int32

    def test_when_game_state_constructed_then_eliminated_is_int32_array(self):
        state = GameState()
        assert isinstance(state.eliminated, np.ndarray)
        assert state.eliminated.dtype == np.int32

    def test_when_game_state_constructed_then_rng_is_numpy_generator(self):
        state = GameState()
        assert isinstance(state.rng, np.random.Generator)

    def test_when_game_state_constructed_then_cards_is_list(self):
        state = GameState()
        assert isinstance(state.cards, list)

    def test_when_game_state_constructed_then_deck_is_list(self):
        state = GameState()
        assert isinstance(state.deck, list)

    def test_when_game_state_constructed_then_discard_pile_is_list(self):
        state = GameState()
        assert isinstance(state.discard_pile, list)

    def test_when_game_state_constructed_then_trade_count_is_zero(self):
        assert GameState().trade_count == 0

    def test_when_game_state_constructed_then_current_player_is_zero(self):
        assert GameState().current_player == 0

    def test_when_game_state_constructed_then_reinforcements_remaining_is_zero(self):
        assert GameState().reinforcements_remaining == 0

    def test_when_game_state_constructed_then_turn_capture_is_zero(self):
        assert GameState().turn_capture == 0

    def test_when_game_state_constructed_then_n_players_is_zero(self):
        assert GameState().n_players == 0

    def test_when_game_state_constructed_then_last_capture_dice_is_zero(self):
        assert GameState().last_capture_dice == 0

    def test_when_two_instances_constructed_then_rngs_are_independent(self):
        s1 = GameState()
        s2 = GameState()
        assert s1.rng is not s2.rng


class TestGameStateMutability:
    """GameState must NOT be frozen — fields must be directly assignable."""

    def test_when_phase_assigned_then_value_is_updated(self):
        state = GameState()
        state.phase = PHASE_ATTACK
        assert state.phase == PHASE_ATTACK

    def test_when_armies_assigned_then_value_is_updated(self):
        state = GameState()
        state.armies = np.array([3, 1, 2], dtype=np.int32)
        assert state.armies[0] == 3

    def test_when_current_player_assigned_then_value_is_updated(self):
        state = GameState()
        state.current_player = 3
        assert state.current_player == 3
