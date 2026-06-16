"""Example tests for src/env_core/combat.py — Issue #50.

Territory layout used throughout:
    Player 0: 0 (Alaska), 1 (Alberta), 8 (Western US)
    Player 1: 4 (Greenland), 5 (NW Territory), 7 (Quebec)

ADJACENCY facts (from constants.py) relevant to these tests:
    0 (Alaska)           → [1, 5, 31]   — so 0→5 is a valid cross-player attack
    4 (Greenland)        → [5, 6, 7, 14] — 0 is NOT adjacent to 4

Cards are stored as state.cards[player] (list[int]).
Deck must be non-empty for draw_card to function.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.env_core.combat import (
    apply_capture_move,
    resolve_attack,
    validate_attack,
    validate_capture_move,
)
from src.env_core.state import (
    MAX_CARDS,
    PHASE_ATTACK,
    PHASE_CAPTURE_MOVE,
    PHASE_FORTIFY,
    GameState,
)

# --------------------------------------------------------------------------
# Territory constants (real Risiko board — avoids magic numbers in tests)
# --------------------------------------------------------------------------
_ALASKA = 0  # player-0 territory; adjacent to [1, 5, 31]
_ALBERTA = 1  # player-0 territory
_WESTERN_US = 8  # player-0 territory
_GREENLAND = 4  # player-1 territory; NOT adjacent to Alaska
_NW_TERRITORY = 5  # player-1 territory; adjacent to Alaska ✓
_QUEBEC = 7  # player-1 territory

# --------------------------------------------------------------------------
# Sequential mock RNG — returns preset arrays in order
# --------------------------------------------------------------------------


class _SeqRng:
    """Each call to integers() returns the next preset value, repeated *size* times."""

    def __init__(self, *arrays: list[int]) -> None:
        self._queue = [np.asarray(a, dtype=int) for a in arrays]
        self._idx = 0

    def integers(self, low: int, high: int, size=None) -> np.ndarray:
        if self._idx >= len(self._queue):
            raise AssertionError(
                f"Mock RNG exhausted after {self._idx} calls — add more preset arrays."
            )
        preset = self._queue[self._idx]
        self._idx += 1
        n = size if isinstance(size, int) else (size[0] if size else 1)
        return np.full(n, preset[0], dtype=int)


# --------------------------------------------------------------------------
# State fixtures
# --------------------------------------------------------------------------


def _make_state(n_players: int = 2, current_player: int = 0, seed: int = 0) -> GameState:
    """Full GameState with player-0 and player-1 territories (see module docstring)."""
    state = GameState(
        n_players=n_players,
        current_player=current_player,
        territory_owner=np.zeros(42, dtype=np.int32),
        armies=np.full(42, 3, dtype=np.int32),
        cards=[[] for _ in range(n_players)],
        deck=list(range(42)),  # non-empty so draw_card always has a card
        discard_pile=[],
        eliminated=np.zeros(n_players, dtype=np.int32),
        rng=np.random.default_rng(seed),
    )
    for t in (_ALASKA, _ALBERTA, _WESTERN_US):
        state.territory_owner[t] = 0
    for t in (_GREENLAND, _NW_TERRITORY, _QUEBEC):
        state.territory_owner[t] = 1
    return state


def _make_capture_state(
    last_capture_dice: int = 1,
    attacker_armies: int = 6,
) -> GameState:
    """State immediately after capture — awaiting apply_capture_move."""
    state = _make_state()
    state.last_capture_dice = last_capture_dice
    state.last_attacker = _ALASKA
    state.last_defender = _NW_TERRITORY
    state.armies[_ALASKA] = attacker_armies
    state.armies[_NW_TERRITORY] = 0
    state.territory_owner[_NW_TERRITORY] = 0  # captured by player 0
    state.turn_capture = 1
    state.phase = PHASE_CAPTURE_MOVE
    return state


# ==========================================================================
# validate_attack — six guard conditions
# ==========================================================================


class TestValidateAttack:
    """Criterion: validate_attack is False unless ALL six conditions hold."""

    def test_when_attacker_owned_by_other_player_then_attack_is_invalid(self):
        state = _make_state()
        state.current_player = 0
        # NW Territory belongs to player 1, not 0
        assert validate_attack(state, _NW_TERRITORY, _ALASKA, dice=1) is False

    def test_when_defender_owned_by_current_player_then_attack_is_invalid(self):
        state = _make_state()
        state.current_player = 0
        # Alberta also belongs to player 0 — cannot attack own territory
        assert validate_attack(state, _ALASKA, _ALBERTA, dice=1) is False

    def test_when_territories_not_adjacent_then_attack_is_invalid(self):
        state = _make_state()
        state.current_player = 0
        # Alaska (0) is NOT adjacent to Greenland (4)
        assert validate_attack(state, _ALASKA, _GREENLAND, dice=1) is False

    def test_when_attacker_has_exactly_one_army_then_attack_is_invalid(self):
        state = _make_state()
        state.current_player = 0
        state.armies[_ALASKA] = 1
        assert validate_attack(state, _ALASKA, _NW_TERRITORY, dice=1) is False

    def test_when_dice_is_zero_then_attack_is_invalid(self):
        state = _make_state()
        state.current_player = 0
        assert validate_attack(state, _ALASKA, _NW_TERRITORY, dice=0) is False

    def test_when_dice_is_four_then_attack_is_invalid(self):
        state = _make_state()
        state.current_player = 0
        assert validate_attack(state, _ALASKA, _NW_TERRITORY, dice=4) is False

    def test_when_dice_equals_attacker_armies_then_attack_is_invalid(self):
        state = _make_state()
        state.current_player = 0
        state.armies[_ALASKA] = 3
        # dice == armies → no army left behind → invalid
        assert validate_attack(state, _ALASKA, _NW_TERRITORY, dice=3) is False

    def test_when_all_six_conditions_met_then_attack_is_valid(self):
        state = _make_state()
        state.current_player = 0
        state.armies[_ALASKA] = 4  # 4 armies, 3 dice → 1 left behind ✓
        assert validate_attack(state, _ALASKA, _NW_TERRITORY, dice=3) is True


# --------------------------------------------------------------------------
# Property: any dice outside [1, 3] always fails
# --------------------------------------------------------------------------


@given(st.integers().filter(lambda d: not (1 <= d <= 3)))
def test_when_dice_outside_1_to_3_then_attack_is_always_invalid(dice: int) -> None:
    """Invariant: out-of-range dice → False regardless of other state."""
    state = _make_state()
    state.current_player = 0
    state.armies[_ALASKA] = 10
    assert validate_attack(state, _ALASKA, _NW_TERRITORY, dice=dice) is False


# ==========================================================================
# resolve_attack — dice mechanics and army decrements
# ==========================================================================


class TestResolveAttack:
    """Criterion: rolls via state.rng, sorts descending, ties → defender."""

    def test_when_attacker_rolls_higher_then_defender_loses_one_army(self):
        state = _make_state()
        state.armies[_ALASKA] = 5
        state.armies[_NW_TERRITORY] = 3
        state.rng = _SeqRng([6], [1])  # attacker wins → defender loses 1
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert state.armies[_NW_TERRITORY] == 2

    def test_when_defender_rolls_higher_then_attacker_loses_one_army(self):
        state = _make_state()
        state.armies[_ALASKA] = 5
        state.armies[_NW_TERRITORY] = 3
        state.rng = _SeqRng([1], [6])  # defender wins → attacker loses 1
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert state.armies[_ALASKA] == 4

    def test_when_attacker_and_defender_tie_then_defender_keeps_army(self):
        state = _make_state()
        state.armies[_ALASKA] = 5
        state.armies[_NW_TERRITORY] = 3
        state.rng = _SeqRng([6], [6])  # tie → defender wins
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert state.armies[_NW_TERRITORY] == 3

    def test_when_attacker_and_defender_tie_then_attacker_loses_one_army(self):
        state = _make_state()
        state.armies[_ALASKA] = 5
        state.armies[_NW_TERRITORY] = 3
        state.rng = _SeqRng([6], [6])  # tie → attacker loses 1
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert state.armies[_ALASKA] == 4


# ==========================================================================
# On capture: state fields set by resolve_attack
# ==========================================================================


class TestCaptureStateAfterResolveAttack:
    """Criteria: territory_owner, armies==0, turn_capture==1, last_capture_* set."""

    def _state_for_capture(self) -> GameState:
        """Defender has exactly 1 army; attacker wins → capture on first roll."""
        state = _make_state()
        state.armies[_ALASKA] = 5
        state.armies[_NW_TERRITORY] = 1
        state.rng = _SeqRng([6], [1])
        return state

    def test_when_capture_occurs_then_territory_owner_is_current_player(self):
        state = self._state_for_capture()
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert state.territory_owner[_NW_TERRITORY] == state.current_player

    def test_when_capture_occurs_then_armies_at_defender_are_zero(self):
        state = self._state_for_capture()
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert state.armies[_NW_TERRITORY] == 0

    def test_when_capture_occurs_then_turn_capture_flag_is_one(self):
        state = self._state_for_capture()
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert state.turn_capture == 1

    def test_when_capture_occurs_then_last_capture_dice_recorded(self):
        state = self._state_for_capture()
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert state.last_capture_dice == 1

    def test_when_capture_occurs_then_last_attacker_recorded(self):
        state = self._state_for_capture()
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert state.last_attacker == _ALASKA

    def test_when_capture_occurs_then_last_defender_recorded(self):
        state = self._state_for_capture()
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert state.last_defender == _NW_TERRITORY


# ==========================================================================
# Card draw on capture
# ==========================================================================


class TestCardDrawOnCapture:
    """Criterion: a card is drawn on capture when the hand is under MAX_CARDS."""

    def _capture_state(self, hand_count: int) -> GameState:
        state = _make_state()
        state.armies[_ALASKA] = 5
        state.armies[_NW_TERRITORY] = 1
        state.cards[0] = list(range(hand_count))  # dummy card symbols
        state.rng = _SeqRng([6], [1])  # attacker always wins
        return state

    def test_when_hand_below_max_then_card_is_drawn_on_capture(self):
        state = self._capture_state(hand_count=MAX_CARDS - 1)
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert len(state.cards[0]) == MAX_CARDS

    def test_when_hand_at_max_then_no_card_is_drawn_on_capture(self):
        state = self._capture_state(hand_count=MAX_CARDS)
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert len(state.cards[0]) == MAX_CARDS

    def test_when_hand_empty_then_one_card_drawn_on_capture(self):
        state = self._capture_state(hand_count=0)
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert len(state.cards[0]) == 1


# ==========================================================================
# Elimination on capture of a player's final territory
# ==========================================================================


class TestEliminationOnCapture:
    """Criteria: eliminated flag set; cards transferred to attacker."""

    def _single_territory_state(self, defender_cards: int = 3) -> GameState:
        """Player 1 owns ONLY NW Territory; player 0 is about to capture it."""
        state = _make_state()
        # Reset ownership — player 1 owns only territory 5
        state.territory_owner[:] = 0
        state.territory_owner[_ALASKA] = 0
        state.territory_owner[_NW_TERRITORY] = 1
        state.armies[_ALASKA] = 5
        state.armies[_NW_TERRITORY] = 1
        state.cards[1] = list(range(defender_cards))
        state.cards[0] = []
        state.deck = []  # empty deck → draw_card is a no-op after transfer
        state.discard_pile = []
        state.rng = _SeqRng([6], [1])
        return state

    def test_when_last_territory_captured_then_player_is_eliminated(self):
        state = self._single_territory_state()
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert state.eliminated[1] == 1

    def test_when_last_territory_captured_then_cards_transfer_to_attacker(self):
        state = self._single_territory_state(defender_cards=3)
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert len(state.cards[0]) == 3

    def test_when_last_territory_captured_then_eliminated_player_has_no_cards(self):
        state = self._single_territory_state(defender_cards=3)
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert state.cards[1] == []

    def test_when_non_final_territory_captured_then_player_is_not_eliminated(self):
        state = _make_state()
        state.armies[_ALASKA] = 5
        state.armies[_NW_TERRITORY] = 1
        # Player 1 still holds Greenland (4) and Quebec (7) after capture of 5
        state.rng = _SeqRng([6], [1])
        resolve_attack(state, _ALASKA, _NW_TERRITORY, dice=1)
        assert state.eliminated[1] == 0


# ==========================================================================
# validate_capture_move
# ==========================================================================


class TestValidateCaptureMove:
    """Criterion: last_capture_dice <= move <= armies[last_attacker] - 1."""

    def test_when_move_below_last_capture_dice_then_invalid(self):
        state = _make_capture_state(last_capture_dice=2, attacker_armies=6)
        assert validate_capture_move(state, move=1) is False

    def test_when_move_exceeds_armies_minus_one_then_invalid(self):
        state = _make_capture_state(last_capture_dice=1, attacker_armies=4)
        # max valid = 4-1=3; move=4 leaves 0 behind → invalid
        assert validate_capture_move(state, move=4) is False

    def test_when_move_equals_last_capture_dice_then_valid(self):
        state = _make_capture_state(last_capture_dice=2, attacker_armies=6)
        assert validate_capture_move(state, move=2) is True

    def test_when_move_equals_armies_minus_one_then_valid(self):
        state = _make_capture_state(last_capture_dice=1, attacker_armies=5)
        assert validate_capture_move(state, move=4) is True  # 5-1=4

    def test_when_move_within_range_then_valid(self):
        state = _make_capture_state(last_capture_dice=1, attacker_armies=6)
        assert validate_capture_move(state, move=3) is True


# --------------------------------------------------------------------------
# Property: every integer in [last_capture_dice, armies[attacker]-1] is valid
# --------------------------------------------------------------------------


@given(
    last_capture_dice=st.integers(min_value=1, max_value=3),
    extra=st.integers(min_value=2, max_value=20),
)
def test_when_move_within_valid_range_then_capture_move_is_always_valid(
    last_capture_dice: int, extra: int
) -> None:
    """Invariant: every move in [last_capture_dice, armies-1] is valid."""
    attacker_armies = last_capture_dice + extra
    state = _make_capture_state(
        last_capture_dice=last_capture_dice, attacker_armies=attacker_armies
    )
    for move in range(last_capture_dice, attacker_armies):
        assert validate_capture_move(state, move=move) is True, (
            f"move={move} last_capture_dice={last_capture_dice} attacker_armies={attacker_armies}"
        )


# ==========================================================================
# apply_capture_move
# ==========================================================================


class TestApplyCaptureMove:
    """Criterion: transfer armies leaving ≥1 behind; set next phase."""

    def test_when_move_applied_then_armies_moved_to_defender_territory(self):
        state = _make_capture_state(last_capture_dice=1, attacker_armies=6)
        apply_capture_move(state, move=3)
        assert state.armies[_NW_TERRITORY] == 3

    def test_when_move_applied_then_attacker_armies_decrease(self):
        state = _make_capture_state(last_capture_dice=1, attacker_armies=6)
        apply_capture_move(state, move=3)
        assert state.armies[_ALASKA] == 3

    def test_when_move_applied_then_attacker_retains_at_least_one_army(self):
        state = _make_capture_state(last_capture_dice=1, attacker_armies=5)
        apply_capture_move(state, move=4)  # max legal: 5-1=4
        assert state.armies[_ALASKA] >= 1

    def test_when_move_applied_then_total_armies_conserved(self):
        state = _make_capture_state(last_capture_dice=1, attacker_armies=7)
        apply_capture_move(state, move=4)
        assert state.armies[_ALASKA] + state.armies[_NW_TERRITORY] == 7

    def test_when_move_applied_and_attacker_has_enough_then_phase_is_attack(self):
        state = _make_capture_state(last_capture_dice=1, attacker_armies=5)
        apply_capture_move(state, move=2)  # 5-2=3 ≥ 2 → PHASE_ATTACK
        assert state.phase == PHASE_ATTACK

    def test_when_move_applied_and_attacker_has_one_army_then_phase_is_fortify(self):
        state = _make_capture_state(last_capture_dice=1, attacker_armies=2)
        apply_capture_move(state, move=1)  # 2-1=1 < 2 → PHASE_FORTIFY
        assert state.phase == PHASE_FORTIFY


# --------------------------------------------------------------------------
# Property: army conservation for all valid moves
# --------------------------------------------------------------------------


@given(
    attacker_armies=st.integers(min_value=3, max_value=30),
    move_offset=st.integers(min_value=0, max_value=27),
)
def test_when_capture_move_applied_then_armies_are_conserved(
    attacker_armies: int, move_offset: int
) -> None:
    """Invariant: total armies unchanged after apply_capture_move."""
    move = 1 + move_offset  # min move = last_capture_dice = 1
    if move >= attacker_armies:
        pytest.skip("move outside valid range — boundary tests cover this")
    state = _make_capture_state(last_capture_dice=1, attacker_armies=attacker_armies)
    apply_capture_move(state, move=move)
    assert state.armies[_ALASKA] + state.armies[_NW_TERRITORY] == attacker_armies
