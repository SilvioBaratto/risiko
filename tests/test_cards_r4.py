"""Source-blind example tests for env_core/cards.py — Issue #49.

Every test is derived from an acceptance criterion only.  No implementation
source was read.  All tests fail today (red phase of TDD) and should pass
once each criterion is genuinely met.

Skipped criteria (not runtime-verifiable per oracle report):
  - "All tests pass" (boilerplate gate)
  - "SOLID / clean code" (subjective prose)
"""

import types

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.env_core.cards import (
    MAX_CARDS,
    apply_trade,
    draw_card,
    forced_trade,
    forms_valid_set,
    get_trade_value,
    has_tradeable_cards,
    is_valid_trade,
    transfer_cards,
)

# ── Symbol constants ──────────────────────────────────────────────────────────
# Cards are stored as integers: 0, 1, 2 = non-wildcard types; 3 = wildcard.
WILDCARD = 3


# ── Minimal state fixture ─────────────────────────────────────────────────────


def _make_state(
    *,
    hands=None,
    deck=None,
    discard_pile=None,
    trade_count: int = 0,
    reinforcements_remaining: int = 0,
    rng=None,
    n_players: int = 2,
):
    """Build the smallest SimpleNamespace that card functions need.

    hands — list-of-lists indexed by player_id; stored as state.cards to match
    GameState. Cards are symbol integers (0=Infantry, 1=Cavalry, 2=Artillery,
    3=Wildcard).
    """
    s = types.SimpleNamespace()
    s.cards = hands if hands is not None else [[] for _ in range(n_players)]
    s.deck = list(deck) if deck is not None else []
    s.discard_pile = list(discard_pile) if discard_pile is not None else []
    s.trade_count = trade_count
    s.reinforcements_remaining = reinforcements_remaining
    s.rng = rng if rng is not None else np.random.default_rng(42)
    return s


# ── forms_valid_set ───────────────────────────────────────────────────────────
# Criterion: True for all-same, all-different, or any set containing wildcard (3);
#            False otherwise — e.g. [0,0,1].


class TestFormsValidSet:
    """Criterion: True for all-same, all-different, or any set containing wildcard (3)."""

    def test_when_all_symbols_are_same_forms_valid_set_returns_true(self):
        assert forms_valid_set([0, 0, 0]) is True

    def test_when_all_symbols_are_different_forms_valid_set_returns_true(self):
        assert forms_valid_set([0, 1, 2]) is True

    def test_when_set_contains_one_wildcard_forms_valid_set_returns_true(self):
        assert forms_valid_set([WILDCARD, 0, 0]) is True

    def test_when_all_three_are_wildcards_forms_valid_set_returns_true(self):
        assert forms_valid_set([WILDCARD, WILDCARD, WILDCARD]) is True

    def test_when_wildcard_with_two_different_forms_valid_set_returns_true(self):
        assert forms_valid_set([WILDCARD, 0, 1]) is True

    def test_when_mixed_non_wildcard_canonical_example_forms_valid_set_returns_false(self):
        # The criterion explicitly names [0,0,1] as the canonical False example.
        assert forms_valid_set([0, 0, 1]) is False

    def test_when_two_same_one_different_no_wildcard_forms_valid_set_returns_false(self):
        assert forms_valid_set([1, 1, 2]) is False

    def test_when_two_same_one_different_variant_forms_valid_set_returns_false(self):
        assert forms_valid_set([2, 0, 2]) is False

    # ── Property: every all-same triple is valid ──────────────────────────────
    @given(sym=st.integers(min_value=0, max_value=2))
    def test_when_all_same_symbol_forms_valid_set_always_returns_true(self, sym):
        assert forms_valid_set([sym, sym, sym]) is True

    # ── Property: wildcard always makes a triple valid ────────────────────────
    @given(
        a=st.integers(min_value=0, max_value=2),
        b=st.integers(min_value=0, max_value=2),
    )
    def test_when_set_contains_wildcard_forms_valid_set_always_returns_true(self, a, b):
        assert forms_valid_set([WILDCARD, a, b]) is True


# ── is_valid_trade ────────────────────────────────────────────────────────────
# Criterion: rejects duplicate or out-of-bounds indices,
#            then delegates to forms_valid_set.


class TestIsValidTrade:
    """Criterion: rejects duplicate/OOB indices; delegates symbol check to forms_valid_set."""

    def test_when_indices_are_duplicate_is_valid_trade_returns_false(self):
        state = _make_state(hands=[[0, 1, 2]])
        assert is_valid_trade(state, [0, 0, 1], player=0) is False

    def test_when_index_is_out_of_bounds_is_valid_trade_returns_false(self):
        # hand has 3 cards (indices 0–2); index 5 is out of range
        state = _make_state(hands=[[0, 1, 2]])
        assert is_valid_trade(state, [0, 1, 5], player=0) is False

    def test_when_valid_indices_form_all_same_is_valid_trade_returns_true(self):
        state = _make_state(hands=[[0, 0, 0]])
        assert is_valid_trade(state, [0, 1, 2], player=0) is True

    def test_when_valid_indices_form_all_different_is_valid_trade_returns_true(self):
        state = _make_state(hands=[[0, 1, 2]])
        assert is_valid_trade(state, [0, 1, 2], player=0) is True

    def test_when_valid_indices_but_invalid_set_is_valid_trade_returns_false(self):
        # hand [0,0,1] — valid indices but forms_valid_set([0,0,1]) is False
        state = _make_state(hands=[[0, 0, 1]])
        assert is_valid_trade(state, [0, 1, 2], player=0) is False

    def test_when_wildcard_in_hand_and_valid_indices_is_valid_trade_returns_true(self):
        state = _make_state(hands=[[WILDCARD, 0, 0]])
        assert is_valid_trade(state, [0, 1, 2], player=0) is True


# ── has_tradeable_cards ───────────────────────────────────────────────────────
# Criterion: True iff some 3-card combination of the hand is valid.


class TestHasTradeableCards:
    """Criterion: True iff some 3-card combination of the hand forms a valid set."""

    def test_when_hand_is_empty_has_tradeable_cards_returns_false(self):
        state = _make_state(hands=[[]])
        assert has_tradeable_cards(state, 0) is False

    def test_when_hand_has_fewer_than_three_cards_has_tradeable_cards_returns_false(self):
        state = _make_state(hands=[[0, 1]])
        assert has_tradeable_cards(state, 0) is False

    def test_when_three_all_same_cards_has_tradeable_cards_returns_true(self):
        state = _make_state(hands=[[0, 0, 0]])
        assert has_tradeable_cards(state, 0) is True

    def test_when_three_all_different_cards_has_tradeable_cards_returns_true(self):
        state = _make_state(hands=[[0, 1, 2]])
        assert has_tradeable_cards(state, 0) is True

    def test_when_hand_has_wildcard_has_tradeable_cards_returns_true(self):
        state = _make_state(hands=[[WILDCARD, 0, 1]])
        assert has_tradeable_cards(state, 0) is True

    def test_when_only_available_combo_is_invalid_has_tradeable_cards_returns_false(self):
        # [0,0,1]: the only 3-combo is [0,0,1] — not same, not all-diff, no wildcard
        state = _make_state(hands=[[0, 0, 1]])
        assert has_tradeable_cards(state, 0) is False

    def test_when_four_cards_all_combos_invalid_has_tradeable_cards_returns_false(self):
        # [0,0,1,1]: all C(4,3)=4 combos are mixed non-wildcard → all invalid
        state = _make_state(hands=[[0, 0, 1, 1]])
        assert has_tradeable_cards(state, 0) is False

    def test_when_five_cards_containing_valid_combo_has_tradeable_cards_returns_true(self):
        # [0,0,1,1,2]: contains [0,1,2] (indices 0,2,4) — all-different → valid
        state = _make_state(hands=[[0, 0, 1, 1, 2]])
        assert has_tradeable_cards(state, 0) is True


# ── draw_card ─────────────────────────────────────────────────────────────────
# Criterion: pops one card from the deck into the hand;
#            reshuffles discard_pile into deck (via state.rng) when deck is empty;
#            no-ops when both are empty.


class TestDrawCard:
    """Criterion: pops from deck; reshuffles discard when empty; no-ops when both empty."""

    def test_when_deck_nonempty_draw_card_adds_one_card_to_hand(self):
        state = _make_state(hands=[[]], deck=[0, 1, 2])
        draw_card(state, 0)
        assert len(state.cards[0]) == 1

    def test_when_deck_nonempty_draw_card_removes_one_card_from_deck(self):
        state = _make_state(hands=[[]], deck=[0, 1, 2])
        draw_card(state, 0)
        assert len(state.deck) == 2

    def test_when_deck_nonempty_drawn_card_was_in_original_deck(self):
        original = [0, 1, 2]
        state = _make_state(hands=[[]], deck=list(original))
        draw_card(state, 0)
        assert state.cards[0][0] in original

    def test_when_deck_empty_and_discard_nonempty_draw_card_reshuffles_then_draws(self):
        state = _make_state(hands=[[]], deck=[], discard_pile=[0, 1, 2])
        draw_card(state, 0)
        assert len(state.cards[0]) == 1

    def test_when_deck_empty_and_discard_nonempty_total_cards_are_conserved(self):
        state = _make_state(hands=[[]], deck=[], discard_pile=[0, 1, 2])
        draw_card(state, 0)
        total = len(state.cards[0]) + len(state.deck) + len(state.discard_pile)
        assert total == 3  # no cards created or destroyed

    def test_when_deck_empty_and_discard_nonempty_discard_pile_is_cleared(self):
        state = _make_state(hands=[[]], deck=[], discard_pile=[0, 1, 2])
        draw_card(state, 0)
        assert len(state.discard_pile) == 0

    def test_when_both_deck_and_discard_empty_draw_card_does_not_raise(self):
        state = _make_state(hands=[[]], deck=[], discard_pile=[])
        draw_card(state, 0)  # must not raise

    def test_when_both_deck_and_discard_empty_hand_remains_unchanged(self):
        state = _make_state(hands=[[2]], deck=[], discard_pile=[])
        draw_card(state, 0)
        assert len(state.cards[0]) == 1  # no card added


# ── get_trade_value ───────────────────────────────────────────────────────────
# Criterion (implicit from apply_trade): escalating values 4, 6, 8, 10, 12, 15,
#            then +5 each trade thereafter.


class TestGetTradeValue:
    """Criterion: escalating values 4,6,8,10,12,15 then +5 each trade thereafter."""

    @pytest.mark.parametrize(
        "count,expected",
        [
            (0, 4),
            (1, 6),
            (2, 8),
            (3, 10),
            (4, 12),
            (5, 15),
            (6, 20),
            (7, 25),
            (8, 30),
        ],
    )
    def test_when_prev_trade_count_is_n_get_trade_value_returns_expected(self, count, expected):
        assert get_trade_value(count) == expected

    # ── Property: strictly increasing ────────────────────────────────────────
    @given(n=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_when_count_increases_get_trade_value_is_strictly_increasing(self, n):
        assert get_trade_value(n + 1) > get_trade_value(n)

    # ── Property: at least 20 after first 6 trades ────────────────────────────
    @given(n=st.integers(min_value=6, max_value=200))
    @settings(max_examples=50)
    def test_when_count_exceeds_5_get_trade_value_is_at_least_20(self, n):
        assert get_trade_value(n) >= 20


# ── apply_trade ───────────────────────────────────────────────────────────────
# Criterion: increments state.trade_count, adds get_trade_value(prev_count)
#            armies to reinforcements_remaining, moves traded cards to
#            discard_pile; returns False on invalid set.


class TestApplyTrade:
    """Criterion: increments trade_count, awards armies, moves cards; False on invalid."""

    def test_when_valid_trade_apply_trade_increments_trade_count_by_one(self):
        state = _make_state(hands=[[0, 0, 0]], trade_count=0)
        apply_trade(state, [0, 1, 2], player=0)
        assert state.trade_count == 1

    def test_when_valid_trade_apply_trade_uses_prev_count_0_adds_4_armies(self):
        state = _make_state(hands=[[0, 0, 0]], trade_count=0, reinforcements_remaining=0)
        apply_trade(state, [0, 1, 2], player=0)
        assert state.reinforcements_remaining == 4  # get_trade_value(0) == 4

    def test_when_valid_trade_with_prev_count_1_apply_trade_adds_6_armies(self):
        state = _make_state(hands=[[0, 0, 0]], trade_count=1, reinforcements_remaining=0)
        apply_trade(state, [0, 1, 2], player=0)
        assert state.reinforcements_remaining == 6  # get_trade_value(1) == 6

    def test_when_valid_trade_apply_trade_removes_traded_cards_from_hand(self):
        # 5-card hand; trade 3 → 2 remain
        state = _make_state(hands=[[0, 0, 0, 1, 2]], discard_pile=[])
        apply_trade(state, [0, 1, 2], player=0)
        assert len(state.cards[0]) == 2

    def test_when_valid_trade_apply_trade_moves_three_cards_to_discard_pile(self):
        state = _make_state(hands=[[0, 0, 0]], discard_pile=[])
        apply_trade(state, [0, 1, 2], player=0)
        assert len(state.discard_pile) == 3

    def test_when_invalid_set_apply_trade_returns_false(self):
        # [0,0,1] is not a valid set
        state = _make_state(hands=[[0, 0, 1]])
        result = apply_trade(state, [0, 1, 2], player=0)
        assert result is False

    def test_when_invalid_set_apply_trade_does_not_mutate_trade_count(self):
        state = _make_state(hands=[[0, 0, 1]], trade_count=3)
        apply_trade(state, [0, 1, 2], player=0)
        assert state.trade_count == 3

    def test_when_invalid_set_apply_trade_does_not_mutate_reinforcements(self):
        state = _make_state(hands=[[0, 0, 1]], reinforcements_remaining=7)
        apply_trade(state, [0, 1, 2], player=0)
        assert state.reinforcements_remaining == 7

    def test_when_invalid_set_apply_trade_leaves_hand_unchanged(self):
        state = _make_state(hands=[[0, 0, 1]], discard_pile=[])
        apply_trade(state, [0, 1, 2], player=0)
        assert len(state.cards[0]) == 3
        assert len(state.discard_pile) == 0


# ── forced_trade ──────────────────────────────────────────────────────────────
# Criterion: applies the first valid combo; no-ops (without error) when none exists.


class TestForcedTrade:
    """Criterion: applies first valid 3-card combo; no-ops without error when none exists."""

    def test_when_valid_combo_exists_forced_trade_removes_three_cards(self):
        state = _make_state(hands=[[0, 0, 0]], trade_count=0, reinforcements_remaining=0)
        forced_trade(state, 0)
        assert len(state.cards[0]) == 0

    def test_when_valid_combo_exists_forced_trade_increments_trade_count(self):
        state = _make_state(hands=[[0, 1, 2]], trade_count=0)
        forced_trade(state, 0)
        assert state.trade_count == 1

    def test_when_valid_combo_exists_forced_trade_adds_armies_for_prev_count(self):
        state = _make_state(hands=[[0, 1, 2]], trade_count=0, reinforcements_remaining=0)
        forced_trade(state, 0)
        assert state.reinforcements_remaining == get_trade_value(0)

    def test_when_no_valid_combo_forced_trade_does_not_raise(self):
        # [0,0,1]: no valid 3-card combo
        state = _make_state(hands=[[0, 0, 1]])
        forced_trade(state, 0)  # must not raise

    def test_when_no_valid_combo_forced_trade_leaves_hand_unchanged(self):
        state = _make_state(hands=[[0, 0, 1]], trade_count=2, reinforcements_remaining=5)
        forced_trade(state, 0)
        assert len(state.cards[0]) == 3

    def test_when_no_valid_combo_forced_trade_leaves_trade_count_unchanged(self):
        state = _make_state(hands=[[0, 0, 1]], trade_count=2)
        forced_trade(state, 0)
        assert state.trade_count == 2

    def test_when_no_valid_combo_forced_trade_leaves_reinforcements_unchanged(self):
        state = _make_state(hands=[[0, 0, 1]], reinforcements_remaining=5)
        forced_trade(state, 0)
        assert state.reinforcements_remaining == 5

    def test_when_hand_is_empty_forced_trade_does_not_raise(self):
        state = _make_state(hands=[[]])
        forced_trade(state, 0)  # must not raise


# ── transfer_cards ────────────────────────────────────────────────────────────
# Criterion: moves all cards from from_player to to_player, then force-trades
#            while the hand exceeds MAX_CARDS, terminating safely.


class TestTransferCards:
    """Criterion: moves all cards to to_player; force-trades while over MAX_CARDS."""

    def test_when_from_player_has_cards_all_are_removed_from_source(self):
        state = _make_state(hands=[[0, 1, 2], []])
        transfer_cards(state, from_player=0, to_player=1)
        assert len(state.cards[0]) == 0

    def test_when_transfer_occurs_total_card_count_is_conserved(self):
        state = _make_state(hands=[[0, 1, 2], [0, 0, 0]], discard_pile=[])
        total_before = sum(len(h) for h in state.cards) + len(state.discard_pile)
        transfer_cards(state, from_player=0, to_player=1)
        total_after = sum(len(h) for h in state.cards) + len(state.discard_pile)
        assert total_after == total_before

    def test_when_transfer_pushes_to_player_over_max_cards_hand_is_reduced(self):
        # to_player starts with MAX_CARDS cards, receives 3 more → 8 > MAX_CARDS
        # [0,0,0] is a valid all-same set; forced_trade should reduce the hand
        state = _make_state(
            hands=[[0, 0, 0], [0, 0, 0, 1, 2]],  # from=3 cards, to=5 cards
            trade_count=0,
            reinforcements_remaining=0,
            discard_pile=[],
        )
        transfer_cards(state, from_player=0, to_player=1)
        assert len(state.cards[1]) <= MAX_CARDS

    def test_when_transfer_does_not_cause_excess_no_forced_trade_occurs(self):
        # Transfer 2 cards to a player with 2 → 4 total < MAX_CARDS; no forced trade
        state = _make_state(
            hands=[[0, 1], [2, 0]],
            trade_count=0,
            reinforcements_remaining=0,
        )
        transfer_cards(state, from_player=0, to_player=1)
        assert state.trade_count == 0

    def test_when_from_player_has_no_cards_transfer_does_not_raise(self):
        state = _make_state(hands=[[], [0]])
        transfer_cards(state, from_player=0, to_player=1)  # must not raise

    def test_when_from_player_has_no_cards_to_player_hand_is_unchanged(self):
        state = _make_state(hands=[[], [0, 1]])
        transfer_cards(state, from_player=0, to_player=1)
        assert len(state.cards[1]) == 2

    def test_max_cards_constant_equals_five(self):
        # Criterion: max 5 cards in hand
        assert MAX_CARDS == 5
