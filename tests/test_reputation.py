"""Spec-blind example tests for ReputationBook (issue #98).

Derived directly from acceptance criteria; no implementation source was read.
All tests are in the Red phase: they will fail until the implementation exists.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from src.agents.reputation import ReputationBook

# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------


def _make_book() -> ReputationBook:
    return ReputationBook()


# ---------------------------------------------------------------------------
# Criterion: ReputationBook accumulates across games
# ---------------------------------------------------------------------------


def test_when_one_betrayal_recorded_then_count_is_one():
    book = _make_book()
    book.record_betrayal(player=3)
    assert book.betrayal_count(player=3) == 1


def test_when_three_betrayals_recorded_then_count_is_three():
    book = _make_book()
    book.record_betrayal(player=1)
    book.record_betrayal(player=1)
    book.record_betrayal(player=1)
    assert book.betrayal_count(player=1) == 3


def test_when_no_betrayal_recorded_then_count_is_zero():
    book = _make_book()
    assert book.betrayal_count(player=0) == 0


def test_when_betrayals_recorded_for_different_players_then_counts_are_independent():
    book = _make_book()
    book.record_betrayal(player=0)
    book.record_betrayal(player=0)
    book.record_betrayal(player=2)
    assert book.betrayal_count(player=0) == 2
    assert book.betrayal_count(player=2) == 1


# ---------------------------------------------------------------------------
# Criterion: trust_penalty is monotone
# ---------------------------------------------------------------------------


def test_when_no_betrayal_then_trust_penalty_is_zero():
    """A player with no betrayals has the minimum (zero) trust penalty."""
    book = _make_book()
    assert book.trust_penalty(player=5) == 0.0


def test_when_one_betrayal_then_trust_penalty_is_positive():
    book = _make_book()
    book.record_betrayal(player=0)
    assert book.trust_penalty(player=0) > 0.0


def test_when_two_betrayals_then_trust_penalty_is_higher_or_equal_to_one():
    """More betrayals must not lower the trust penalty (monotone)."""
    book_one = _make_book()
    book_one.record_betrayal(player=0)
    penalty_one = book_one.trust_penalty(player=0)

    book_two = _make_book()
    book_two.record_betrayal(player=0)
    book_two.record_betrayal(player=0)
    penalty_two = book_two.trust_penalty(player=0)

    assert penalty_two >= penalty_one


@given(st.integers(min_value=0, max_value=30))
def test_when_n_betrayals_recorded_then_trust_penalty_is_nonnegative(n: int):
    """trust_penalty is always non-negative for any betrayal count."""
    book = _make_book()
    for _ in range(n):
        book.record_betrayal(player=0)
    assert book.trust_penalty(player=0) >= 0.0


@given(st.integers(min_value=1, max_value=20))
def test_when_one_more_betrayal_added_then_penalty_does_not_decrease(n: int):
    """Adding another betrayal never lowers the trust penalty (strict monotone direction)."""
    book = _make_book()
    for _ in range(n):
        book.record_betrayal(player=2)
    penalty_n = book.trust_penalty(player=2)
    book.record_betrayal(player=2)
    penalty_n_plus_1 = book.trust_penalty(player=2)
    assert penalty_n_plus_1 >= penalty_n


# ---------------------------------------------------------------------------
# Criterion: reset() clears the book
# ---------------------------------------------------------------------------


def test_when_reset_then_betrayal_count_returns_to_zero():
    book = _make_book()
    book.record_betrayal(player=2)
    book.record_betrayal(player=2)
    book.reset()
    assert book.betrayal_count(player=2) == 0


def test_when_reset_then_trust_penalty_returns_to_zero():
    """After reset the trust penalty must also be cleared."""
    book = _make_book()
    book.record_betrayal(player=4)
    book.reset()
    assert book.trust_penalty(player=4) == 0.0


def test_when_reset_then_subsequent_betrayal_count_starts_fresh():
    book = _make_book()
    book.record_betrayal(player=0)
    book.reset()
    book.record_betrayal(player=0)
    assert book.betrayal_count(player=0) == 1


# ---------------------------------------------------------------------------
# Criterion: two books are independent
# ---------------------------------------------------------------------------


def test_when_two_books_updated_separately_then_counts_do_not_bleed():
    """Two ReputationBook instances must not share any internal state."""
    book_a = _make_book()
    book_b = _make_book()

    book_a.record_betrayal(player=0)
    book_a.record_betrayal(player=0)

    assert book_a.betrayal_count(player=0) == 2
    assert book_b.betrayal_count(player=0) == 0


def test_when_one_book_reset_then_other_book_is_unaffected():
    book_a = _make_book()
    book_b = _make_book()

    book_a.record_betrayal(player=3)
    book_b.record_betrayal(player=3)

    book_a.reset()

    assert book_a.betrayal_count(player=3) == 0
    assert book_b.betrayal_count(player=3) == 1


def test_when_two_books_accumulate_independently_then_penalties_differ():
    """Books with different betrayal histories yield different trust penalties."""
    book_a = _make_book()
    book_b = _make_book()

    book_a.record_betrayal(player=1)
    # book_b has no betrayals
    assert book_a.trust_penalty(player=1) >= book_b.trust_penalty(player=1)
