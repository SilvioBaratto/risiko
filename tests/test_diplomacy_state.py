"""Spec-blind example tests for DiplomacyState (issue #98).

Derived directly from acceptance criteria; no implementation source was read.
All tests are in the Red phase: they will fail until the implementation exists.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from src.agents.diplomacy import DiplomacyState

# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------


def _make_state(decay: float = 0.9) -> DiplomacyState:
    """Fresh DiplomacyState with a given decay coefficient."""
    return DiplomacyState(decay=decay)


# ---------------------------------------------------------------------------
# Criterion: form_alliance/break_alliance are symmetric and idempotent;
#            are_allied works both directions
# ---------------------------------------------------------------------------


def test_when_alliance_formed_then_are_allied_true_in_both_directions():
    state = _make_state()
    state.form_alliance(0, 1)
    assert state.are_allied(0, 1) is True
    assert state.are_allied(1, 0) is True


def test_when_alliance_broken_then_are_allied_false_in_both_directions():
    state = _make_state()
    state.form_alliance(2, 3)
    state.break_alliance(2, 3)
    assert state.are_allied(2, 3) is False
    assert state.are_allied(3, 2) is False


def test_when_form_alliance_called_twice_then_allied_state_unchanged():
    """form_alliance is idempotent: a second call on the same pair is a no-op."""
    state = _make_state()
    state.form_alliance(0, 2)
    state.form_alliance(0, 2)
    assert state.are_allied(0, 2) is True


def test_when_break_alliance_called_twice_then_no_error_and_not_allied():
    """break_alliance is idempotent: a second call on an absent alliance is a no-op."""
    state = _make_state()
    state.form_alliance(1, 3)
    state.break_alliance(1, 3)
    state.break_alliance(1, 3)  # must not raise
    assert state.are_allied(1, 3) is False


def test_when_no_alliance_formed_then_are_allied_returns_false():
    state = _make_state()
    assert state.are_allied(0, 5) is False


@given(
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=0, max_value=5),
)
def test_when_alliance_formed_symmetry_holds_for_all_valid_pairs(a: int, b: int):
    """are_allied(a, b) == are_allied(b, a) after form_alliance(a, b)."""
    if a == b:
        return
    state = _make_state()
    state.form_alliance(a, b)
    assert state.are_allied(a, b) == state.are_allied(b, a)


@given(
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=0, max_value=5),
)
def test_when_form_alliance_twice_result_equals_once_for_any_pair(a: int, b: int):
    """form_alliance is idempotent for all valid player pairs."""
    if a == b:
        return
    once = _make_state()
    once.form_alliance(a, b)

    twice = _make_state()
    twice.form_alliance(a, b)
    twice.form_alliance(a, b)

    assert once.are_allied(a, b) == twice.are_allied(a, b)


# ---------------------------------------------------------------------------
# Criterion: record_attack appends a grudge; grudge_score decays with turn
#            distance and is computed on read (log stays immutable)
# ---------------------------------------------------------------------------


def test_when_attack_recorded_then_grudge_score_is_positive():
    state = _make_state()
    state.record_attack(offender=1, victim=0, turn=5, base_severity=1.0)
    score = state.grudge_score(offender=1, victim=0, now=5)
    assert score > 0.0


def test_when_grudge_recorded_at_earlier_turn_then_score_is_lower_at_later_now():
    """grudge_score decays as (now - turn) grows; value at now=10 < value at now=0."""
    state = _make_state(decay=0.9)
    state.record_attack(offender=2, victim=0, turn=0, base_severity=1.0)
    score_now0 = state.grudge_score(offender=2, victim=0, now=0)
    score_now10 = state.grudge_score(offender=2, victim=0, now=10)
    assert score_now10 < score_now0


def test_when_grudge_score_read_multiple_times_then_grudge_log_is_unchanged():
    """grudge_score is read-derived; reading it must not mutate the stored log."""
    state = _make_state()
    state.record_attack(offender=3, victim=1, turn=2, base_severity=0.8)
    length_before = len(state.grudges)
    state.grudge_score(offender=3, victim=1, now=2)
    state.grudge_score(offender=3, victim=1, now=8)
    assert len(state.grudges) == length_before


def test_when_no_attacks_recorded_then_grudge_score_is_zero():
    state = _make_state()
    assert state.grudge_score(offender=0, victim=1, now=10) == 0.0


def test_when_two_attacks_recorded_then_grudge_log_has_two_entries():
    """record_attack is append-only: each call grows the log by one."""
    state = _make_state()
    state.record_attack(offender=1, victim=0, turn=1, base_severity=1.0)
    state.record_attack(offender=1, victim=0, turn=3, base_severity=1.0)
    assert len(state.grudges) == 2


def test_when_grudge_score_evaluated_at_same_turn_as_attack_no_decay_applied():
    """At now==turn the decay exponent is 0, so score equals base_severity."""
    state = _make_state(decay=0.5)
    state.record_attack(offender=4, victim=2, turn=7, base_severity=0.6)
    score = state.grudge_score(offender=4, victim=2, now=7)
    assert abs(score - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# Criterion: leader() names a leader only when there is one;
#            trust() is monotonic in grudge score
# ---------------------------------------------------------------------------


def test_when_players_tied_in_territories_then_there_is_no_leader():
    """A tie has no leader.

    This used to break ties by lowest player id "for determinism". Determinism was
    never the problem — the problem is that the result is announced to the models as
    ``Leader (most territories): Player N`` and they act on it. Every game starts with
    every player on an equal share, so the tie-break crowned seat 0 in every single
    game and the table duly ganged up on it. Returning None is just as deterministic
    and does not invent a fact.

    (Resolving a *drawn game* still needs a winner; that is ``_territory_leader`` in
    training/tournament.py, which keeps its tie-break.)
    """
    # Players 0 and 1 each own 3 territories out of 6.
    territory_owner = [0, 1, 0, 1, 0, 1]
    state = _make_state()
    assert state.leader(territory_owner) is None


def test_when_one_player_dominates_then_leader_returns_dominant_player():
    territory_owner = [2, 2, 2, 2, 0, 1]  # player 2 owns 4
    state = _make_state()
    assert state.leader(territory_owner) == 2


def test_when_single_player_owns_all_then_leader_is_that_player():
    territory_owner = [5] * 10
    state = _make_state()
    assert state.leader(territory_owner) == 5


def test_when_three_way_tie_then_there_is_still_no_leader():
    """Any tie at the top means nobody is ahead — not even a three-way one."""
    territory_owner = [0, 1, 2, 0, 1, 2]  # 2 territories each
    state = _make_state()
    assert state.leader(territory_owner) is None


def test_when_no_attack_history_then_trust_is_nonnegative():
    """With no grudges, trust should be at a non-negative neutral value."""
    state = _make_state()
    t = state.trust(a=0, b=1, now=0)
    assert t >= 0.0


def test_when_attack_recorded_then_trust_is_lower_than_baseline():
    """trust() is monotonically reduced by a grudge."""
    baseline_state = _make_state()
    baseline = baseline_state.trust(a=1, b=0, now=5)

    grudge_state = _make_state()
    grudge_state.record_attack(offender=1, victim=0, turn=3, base_severity=1.0)
    with_grudge = grudge_state.trust(a=1, b=0, now=5)

    assert with_grudge <= baseline


def test_when_two_attacks_recorded_then_trust_is_lower_or_equal_to_one():
    """Accumulating more attacks on the same victim never raises trust."""
    one_attack = _make_state()
    one_attack.record_attack(offender=2, victim=0, turn=0, base_severity=1.0)
    trust_one = one_attack.trust(a=2, b=0, now=0)

    two_attacks = _make_state()
    two_attacks.record_attack(offender=2, victim=0, turn=0, base_severity=1.0)
    two_attacks.record_attack(offender=2, victim=0, turn=0, base_severity=1.0)
    trust_two = two_attacks.trust(a=2, b=0, now=0)

    assert trust_two <= trust_one


# ---------------------------------------------------------------------------
# Criterion: Deterministic — identical inputs ⇒ identical state
#            (no set/dict iteration-order leakage)
# ---------------------------------------------------------------------------


def test_when_same_operations_applied_then_grudge_scores_are_identical():
    """Determinism: replaying the same record_attack sequence yields equal scores."""

    def _build() -> DiplomacyState:
        s = DiplomacyState(decay=0.85)
        s.record_attack(offender=4, victim=1, turn=2, base_severity=0.7)
        s.record_attack(offender=3, victim=2, turn=5, base_severity=1.2)
        return s

    s1, s2 = _build(), _build()
    assert s1.grudge_score(4, 1, now=10) == s2.grudge_score(4, 1, now=10)
    assert s1.grudge_score(3, 2, now=10) == s2.grudge_score(3, 2, now=10)


def test_when_alliances_formed_in_different_insertion_order_then_are_allied_is_same():
    """Order of form_alliance calls must not affect the resulting allied state."""
    s1 = _make_state()
    s1.form_alliance(0, 1)
    s1.form_alliance(2, 3)
    s1.form_alliance(4, 5)

    s2 = _make_state()
    s2.form_alliance(4, 5)
    s2.form_alliance(0, 1)
    s2.form_alliance(2, 3)

    for a, b in [(0, 1), (2, 3), (4, 5)]:
        assert s1.are_allied(a, b) == s2.are_allied(a, b)
        assert s1.are_allied(b, a) == s2.are_allied(b, a)


def test_when_same_alliance_sequence_applied_then_trust_scores_are_identical():
    """Determinism: identical alliance + grudge history → identical trust values."""

    def _build() -> DiplomacyState:
        s = DiplomacyState(decay=0.9)
        s.form_alliance(0, 1)
        s.record_attack(offender=2, victim=0, turn=3, base_severity=0.5)
        return s

    s1, s2 = _build(), _build()
    assert s1.trust(a=2, b=0, now=6) == s2.trust(a=2, b=0, now=6)
    assert s1.trust(a=0, b=1, now=6) == s2.trust(a=0, b=1, now=6)
