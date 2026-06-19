"""Source-blind example tests for DiplomacyState (issue #101).

Derived solely from the acceptance criteria:
  - `declare_war_on` on an ally breaks the alliance and records a grudge.

These tests fail until the criterion is genuinely met.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.diplomacy import DiplomacyState

# ---------------------------------------------------------------------------
# Helpers — build minimal test state without touching internals
# ---------------------------------------------------------------------------


def _allied_state(player_a: int, player_b: int) -> DiplomacyState:
    """Return a DiplomacyState where player_a and player_b are allied."""
    state = DiplomacyState()
    state.form_alliance(player_a, player_b)
    return state


# ---------------------------------------------------------------------------
# Example tests — one observable behaviour per test
# ---------------------------------------------------------------------------


class TestDeclareWarOnBreaksAlliance:
    """declare_war_on on an ally removes the alliance."""

    def test_when_declare_war_on_ally_then_alliance_is_broken(self) -> None:
        """After declare_war_on(1, 2), players 1 and 2 are no longer allied."""
        state = _allied_state(1, 2)
        assert state.are_allied(1, 2), "pre-condition: players must be allied before the call"

        state.declare_war_on(1, 2)

        assert not state.are_allied(1, 2)

    def test_when_declare_war_on_ally_then_alliance_is_symmetric_both_ways(self) -> None:
        """Alliance removal is symmetric: neither direction shows allied after declare_war_on."""
        state = _allied_state(1, 2)
        state.declare_war_on(1, 2)

        assert not state.are_allied(1, 2)
        assert not state.are_allied(2, 1)


class TestDeclareWarOnRecordsGrudge:
    """declare_war_on on an ally records a grudge against the attacker."""

    def test_when_declare_war_on_ally_then_grudge_is_recorded(self) -> None:
        """After declare_war_on(1, 2), player 2 holds a grudge against player 1."""
        state = _allied_state(1, 2)
        state.declare_war_on(1, 2)

        assert state.has_grudge(victim=2, aggressor=1)

    def test_when_declare_war_on_non_ally_then_grudge_is_still_recorded(self) -> None:
        """Attacking an enemy (not ally) also records a grudge against the attacker."""
        state = DiplomacyState()
        # no prior alliance
        state.declare_war_on(3, 4)

        assert state.has_grudge(victim=4, aggressor=3)

    def test_when_declare_war_on_ally_then_attacker_holds_no_grudge_against_victim(self) -> None:
        """The attacker does not auto-receive a grudge against the victim."""
        state = _allied_state(1, 2)
        state.declare_war_on(1, 2)

        assert not state.has_grudge(victim=1, aggressor=2)


# ---------------------------------------------------------------------------
# Property-based tests — invariants derived from criterion text
# ---------------------------------------------------------------------------


@settings(max_examples=60)
@given(
    player_a=st.integers(min_value=0, max_value=5),
    player_b=st.integers(min_value=0, max_value=5),
)
def test_when_declare_war_on_any_allied_pair_then_alliance_is_always_broken(
    player_a: int,
    player_b: int,
) -> None:
    """Invariant: for ALL pairs (a, b) where a != b and they are allied,
    declare_war_on(a, b) always removes the alliance.

    Derived from: 'declare_war_on on an ally breaks the alliance'.
    """
    if player_a == player_b:
        return  # self-declaration is out of domain

    state = _allied_state(player_a, player_b)
    state.declare_war_on(player_a, player_b)

    assert not state.are_allied(player_a, player_b)


@settings(max_examples=60)
@given(
    player_a=st.integers(min_value=0, max_value=5),
    player_b=st.integers(min_value=0, max_value=5),
)
def test_when_declare_war_on_any_allied_pair_then_grudge_is_always_recorded(
    player_a: int,
    player_b: int,
) -> None:
    """Invariant: for ALL allied pairs (a, b), declare_war_on(a, b) always records a grudge.

    Derived from: 'declare_war_on on an ally ... records a grudge'.
    """
    if player_a == player_b:
        return

    state = _allied_state(player_a, player_b)
    state.declare_war_on(player_a, player_b)

    assert state.has_grudge(victim=player_b, aggressor=player_a)
