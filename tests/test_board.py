"""Tests for src/env_core/board.py — Issue #48.

Board setup: territory distribution, starting armies, reinforcements.

Attribute names match the actual ``GameState`` dataclass in src/env_core/state.py:
  state.territory_owner          – int32 array, length 42
  state.armies                   – int32 array, length 42
  state.deck                     – list[int], fresh = list(range(44))
  state.rng                      – np.random.Generator
  state.current_player           – int
  state.reinforcements_remaining – int (scalar, current player only)
  state.phase                    – int (Phase.TRADE or Phase.REINFORCE)
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.env_core.board import (
    STARTING_ARMIES,
    Phase,
    compute_reinforcements,
    controls_continent,
    distribute_territories,
    new_state,
    place_initial_armies,
    set_initial_phase,
)

N_TERRITORIES = 42
N_DECK = 44
VALID_N_PLAYERS = [2, 3, 4, 5, 6]

CONTINENTS = [
    "North America",
    "South America",
    "Europe",
    "Africa",
    "Asia",
    "Australia",
]

# ---------------------------------------------------------------------------
# new_state
# ---------------------------------------------------------------------------


def test_when_new_state_called_then_territory_owner_has_42_entries():
    state = new_state(3, seed=42)
    assert len(state.territory_owner) == N_TERRITORIES


def test_when_new_state_called_then_territory_owner_is_zeroed():
    state = new_state(3, seed=42)
    assert all(int(o) == 0 for o in state.territory_owner)


def test_when_new_state_called_then_armies_has_42_entries():
    state = new_state(3, seed=42)
    assert len(state.armies) == N_TERRITORIES


def test_when_new_state_called_then_armies_are_zeroed():
    state = new_state(3, seed=42)
    assert all(int(a) == 0 for a in state.armies)


def test_when_new_state_called_then_deck_is_list_range_44():
    state = new_state(3, seed=42)
    assert state.deck == list(range(N_DECK))


def test_when_new_state_called_with_same_seed_then_rng_produces_identical_first_draw():
    state1 = new_state(3, seed=99)
    state2 = new_state(3, seed=99)
    assert state1.rng.integers(0, 10_000) == state2.rng.integers(0, 10_000)


def test_when_new_state_called_with_different_seeds_then_rng_diverges():
    state1 = new_state(3, seed=1)
    state2 = new_state(3, seed=2)
    draws_1 = [state1.rng.integers(0, 100_000) for _ in range(10)]
    draws_2 = [state2.rng.integers(0, 100_000) for _ in range(10)]
    assert draws_1 != draws_2


@given(
    n_players=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_when_new_state_called_with_any_valid_args_then_deck_is_always_range_44(n_players, seed):
    """Invariant: the deck is always list(range(44)) regardless of n_players or seed."""
    state = new_state(n_players, seed=seed)
    assert state.deck == list(range(N_DECK))


# ---------------------------------------------------------------------------
# distribute_territories
# ---------------------------------------------------------------------------


def test_when_distribute_territories_called_then_all_owners_are_valid_player_indices():
    state = new_state(3, seed=42)
    distribute_territories(state)
    assert all(0 <= int(o) < 3 for o in state.territory_owner)


def test_when_distribute_territories_called_then_every_territory_has_at_least_1_army():
    state = new_state(3, seed=42)
    distribute_territories(state)
    assert all(int(a) >= 1 for a in state.armies)


def test_when_distribute_territories_called_then_all_42_territory_ids_are_covered():
    state = new_state(4, seed=7)
    distribute_territories(state)
    assert len(state.territory_owner) == N_TERRITORIES
    assert all(0 <= int(o) < 4 for o in state.territory_owner)


def test_when_distribute_territories_called_then_deck_is_shuffled():
    """Issue comment: distribute_territories must shuffle the deck via state.rng."""
    state = new_state(3, seed=1)
    distribute_territories(state)
    assert state.deck != list(range(N_DECK)), (
        "deck must be shuffled during setup; unshuffled deck breaks per-seed card diversity"
    )


def test_when_distribute_territories_called_twice_with_same_seed_then_deck_orders_match():
    """Deck shuffle is reproducible: same seed → same shuffled deck."""
    s1 = new_state(3, seed=42)
    distribute_territories(s1)
    s2 = new_state(3, seed=42)
    distribute_territories(s2)
    assert s1.deck == s2.deck


@given(
    n_players=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50)
def test_when_distribute_territories_called_then_army_and_owner_invariants_hold(n_players, seed):
    """Property: every territory has owner in [0, n_players) and ≥ 1 army."""
    state = new_state(n_players, seed=seed)
    distribute_territories(state)
    assert all(int(a) >= 1 for a in state.armies)
    assert all(0 <= int(o) < n_players for o in state.territory_owner)


# ---------------------------------------------------------------------------
# place_initial_armies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_players", VALID_N_PLAYERS)
def test_when_place_initial_armies_called_then_each_player_totals_starting_armies(
    n_players,
):
    """Criterion: total armies per player == STARTING_ARMIES[n_players]."""
    state = new_state(n_players, seed=42)
    distribute_territories(state)
    place_initial_armies(state)
    expected = STARTING_ARMIES[n_players]
    for player in range(n_players):
        owned = [i for i, o in enumerate(state.territory_owner) if int(o) == player]
        total = sum(int(state.armies[t]) for t in owned)
        assert total == expected, (
            f"n_players={n_players} player={player}: got {total} armies, expected {expected}"
        )


def test_when_place_initial_armies_called_then_no_territory_is_left_empty():
    state = new_state(3, seed=42)
    distribute_territories(state)
    place_initial_armies(state)
    assert all(int(a) >= 1 for a in state.armies)


# ---------------------------------------------------------------------------
# compute_reinforcements
# ---------------------------------------------------------------------------


def test_when_player_owns_all_territories_then_reinforcements_equals_full_formula():
    """max(3, 42//3)=14 plus all-continent bonuses (NA+5 SA+2 EU+5 AF+3 AS+7 AU+2=24) = 38."""
    state = new_state(2, seed=0)
    state.territory_owner[:] = 0
    state.armies[:] = 1
    compute_reinforcements(state)
    continent_bonus = 5 + 2 + 5 + 3 + 7 + 2  # 24
    expected = max(3, 42 // 3) + continent_bonus  # 14 + 24 = 38
    assert state.reinforcements_remaining == expected


def test_when_player_owns_fewer_than_9_territories_then_minimum_reinforcements_is_3():
    """max(3, owned // 3) enforces a floor of 3 for owned < 9."""
    state = new_state(2, seed=0)
    state.territory_owner[:] = 1
    state.territory_owner[:6] = 0  # player 0 owns exactly 6 territories
    state.armies[:] = 1
    compute_reinforcements(state)
    assert state.reinforcements_remaining >= 3


def test_when_player_owns_12_territories_then_reinforcements_is_at_least_4():
    """max(3, 12 // 3) = 4; continent bonus may add more."""
    state = new_state(2, seed=0)
    state.territory_owner[:] = 1
    state.territory_owner[:12] = 0
    state.armies[:] = 1
    compute_reinforcements(state)
    assert state.reinforcements_remaining >= 4


def test_when_player_owns_zero_territories_then_reinforcements_is_3():
    """max(3, 0 // 3) + 0 = 3."""
    state = new_state(2, seed=0)
    state.territory_owner[:] = 1
    state.armies[:] = 1
    compute_reinforcements(state)
    assert state.reinforcements_remaining == 3


# ---------------------------------------------------------------------------
# controls_continent
# ---------------------------------------------------------------------------


def test_when_player_owns_all_territories_then_controls_every_continent():
    state = new_state(2, seed=0)
    state.territory_owner[:] = 0
    state.armies[:] = 1
    for continent in CONTINENTS:
        assert controls_continent(state, 0, continent) is True, (
            f"Player 0 should control {continent} when owning all territories"
        )


def test_when_player_owns_no_territories_then_controls_no_continent():
    state = new_state(2, seed=0)
    state.territory_owner[:] = 1
    state.armies[:] = 1
    for continent in CONTINENTS:
        assert controls_continent(state, 0, continent) is False, (
            f"Player 0 should not control {continent} when owning no territories"
        )


def test_when_ownership_is_split_then_no_continent_is_controlled_by_both_players():
    """Invariant: a continent cannot be controlled by two distinct players at once."""
    state = new_state(2, seed=0)
    distribute_territories(state)
    state.armies[:] = 1
    for continent in CONTINENTS:
        p0 = controls_continent(state, 0, continent)
        p1 = controls_continent(state, 1, continent)
        assert not (p0 and p1), f"Both players 0 and 1 cannot simultaneously control {continent}"


@given(
    owners_list=st.lists(
        st.integers(min_value=0, max_value=1),
        min_size=N_TERRITORIES,
        max_size=N_TERRITORIES,
    )
)
def test_when_any_2player_ownership_pattern_then_no_continent_is_doubly_controlled(
    owners_list,
):
    """Property: mutual exclusivity of continent control for any ownership assignment."""
    state = new_state(2, seed=0)
    state.territory_owner = np.array(owners_list, dtype=np.int32)
    state.armies = np.ones(N_TERRITORIES, dtype=np.int32)
    for continent in CONTINENTS:
        assert not (
            controls_continent(state, 0, continent) and controls_continent(state, 1, continent)
        ), f"Both players cannot control {continent} simultaneously"


# ---------------------------------------------------------------------------
# set_initial_phase
# ---------------------------------------------------------------------------


def test_when_has_tradeable_is_always_true_then_phase_is_set_to_trade():
    state = new_state(3, seed=42)
    set_initial_phase(state, lambda s, p: True)
    assert state.phase == Phase.TRADE


def test_when_has_tradeable_is_always_false_then_phase_is_set_to_reinforce():
    state = new_state(3, seed=42)
    set_initial_phase(state, lambda s, p: False)
    assert state.phase == Phase.REINFORCE


def test_when_has_tradeable_is_true_only_for_current_player_then_phase_is_trade():
    """set_initial_phase calls has_tradeable(state, current_player)."""
    state = new_state(3, seed=42)
    current = state.current_player
    set_initial_phase(state, lambda s, p: p == current)
    assert state.phase == Phase.TRADE


def test_when_has_tradeable_excludes_current_player_then_phase_is_reinforce():
    state = new_state(3, seed=42)
    current = state.current_player
    set_initial_phase(state, lambda s, p: p != current)
    assert state.phase == Phase.REINFORCE


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_when_two_new_states_share_seed_then_distribute_produces_identical_owners():
    """Criterion: Two new_state(3, 42) + distribute produce identical assignments."""
    s1 = new_state(3, seed=42)
    distribute_territories(s1)
    s2 = new_state(3, seed=42)
    distribute_territories(s2)
    np.testing.assert_array_equal(s1.territory_owner, s2.territory_owner)


def test_when_two_new_states_share_seed_then_distribute_produces_identical_armies():
    s1 = new_state(3, seed=42)
    distribute_territories(s1)
    s2 = new_state(3, seed=42)
    distribute_territories(s2)
    np.testing.assert_array_equal(s1.armies, s2.armies)


def test_when_two_new_states_share_seed_then_distribute_produces_identical_deck_order():
    """Issue comment: deck shuffle must also be reproducible for the same seed."""
    s1 = new_state(3, seed=42)
    distribute_territories(s1)
    s2 = new_state(3, seed=42)
    distribute_territories(s2)
    assert s1.deck == s2.deck


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50)
def test_when_any_seed_is_used_then_repeated_distribute_is_always_identical(seed):
    """Property: determinism holds for all seed values."""
    s1 = new_state(3, seed=seed)
    distribute_territories(s1)
    s2 = new_state(3, seed=seed)
    distribute_territories(s2)
    np.testing.assert_array_equal(s1.territory_owner, s2.territory_owner)
    np.testing.assert_array_equal(s1.armies, s2.armies)
    assert s1.deck == s2.deck
