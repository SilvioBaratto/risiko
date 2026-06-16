"""Source-blind tests for env_core/observation.py — Issue #52.

Authored from acceptance criteria only; no implementation source was read.
All tests are expected to FAIL until the implementation is written (Red phase).
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.env_core.observation import CONTINENTS, build_info, build_obs

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


class MockState:
    """Minimal state stub whose attributes mirror the spec contract."""

    def __init__(self, n_players: int = 6, current_player: int = 0):  # noqa: D107
        self.n_players = n_players
        self.current_player = current_player
        # One integer per territory: player index who owns it (0 … n_players-1)
        self.territory_owner = np.zeros(42, dtype=np.int32)
        # Army count per territory (≥1 invariant enforced by the env, not obs)
        self.armies = np.ones(42, dtype=np.int32)
        # Per-player hand: list of symbol integers (0=Infantry,1=Cavalry,2=Artillery,3=Wildcard)
        self.cards: list[list[int]] = [[] for _ in range(n_players)]
        # Per-player elimination status (0 = alive, 1 = eliminated)
        self.eliminated = np.zeros(n_players, dtype=np.int32)
        # Turn phase (0=TRADE … 4=FORTIFY)
        self.phase = 0
        self.reinforcements_remaining = 3
        self.trade_count = 0
        self.turn_capture = 0


def make_state(**kwargs) -> MockState:
    return MockState(**kwargs)


# ---------------------------------------------------------------------------
# build_obs — structural contract
# ---------------------------------------------------------------------------


def test_when_build_obs_called_then_result_is_a_dict():
    obs = build_obs(make_state())
    assert isinstance(obs, dict)


def test_when_build_obs_called_then_dict_contains_exactly_11_keys():
    obs = build_obs(make_state())
    assert len(obs) == 11


def test_when_build_obs_called_then_territory_owner_key_is_present():
    obs = build_obs(make_state())
    assert "territory_owner" in obs


def test_when_build_obs_called_then_territory_owner_is_shape_42_int32():
    obs = build_obs(make_state())
    assert obs["territory_owner"].shape == (42,)
    assert obs["territory_owner"].dtype == np.int32


def test_when_build_obs_called_then_armies_is_shape_42_int32():
    obs = build_obs(make_state())
    assert obs["armies"].shape == (42,)
    assert obs["armies"].dtype == np.int32


def test_when_build_obs_called_then_cards_is_shape_5x4_int32():
    obs = build_obs(make_state())
    assert obs["cards"].shape == (5, 4)
    assert obs["cards"].dtype == np.int32


def test_when_cards_have_binary_values_then_obs_cards_values_are_in_zero_and_one():
    """Values in the cards array must be constrained to {0, 1}."""
    state = make_state()
    # Give the current player one card of each symbol type (symbols stored as ints 0-3)
    state.cards[state.current_player] = [0, 1, 2, 3, 0]
    obs = build_obs(state)
    unique_values = set(obs["cards"].flatten().tolist())
    assert unique_values.issubset({0, 1})


def test_when_build_obs_called_then_continent_control_is_shape_6_int32():
    obs = build_obs(make_state())
    assert obs["continent_control"].shape == (6,)
    assert obs["continent_control"].dtype == np.int32


def test_when_build_obs_called_then_eliminated_key_is_present():
    obs = build_obs(make_state())
    assert "eliminated" in obs


def test_when_build_obs_called_then_eliminated_is_shape_6_int32():
    obs = build_obs(make_state())
    assert obs["eliminated"].shape == (6,)
    assert obs["eliminated"].dtype == np.int32


# ---------------------------------------------------------------------------
# continent_control[5] — Australia (index 5 in list(CONTINENTS.values()))
# ---------------------------------------------------------------------------


def test_when_current_player_owns_all_australia_then_continent_control_index_5_is_1():
    """CONTINENTS is ordered [NA, SA, EU, AF, AS, AU]; index 5 is Australia.

    When the current player (0) owns every territory in list(CONTINENTS.values())[5],
    continent_control[5] must equal 1.
    """
    australia_territory_indices = list(CONTINENTS.values())[5]
    state = make_state(current_player=0)
    state.territory_owner[:] = 1  # player 1 starts with everything
    for t in australia_territory_indices:
        state.territory_owner[t] = 0  # current player owns Australia
    obs = build_obs(state)
    assert obs["continent_control"][5] == 1


def test_when_current_player_owns_only_part_of_australia_then_continent_control_index_5_is_0():
    """Partial ownership must not set the continent flag."""
    australia_territory_indices = list(CONTINENTS.values())[5]
    state = make_state(current_player=0)
    state.territory_owner[:] = 1
    # Give current player all but the last territory in Australia
    for t in australia_territory_indices[:-1]:
        state.territory_owner[t] = 0
    obs = build_obs(state)
    assert obs["continent_control"][5] == 0


def test_when_current_player_owns_no_australia_territory_then_continent_control_index_5_is_0():
    state = make_state(current_player=0)
    state.territory_owner[:] = 1  # all belong to another player
    obs = build_obs(state)
    assert obs["continent_control"][5] == 0


# ---------------------------------------------------------------------------
# eliminated — padding invariant
# ---------------------------------------------------------------------------


def test_when_2_player_game_then_eliminated_slots_2_to_5_are_padded_with_1():
    """Slots ≥ n_players must be 1 (treated as eliminated).

    The fixed-length array remains meaningful when fewer than 6 players are active.
    """
    obs = build_obs(make_state(n_players=2))
    assert np.all(obs["eliminated"][2:] == 1)


def test_when_3_player_game_then_eliminated_slots_3_to_5_are_padded_with_1():
    obs = build_obs(make_state(n_players=3))
    assert np.all(obs["eliminated"][3:] == 1)


def test_when_4_player_game_then_eliminated_slots_4_and_5_are_padded_with_1():
    obs = build_obs(make_state(n_players=4))
    assert np.all(obs["eliminated"][4:] == 1)


def test_when_6_player_game_then_no_padding_is_required_and_length_is_6():
    """With n_players == 6, padding rule is vacuously satisfied; length must still be 6."""
    state = make_state(n_players=6)
    state.eliminated = np.zeros(6, dtype=np.int32)
    obs = build_obs(state)
    assert obs["eliminated"].shape == (6,)


# ---------------------------------------------------------------------------
# Copy semantics
# ---------------------------------------------------------------------------


def test_when_state_armies_mutated_after_build_obs_then_obs_armies_is_unchanged():
    """build_obs must copy the armies array.

    Subsequent mutation of state.armies must not affect the observation already returned.
    """
    state = make_state()
    state.armies = np.full(42, 5, dtype=np.int32)
    obs = build_obs(state)
    state.armies[0] = 9999
    assert obs["armies"][0] == 5, (
        "obs['armies'][0] changed after mutating state.armies — "
        "build_obs must return a copy, not a view"
    )


def test_when_state_territory_owner_mutated_after_build_obs_then_obs_is_unchanged():
    """Copy semantics extend to territory_owner as well."""
    state = make_state()
    state.territory_owner[0] = 2
    obs = build_obs(state)
    original = int(obs["territory_owner"][0])
    state.territory_owner[0] = 5
    assert obs["territory_owner"][0] == original


# ---------------------------------------------------------------------------
# build_info
# ---------------------------------------------------------------------------


def test_when_build_info_called_then_result_is_a_dict():
    assert isinstance(build_info(make_state(), []), dict)


def test_when_build_info_called_then_result_contains_legal_actions_key():
    assert "legal_actions" in build_info(make_state(), [1, 2])


def test_when_build_info_called_with_non_empty_list_then_legal_actions_are_preserved():
    legal_actions = [10, 20, 30]
    result = build_info(make_state(), legal_actions)
    assert result == {"legal_actions": legal_actions}


def test_when_build_info_called_with_empty_list_then_legal_actions_is_empty():
    result = build_info(make_state(), [])
    assert result == {"legal_actions": []}


def test_when_build_info_called_then_result_has_exactly_one_key():
    """build_info returns only {"legal_actions": ...} — no extra keys."""
    result = build_info(make_state(), [0])
    assert set(result.keys()) == {"legal_actions"}


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


@given(st.integers(min_value=2, max_value=6))
def test_when_n_players_varies_then_build_obs_always_returns_11_keys(n_players):
    """Invariant: the dict returned by build_obs always has exactly 11 keys."""
    obs = build_obs(make_state(n_players=n_players))
    assert len(obs) == 11


@given(st.integers(min_value=2, max_value=6))
def test_when_n_players_varies_then_eliminated_is_always_length_6(n_players):
    """Invariant: eliminated is always (6,) regardless of the player count."""
    obs = build_obs(make_state(n_players=n_players))
    assert obs["eliminated"].shape == (6,)


@given(st.integers(min_value=2, max_value=6))
def test_when_n_players_is_n_then_slots_from_n_onwards_are_always_padded_with_1(n_players):
    """Invariant: every slot ≥ n_players must be 1 for all valid game sizes."""
    obs = build_obs(make_state(n_players=n_players))
    assert np.all(obs["eliminated"][n_players:] == 1)


@given(armies=st.lists(st.integers(min_value=1, max_value=50), min_size=42, max_size=42))
@settings(max_examples=50)
def test_when_armies_mutated_after_build_obs_then_obs_is_always_unchanged(armies):
    """Copy-semantics invariant: for any valid armies configuration.

    Mutating state.armies after the call must never change the returned observation.
    """
    state = make_state()
    state.armies = np.array(armies, dtype=np.int32)
    original_first = int(state.armies[0])
    obs = build_obs(state)
    state.armies[0] = original_first + 1000
    assert obs["armies"][0] == original_first


@given(legal_actions=st.lists(st.integers(min_value=0, max_value=999), min_size=0, max_size=100))
def test_when_any_legal_actions_list_is_passed_then_build_info_round_trips_it(
    legal_actions,
):
    """Round-trip invariant: build_info preserves legal_actions without modification."""
    state = make_state()
    result = build_info(state, legal_actions)
    assert result["legal_actions"] == legal_actions


@given(hand=st.lists(st.integers(min_value=0, max_value=3), min_size=0, max_size=5))
@settings(max_examples=50)
def test_when_any_valid_hand_is_set_then_obs_cards_values_are_always_binary(hand):
    """Invariant: for any hand of valid symbol integers (0-3), output cards are binary."""
    state = make_state()
    state.cards[0] = hand
    obs = build_obs(state)
    unique = set(obs["cards"].flatten().tolist())
    assert unique.issubset({0, 1})
