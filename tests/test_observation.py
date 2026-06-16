"""Tests for src/env_core/observation.py (Issue #52)."""

from __future__ import annotations

import numpy as np
import pytest

from src.env_core.observation import CONTINENTS, build_info, build_obs
from src.env_core.state import PHASE_REINFORCE, GameState

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _make_state(n_players: int = 6, current_player: int = 0) -> GameState:
    return GameState(
        territory_owner=np.zeros(42, dtype=np.int32),
        armies=np.ones(42, dtype=np.int32),
        cards=[[] for _ in range(n_players)],
        eliminated=np.zeros(n_players, dtype=np.int32),
        n_players=n_players,
        current_player=current_player,
        phase=PHASE_REINFORCE,
        reinforcements_remaining=3,
        trade_count=0,
        turn_capture=0,
    )


# ---------------------------------------------------------------------------
# Key set
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "territory_owner",
    "armies",
    "phase",
    "current_player",
    "cards",
    "continent_control",
    "trade_count",
    "reinforcements_remaining",
    "turn_capture",
    "n_players",
    "eliminated",
}


def test_when_build_obs_called_then_result_is_dict():
    assert isinstance(build_obs(_make_state()), dict)


def test_when_build_obs_called_then_exactly_11_keys_returned():
    assert len(build_obs(_make_state())) == 11


def test_when_build_obs_called_then_all_11_keys_are_present():
    assert set(build_obs(_make_state()).keys()) == EXPECTED_KEYS


# ---------------------------------------------------------------------------
# Shapes and dtypes
# ---------------------------------------------------------------------------


def test_when_build_obs_called_then_territory_owner_is_shape_42_int32():
    obs = build_obs(_make_state())
    assert obs["territory_owner"].shape == (42,)
    assert obs["territory_owner"].dtype == np.int32


def test_when_build_obs_called_then_armies_is_shape_42_int32():
    obs = build_obs(_make_state())
    assert obs["armies"].shape == (42,)
    assert obs["armies"].dtype == np.int32


def test_when_build_obs_called_then_cards_is_shape_5x4_int32():
    obs = build_obs(_make_state())
    assert obs["cards"].shape == (5, 4)
    assert obs["cards"].dtype == np.int32


def test_when_build_obs_called_then_continent_control_is_shape_6_int32():
    obs = build_obs(_make_state())
    assert obs["continent_control"].shape == (6,)
    assert obs["continent_control"].dtype == np.int32


def test_when_build_obs_called_then_eliminated_is_shape_6_int32():
    obs = build_obs(_make_state())
    assert obs["eliminated"].shape == (6,)
    assert obs["eliminated"].dtype == np.int32


def test_when_player_holds_one_card_per_symbol_then_obs_cards_is_binary():
    state = _make_state()
    state.cards[0] = [0, 1, 2, 3, 0]  # infantry, cavalry, artillery, wildcard, infantry
    obs = build_obs(state)
    assert set(obs["cards"].flatten().tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# continent_control[5] — Australia
# ---------------------------------------------------------------------------


def test_when_current_player_owns_all_australia_then_continent_control_5_is_1():
    """Index 5 in list(CONTINENTS.values()) is Australia (NA, SA, EU, AF, AS, AU)."""
    australia = list(CONTINENTS.values())[5]
    state = _make_state(current_player=0)
    state.territory_owner[:] = 1
    for t in australia:
        state.territory_owner[t] = 0
    obs = build_obs(state)
    assert obs["continent_control"][5] == 1


def test_when_current_player_owns_partial_australia_then_continent_control_5_is_0():
    australia = list(CONTINENTS.values())[5]
    state = _make_state(current_player=0)
    state.territory_owner[:] = 1
    for t in australia[:-1]:
        state.territory_owner[t] = 0
    obs = build_obs(state)
    assert obs["continent_control"][5] == 0


def test_when_current_player_owns_no_australia_then_continent_control_5_is_0():
    state = _make_state(current_player=0)
    state.territory_owner[:] = 1
    obs = build_obs(state)
    assert obs["continent_control"][5] == 0


def test_when_continents_ordered_correctly_then_na_is_index_0_and_au_is_index_5():
    names = list(CONTINENTS.keys())
    assert names[0] == "North America"
    assert names[5] == "Australia"


# ---------------------------------------------------------------------------
# eliminated — padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_players", [2, 3, 4, 5])
def test_when_n_players_is_less_than_6_then_slots_from_n_onwards_are_padded_with_1(n_players):
    obs = build_obs(_make_state(n_players=n_players))
    assert np.all(obs["eliminated"][n_players:] == 1)


def test_when_6_player_game_then_eliminated_shape_is_still_6():
    state = _make_state(n_players=6)
    obs = build_obs(state)
    assert obs["eliminated"].shape == (6,)


def test_when_2_player_game_active_player_alive_then_active_slots_reflect_eliminated():
    state = _make_state(n_players=2)
    state.eliminated = np.array([0, 1], dtype=np.int32)
    obs = build_obs(state)
    assert obs["eliminated"][0] == 0  # alive
    assert obs["eliminated"][1] == 1  # eliminated
    assert np.all(obs["eliminated"][2:] == 1)  # padding


# ---------------------------------------------------------------------------
# Copy semantics
# ---------------------------------------------------------------------------


def test_when_armies_mutated_after_build_obs_then_obs_armies_unchanged():
    state = _make_state()
    state.armies[0] = 7
    obs = build_obs(state)
    state.armies[0] = 999
    assert obs["armies"][0] == 7


def test_when_territory_owner_mutated_after_build_obs_then_obs_unchanged():
    state = _make_state()
    state.territory_owner[0] = 3
    obs = build_obs(state)
    state.territory_owner[0] = 5
    assert obs["territory_owner"][0] == 3


# ---------------------------------------------------------------------------
# build_info
# ---------------------------------------------------------------------------


def test_when_build_info_called_then_returns_legal_actions_dict():
    result = build_info(_make_state(), [10, 20, 30])
    assert result == {"legal_actions": [10, 20, 30]}


def test_when_build_info_called_with_empty_list_then_returns_empty():
    assert build_info(_make_state(), []) == {"legal_actions": []}


def test_when_build_info_called_then_no_extra_keys():
    assert set(build_info(_make_state(), [0]).keys()) == {"legal_actions"}
