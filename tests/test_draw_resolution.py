"""Tests for tournament draw-resolution: territory-leader fallback at the turn cap.

6-player games rarely eliminate down to one survivor within max_turns, so a game
that hits the cap is scored by board leadership (most territories, tie-break by
armies) when draw_resolution="territory". True elimination wins are unaffected.
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import numpy as np

from src.agents.diplomacy import DiplomacyState
from training.tournament import (
    LedgerRecord,
    _final_ranking,
    _territory_leader,
    load_tournament_config,
)
from training.tournament_stats import placements

CONFIG_PATH = pathlib.Path("config/tournament.yaml")


def _state(owner: list[int], armies: list[int], eliminated: list[int], n: int) -> SimpleNamespace:
    return SimpleNamespace(
        territory_owner=np.array(owner),
        armies=np.array(armies),
        eliminated=np.array(eliminated),
        n_players=n,
    )


# ── _territory_leader ────────────────────────────────────────────────────────


def test_when_one_player_holds_most_territories_then_that_player_leads():
    st = _state([0, 1, 1, 1, 2], [9, 1, 1, 1, 1], [0, 0, 0], 3)
    assert _territory_leader(st) == 1


def test_when_territories_tied_then_armies_break_the_tie():
    st = _state([0, 1], [2, 5], [0, 0], 2)
    assert _territory_leader(st) == 1


def test_when_leader_is_eliminated_then_excluded_from_ranking():
    # p1 holds the most land but is eliminated → leader is p0 (1 terr, most armies).
    st = _state([1, 1, 1, 0, 2], [1, 1, 1, 9, 1], [0, 1, 0], 3)
    assert _territory_leader(st) == 0


def test_when_no_active_player_holds_territory_then_none():
    st = _state([0, 0], [3, 3], [1, 1], 2)
    assert _territory_leader(st) is None


def test_when_territories_and_armies_tie_then_lowest_index_wins():
    st = _state([0, 1], [3, 3], [0, 0], 2)
    assert _territory_leader(st) == 0


# ── _final_ranking ───────────────────────────────────────────────────────────


def test_when_no_eliminations_then_ranking_orders_survivors_by_territory():
    # p1 holds 3 terr, p0 1, p2 1 (p0 more armies than p2) → [1, 0, 2].
    st = _state([0, 1, 1, 1, 2], [9, 1, 1, 1, 1], [0, 0, 0], 3)
    assert _final_ranking(st, []) == [1, 0, 2]


def test_when_players_eliminated_then_they_rank_after_survivors_reverse_order():
    # p0 survives (all land); p1 then p2 eliminated → survivors first, then
    # eliminated in reverse elimination order (last out ranks higher): [0, 2, 1].
    st = _state([0, 0, 0, 0], [4, 4, 4, 4], [0, 1, 1], 3)
    assert _final_ranking(st, [1, 2]) == [0, 2, 1]


def test_when_ranking_first_then_matches_territory_leader():
    st = _state([0, 1, 1, 1, 2], [9, 1, 1, 1, 1], [0, 0, 0], 3)
    assert _final_ranking(st, [])[0] == _territory_leader(st)


# ── placements use final_ranking ─────────────────────────────────────────────


def test_when_record_has_final_ranking_then_placements_are_distinct():
    rec = {
        "seat_strategies": {"0": "a", "1": "b", "2": "c"},
        "elimination_order": [],
        "final_ranking": [2, 0, 1],
    }
    assert placements(rec) == {2: 1, 0: 2, 1: 3}


def test_when_no_final_ranking_then_falls_back_to_elimination_logic():
    rec = {
        "seat_strategies": {"0": "a", "1": "b"},
        "elimination_order": [1],
    }
    # p0 survives → 1; p1 eliminated (last out) → 2.
    assert placements(rec) == {0: 1, 1: 2}


# ── alliance formation counter ───────────────────────────────────────────────


def test_when_alliance_formed_then_both_members_counted_once():
    st = DiplomacyState()
    st.form_alliance(0, 3)
    st.form_alliance(0, 3)  # re-forming the same pair must not double-count
    assert st.alliance_formations[0] == 1
    assert st.alliance_formations[3] == 1


def test_when_player_forms_two_alliances_then_count_is_two():
    st = DiplomacyState()
    st.form_alliance(0, 1)
    st.form_alliance(0, 2)
    assert st.alliance_formations[0] == 2


# ── config field ─────────────────────────────────────────────────────────────


def test_when_config_loaded_then_draw_resolution_is_territory():
    cfg = load_tournament_config(CONFIG_PATH)
    assert cfg.draw_resolution == "territory"


# ── LedgerRecord.resolution ──────────────────────────────────────────────────


def _record(resolution: str = "draw") -> LedgerRecord:
    return LedgerRecord(
        game_index=0,
        seed=1,
        winner=2,
        winner_strategy="aggressive_blitz",
        winner_model="glm-5.2:cloud",
        n_turns=200,
        elimination_order=[],
        seat_strategies={},
        seat_models={},
        assignment={},
        card_trade_turns=[],
        llm_call_count=10,
        resolution=resolution,
    )


def test_when_resolution_set_then_round_trips_through_json():
    rec = _record(resolution="territory_cap")
    restored = LedgerRecord.from_json(json.loads(rec.to_json_line()))
    assert restored.resolution == "territory_cap"


def test_when_old_ledger_line_lacks_resolution_then_defaults_to_draw():
    """Back-compat: a record dict without `resolution` (pre-field ledger) loads."""
    data = {
        "game_index": 0,
        "seed": 1,
        "winner": None,
        "winner_strategy": None,
        "winner_model": None,
        "n_turns": 200,
        "elimination_order": [],
        "seat_strategies": {},
        "seat_models": {},
        "assignment": {},
        "card_trade_turns": [],
        "llm_call_count": 5,
    }
    assert LedgerRecord.from_json(data).resolution == "draw"
