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

from training.tournament import (
    LedgerRecord,
    _territory_leader,
    load_tournament_config,
)

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
