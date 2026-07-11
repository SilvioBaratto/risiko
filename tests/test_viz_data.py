"""Tests for the tournament-results loaders.

The traps these cover are the ones the raw ledger actually sets: ``alliances`` and
``betrayals`` are keyed by *string* seat ids and omit zeros entirely, and ``winner`` is
a seat number rather than a strategy name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visualization import data as vdata

_STRATEGIES = [
    "diplomat_coalition",
    "card_cycle_hunter",
    "aggressive_blitz",
    "australia_lock",
    "south_america_lock",
    "turtle_defensive",
]


def _record(game_index: int, winner: int, *, betrayals: dict[str, int] | None = None) -> dict:
    """Build one ledger line shaped exactly like the tournament writes it."""
    return {
        "game_index": game_index,
        "seed": 42 + game_index,
        "winner": winner,
        "winner_strategy": _STRATEGIES[winner],
        "winner_model": "gemma4:cloud",
        "n_turns": 1200,
        "card_trade_turns": [3, 9],
        "resolution": "territory",
        "llm_call_count": 500,
        "seat_strategies": {str(i): s for i, s in enumerate(_STRATEGIES)},
        "seat_models": {str(i): "gemma4:cloud" for i in range(6)},
        "final_ranking": [winner, *[i for i in range(6) if i != winner]],
        "alliances": {"0": 3},
        # Sparse and string-keyed: seats 1-5 committed none and are simply absent.
        "betrayals": betrayals if betrayals is not None else {"2": 7},
    }


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Write a two-game run to disk and point the loaders at it."""
    root = tmp_path / "tournament"
    run = root / "fake"
    run.mkdir(parents=True)
    records = [_record(0, winner=0), _record(1, winner=2)]
    run.joinpath("games.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    run.joinpath("leaderboard.json").write_text(
        json.dumps(
            {
                "run": "fake",
                "n_games": 2,
                "n_players": 6,
                "strategies": [],
                "models": [],
                "matrix": [
                    {
                        "strategy": "diplomat_coalition",
                        "model": "a",
                        "games": 2,
                        "wins": 1,
                        "win_rate": 0.5,
                    },
                    {
                        "strategy": "diplomat_coalition",
                        "model": "b",
                        "games": 2,
                        "wins": 0,
                        "win_rate": 0.0,
                    },
                    {
                        "strategy": "aggressive_blitz",
                        "model": "a",
                        "games": 2,
                        "wins": 0,
                        "win_rate": 0.0,
                    },
                    {
                        "strategy": "aggressive_blitz",
                        "model": "b",
                        "games": 2,
                        "wins": 2,
                        "win_rate": 1.0,
                    },
                ],
            }
        )
    )
    monkeypatch.setattr(vdata, "RESULTS_DIR", root)
    return "fake"


def test_missing_run_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vdata, "RESULTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        vdata.load_leaderboard("nope")


def test_load_games_is_sorted_and_typed(run_dir: str) -> None:
    games = vdata.load_games(run_dir)
    assert list(games["game_index"]) == [0, 1]
    assert list(games["winner_strategy"]) == ["diplomat_coalition", "aggressive_blitz"]
    assert list(games["n_card_trades"]) == [2, 2]


def test_seat_stats_expands_sparse_string_keyed_counts(run_dir: str) -> None:
    seats = vdata.load_seat_stats(run_dir)
    assert len(seats) == 12  # 2 games x 6 seats

    game0 = seats[seats["game_index"] == 0].set_index("seat")
    # seat "2" is the only betrayer in the record; every other seat must read 0, not NaN.
    assert game0.loc[2, "betrayals"] == 7
    assert game0.loc[3, "betrayals"] == 0
    assert game0["betrayals"].sum() == 7
    assert game0.loc[0, "alliances"] == 3
    assert game0.loc[1, "alliances"] == 0


def test_seat_stats_derives_one_based_placement(run_dir: str) -> None:
    seats = vdata.load_seat_stats(run_dir)
    game1 = seats[seats["game_index"] == 1].set_index("seat")
    assert game1.loc[2, "placement"] == 1  # seat 2 won game 1
    assert bool(game1.loc[2, "won"]) is True
    assert set(game1["placement"]) == {1, 2, 3, 4, 5, 6}


def test_cumulative_win_rate_denominator_is_games_played(run_dir: str) -> None:
    games = vdata.load_games(run_dir)
    curves = vdata.cumulative_win_rate(games, _STRATEGIES)
    assert list(curves.index) == [1, 2]
    assert curves.loc[1, "diplomat_coalition"] == 1.0  # 1 win out of 1 game
    assert curves.loc[2, "diplomat_coalition"] == 0.5  # still 1 win, now out of 2
    assert curves.loc[2, "turtle_defensive"] == 0.0


def test_strategy_matrix_pivots_three_parallel_frames(run_dir: str) -> None:
    rates, wins, played = vdata.strategy_matrix(vdata.load_leaderboard(run_dir))
    assert rates.loc["diplomat_coalition", "a"] == 0.5
    assert wins.loc["aggressive_blitz", "b"] == 2
    assert played.loc["aggressive_blitz", "b"] == 2
    # The three frames must stay aligned — the plots index them together.
    assert list(rates.index) == list(wins.index) == list(played.index)
    assert list(rates.columns) == list(wins.columns) == list(played.columns)
