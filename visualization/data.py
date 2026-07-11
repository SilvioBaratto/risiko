"""Loaders for the tournament results consumed by the charts.

Single Responsibility: read ``results/tournament/<run>/`` off disk and hand back
tidy structures. No plotting, no styling, no network.

Two traps live in the raw ledger and are absorbed here so no caller repeats them:

* ``alliances`` / ``betrayals`` are keyed by **string** seat ids and are **sparse** —
  a seat with zero is simply absent. Reading them naively silently drops zeros.
* ``n_turns`` counts **env steps** (~1000-2200), while ``card_trade_turns`` are on the
  **player-turn** scale (0-200). They must never share an axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

__all__ = [
    "DEFAULT_RUN",
    "RESULTS_DIR",
    "load_leaderboard",
    "load_games",
    "load_seat_stats",
    "cumulative_win_rate",
    "strategy_matrix",
]

RESULTS_DIR: Path = Path(__file__).resolve().parents[1] / "results" / "tournament"
DEFAULT_RUN: str = "tourney300"  # the 100-game run


def _run_dir(run: str) -> Path:
    """Return the directory for *run*, raising if it is not on disk."""
    path = RESULTS_DIR / run
    if not path.is_dir():
        raise FileNotFoundError(f"tournament run not found: {path}")
    return path


def load_leaderboard(run: str = DEFAULT_RUN) -> dict:
    """Load the aggregated leaderboard for a run.

    Args:
        run: Run directory name under ``results/tournament/``.

    Returns:
        The parsed ``leaderboard.json``: ``run``, ``n_games``, ``n_players`` plus the
        ``strategies``, ``models`` and ``matrix`` lists.

    Raises:
        FileNotFoundError: If the run or its leaderboard is missing.
    """
    path = _run_dir(run) / "leaderboard.json"
    if not path.is_file():
        raise FileNotFoundError(f"leaderboard not found: {path}")
    return json.loads(path.read_text())


def load_games(run: str = DEFAULT_RUN) -> pd.DataFrame:
    """Load one row per finished game, sorted by ``game_index``.

    Args:
        run: Run directory name under ``results/tournament/``.

    Returns:
        A DataFrame with ``game_index``, ``seed``, ``winner``, ``winner_strategy``,
        ``winner_model``, ``n_turns`` (env steps), ``n_card_trades``, ``resolution``
        and ``llm_call_count``.

    Raises:
        FileNotFoundError: If the ledger is missing.
    """
    records = _read_ledger(run)
    rows = [
        {
            "game_index": r["game_index"],
            "seed": r["seed"],
            "winner": r["winner"],
            "winner_strategy": r["winner_strategy"],
            "winner_model": r["winner_model"],
            "n_turns": r["n_turns"],
            "n_card_trades": len(r.get("card_trade_turns") or []),
            "resolution": r["resolution"],
            "llm_call_count": r["llm_call_count"],
        }
        for r in records
    ]
    return pd.DataFrame(rows).sort_values("game_index").reset_index(drop=True)


def load_seat_stats(run: str = DEFAULT_RUN) -> pd.DataFrame:
    """Load a tidy row per (game, seat), joining strategy, model and social counts.

    This is where the sparse string-keyed ``alliances`` / ``betrayals`` dicts are
    expanded to a dense 0 for every seat, and where ``final_ranking`` becomes a
    1-based placement.

    Args:
        run: Run directory name under ``results/tournament/``.

    Returns:
        A DataFrame with ``game_index``, ``seat``, ``strategy``, ``model``,
        ``alliances``, ``betrayals``, ``placement`` (1 = best) and ``won``.
    """
    rows: list[dict] = []
    for record in _read_ledger(run):
        ranking = record.get("final_ranking") or []
        placement = {seat: i + 1 for i, seat in enumerate(ranking)}
        alliances = record.get("alliances") or {}
        betrayals = record.get("betrayals") or {}
        for seat_key, strategy in record["seat_strategies"].items():
            seat = int(seat_key)
            rows.append(
                {
                    "game_index": record["game_index"],
                    "seat": seat,
                    "strategy": strategy,
                    "model": record["seat_models"][seat_key],
                    # keys are strings and zeros are omitted — hence .get(str(seat), 0)
                    "alliances": int(alliances.get(seat_key, 0)),
                    "betrayals": int(betrayals.get(seat_key, 0)),
                    "placement": placement.get(seat),
                    "won": record["winner"] == seat,
                }
            )
    return pd.DataFrame(rows)


def cumulative_win_rate(games: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    """Return the running win rate of every strategy after each game.

    Every strategy is present in every game, so the denominator is simply the number
    of games played so far.

    Args:
        games: Output of :func:`load_games`.
        strategies: Strategy slugs to track.

    Returns:
        A DataFrame indexed by game number (1..N) with one column per strategy.
    """
    ordered = games.sort_values("game_index")
    wins = dict.fromkeys(strategies, 0)
    rows: list[dict] = []
    for played, winner in enumerate(ordered["winner_strategy"], start=1):
        if winner in wins:
            wins[winner] += 1
        rows.append({"games": played, **{s: wins[s] / played for s in strategies}})
    return pd.DataFrame(rows).set_index("games")


def strategy_matrix(
    leaderboard: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pivot the strategy x model cells into win-rate, wins and games matrices.

    Returned as three parallel frames rather than one frame with ``.attrs``, because
    pandas drops ``.attrs`` on ``reindex`` — the caller always reorders these.

    Args:
        leaderboard: Output of :func:`load_leaderboard`.

    Returns:
        ``(win_rate, wins, games)``, each with strategies as rows and models as columns.
    """
    cells = pd.DataFrame(leaderboard["matrix"])
    rates = cells.pivot(index="strategy", columns="model", values="win_rate")
    wins = cells.pivot(index="strategy", columns="model", values="wins")
    games = cells.pivot(index="strategy", columns="model", values="games")
    return rates, wins, games


def _read_ledger(run: str) -> list[dict]:
    """Parse ``games.jsonl`` into a list of records."""
    path = _run_dir(run) / "games.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"game ledger not found: {path}")
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]
