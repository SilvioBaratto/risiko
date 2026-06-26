"""Tournament analysis facade and strategy × model win-matrix builder.

Public API (facade):
  build_leaderboard(records, run, n_malformed=0) -> Leaderboard   # pure
  analyze_run(run_dir: Path) -> Leaderboard                        # I/O

Also re-exports build_matrix (kept for backward-compatibility with issue #126 tests).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from training.tournament_stats import (
    Leaderboard,
    MatrixCell,
    StrategyStat,
    aggregate_diplomacy,
    aggregate_models,
    aggregate_strategies,
    read_ledger,
    with_diplomacy,
)

__all__ = [
    "build_leaderboard",
    "analyze_run",
    "build_matrix",
    # backward-compat re-exports
    "MatrixCell",
    "StrategyStat",
    "aggregate_diplomacy",
    "with_diplomacy",
]


# ---------------------------------------------------------------------------
# Facade — pure builder
# ---------------------------------------------------------------------------


def _infer_n_players(records: list[dict]) -> int:
    """Infer player count from seat_strategies length; default 6 when empty."""
    if not records:
        return 6
    return max(len(r.get("seat_strategies", {})) for r in records) or 6


def build_leaderboard(
    records: list[dict],
    run: str,
    n_malformed: int = 0,
) -> Leaderboard:
    """Compose a full :class:`Leaderboard` from well-formed ledger records (pure).

    Args:
        records: Well-formed game records (output of ``read_ledger``).
        run: Run identifier (usually the run directory name).
        n_malformed: Count of lines skipped before this call.

    Returns:
        Immutable :class:`Leaderboard` with strategy/model stats and win matrix.
    """
    strategy_stats = with_diplomacy(aggregate_strategies(records), records)
    model_stats = aggregate_models(records)
    return Leaderboard(
        run=run,
        n_games=len(records),
        n_malformed=n_malformed,
        n_players=_infer_n_players(records),
        strategies=strategy_stats,
        models=model_stats,
        matrix=build_matrix(records),
    )


# ---------------------------------------------------------------------------
# Facade — I/O entry point
# ---------------------------------------------------------------------------


def analyze_run(run_dir: Path) -> Leaderboard:
    """Read a run's ledger, build the leaderboard, and write both artefacts.

    Args:
        run_dir: Directory containing ``games.jsonl``; also the output destination.

    Returns:
        The constructed :class:`Leaderboard` (also written to ``leaderboard.json``
        and ``report.md`` under *run_dir*).
    """
    from training.tournament_report import write_leaderboard_json, write_report_md

    records, n_malformed = read_ledger(run_dir / "games.jsonl")
    board = build_leaderboard(records, run=run_dir.name, n_malformed=n_malformed)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_leaderboard_json(board, run_dir / "leaderboard.json")
    write_report_md(board, run_dir / "report.md")
    return board


# ---------------------------------------------------------------------------
# Strategy × model win-matrix builder (issue #126, kept for compat)
# ---------------------------------------------------------------------------


def build_matrix(records: Iterable[dict]) -> tuple[MatrixCell, ...]:
    """Return one :class:`MatrixCell` per observed ``(strategy, model)`` pairing.

    Args:
        records: Iterable of well-formed ledger dicts.

    Returns:
        Sorted ``tuple[MatrixCell, ...]`` (strategy asc, model asc).
    """
    accum: dict[tuple[str, str], dict[str, int]] = {}
    for rec in records:
        winner_strategy = rec.get("winner_strategy")
        winner_model = rec.get("winner_model")
        for strategy, model in _extract_pairs(rec):
            cell = accum.setdefault((strategy, model), {"games": 0, "wins": 0})
            cell["games"] += 1
            if strategy == winner_strategy and model == winner_model:
                cell["wins"] += 1
    return tuple(
        _make_cell(strategy, model, data)
        for (strategy, model), data in sorted(accum.items())
        if data["games"] > 0
    )


# ---------------------------------------------------------------------------
# Private helpers for build_matrix
# ---------------------------------------------------------------------------


def _extract_pairs(rec: dict) -> list[tuple[str, str]]:
    assignment = rec.get("assignment")
    if assignment is not None:
        return _pairs_from_assignment(assignment)
    return _pairs_from_seats(rec)


def _pairs_from_assignment(assignment: object) -> list[tuple[str, str]]:
    if isinstance(assignment, list):
        return [
            (e["strategy"], e["model"])
            for e in assignment
            if isinstance(e, dict) and "strategy" in e and "model" in e
        ]
    if isinstance(assignment, dict):
        return list(assignment.items())
    return []


def _pairs_from_seats(rec: dict) -> list[tuple[str, str]]:
    strategies = rec.get("seat_strategies")
    models = rec.get("seat_models")
    if strategies is None or models is None:
        return []
    if isinstance(strategies, list) and isinstance(models, list):
        return list(zip(strategies, models, strict=False))
    if isinstance(strategies, dict) and isinstance(models, dict):
        return [(strategies[k], models[k]) for k in sorted(strategies.keys()) if k in models]
    return []


def _make_cell(strategy: str, model: str, data: dict[str, int]) -> MatrixCell:
    games, wins = data["games"], data["wins"]
    return MatrixCell(strategy=strategy, model=model, games=games, wins=wins, win_rate=wins / games)
