"""Tournament leaderboard serialisation: JSON dict and Markdown report.

Pure producers (to_json_dict, to_markdown) are wrapped by thin I/O writers
(write_leaderboard_json, write_report_md) so tests can assert on content
without touching the filesystem.

JSON schema top-level keys: run, n_games, n_malformed, n_players,
                             strategies[], models[], matrix[]
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from training.tournament_stats import Leaderboard, MatrixCell

__all__ = [
    "to_json_dict",
    "to_markdown",
    "write_leaderboard_json",
    "write_report_md",
]

_ROUND = 4  # decimal places for rates / CIs / placements in JSON


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def _round_dict(d: dict) -> dict:
    return {k: round(v, _ROUND) if isinstance(v, float) else v for k, v in d.items()}


def to_json_dict(board: Leaderboard) -> dict:
    """Return a JSON-serialisable dict for *board*.

    Rates, CIs, and placements are rounded to :data:`_ROUND` decimal places.
    Serialises via ``json.dumps(..., default=str)`` to guard against any
    non-native type that may leak from numpy or similar.
    """
    return {
        "run": board.run,
        "n_games": board.n_games,
        "n_malformed": board.n_malformed,
        "n_players": board.n_players,
        "strategies": [_round_dict(dataclasses.asdict(s)) for s in board.strategies],
        "models": [_round_dict(dataclasses.asdict(m)) for m in board.models],
        "matrix": [_round_dict(dataclasses.asdict(c)) for c in board.matrix],
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def to_markdown(board: Leaderboard) -> str:
    """Return a human-readable Markdown report for *board*."""
    return "\n".join(
        [
            f"# Tournament Report — {board.run}\n",
            _md_summary(board),
            _md_strategy_table(board),
            _md_model_table(board),
            _md_matrix_section(board),
            _md_survival_section(board),
            _md_diplomacy_section(board),
        ]
    )


def _md_summary(board: Leaderboard) -> str:
    return (
        f"**Games completed:** {board.n_games}  \n"
        f"**Malformed skipped:** {board.n_malformed}  \n"
        f"**Players per game:** {board.n_players}\n"
    )


def _md_strategy_table(board: Leaderboard) -> str:
    header = "## Strategy Leaderboard\n"
    cols = "| Rank | Strategy | Games | Wins | Win% | 95% CI | Mean Placement | Betrayals | Alliances |"  # noqa: E501
    sep = "|------|----------|-------|------|------|--------|----------------|-----------|-----------|"  # noqa: E501
    rows = [header, cols, sep]
    for rank, s in enumerate(board.strategies, 1):
        ci = f"[{s.ci_low:.4f}, {s.ci_high:.4f}]"
        rows.append(
            f"| {rank} | {s.strategy} | {s.games} | {s.wins} "
            f"| {s.win_rate * 100:.2f}% | {ci} | {s.mean_placement:.2f} "
            f"| {s.betrayals} | {s.alliances} |"
        )
    return "\n".join(rows) + "\n"


def _md_model_table(board: Leaderboard) -> str:
    header = "## Model Leaderboard\n"
    cols = "| Rank | Model | Games | Wins | Win% | 95% CI | Mean Placement |"
    sep = "|------|-------|-------|------|------|--------|----------------|"
    rows = [header, cols, sep]
    for rank, m in enumerate(board.models, 1):
        ci = f"[{m.ci_low:.4f}, {m.ci_high:.4f}]"
        rows.append(
            f"| {rank} | {m.model} | {m.games} | {m.wins} "
            f"| {m.win_rate * 100:.2f}% | {ci} | {m.mean_placement:.2f} |"
        )
    return "\n".join(rows) + "\n"


def _md_matrix_section(board: Leaderboard) -> str:
    return "## Strategy × Model Win Matrix\n\n" + _pivot_matrix(board.matrix)


def _pivot_matrix(cells: tuple[MatrixCell, ...]) -> str:
    """Build a pivot table (strategies as rows, models as columns)."""
    if not cells:
        return "*(no games recorded)*\n"
    strategies = sorted({c.strategy for c in cells})
    models = sorted({c.model for c in cells})
    lookup = {(c.strategy, c.model): c for c in cells}
    header = "| Strategy | " + " | ".join(models) + " |"
    sep = "|---|" + "---|" * len(models)
    rows = [header, sep]
    for strategy in strategies:
        cells_row = [_cell_label(lookup.get((strategy, m))) for m in models]
        rows.append(f"| {strategy} | " + " | ".join(cells_row) + " |")
    return "\n".join(rows) + "\n"


def _cell_label(cell: MatrixCell | None) -> str:
    if cell is None:
        return "—"
    return f"{cell.win_rate * 100:.1f}% ({cell.wins}/{cell.games})"


def _md_survival_section(board: Leaderboard) -> str:
    header = "## Survival & Placement\n"
    cols = "| Rank | Strategy | Mean Placement |"
    sep = "|------|----------|----------------|"
    rows = [header, cols, sep]
    for rank, s in enumerate(sorted(board.strategies, key=lambda x: x.mean_placement), 1):
        rows.append(f"| {rank} | {s.strategy} | {s.mean_placement:.2f} |")
    return "\n".join(rows) + "\n"


def _md_diplomacy_section(board: Leaderboard) -> str:
    header = "## Diplomacy (Alliances & Betrayals)\n"
    cols = "| Strategy | Betrayals | Alliances |"
    sep = "|----------|-----------|-----------|"
    rows = [header, cols, sep]
    for s in board.strategies:
        rows.append(f"| {s.strategy} | {s.betrayals} | {s.alliances} |")
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# I/O writers — thin wrappers (≤ 3 lines each)
# ---------------------------------------------------------------------------


def write_leaderboard_json(board: Leaderboard, path: Path) -> None:
    """Write *board* as JSON to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_dict(board), indent=2, default=str))


def write_report_md(board: Leaderboard, path: Path) -> None:
    """Write the Markdown report for *board* to *path*, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_markdown(board))
