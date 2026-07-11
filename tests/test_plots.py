"""Smoke tests for the figures.

They do not assert on pixels — they assert that every figure builds from the real
committed run and writes a non-trivial PNG. That is enough to catch the failure mode
that actually happens here: a renamed leaderboard field or a missing strategy colour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from visualization import data as vdata
from visualization import plots
from visualization.theme import STRATEGY_COLORS, STRATEGY_LABELS

pytestmark = pytest.mark.skipif(
    not (vdata.RESULTS_DIR / vdata.DEFAULT_RUN / "leaderboard.json").is_file(),
    reason=f"tournament run {vdata.DEFAULT_RUN!r} is not on disk",
)

MIN_PNG_BYTES = 10_000


@pytest.fixture(scope="module")
def leaderboard() -> dict:
    return vdata.load_leaderboard()


def _assert_png(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > MIN_PNG_BYTES
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_world_map(tmp_path: Path) -> None:
    _assert_png(plots.plot_world_map(tmp_path / "mappa.png"))


def test_strategy_win_rates(tmp_path: Path, leaderboard: dict) -> None:
    _assert_png(plots.plot_strategy_win_rates(leaderboard, tmp_path / "vittorie.png"))


def test_betrayals_vs_winrate(tmp_path: Path, leaderboard: dict) -> None:
    _assert_png(plots.plot_betrayals_vs_winrate(leaderboard, tmp_path / "tradimenti.png"))


def test_strategy_model_matrix(tmp_path: Path, leaderboard: dict) -> None:
    _assert_png(plots.plot_strategy_model_matrix(leaderboard, tmp_path / "matrice.png"))


def test_model_win_rates(tmp_path: Path, leaderboard: dict) -> None:
    _assert_png(plots.plot_model_win_rates(leaderboard, tmp_path / "modelli.png"))


def test_mean_placement(tmp_path: Path, leaderboard: dict) -> None:
    _assert_png(plots.plot_mean_placement(leaderboard, tmp_path / "piazzamento.png"))


def test_convergence(tmp_path: Path) -> None:
    _assert_png(plots.plot_convergence(vdata.load_games(), tmp_path / "convergenza.png"))


def test_plot_all_writes_every_figure(tmp_path: Path) -> None:
    written = plots.plot_all(figures_dir=tmp_path)
    assert len(written) == 7
    for path in written:
        _assert_png(path)


def test_every_strategy_in_the_run_has_a_colour_and_a_label(leaderboard: dict) -> None:
    """A new strategy in a future run must not silently fall back to a default hue."""
    for row in leaderboard["strategies"]:
        assert row["strategy"] in STRATEGY_COLORS
        assert row["strategy"] in STRATEGY_LABELS
