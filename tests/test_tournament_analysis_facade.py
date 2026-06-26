"""
Source-blind example tests for training/tournament_analysis.py — issue #128.

Every test is derived from the acceptance-criteria text only; no implementation
source was read. Tests are in the Red phase of TDD: they must fail until the
module satisfies every criterion.

Assumed JSONL record structure (derived from spec, not from implementation):
  {
    "game_id": <int>,
    "assignments": [{"model": <str>, "strategy": <str>}, ...],   # 6 entries
    "winner_model": <str>,
    "winner_strategy": <str>,
    "placements": [{"model": <str>, "strategy": <str>, "rank": <int>}, ...],
    "diplomacy": {<strategy>: {"alliances_proposed": <int>,
                               "alliances_honored": <int>,
                               "betrayals": <int>}, ...},
    "seed": <int>
  }
"""

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ─── Canonical strategy / model names matching the ledger format ─────────────

_STRATEGIES = [
    "australia_lock",
    "south_america_lock",
    "aggressive_blitz",
    "turtle_defensive",
    "card_cycle_hunter",
    "diplomat_coalition",
]

_MODELS = [
    "glm-5.2:cloud",
    "minimax-m3:cloud",
    "glm-5.1:cloud",
    "kimi-k2.6:cloud",
    "deepseek-v4-flash:cloud",
    "qwen3.5:cloud",
]


# ─── Fixture helpers ──────────────────────────────────────────────────────────


def _make_record(game_id: int, winner_idx: int = 0) -> dict:
    """Return a well-formed 6-player game record in the ledger format.

    Uses seat_strategies / seat_models / elimination_order as required by
    read_ledger and aggregate_strategies.
    """
    seat_strategies = {str(i): _STRATEGIES[i] for i in range(6)}
    seat_models = {str(i): _MODELS[i] for i in range(6)}
    # Eliminate all non-winner seats in reverse seat order (winner not in list)
    elimination_order = [i for i in range(5, -1, -1) if i != winner_idx]
    return {
        "game_id": game_id,
        "seed": game_id,
        "winner": winner_idx,
        "winner_strategy": _STRATEGIES[winner_idx],
        "winner_model": _MODELS[winner_idx],
        "seat_strategies": seat_strategies,
        "seat_models": seat_models,
        "elimination_order": elimination_order,
        "betrayals": {},
        "alliances": {},
    }


def _write_jsonl(path: Path, records: list) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) if records else "")


# ─── Module API — __all__ and callable surface ───────────────────────────────


class TestModuleAPI:
    """Criterion: module defines __all__ containing build_leaderboard and analyze_run."""

    def test_when_module_imported_then_all_is_a_sequence(self):
        import training.tournament_analysis as ta

        assert isinstance(ta.__all__, (list, tuple))

    def test_when_module_imported_then_build_leaderboard_in_all(self):
        import training.tournament_analysis as ta

        assert "build_leaderboard" in ta.__all__

    def test_when_module_imported_then_analyze_run_in_all(self):
        import training.tournament_analysis as ta

        assert "analyze_run" in ta.__all__

    def test_when_build_leaderboard_called_with_empty_records_then_no_type_error(self):
        """Signature: build_leaderboard(records, run, n_malformed=0) must be callable."""
        from training.tournament_analysis import build_leaderboard

        board = build_leaderboard([], run="smoke")
        assert board is not None

    def test_when_analyze_run_called_then_accepts_path_argument(self, tmp_path):
        """Signature: analyze_run(run_dir: Path) must accept a Path."""
        from training.tournament_analysis import analyze_run

        _write_jsonl(tmp_path / "games.jsonl", [])
        try:
            analyze_run(tmp_path)
        except TypeError as exc:
            pytest.fail(f"analyze_run must accept a Path argument: {exc}")


# ─── build_leaderboard — run name propagation ────────────────────────────────


class TestBuildLeaderboardRunName:
    """Criterion: run is stored verbatim on the returned Leaderboard."""

    def test_when_run_given_then_board_run_matches(self):
        from training.tournament_analysis import build_leaderboard

        board = build_leaderboard([_make_record(0)], run="alpha-001")
        assert board.run == "alpha-001"

    def test_when_run_is_empty_string_then_board_run_is_empty_string(self):
        from training.tournament_analysis import build_leaderboard

        board = build_leaderboard([], run="")
        assert board.run == ""


# ─── build_leaderboard — n_malformed propagation ─────────────────────────────


class TestBuildLeaderboardNMalformed:
    """Criterion: n_malformed=0 default; any supplied value is stored unchanged."""

    def test_when_n_malformed_omitted_then_defaults_to_zero(self):
        from training.tournament_analysis import build_leaderboard

        board = build_leaderboard([_make_record(0)], run="r")
        assert board.n_malformed == 0

    def test_when_n_malformed_given_then_board_records_exact_value(self):
        from training.tournament_analysis import build_leaderboard

        board = build_leaderboard([_make_record(0)], run="r", n_malformed=13)
        assert board.n_malformed == 13

    @given(st.integers(min_value=0, max_value=999))
    @settings(max_examples=30)
    def test_when_any_non_negative_n_malformed_given_then_it_propagates_unchanged(self, n: int):
        """Invariant: n_malformed is stored verbatim for all non-negative counts."""
        from training.tournament_analysis import build_leaderboard

        board = build_leaderboard([], run="prop", n_malformed=n)
        assert board.n_malformed == n


# ─── build_leaderboard — n_players inference ─────────────────────────────────


class TestBuildLeaderboardNPlayers:
    """Criterion: n_players inferred from records; default 6 when records are empty."""

    def test_when_records_have_6_players_each_then_n_players_is_6(self):
        from training.tournament_analysis import build_leaderboard

        records = [_make_record(i) for i in range(4)]
        board = build_leaderboard(records, run="r")
        assert board.n_players == 6

    def test_when_empty_records_given_then_n_players_defaults_to_6(self):
        from training.tournament_analysis import build_leaderboard

        board = build_leaderboard([], run="empty")
        assert board.n_players == 6


# ─── build_leaderboard — composed output structure ────────────────────────────


class TestBuildLeaderboardStructure:
    """Criterion: composes marginal strategy stats (with diplomacy), model stats, matrix."""

    def _board(self, n: int = 3):
        from training.tournament_analysis import build_leaderboard

        return build_leaderboard(
            [_make_record(i, winner_idx=i % 6) for i in range(n)], run="struct"
        )

    def test_when_records_given_then_board_exposes_strategy_stats(self):
        board = self._board()
        attr = getattr(board, "strategies", None) or getattr(board, "strategy_stats", None)
        assert attr is not None, "Leaderboard must expose per-strategy statistics"

    def test_when_records_given_then_board_exposes_model_stats(self):
        board = self._board()
        attr = getattr(board, "models", None) or getattr(board, "model_stats", None)
        assert attr is not None, "Leaderboard must expose per-model statistics"

    def test_when_records_given_then_board_exposes_win_matrix(self):
        board = self._board()
        attr = getattr(board, "matrix", None) or getattr(board, "win_matrix", None)
        assert attr is not None, "Leaderboard must expose a strategy×model win matrix"

    def test_when_same_args_called_twice_then_boards_share_identical_metadata(self):
        """build_leaderboard is pure: same inputs → same run/n_malformed/n_players."""
        from training.tournament_analysis import build_leaderboard

        records = [_make_record(i, winner_idx=i % 6) for i in range(5)]
        b1 = build_leaderboard(records, run="pur", n_malformed=2)
        b2 = build_leaderboard(records, run="pur", n_malformed=2)
        assert (b1.run, b1.n_malformed, b1.n_players) == (
            b2.run,
            b2.n_malformed,
            b2.n_players,
        )


# ─── analyze_run — file emission ─────────────────────────────────────────────


class TestAnalyzeRunFileEmission:
    """Criterion: analyze_run writes leaderboard.json and report.md under run_dir."""

    def test_when_valid_games_jsonl_present_then_leaderboard_json_is_created(self, tmp_path):
        from training.tournament_analysis import analyze_run

        _write_jsonl(tmp_path / "games.jsonl", [_make_record(0)])
        analyze_run(tmp_path)
        assert (tmp_path / "leaderboard.json").exists()

    def test_when_valid_games_jsonl_present_then_report_md_is_created(self, tmp_path):
        from training.tournament_analysis import analyze_run

        _write_jsonl(tmp_path / "games.jsonl", [_make_record(0)])
        analyze_run(tmp_path)
        assert (tmp_path / "report.md").exists()

    def test_when_analyze_run_returns_then_result_has_run_attribute(self, tmp_path):
        """Return value must be a Leaderboard (has .run at minimum)."""
        from training.tournament_analysis import analyze_run

        _write_jsonl(tmp_path / "games.jsonl", [_make_record(0)])
        board = analyze_run(tmp_path)
        assert hasattr(board, "run"), "analyze_run must return a Leaderboard object"

    def test_when_analyze_run_called_then_run_equals_dir_name(self, tmp_path):
        """Criterion: run equals the dir name."""
        from training.tournament_analysis import analyze_run

        _write_jsonl(tmp_path / "games.jsonl", [_make_record(0)])
        board = analyze_run(tmp_path)
        assert board.run == tmp_path.name


# ─── analyze_run — resilience to empty / malformed input ─────────────────────


class TestAnalyzeRunResilience:
    """Criterion: empty/all-malformed ledger renders both artefacts without crashing."""

    def test_when_games_jsonl_is_empty_then_no_crash_and_both_files_created(self, tmp_path):
        from training.tournament_analysis import analyze_run

        (tmp_path / "games.jsonl").write_text("")
        analyze_run(tmp_path)
        assert (tmp_path / "leaderboard.json").exists()
        assert (tmp_path / "report.md").exists()

    def test_when_all_lines_are_malformed_json_then_no_crash_and_both_files_created(self, tmp_path):
        from training.tournament_analysis import analyze_run

        (tmp_path / "games.jsonl").write_text("not-json\n{broken\n{}")
        analyze_run(tmp_path)
        assert (tmp_path / "leaderboard.json").exists()
        assert (tmp_path / "report.md").exists()

    def test_when_games_jsonl_is_missing_then_both_files_written_and_round_trip(self, tmp_path):
        """Missing ledger (no games.jsonl) must still emit valid empty artefacts."""
        from training.tournament_analysis import analyze_run
        from training.tournament_report import to_json_dict

        run_dir = tmp_path / "run_without_ledger"  # does not exist yet
        board = analyze_run(run_dir)
        leaderboard = run_dir / "leaderboard.json"
        assert leaderboard.exists()
        assert (run_dir / "report.md").exists()
        assert board.n_games == 0
        assert json.loads(leaderboard.read_text()) == to_json_dict(board)

    def test_when_mix_of_valid_and_malformed_then_malformed_count_propagates(self, tmp_path):
        """Criterion: n_malformed propagates (counted from unreadable lines)."""
        from training.tournament_analysis import analyze_run

        good = [json.dumps(_make_record(i)) for i in range(3)]
        bad = ["not-valid-json", "{broken-json"]
        (tmp_path / "games.jsonl").write_text("\n".join(good + bad))
        board = analyze_run(tmp_path)
        assert board.n_malformed == 2


# ─── Existing RL training path not broken ────────────────────────────────────


class TestRLPathUntouched:
    """Criterion: the existing RL train path is unchanged and still imports cleanly."""

    def test_when_tournament_analysis_imported_then_self_play_still_importable(self):
        import training.self_play  # must not raise
        import training.tournament_analysis  # noqa: F401

    def test_when_tournament_analysis_imported_then_evaluate_still_importable(self):
        import training.evaluate  # must not raise
        import training.tournament_analysis  # noqa: F401
