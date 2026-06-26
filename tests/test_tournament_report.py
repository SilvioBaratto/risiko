"""
Source-blind example tests for training/tournament_report.py — issue #128.

Every test is derived from the acceptance-criteria text only; no implementation
source was read. Tests are in the Red phase of TDD: they must fail until the
module satisfies every criterion.

Criteria covered:
  - Module exports to_json_dict, to_markdown, write_leaderboard_json, write_report_md
  - Rates/CIs/placements rounded to 4 decimal places in the JSON dict
  - to_json_dict round-trips through json.loads(json.dumps(...))
  - to_json_dict contains documented top-level keys: run, strategies, models, matrix,
    n_malformed (at minimum)
  - to_markdown contains all required section headers (strategy, model, win matrix),
    the strategy table (all strategy names present), and a pivoted win matrix section
  - write_leaderboard_json writes a file whose JSON content equals to_json_dict(board)
  - write_report_md writes a file whose text content equals to_markdown(board)
  - End-to-end: games.jsonl → analyze_run → leaderboard.json parses == to_json_dict;
    run == dir name; n_malformed propagates; empty/all-malformed no crash
"""

import json
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# ─── Canonical names matching the ledger format ───────────────────────────────

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
    """Well-formed 6-player game record in the ledger format.

    Uses seat_strategies / seat_models / elimination_order as required by
    read_ledger and the aggregation functions.
    """
    seat_strategies = {str(i): _STRATEGIES[i] for i in range(6)}
    seat_models = {str(i): _MODELS[i] for i in range(6)}
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


def _make_board(n: int = 6, run: str = "test-run", n_malformed: int = 0):
    """Build a Leaderboard via the pure function, cycling winners across strategies."""
    from training.tournament_analysis import build_leaderboard

    records = [_make_record(i, winner_idx=i % 6) for i in range(n)]
    return build_leaderboard(records, run=run, n_malformed=n_malformed)


def _write_jsonl(path: Path, records: list) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) if records else "")


# ─── Module API ───────────────────────────────────────────────────────────────


class TestReportModuleAPI:
    """Criterion: module exports to_json_dict, to_markdown, write_leaderboard_json,
    write_report_md.
    """

    def test_when_module_imported_then_to_json_dict_is_callable(self):
        from training import tournament_report as tr

        assert callable(tr.to_json_dict)

    def test_when_module_imported_then_to_markdown_is_callable(self):
        from training import tournament_report as tr

        assert callable(tr.to_markdown)

    def test_when_module_imported_then_write_leaderboard_json_is_callable(self):
        from training import tournament_report as tr

        assert callable(tr.write_leaderboard_json)

    def test_when_module_imported_then_write_report_md_is_callable(self):
        from training import tournament_report as tr

        assert callable(tr.write_report_md)


# ─── to_json_dict — documented top-level keys ─────────────────────────────────


class TestToJsonDictTopLevelKeys:
    """Criterion: dict contains the documented top-level keys."""

    def _d(self, **kw):
        from training.tournament_report import to_json_dict

        return to_json_dict(_make_board(**kw))

    def test_when_board_given_then_dict_contains_run_key(self):
        assert "run" in self._d()

    def test_when_board_given_then_dict_contains_strategies_key(self):
        assert "strategies" in self._d()

    def test_when_board_given_then_dict_contains_models_key(self):
        assert "models" in self._d()

    def test_when_board_given_then_dict_contains_matrix_key(self):
        assert "matrix" in self._d()

    def test_when_board_given_then_dict_contains_n_malformed_key(self):
        d = self._d(n_malformed=4)
        assert "n_malformed" in d

    def test_when_n_malformed_set_then_dict_value_matches(self):
        d = self._d(n_malformed=4)
        assert d["n_malformed"] == 4

    def test_when_run_set_then_dict_run_value_matches(self):
        d = self._d(run="run-xyz-99")
        assert d["run"] == "run-xyz-99"


# ─── to_json_dict — 4-decimal rounding ───────────────────────────────────────


class TestToJsonDictRounding:
    """Criterion: rates/CIs/placements rounded to 4 decimal places."""

    # Use n=12 records so win-rates are non-trivial non-exact decimals.
    def _strategy_entries(self) -> list:
        from training.tournament_report import to_json_dict

        return to_json_dict(_make_board(n=12)).get("strategies", [])

    def _model_entries(self) -> list:
        from training.tournament_report import to_json_dict

        return to_json_dict(_make_board(n=12)).get("models", [])

    def test_when_strategy_float_values_present_then_at_most_4_decimal_places(self):
        for entry in self._strategy_entries():
            for key in ("win_rate", "rate", "ci_low", "ci_high"):
                val = entry.get(key)
                if isinstance(val, float):
                    assert round(val, 4) == val, (
                        f"strategies[].{key}={val!r} has more than 4 decimal places"
                    )

    def test_when_model_float_values_present_then_at_most_4_decimal_places(self):
        for entry in self._model_entries():
            for key in ("win_rate", "rate", "ci_low", "ci_high"):
                val = entry.get(key)
                if isinstance(val, float):
                    assert round(val, 4) == val, (
                        f"models[].{key}={val!r} has more than 4 decimal places"
                    )


# ─── to_json_dict — JSON round-trip ──────────────────────────────────────────


class TestToJsonDictRoundTrip:
    """Criterion: to_json_dict(board) round-trips through json.loads(json.dumps(...))."""

    def test_when_serialised_and_deserialised_then_result_equals_original(self):
        from training.tournament_report import to_json_dict

        d = to_json_dict(_make_board(n=6))
        assert json.loads(json.dumps(d)) == d

    @given(st.integers(min_value=0, max_value=6))
    @settings(max_examples=25)
    def test_when_n_malformed_varies_then_json_round_trip_always_holds(self, n: int):
        """Invariant: JSON serialisation round-trips for any valid n_malformed."""
        from training.tournament_analysis import build_leaderboard
        from training.tournament_report import to_json_dict

        board = build_leaderboard([_make_record(0)], run="rt-prop", n_malformed=n)
        d = to_json_dict(board)
        assert json.loads(json.dumps(d)) == d


# ─── to_markdown — section headers and content ───────────────────────────────


class TestToMarkdown:
    """Criterion: contains all required section headers, strategy table, pivoted win matrix."""

    def _md(self):
        from training.tournament_report import to_markdown

        return to_markdown(_make_board(n=6))

    def _headings(self):
        return [line for line in self._md().splitlines() if line.startswith("#")]

    def test_when_board_given_then_output_is_a_non_empty_string(self):
        md = self._md()
        assert isinstance(md, str) and len(md) > 0

    def test_when_board_given_then_markdown_has_at_least_one_heading(self):
        assert self._headings(), "to_markdown must include at least one markdown heading"

    def test_when_board_given_then_markdown_has_a_strategy_section_heading(self):
        headings = self._headings()
        assert any("strategy" in h.lower() for h in headings), (
            "Markdown must contain a section heading for strategies"
        )

    def test_when_board_given_then_markdown_has_a_model_section_heading(self):
        headings = self._headings()
        assert any("model" in h.lower() for h in headings), (
            "Markdown must contain a section heading for models"
        )

    def test_when_board_given_then_markdown_has_a_matrix_section_heading(self):
        headings = self._headings()
        assert any("matrix" in h.lower() or "win" in h.lower() for h in headings), (
            "Markdown must contain a section heading for the win matrix"
        )

    def test_when_board_given_then_markdown_contains_all_strategy_names(self):
        """Strategy table must mention every strategy name from the catalog."""
        md = self._md()
        for strategy in _STRATEGIES:
            assert strategy in md, f"Strategy '{strategy}' missing from markdown output"


# ─── write_leaderboard_json ───────────────────────────────────────────────────


class TestWriteLeaderboardJson:
    def test_when_called_then_output_file_is_created(self, tmp_path):
        from training.tournament_report import write_leaderboard_json

        out = tmp_path / "leaderboard.json"
        write_leaderboard_json(_make_board(), out)
        assert out.exists()

    def test_when_called_then_output_file_contains_valid_json_object(self, tmp_path):
        from training.tournament_report import write_leaderboard_json

        out = tmp_path / "leaderboard.json"
        write_leaderboard_json(_make_board(), out)
        assert isinstance(json.loads(out.read_text()), dict)

    def test_when_called_then_file_content_equals_to_json_dict(self, tmp_path):
        """write_leaderboard_json is consistent with to_json_dict."""
        from training.tournament_report import to_json_dict, write_leaderboard_json

        board = _make_board(run="writer-check", n_malformed=1)
        out = tmp_path / "leaderboard.json"
        write_leaderboard_json(board, out)
        assert json.loads(out.read_text()) == to_json_dict(board)


# ─── write_report_md ─────────────────────────────────────────────────────────


class TestWriteReportMd:
    def test_when_called_then_output_file_is_created(self, tmp_path):
        from training.tournament_report import write_report_md

        out = tmp_path / "report.md"
        write_report_md(_make_board(), out)
        assert out.exists()

    def test_when_called_then_file_contains_at_least_one_markdown_heading(self, tmp_path):
        from training.tournament_report import write_report_md

        out = tmp_path / "report.md"
        write_report_md(_make_board(), out)
        assert "#" in out.read_text(), "report.md must contain at least one heading"

    def test_when_called_then_file_content_equals_to_markdown(self, tmp_path):
        """write_report_md is consistent with to_markdown."""
        from training.tournament_report import to_markdown, write_report_md

        board = _make_board(run="md-writer-check")
        out = tmp_path / "report.md"
        write_report_md(board, out)
        assert out.read_text() == to_markdown(board)


# ─── End-to-end: analyze_run → both files coherent ──────────────────────────


class TestEndToEnd:
    """T3 criterion: games.jsonl fixture → analyze_run → both output files are coherent."""

    def _write(self, path: Path, n: int, extra_lines: list | None = None) -> None:
        lines = [json.dumps(_make_record(i, winner_idx=i % 6)) for i in range(n)]
        if extra_lines:
            lines.extend(extra_lines)
        path.write_text("\n".join(lines))

    def test_when_games_jsonl_given_then_leaderboard_json_equals_to_json_dict(self, tmp_path):
        """Criterion: leaderboard.json parses and equals to_json_dict(board)."""
        from training.tournament_analysis import analyze_run
        from training.tournament_report import to_json_dict

        self._write(tmp_path / "games.jsonl", n=6)
        board = analyze_run(tmp_path)
        parsed = json.loads((tmp_path / "leaderboard.json").read_text())
        assert parsed == to_json_dict(board)

    def test_when_games_jsonl_given_then_run_equals_dir_name(self, tmp_path):
        """Criterion: run equals the dir name."""
        from training.tournament_analysis import analyze_run

        self._write(tmp_path / "games.jsonl", n=3)
        board = analyze_run(tmp_path)
        assert board.run == tmp_path.name

    def test_when_malformed_lines_present_then_n_malformed_appears_in_leaderboard_json(
        self, tmp_path
    ):
        """Criterion: n_malformed propagates into the emitted JSON artefact."""
        from training.tournament_analysis import analyze_run

        self._write(
            tmp_path / "games.jsonl",
            n=3,
            extra_lines=["not-valid-json", "{broken-json"],
        )
        board = analyze_run(tmp_path)
        assert board.n_malformed == 2
        parsed = json.loads((tmp_path / "leaderboard.json").read_text())
        assert parsed["n_malformed"] == 2

    def test_when_games_jsonl_is_empty_then_both_files_created_without_crash(self, tmp_path):
        """Criterion: empty ledger renders both artefacts without crashing."""
        from training.tournament_analysis import analyze_run

        (tmp_path / "games.jsonl").write_text("")
        analyze_run(tmp_path)
        assert (tmp_path / "leaderboard.json").exists()
        assert (tmp_path / "report.md").exists()

    def test_when_all_records_are_malformed_then_both_files_created_without_crash(self, tmp_path):
        """Criterion: all-malformed ledger renders both artefacts without crashing."""
        from training.tournament_analysis import analyze_run

        (tmp_path / "games.jsonl").write_text("garbage\n{bad\n")
        analyze_run(tmp_path)
        assert (tmp_path / "leaderboard.json").exists()
        assert (tmp_path / "report.md").exists()
