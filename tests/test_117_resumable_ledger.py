"""Example tests for the resumable tournament ledger — Issue #117.

Authored source-blind from acceptance criteria only:
  - pending_game_indices returns the exact complement of completed
  - load_completed_indices: empty for missing file; set of game_index values
    otherwise; malformed lines skipped (not fatal)
  - append_record: exactly one JSON line, flushes+fsyncs; append-then-load
    round-trips the index
  - TestResumability carries NO @pytest.mark.xfail (criterion: remove xfail)
  - TournamentRunner honours configurable max_concurrency
  - Existing RL training imports are unaffected

All tests start in Red (TDD). No implementation source was read.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile

from hypothesis import given
from hypothesis import strategies as st

_VALID_STRATEGIES = [
    "australia_lock",
    "south_america_lock",
    "aggressive_blitz",
    "turtle_defensive",
    "card_cycle_hunter",
    "diplomat_coalition",
]

_VALID_MODELS = [
    "glm-5.2:cloud",
    "minimax-m3:cloud",
    "glm-5.1:cloud",
    "kimi-k2.6:cloud",
    "deepseek-v4-flash:cloud",
    "qwen3.5:cloud",
]


def _make_ledger_record(game_index: int = 0, seed: int = 42):
    """Build a minimal LedgerRecord with all required fields."""
    from training.tournament import LedgerRecord

    return LedgerRecord(
        game_index=game_index,
        seed=seed,
        winner=0,
        winner_strategy="australia_lock",
        winner_model="glm-5.2:cloud",
        n_turns=100,
        elimination_order=[1, 2, 3, 4, 5],
        seat_strategies={str(i): s for i, s in enumerate(_VALID_STRATEGIES)},
        seat_models={str(i): m for i, m in enumerate(_VALID_MODELS)},
        assignment=dict(zip(_VALID_STRATEGIES, _VALID_MODELS, strict=True)),
        card_trade_turns=[10, 35],
        llm_call_count=200,
    )


# ---------------------------------------------------------------------------
# TestResumability — NO @pytest.mark.xfail marks (criterion: remove xfail)
# Criterion: pending_game_indices returns the exact complement of completed
# ---------------------------------------------------------------------------


class TestResumability:
    """pending_game_indices must equal set(range(total)) - completed, always."""

    def test_when_completed_is_empty_then_all_indices_are_pending(self):
        """When no games have completed, every index in [0, total) is pending."""
        from training.tournament import pending_game_indices

        result = pending_game_indices(10, set())
        assert result == set(range(10))

    def test_when_some_games_completed_then_pending_is_exact_complement(self):
        """When games 0-2 are done, pending is {3..9} — the exact complement."""
        from training.tournament import pending_game_indices

        result = pending_game_indices(10, {0, 1, 2})
        assert result == {3, 4, 5, 6, 7, 8, 9}

    def test_when_all_games_completed_then_pending_is_empty(self):
        """When every index is in completed, nothing is pending."""
        from training.tournament import pending_game_indices

        result = pending_game_indices(5, {0, 1, 2, 3, 4})
        assert result == set()

    def test_when_total_is_zero_then_pending_is_empty(self):
        """No games scheduled ⇒ pending is always empty."""
        from training.tournament import pending_game_indices

        assert pending_game_indices(0, set()) == set()

    def test_when_single_game_not_completed_then_pending_contains_it(self):
        """Single scheduled game that has not run appears in pending."""
        from training.tournament import pending_game_indices

        assert pending_game_indices(1, set()) == {0}

    def test_when_single_game_completed_then_pending_is_empty(self):
        """Single scheduled game that already ran is not pending."""
        from training.tournament import pending_game_indices

        assert pending_game_indices(1, {0}) == set()

    def test_when_resuming_then_pending_excludes_all_completed(self):
        """Re-invoking with the same completed set skips those game indices."""
        from training.tournament import pending_game_indices

        completed = {0, 3, 7}
        pending = pending_game_indices(10, completed)
        # No completed index appears in pending (disjoint)
        assert pending.isdisjoint(completed)
        # Union covers all indices exactly once
        assert pending | completed == set(range(10))

    # --- Hypothesis complement-invariant ---

    @given(
        total_games=st.integers(min_value=0, max_value=200),
        extra_indices=st.frozensets(st.integers(min_value=0, max_value=199)),
    )
    def test_complement_invariant_holds_for_all_valid_inputs(self, total_games, extra_indices):
        """Invariant: pending == set(range(total)) − completed for any inputs."""
        from training.tournament import pending_game_indices

        # Keep only indices that are valid game indices for this total
        completed = {i for i in extra_indices if i < total_games}
        result = pending_game_indices(total_games, completed)
        expected = set(range(total_games)) - completed
        assert result == expected


# ---------------------------------------------------------------------------
# TestLoadCompletedIndices
# Criterion: empty for missing file; set of game_index values otherwise;
#            malformed lines skipped, not fatal
# ---------------------------------------------------------------------------


class TestLoadCompletedIndices:
    """Tests for load_completed_indices — JSONL ledger reader."""

    def test_when_ledger_file_is_missing_then_returns_empty_set(self, tmp_path):
        """Criterion: a non-existent file path → empty set, no exception."""
        from training.tournament import load_completed_indices

        missing = tmp_path / "does_not_exist.jsonl"
        result = load_completed_indices(missing)
        assert result == set()

    def test_when_ledger_has_one_valid_line_then_returns_its_game_index(self, tmp_path):
        """One valid JSONL line with game_index → {game_index}."""
        from training.tournament import load_completed_indices

        ledger = tmp_path / "games.jsonl"
        ledger.write_text(json.dumps({"game_index": 7}) + "\n")
        assert load_completed_indices(ledger) == {7}

    def test_when_ledger_has_multiple_valid_lines_then_returns_all_indices(self, tmp_path):
        """N valid JSONL lines → set of all N game_index values."""
        from training.tournament import load_completed_indices

        ledger = tmp_path / "games.jsonl"
        lines = "\n".join(json.dumps({"game_index": i, "seed": 42}) for i in range(5))
        ledger.write_text(lines + "\n")
        assert load_completed_indices(ledger) == {0, 1, 2, 3, 4}

    def test_when_ledger_has_malformed_lines_then_they_are_skipped(self, tmp_path):
        """Malformed (non-JSON) lines are silently skipped; valid lines collected."""
        from training.tournament import load_completed_indices

        ledger = tmp_path / "games.jsonl"
        ledger.write_text(
            json.dumps({"game_index": 3})
            + "\n"
            + "THIS IS NOT VALID JSON\n"
            + "}{broken\n"
            + json.dumps({"game_index": 7})
            + "\n"
        )
        result = load_completed_indices(ledger)
        # Must not raise; valid indices are present
        assert result == {3, 7}

    def test_when_ledger_is_empty_file_then_returns_empty_set(self, tmp_path):
        """A zero-byte ledger file → empty set."""
        from training.tournament import load_completed_indices

        ledger = tmp_path / "games.jsonl"
        ledger.write_text("")
        assert load_completed_indices(ledger) == set()

    def test_when_line_has_no_game_index_key_then_line_is_skipped(self, tmp_path):
        """A valid-JSON line without a game_index key is skipped, not fatal."""
        from training.tournament import load_completed_indices

        ledger = tmp_path / "games.jsonl"
        ledger.write_text(
            json.dumps({"winner": 2, "n_turns": 80})
            + "\n"  # no game_index
            + json.dumps({"game_index": 5})
            + "\n"
        )
        result = load_completed_indices(ledger)
        assert result == {5}

    def test_when_ledger_has_duplicate_indices_then_set_deduplicates(self, tmp_path):
        """Duplicate game_index values in the file are deduplicated in the result."""
        from training.tournament import load_completed_indices

        ledger = tmp_path / "games.jsonl"
        ledger.write_text(
            json.dumps({"game_index": 3})
            + "\n"
            + json.dumps({"game_index": 3})
            + "\n"  # duplicate
            + json.dumps({"game_index": 9})
            + "\n"
        )
        result = load_completed_indices(ledger)
        assert result == {3, 9}


# ---------------------------------------------------------------------------
# TestAppendRecord
# Criterion: appends exactly one JSON line, flushes+fsyncs;
#            append-then-load round-trips the index
# ---------------------------------------------------------------------------


class TestAppendRecord:
    """Tests for append_record — JSONL ledger writer."""

    def test_when_append_record_then_file_gains_exactly_one_line(self, tmp_path):
        """One call to append_record → file contains exactly 1 non-empty line."""
        from training.tournament import append_record

        ledger = tmp_path / "games.jsonl"
        append_record(ledger, _make_ledger_record(game_index=0))
        lines = [ln for ln in ledger.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1

    def test_when_two_records_appended_then_file_has_two_lines(self, tmp_path):
        """Two appends accumulate (no overwrite) → exactly 2 non-empty lines."""
        from training.tournament import append_record

        ledger = tmp_path / "games.jsonl"
        append_record(ledger, _make_ledger_record(game_index=0))
        append_record(ledger, _make_ledger_record(game_index=1))
        lines = [ln for ln in ledger.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2

    def test_when_append_then_load_then_game_index_round_trips(self, tmp_path):
        """Criterion: append-then-load round-trips the game_index."""
        from training.tournament import append_record, load_completed_indices

        ledger = tmp_path / "games.jsonl"
        append_record(ledger, _make_ledger_record(game_index=42))
        assert 42 in load_completed_indices(ledger)

    def test_when_appending_n_records_then_all_indices_round_trip(self, tmp_path):
        """N appends → load_completed_indices returns all N game indices."""
        from training.tournament import append_record, load_completed_indices

        ledger = tmp_path / "games.jsonl"
        for i in range(6):
            append_record(ledger, _make_ledger_record(game_index=i))
        assert load_completed_indices(ledger) == set(range(6))

    def test_when_append_creates_parent_directories(self, tmp_path):
        """append_record must create missing intermediate parent directories."""
        from training.tournament import append_record

        ledger = tmp_path / "results" / "tournament" / "run01" / "games.jsonl"
        append_record(ledger, _make_ledger_record(game_index=0))
        assert ledger.exists()

    def test_when_appended_line_is_parsed_then_it_is_valid_json_dict(self, tmp_path):
        """Every line written by append_record must be valid JSON (a dict)."""
        from training.tournament import append_record

        ledger = tmp_path / "games.jsonl"
        append_record(ledger, _make_ledger_record(game_index=5))
        for line in ledger.read_text().splitlines():
            if line.strip():
                parsed = json.loads(line)  # must not raise
                assert isinstance(parsed, dict)
                assert "game_index" in parsed

    def test_when_appended_line_is_parsed_then_game_index_value_is_correct(self, tmp_path):
        """The game_index field in the written JSON matches the record's value."""
        from training.tournament import append_record

        ledger = tmp_path / "games.jsonl"
        append_record(ledger, _make_ledger_record(game_index=99))
        line = ledger.read_text().strip()
        assert json.loads(line)["game_index"] == 99

    # --- Hypothesis round-trip invariant ---

    @given(game_index=st.integers(min_value=0, max_value=9999))
    def test_append_then_load_round_trips_any_valid_game_index(self, game_index):
        """Invariant: append_then_load round-trips any game_index in [0, 9999]."""
        from training.tournament import append_record, load_completed_indices

        tmpdir = pathlib.Path(tempfile.mkdtemp())
        try:
            ledger = tmpdir / "games.jsonl"
            append_record(ledger, _make_ledger_record(game_index=game_index))
            loaded = load_completed_indices(ledger)
            assert game_index in loaded
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# TestConcurrentRunner
# Criterion: games execute concurrently (configurable max_concurrency)
# ---------------------------------------------------------------------------


class TestConcurrentRunner:
    """Tests for TournamentRunner — concurrent game scheduler."""

    def test_when_runner_created_with_max_concurrency_then_attribute_is_set(self):
        """Criterion: TournamentRunner accepts and stores configurable max_concurrency."""
        from training.tournament import TournamentRunner

        runner = TournamentRunner(max_concurrency=4)
        assert runner.max_concurrency == 4

    def test_when_runner_runs_empty_plan_list_then_completed_count_is_zero(self):
        """No plans → run returns 0 (no games completed)."""
        from training.tournament import TournamentRunner

        runner = TournamentRunner(max_concurrency=2)
        count = runner.run([], lambda p: None, lambda r: None)
        assert count == 0

    def test_when_runner_runs_three_plans_then_on_result_called_three_times(self):
        """on_result callback is invoked exactly once per completed game."""
        from training.tournament import GamePlan, TournamentRunner

        plans = [GamePlan(game_index=i, seed=i, assignment={}) for i in range(3)]
        results = []

        def play_one(plan):
            return _make_ledger_record(game_index=plan.game_index)

        runner = TournamentRunner(max_concurrency=3)
        count = runner.run(plans, play_one, results.append)
        assert count == 3
        assert len(results) == 3

    def test_when_runner_runs_plans_then_all_game_indices_appear_in_results(self):
        """All scheduled game indices appear in the on_result records."""
        from training.tournament import GamePlan, TournamentRunner

        plans = [GamePlan(game_index=i, seed=i, assignment={}) for i in range(5)]
        seen_indices = []

        def play_one(plan):
            return _make_ledger_record(game_index=plan.game_index)

        TournamentRunner(max_concurrency=2).run(
            plans, play_one, lambda r: seen_indices.append(r.game_index)
        )
        assert sorted(seen_indices) == list(range(5))

    def test_when_max_concurrency_is_one_then_all_plans_still_execute(self):
        """Even with max_concurrency=1 all plans complete (sequential fallback)."""
        from training.tournament import GamePlan, TournamentRunner

        plans = [GamePlan(game_index=i, seed=i, assignment={}) for i in range(4)]
        results = []

        runner = TournamentRunner(max_concurrency=1)
        count = runner.run(
            plans,
            lambda p: _make_ledger_record(game_index=p.game_index),
            results.append,
        )
        assert count == 4
        assert len(results) == 4


# ---------------------------------------------------------------------------
# RL train path smoke — existing path unchanged
# Criterion: The existing RL `train` path is unchanged and still passes tests
# ---------------------------------------------------------------------------


def test_when_rl_training_modules_imported_then_no_error_is_raised():
    """Importing core RL modules must succeed; tournament must not break them."""
    import src.checkpoint  # noqa: F401
    import src.models.actor_critic  # noqa: F401
    import src.models.ppo  # noqa: F401


def test_when_tournament_module_imported_then_rl_config_still_importable():
    """Tournament import must not shadow or break src.config."""
    import src.config  # noqa: F401
    import training.tournament  # noqa: F401

    assert hasattr(src.config, "TrainingConfig")
