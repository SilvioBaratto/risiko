"""Tournament driver resumability and concurrency tests — issue #127.

Tests derived from acceptance criteria, updated for the actual API in
training/tournament.py:
  - load_completed_indices  (ledger-based completed-game tracking)
  - pending_game_indices    (compute remaining work)
  - TournamentRunner        (configurable max_concurrency)
"""

from __future__ import annotations

import json

from hypothesis import given
from hypothesis import strategies as st

from training.tournament import (
    TournamentRunner,
    load_completed_indices,
    pending_game_indices,
)

# ---------------------------------------------------------------------------
# load_completed_indices — reads ledger, returns set of completed game indices
# ---------------------------------------------------------------------------


class TestLoadCompletedIndices:
    """Criterion: ledger-based tracking of completed games."""

    def test_when_ledger_has_two_entries_then_both_indices_returned(self, tmp_path):
        ledger = tmp_path / "games.jsonl"
        lines = [
            json.dumps({"game_index": 0, "winner": 0}),
            json.dumps({"game_index": 3, "winner": 1}),
        ]
        ledger.write_text("\n".join(lines))
        completed = load_completed_indices(ledger)
        assert 0 in completed
        assert 3 in completed

    def test_when_ledger_has_entries_then_absent_indices_are_not_returned(self, tmp_path):
        ledger = tmp_path / "games.jsonl"
        ledger.write_text(json.dumps({"game_index": 0, "winner": 0}))
        completed = load_completed_indices(ledger)
        assert 1 not in completed
        assert 2 not in completed

    def test_when_ledger_is_empty_then_empty_set_returned(self, tmp_path):
        ledger = tmp_path / "games.jsonl"
        ledger.write_text("")
        completed = load_completed_indices(ledger)
        assert len(completed) == 0

    def test_when_ledger_does_not_exist_then_empty_set_returned(self, tmp_path):
        ledger = tmp_path / "no_such_file.jsonl"
        completed = load_completed_indices(ledger)
        assert len(completed) == 0


# ---------------------------------------------------------------------------
# pending_game_indices — derives remaining work from completed set
# ---------------------------------------------------------------------------


class TestPendingGameIndices:
    """Criterion: re-invoking skips completed games and continues to target count."""

    def test_when_no_games_completed_then_all_indices_are_pending(self):
        pending = pending_game_indices(5, frozenset())
        assert set(pending) == {0, 1, 2, 3, 4}

    def test_when_some_games_completed_then_those_indices_are_excluded(self):
        pending = pending_game_indices(5, frozenset({0, 1, 3}))
        assert set(pending) == {2, 4}
        assert 0 not in pending
        assert 1 not in pending
        assert 3 not in pending

    def test_when_all_games_completed_then_pending_is_empty(self):
        completed = frozenset(range(10))
        pending = pending_game_indices(10, completed)
        assert len(pending) == 0

    def test_when_no_games_requested_then_pending_is_empty(self):
        pending = pending_game_indices(0, frozenset())
        assert len(pending) == 0

    def test_when_pending_computed_then_count_equals_n_games_minus_completed(self):
        n_games = 20
        completed = frozenset({0, 5, 10, 15})
        pending = pending_game_indices(n_games, completed)
        assert len(pending) == n_games - len(completed)


# ---------------------------------------------------------------------------
# TournamentRunner — configurable max_concurrency
# ---------------------------------------------------------------------------


class TestTournamentRunnerConcurrency:
    """Criterion: Games execute concurrently (configurable max_concurrency)."""

    def test_when_max_concurrency_passed_then_runner_stores_it(self):
        runner = TournamentRunner(max_concurrency=4)
        assert runner.max_concurrency == 4

    def test_when_different_max_concurrency_values_then_each_is_accepted(self):
        for concurrency in (1, 2, 8, 16):
            runner = TournamentRunner(max_concurrency=concurrency)
            assert runner.max_concurrency == concurrency

    def test_when_max_concurrency_is_one_then_runner_constructed_without_error(self):
        TournamentRunner(max_concurrency=1)  # must not raise


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(
    st.integers(min_value=0, max_value=200),
    st.frozensets(st.integers(min_value=0, max_value=199), max_size=200),
)
def test_when_pending_game_indices_called_then_pending_and_completed_partition_range(
    n_games, completed_raw
):
    """Invariant: pending ∪ (completed ∩ range) == range(n_games), disjoint."""
    completed = frozenset(i for i in completed_raw if i < n_games)
    pending = set(pending_game_indices(n_games, completed))
    full_range = set(range(n_games))

    assert pending.isdisjoint(completed)
    assert pending | completed == full_range


@given(st.integers(min_value=0, max_value=100))
def test_when_no_completed_games_then_pending_count_equals_n_games(n_games):
    """Invariant: with no completed games, all games are pending."""
    pending = pending_game_indices(n_games, frozenset())
    assert len(pending) == n_games
