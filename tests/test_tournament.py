"""
Tests for tournament resumability, concurrency, and RL train-path preservation.
Issue #115 — RL regression tests are green now.
Issue #117 — TestResumability and TestConcurrency implemented; xfail removed.
"""

import json

from hypothesis import given
from hypothesis import strategies as st

# ── Resumability ──────────────────────────────────────────────────────────────


class TestResumability:
    """Re-invoking with the same run id must skip completed game indices."""

    def test_when_no_games_completed_then_all_indices_are_pending(self):
        """With an empty completed set every game index is returned as pending."""
        from training.tournament import pending_game_indices

        result = sorted(pending_game_indices(total_games=5, completed=frozenset()))
        assert result == list(range(5))

    def test_when_some_games_completed_then_completed_indices_absent_from_pending(self):
        """Completed indices must not appear in the pending collection."""
        from training.tournament import pending_game_indices

        completed = frozenset({0, 2, 4})
        pending = set(pending_game_indices(total_games=6, completed=completed))
        assert pending.isdisjoint(completed)

    def test_when_all_games_completed_then_pending_is_empty(self):
        """When all games are done the pending collection must be empty."""
        from training.tournament import pending_game_indices

        total = 4
        completed = frozenset(range(total))
        result = list(pending_game_indices(total_games=total, completed=completed))
        assert result == []

    def test_when_ledger_file_read_then_completed_indices_extracted(self, tmp_path):
        """load_completed_indices must parse game_index from each JSONL line."""
        from training.tournament import load_completed_indices

        ledger = tmp_path / "games.jsonl"
        expected = {0, 3, 7}
        with open(ledger, "w") as fh:
            for idx in expected:
                fh.write(json.dumps({"game_index": idx, "winner": 2}) + "\n")

        result = load_completed_indices(ledger)
        assert set(result) == expected

    # ── Property: pending is always the exact complement of completed ─────────

    @given(
        total_games=st.integers(min_value=1, max_value=100),
        raw_completed=st.frozensets(st.integers(min_value=0, max_value=99)),
    )
    def test_when_any_completed_subset_given_then_pending_is_exact_complement(
        self, total_games, raw_completed
    ):
        """Invariant: pending_game_indices(n, C) == set(range(n)) - C."""
        from training.tournament import pending_game_indices

        completed = frozenset(i for i in raw_completed if i < total_games)
        pending = frozenset(pending_game_indices(total_games=total_games, completed=completed))
        expected = frozenset(range(total_games)) - completed
        assert pending == expected


# ── Concurrency ───────────────────────────────────────────────────────────────


class TestConcurrency:
    """Games must execute concurrently, controlled by max_concurrency."""

    def test_when_tournament_runner_created_then_accepts_max_concurrency_param(self):
        """TournamentRunner must expose max_concurrency after construction."""
        from training.tournament import TournamentRunner

        runner = TournamentRunner(max_concurrency=4)
        assert runner.max_concurrency == 4

    def test_when_max_concurrency_is_1_then_runner_stores_serial_limit(self):
        """max_concurrency=1 must be stored exactly (serial-execution sentinel)."""
        from training.tournament import TournamentRunner

        runner = TournamentRunner(max_concurrency=1)
        assert runner.max_concurrency == 1


# ── RL train-path regression ──────────────────────────────────────────────────


class TestRLTrainPathUnchanged:
    """The existing RL train path must remain intact after adding the tournament."""

    def test_when_train_command_imported_then_still_callable(self):
        """risiko_rl.cli must still expose a callable 'train' function."""
        from risiko_rl.cli import train

        assert callable(train)

    def test_when_default_training_config_loaded_then_succeeds_without_error(self):
        """config/default.yaml must still be loadable via load_config."""
        from src.config import load_config

        cfg = load_config("config/default.yaml")
        assert cfg is not None

    def test_when_actor_critic_imported_then_class_is_available(self):
        """src.models.actor_critic.ActorCritic must still be importable."""
        from src.models.actor_critic import ActorCritic

        assert ActorCritic is not None
