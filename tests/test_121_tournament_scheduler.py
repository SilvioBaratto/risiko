"""
Issue #121 — concurrent scheduler with exponential backoff & call logging.

Source-blind example tests derived solely from acceptance criteria.
All tests are expected to FAIL (red) until the implementation satisfies the criteria.

Skipped criteria (oracle: NOT VERIFIABLE):
- risiko-rl tournament --llm-only flag end-to-end
- Preflight model pull / abort behaviour
- Random strategy assignment distribution over many games
- Leaderboard JSON+report output format
- All-randomness-through-seeds guarantee
- Per-game and cumulative LLM-call count logging
- SOLID / line-length code-quality prose
"""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from training.tournament import TournamentRunner

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _plans(n: int) -> list:
    return [{"game_index": i} for i in range(n)]


def _identity(plan: dict) -> dict:
    return {"game_index": plan["game_index"]}


def _status_error() -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError("429 Too Many Requests", request=MagicMock(), response=MagicMock())


def _http_error() -> httpx.HTTPError:
    return httpx.HTTPError("simulated network failure")


# ---------------------------------------------------------------------------
# Criterion: TournamentRunner.max_concurrency stored exactly
# "TournamentRunner(max_concurrency=4).max_concurrency == 4 and =1 stored exactly"
# ---------------------------------------------------------------------------


class TestMaxConcurrencyAttribute:
    def test_when_max_concurrency_4_then_attribute_equals_4(self):
        assert TournamentRunner(max_concurrency=4).max_concurrency == 4

    def test_when_max_concurrency_1_then_attribute_equals_1(self):
        assert TournamentRunner(max_concurrency=1).max_concurrency == 1

    @given(st.integers(min_value=1, max_value=64))
    def test_when_max_concurrency_k_then_attribute_stored_exactly(self, k):
        """Invariant: constructor stores max_concurrency verbatim for any positive int."""
        assert TournamentRunner(max_concurrency=k).max_concurrency == k


# ---------------------------------------------------------------------------
# Criterion: .run() dispatches plans; on_result called once per plan; disjoint indices
# "invokes on_result exactly once per plan with a disjoint game_index set;
#  with max_concurrency=1 the in-flight count never exceeds 1 (thread-safe counter probe)"
# ---------------------------------------------------------------------------


class TestRunDispatch:
    def test_when_run_with_5_plans_then_on_result_called_5_times(self):
        results: list = []
        TournamentRunner(max_concurrency=2).run(
            _plans(5), play_one=_identity, on_result=results.append
        )
        assert len(results) == 5

    def test_when_run_then_all_game_indices_appear_in_results(self):
        results: list = []
        TournamentRunner(max_concurrency=2).run(
            _plans(5), play_one=_identity, on_result=results.append
        )
        assert {r["game_index"] for r in results} == {0, 1, 2, 3, 4}

    def test_when_run_then_game_indices_in_results_are_disjoint(self):
        """No plan may be processed more than once."""
        results: list = []
        TournamentRunner(max_concurrency=3).run(
            _plans(6), play_one=_identity, on_result=results.append
        )
        indices = [r["game_index"] for r in results]
        assert len(indices) == len(set(indices))

    def test_when_max_concurrency_1_then_inflight_never_exceeds_1(self):
        """
        Thread-safe counter probe: peak concurrent calls to play_one must stay <= 1.
        A short sleep in play_one gives any incorrectly-concurrent scheduler
        a window to expose the bug.
        """
        lock = threading.Lock()
        current = [0]
        peak = [0]

        def play_one(plan: dict) -> dict:
            with lock:
                current[0] += 1
                peak[0] = max(peak[0], current[0])
            time.sleep(0.005)
            with lock:
                current[0] -= 1
            return {"game_index": plan["game_index"]}

        TournamentRunner(max_concurrency=1).run(
            _plans(6), play_one=play_one, on_result=lambda _: None
        )
        assert peak[0] <= 1

    @given(st.integers(min_value=0, max_value=12))
    @settings(max_examples=15)
    def test_when_n_plans_run_then_exactly_n_results_returned(self, n):
        """Invariant: on_result is called exactly once per plan for any n >= 0."""
        results: list = []
        TournamentRunner(max_concurrency=2).run(
            _plans(n), play_one=_identity, on_result=results.append
        )
        assert len(results) == n


# ---------------------------------------------------------------------------
# Criterion: _play_with_backoff retries on HTTP errors, grows delay, re-raises
# "_play_with_backoff retries on httpx.HTTPStatusError/HTTPError, grows the delay
#  exponentially (monkeypatched time.sleep), and re-raises after max_retries"
#
# Design choice: max_retries = number of retry attempts after the first failure,
# so the callable is invoked at most 1 + max_retries times before re-raising.
# The implementation must use time.sleep(x) (not `from time import sleep`)
# so that `patch("time.sleep")` intercepts all delay calls.
# ---------------------------------------------------------------------------


class TestPlayWithBackoff:
    def test_when_fn_raises_http_status_error_then_retried_and_reraised(self):
        runner = TournamentRunner(max_concurrency=1)
        call_count = [0]

        def failing():
            call_count[0] += 1
            raise _status_error()

        with patch("time.sleep"), pytest.raises(httpx.HTTPStatusError):
            runner._play_with_backoff(failing, max_retries=3)

        assert call_count[0] > 1, "expected at least one retry before re-raising"

    def test_when_fn_raises_http_error_then_retried_and_reraised(self):
        """httpx.HTTPError (non-status network errors) must also trigger retry."""
        runner = TournamentRunner(max_concurrency=1)
        call_count = [0]

        def failing():
            call_count[0] += 1
            raise _http_error()

        with patch("time.sleep"), pytest.raises(httpx.HTTPError):
            runner._play_with_backoff(failing, max_retries=2)

        assert call_count[0] > 1

    def test_when_max_retries_exhausted_then_original_exception_is_reraised(self):
        runner = TournamentRunner(max_concurrency=1)

        def always_fails():
            raise _status_error()

        with patch("time.sleep"), pytest.raises((httpx.HTTPStatusError, httpx.HTTPError)):
            runner._play_with_backoff(always_fails, max_retries=2)

    def test_when_retrying_then_sleep_delays_grow_strictly(self):
        """Each successive backoff sleep must be strictly longer than the previous one."""
        runner = TournamentRunner(max_concurrency=1)
        sleep_calls: list = []

        def always_fails():
            raise _status_error()

        with (
            patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)),
            pytest.raises(httpx.HTTPStatusError),
        ):
            runner._play_with_backoff(always_fails, max_retries=4)

        assert len(sleep_calls) >= 2, "expected >=2 sleep calls to check exponential growth"
        for i in range(1, len(sleep_calls)):
            assert sleep_calls[i] > sleep_calls[i - 1], (
                f"sleep delay did not grow at index {i}: {sleep_calls}"
            )

    def test_when_fn_succeeds_on_retry_then_successful_result_is_returned(self):
        """A transient error followed by success must return the successful value."""
        runner = TournamentRunner(max_concurrency=1)
        attempts = [0]

        def flaky():
            attempts[0] += 1
            if attempts[0] == 1:
                raise _status_error()
            return "ok"

        with patch("time.sleep"):
            result = runner._play_with_backoff(flaky, max_retries=3)

        assert result == "ok"

    @given(st.integers(min_value=2, max_value=5))
    @settings(max_examples=10)
    def test_when_max_retries_at_least_2_then_sleep_delays_strictly_increase(self, max_retries):
        """Invariant: exponential growth of sleep delays holds for any max_retries >= 2."""
        runner = TournamentRunner(max_concurrency=1)
        sleep_calls: list = []

        def always_fails():
            raise httpx.HTTPStatusError("429", request=MagicMock(), response=MagicMock())

        with (
            patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)),
            pytest.raises(httpx.HTTPStatusError),
        ):
            runner._play_with_backoff(always_fails, max_retries=max_retries)

        for i in range(1, len(sleep_calls)):
            assert sleep_calls[i] > sleep_calls[i - 1], (
                f"non-exponential delays at index {i}: {sleep_calls}"
            )


# ---------------------------------------------------------------------------
# Criterion: Resumable — completed games in ledger are skipped on re-invocation
# "re-invoking with the same run id skips completed games (ledger-based)"
#
# Design choice: TournamentRunner accepts ledger_path=Path(...); games whose
# game_index appears in the JSONL ledger are skipped without calling play_one.
# Completed results are appended to the ledger by run() so subsequent calls skip them.
# ---------------------------------------------------------------------------


class TestResumability:
    def test_when_ledger_has_completed_games_then_those_indices_are_skipped(self, tmp_path):
        ledger = tmp_path / "games.jsonl"
        with ledger.open("w") as f:
            for idx in (0, 1):
                f.write(json.dumps({"game_index": idx}) + "\n")

        played: list = []
        TournamentRunner(max_concurrency=1, ledger_path=ledger).run(
            _plans(4),
            play_one=_identity,
            on_result=played.append,
        )
        assert {r["game_index"] for r in played} == {2, 3}

    def test_when_ledger_is_empty_then_all_plans_are_played(self, tmp_path):
        ledger = tmp_path / "games.jsonl"
        ledger.write_text("")

        played: list = []
        TournamentRunner(max_concurrency=1, ledger_path=ledger).run(
            _plans(3),
            play_one=_identity,
            on_result=played.append,
        )
        assert {r["game_index"] for r in played} == {0, 1, 2}

    def test_when_all_plans_already_in_ledger_then_play_one_never_called(self, tmp_path):
        ledger = tmp_path / "games.jsonl"
        with ledger.open("w") as f:
            for i in range(3):
                f.write(json.dumps({"game_index": i}) + "\n")

        call_count = [0]

        def counting_play(plan):
            call_count[0] += 1
            return {"game_index": plan["game_index"]}

        TournamentRunner(max_concurrency=1, ledger_path=ledger).run(
            _plans(3), play_one=counting_play, on_result=lambda _: None
        )
        assert call_count[0] == 0

    def test_when_game_completes_then_result_is_appended_to_ledger(self, tmp_path):
        """Completed results must be persisted so re-invocations can skip them."""
        ledger = tmp_path / "games.jsonl"
        TournamentRunner(max_concurrency=1, ledger_path=ledger).run(
            [{"game_index": 0}],
            play_one=lambda p: {"game_index": 0},
            on_result=lambda _: None,
        )
        records = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        assert any(r["game_index"] == 0 for r in records)


# ---------------------------------------------------------------------------
# Criterion: Existing RL train path is unchanged
# "The existing RL train path is unchanged and still passes its tests."
# ---------------------------------------------------------------------------


class TestRLTrainPathUnchanged:
    """Regression guard: importing training.tournament must not break RL modules."""

    def test_when_tournament_module_imported_then_self_play_still_importable(self):
        import training.self_play  # noqa: F401
        import training.tournament  # noqa: F401

    def test_when_tournament_module_imported_then_bc_trainer_still_importable(self):
        import training.bc_trainer  # noqa: F401
        import training.tournament  # noqa: F401
