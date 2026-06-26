"""Example tests for run_tournament orchestration — Issue #122.

Authored source-blind from acceptance criteria only; no implementation source
was read.  All tests start Red (TDD).

Criteria covered (UNIT-verifiable from the oracle report):
  - Fresh run with n_games=3 writes 3 JSONL lines with the documented schema;
    TournamentSummary.total_games_in_ledger == 3
  - Resume: ledger pre-seeded with game_index=0 and n_games=3 calls play_fn
    only for indices 1 and 2; final ledger has 3 lines, no duplicate game_index
  - max_games=2 with n_games=10 plays exactly 2 new games this run
  - Same seed → identical assignment/seat_strategies per game_index across runs
    (assignment determinism; LLM sampling isolated to the opponent)
  - TournamentSummary.total_llm_calls equals the sum of per-game llm_call_count
  - The existing RL train path is unchanged and still passes its tests
"""

from __future__ import annotations

import json
import pathlib

import yaml
from hypothesis import HealthCheck, given, settings
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

# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _base_config_data(n_games: int = 3, seed: int = 42) -> dict:
    return {
        "name": "test_tournament_122",
        "seed": seed,
        "n_games": n_games,
        "max_concurrency": 1,
        "n_rounds": 2,
        "strategies": _VALID_STRATEGIES,
        "models": _VALID_MODELS,
        "temperature": 0.7,
        "top_p": 0.9,
    }


def _write_config(
    tmp_path: pathlib.Path,
    *,
    n_games: int = 3,
    seed: int = 42,
    filename: str = "tournament.yaml",
) -> pathlib.Path:
    cfg_file = tmp_path / filename
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(yaml.dump(_base_config_data(n_games=n_games, seed=seed)))
    return cfg_file


def _load_cfg(tmp_path: pathlib.Path, *, n_games: int = 3, seed: int = 42):
    from training.tournament import load_tournament_config

    return load_tournament_config(_write_config(tmp_path, n_games=n_games, seed=seed))


def _make_ledger_record(game_index: int = 0, llm_call_count: int = 10):
    """Return a minimal but fully-populated LedgerRecord for testing."""
    from training.tournament import LedgerRecord

    return LedgerRecord(
        game_index=game_index,
        seed=game_index + 100,
        winner=0,
        winner_strategy="australia_lock",
        winner_model="glm-5.2:cloud",
        n_turns=80,
        elimination_order=[1, 2, 3, 4, 5],
        seat_strategies={str(i): s for i, s in enumerate(_VALID_STRATEGIES)},
        seat_models={str(i): m for i, m in enumerate(_VALID_MODELS)},
        assignment=dict(zip(_VALID_STRATEGIES, _VALID_MODELS, strict=True)),
        card_trade_turns=[15, 42],
        llm_call_count=llm_call_count,
    )


def _spy_play_fn(call_log: list[int]):
    """Return a play_fn that records every game_index it is invoked with."""

    def _play(plan):
        call_log.append(plan.game_index)
        return _make_ledger_record(game_index=plan.game_index)

    return _play


# ─────────────────────────────────────────────────────────────────────────────
# TestFreshRunWritesLedger
# Criterion: fresh run with n_games=3 writes 3 JSONL lines with the documented
#            schema; TournamentSummary.total_games_in_ledger == 3
# ─────────────────────────────────────────────────────────────────────────────


class TestFreshRunWritesLedger:
    """run_tournament on an empty ledger must write exactly n_games records."""

    def test_when_fresh_run_with_n_games_3_then_3_jsonl_lines_are_written(self, tmp_path):
        """Criterion: a fresh run with n_games=3 writes exactly 3 non-empty JSONL lines."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        lines = [ln for ln in ledger.read_text().splitlines() if ln.strip()]
        assert len(lines) == 3

    def test_when_fresh_run_with_n_games_3_then_summary_total_games_in_ledger_is_3(self, tmp_path):
        """Criterion: TournamentSummary.total_games_in_ledger == 3 after a fresh 3-game run."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        summary = run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        assert summary.total_games_in_ledger == 3

    def test_when_fresh_run_then_each_jsonl_line_is_valid_json_dict(self, tmp_path):
        """Every line written to the ledger must parse as a JSON object (dict)."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        for line in ledger.read_text().splitlines():
            if line.strip():
                parsed = json.loads(line)
                assert isinstance(parsed, dict)

    def test_when_fresh_run_then_each_jsonl_line_has_game_index_field(self, tmp_path):
        """Every JSONL record must contain the documented 'game_index' field."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        for line in ledger.read_text().splitlines():
            if line.strip():
                assert "game_index" in json.loads(line)

    def test_when_fresh_run_then_ledger_game_indices_are_0_through_2(self, tmp_path):
        """A 3-game fresh run must produce records for game_index ∈ {0, 1, 2} (complete)."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        indices = {
            json.loads(ln)["game_index"] for ln in ledger.read_text().splitlines() if ln.strip()
        }
        assert indices == {0, 1, 2}

    def test_when_fresh_run_then_no_duplicate_game_indices_in_ledger(self, tmp_path):
        """No game_index value may appear more than once in the ledger after a fresh run."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        indices = [
            json.loads(ln)["game_index"] for ln in ledger.read_text().splitlines() if ln.strip()
        ]
        assert len(indices) == len(set(indices))

    def test_when_fresh_run_then_play_fn_called_exactly_3_times(self, tmp_path):
        """play_fn must be invoked exactly 3 times for a fresh run with n_games=3."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        call_log: list[int] = []
        run_tournament(cfg, ledger, play_fn=_spy_play_fn(call_log))
        assert len(call_log) == 3

    def test_when_fresh_run_then_each_jsonl_line_has_llm_call_count_field(self, tmp_path):
        """Every JSONL record must include the documented 'llm_call_count' field."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        for line in ledger.read_text().splitlines():
            if line.strip():
                assert "llm_call_count" in json.loads(line)

    def test_when_fresh_run_then_each_jsonl_line_has_seat_strategies_field(self, tmp_path):
        """Every JSONL record must include the documented 'seat_strategies' field."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        for line in ledger.read_text().splitlines():
            if line.strip():
                assert "seat_strategies" in json.loads(line)

    def test_when_fresh_run_then_each_jsonl_line_has_assignment_field(self, tmp_path):
        """Every JSONL record must include the documented 'assignment' field."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        for line in ledger.read_text().splitlines():
            if line.strip():
                assert "assignment" in json.loads(line)

    def test_when_fresh_run_then_summary_total_games_equals_ledger_line_count(self, tmp_path):
        """total_games_in_ledger must equal the count of non-empty lines in the JSONL file."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        summary = run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        lines = [ln for ln in ledger.read_text().splitlines() if ln.strip()]
        assert summary.total_games_in_ledger == len(lines)


# ─────────────────────────────────────────────────────────────────────────────
# TestResumption
# Criterion: a ledger pre-seeded with game_index=0 and n_games=3 plays only
#            indices 1 and 2 (verified via a spy on play_fn); ends with 3 lines,
#            no duplicate game_index.
# ─────────────────────────────────────────────────────────────────────────────


class TestResumption:
    """run_tournament skips completed game indices found in a pre-existing ledger."""

    def _seed_ledger(self, ledger_path: pathlib.Path, game_indices: list[int]) -> None:
        from training.tournament import append_record

        for idx in game_indices:
            append_record(ledger_path, _make_ledger_record(game_index=idx))

    def test_when_game_0_completed_and_n_games_3_then_spy_called_for_index_1(self, tmp_path):
        """When game_index=0 is already in the ledger, play_fn must be called for index 1."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        self._seed_ledger(ledger, [0])
        call_log: list[int] = []
        run_tournament(cfg, ledger, play_fn=_spy_play_fn(call_log))
        assert 1 in call_log

    def test_when_game_0_completed_and_n_games_3_then_spy_called_for_index_2(self, tmp_path):
        """When game_index=0 is already in the ledger, play_fn must be called for index 2."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        self._seed_ledger(ledger, [0])
        call_log: list[int] = []
        run_tournament(cfg, ledger, play_fn=_spy_play_fn(call_log))
        assert 2 in call_log

    def test_when_game_0_completed_and_n_games_3_then_spy_not_called_for_index_0(self, tmp_path):
        """Criterion: play_fn must NOT be called for the already-completed game_index=0."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        self._seed_ledger(ledger, [0])
        call_log: list[int] = []
        run_tournament(cfg, ledger, play_fn=_spy_play_fn(call_log))
        assert 0 not in call_log

    def test_when_game_0_completed_and_n_games_3_then_spy_called_exactly_twice(self, tmp_path):
        """play_fn must be invoked exactly 2 times (only indices 1 and 2 are pending)."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        self._seed_ledger(ledger, [0])
        call_log: list[int] = []
        run_tournament(cfg, ledger, play_fn=_spy_play_fn(call_log))
        assert len(call_log) == 2

    def test_when_game_0_completed_and_n_games_3_then_final_ledger_has_3_lines(self, tmp_path):
        """After resuming from game_index=0, the ledger must contain exactly 3 non-empty lines."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        self._seed_ledger(ledger, [0])
        run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        lines = [ln for ln in ledger.read_text().splitlines() if ln.strip()]
        assert len(lines) == 3

    def test_when_game_0_completed_and_n_games_3_then_no_duplicate_game_index(self, tmp_path):
        """After resumption no game_index must appear more than once in the final ledger."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        self._seed_ledger(ledger, [0])
        run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        indices = [
            json.loads(ln)["game_index"] for ln in ledger.read_text().splitlines() if ln.strip()
        ]
        assert len(indices) == len(set(indices))

    def test_when_game_0_completed_and_n_games_3_then_all_three_indices_present(self, tmp_path):
        """After resumption the ledger must contain records for {0, 1, 2} (none missing)."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        self._seed_ledger(ledger, [0])
        run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        indices = {
            json.loads(ln)["game_index"] for ln in ledger.read_text().splitlines() if ln.strip()
        }
        assert indices == {0, 1, 2}

    def test_when_all_games_already_in_ledger_then_spy_never_called(self, tmp_path):
        """When every game index is already completed, play_fn is never invoked."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        self._seed_ledger(ledger, [0, 1, 2])
        call_log: list[int] = []
        run_tournament(cfg, ledger, play_fn=_spy_play_fn(call_log))
        assert call_log == []

    def test_when_all_games_already_in_ledger_then_total_games_in_ledger_is_3(self, tmp_path):
        """Resuming a fully-complete ledger still reports total_games_in_ledger == 3."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        self._seed_ledger(ledger, [0, 1, 2])
        summary = run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        assert summary.total_games_in_ledger == 3


# ─────────────────────────────────────────────────────────────────────────────
# TestMaxGamesCap
# Criterion: max_games=2 with n_games=10 plays exactly 2 new games this run.
# ─────────────────────────────────────────────────────────────────────────────


class TestMaxGamesCap:
    """run_tournament respects the max_games cap within a single invocation."""

    def test_when_max_games_2_and_n_games_10_then_play_fn_called_exactly_twice(self, tmp_path):
        """Criterion: max_games=2 with n_games=10 → play_fn invoked exactly 2 times."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=10)
        ledger = tmp_path / "games.jsonl"
        call_log: list[int] = []
        run_tournament(cfg, ledger, play_fn=_spy_play_fn(call_log), max_games=2)
        assert len(call_log) == 2

    def test_when_max_games_2_and_n_games_10_then_exactly_2_lines_in_ledger(self, tmp_path):
        """max_games=2 cap: exactly 2 JSONL records are written to the ledger."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=10)
        ledger = tmp_path / "games.jsonl"
        run_tournament(cfg, ledger, play_fn=_spy_play_fn([]), max_games=2)
        lines = [ln for ln in ledger.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2

    def test_when_max_games_equals_n_games_then_all_games_played(self, tmp_path):
        """When max_games == n_games, all games must complete (cap does not truncate)."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=4)
        ledger = tmp_path / "games.jsonl"
        call_log: list[int] = []
        run_tournament(cfg, ledger, play_fn=_spy_play_fn(call_log), max_games=4)
        assert len(call_log) == 4

    def test_when_max_games_0_then_no_games_are_played(self, tmp_path):
        """max_games=0 must prevent any game from running this invocation."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=5)
        ledger = tmp_path / "games.jsonl"
        call_log: list[int] = []
        run_tournament(cfg, ledger, play_fn=_spy_play_fn(call_log), max_games=0)
        assert call_log == []

    def test_when_max_games_larger_than_pending_then_only_pending_played(self, tmp_path):
        """max_games larger than pending count plays only the pending count, not more."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        call_log: list[int] = []
        run_tournament(cfg, ledger, play_fn=_spy_play_fn(call_log), max_games=100)
        assert len(call_log) == 3

    def test_when_max_games_is_none_then_all_pending_games_are_played(self, tmp_path):
        """max_games=None (the default) plays all pending games without capping."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        call_log: list[int] = []
        run_tournament(cfg, ledger, play_fn=_spy_play_fn(call_log), max_games=None)
        assert len(call_log) == 3

    def test_when_max_games_applied_then_two_sequential_runs_cover_all_games(self, tmp_path):
        """Two capped runs of 2 on a 4-game tournament together cover all 4 games."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=4)
        ledger = tmp_path / "games.jsonl"
        all_calls: list[int] = []

        run_tournament(cfg, ledger, play_fn=_spy_play_fn(all_calls), max_games=2)
        run_tournament(cfg, ledger, play_fn=_spy_play_fn(all_calls), max_games=2)

        assert sorted(all_calls) == [0, 1, 2, 3]

    # ── Hypothesis: max_games cap invariant ──────────────────────────────────

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(cap=st.integers(min_value=0, max_value=5))
    def test_when_any_max_games_cap_then_play_fn_not_called_more_than_cap(self, cap, tmp_path):
        """Invariant: play_fn is called at most max_games times for any cap in [0, 5]."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=10)
        ledger = tmp_path / "games.jsonl"
        call_log: list[int] = []
        run_tournament(cfg, ledger, play_fn=_spy_play_fn(call_log), max_games=cap)
        assert len(call_log) <= cap


# ─────────────────────────────────────────────────────────────────────────────
# TestTournamentSummary
# Criterion: TournamentSummary.total_llm_calls equals the sum of per-game
#            llm_call_count.
# ─────────────────────────────────────────────────────────────────────────────


class TestTournamentSummary:
    """TournamentSummary correctly aggregates ledger data."""

    def test_when_3_games_each_10_llm_calls_then_total_llm_calls_is_30(self, tmp_path):
        """Criterion: total_llm_calls == sum of per-game llm_call_count (3 × 10 = 30)."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"

        def play_fn(plan):
            return _make_ledger_record(game_index=plan.game_index, llm_call_count=10)

        summary = run_tournament(cfg, ledger, play_fn=play_fn)
        assert summary.total_llm_calls == 30

    def test_when_games_have_varied_llm_calls_then_total_equals_exact_sum(self, tmp_path):
        """total_llm_calls must equal the exact arithmetic sum, not an approximation."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        # Sum = 7 + 13 + 22 = 42; keyed by game_index for concurrency safety
        call_counts = {0: 7, 1: 13, 2: 22}

        def play_fn(plan):
            return _make_ledger_record(
                game_index=plan.game_index,
                llm_call_count=call_counts[plan.game_index],
            )

        summary = run_tournament(cfg, ledger, play_fn=play_fn)
        assert summary.total_llm_calls == 42

    def test_when_no_games_played_then_total_llm_calls_is_zero(self, tmp_path):
        """When max_games=0 prevents all games, total_llm_calls must be 0."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=5)
        ledger = tmp_path / "games.jsonl"
        summary = run_tournament(cfg, ledger, play_fn=_spy_play_fn([]), max_games=0)
        assert summary.total_llm_calls == 0

    def test_when_summary_has_total_llm_calls_including_preexisting_records(self, tmp_path):
        """
        total_llm_calls reflects ALL records in the ledger (including pre-existing),
        consistent with total_games_in_ledger counting the full ledger, not just this run.

        Choice: TournamentSummary aggregates over the complete ledger, not only the
        games played in the current invocation.
        """
        from training.tournament import append_record, run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        # Pre-seed game 0 with 50 LLM calls
        append_record(ledger, _make_ledger_record(game_index=0, llm_call_count=50))

        def play_fn(plan):
            return _make_ledger_record(game_index=plan.game_index, llm_call_count=10)

        # Resumes: plays games 1 and 2 (10 calls each)
        summary = run_tournament(cfg, ledger, play_fn=play_fn)
        # 50 (pre-existing) + 10 + 10 = 70
        assert summary.total_llm_calls == 70

    def test_when_summary_total_llm_calls_is_non_negative(self, tmp_path):
        """total_llm_calls must never be negative."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        summary = run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        assert summary.total_llm_calls >= 0

    def test_when_summary_has_total_games_in_ledger_attribute(self, tmp_path):
        """TournamentSummary must expose a total_games_in_ledger integer attribute."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        summary = run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        assert isinstance(summary.total_games_in_ledger, int)
        assert summary.total_games_in_ledger >= 0

    def test_when_summary_has_total_llm_calls_attribute(self, tmp_path):
        """TournamentSummary must expose a total_llm_calls integer attribute."""
        from training.tournament import run_tournament

        cfg = _load_cfg(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        summary = run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))
        assert isinstance(summary.total_llm_calls, int)


# ─────────────────────────────────────────────────────────────────────────────
# TestAssignmentDeterminismAcrossRuns
# Criterion: Same seed → identical assignment/seat_strategies per game_index
#            across runs (assignment determinism; LLM sampling isolated).
# ─────────────────────────────────────────────────────────────────────────────


class TestAssignmentDeterminismAcrossRuns:
    """build_game_plan produces identical assignment and seat_strategies across runs."""

    def test_when_same_seed_and_game_index_then_assignment_is_identical_across_runs(self, tmp_path):
        """Criterion: same (seed, game_index) → identical assignment on two separate calls."""
        from training.tournament import build_game_plan, load_tournament_config

        cfg_file = _write_config(tmp_path, seed=7)
        cfg_a = load_tournament_config(cfg_file)
        cfg_b = load_tournament_config(cfg_file)

        plan_a = build_game_plan(cfg_a, game_index=3)
        plan_b = build_game_plan(cfg_b, game_index=3)
        assert plan_a.assignment == plan_b.assignment

    def test_when_same_seed_and_game_index_then_seat_strategies_is_identical_across_runs(
        self, tmp_path
    ):
        """
        Criterion: seat_strategies derived from the same (seed, game_index) is reproducible.

        Choice: if GamePlan exposes a seat_strategies attribute (seat_idx → strategy),
        it must be deterministic.  If not present, the assignment bijection (which
        determines seat_strategies) is checked instead — the invariant is the same.
        """
        from training.tournament import build_game_plan, load_tournament_config

        cfg_file = _write_config(tmp_path, seed=7)
        cfg_a = load_tournament_config(cfg_file)
        cfg_b = load_tournament_config(cfg_file)

        plan_a = build_game_plan(cfg_a, game_index=5)
        plan_b = build_game_plan(cfg_b, game_index=5)

        seat_a = getattr(plan_a, "seat_strategies", plan_a.assignment)
        seat_b = getattr(plan_b, "seat_strategies", plan_b.assignment)
        assert seat_a == seat_b

    def test_when_same_seed_then_game_index_0_assignment_is_reproducible(self, tmp_path):
        """game_index=0 assignment is stable across separate load_tournament_config calls."""
        from training.tournament import build_game_plan, load_tournament_config

        cfg_file = _write_config(tmp_path, seed=42)
        plan_first = build_game_plan(load_tournament_config(cfg_file), game_index=0)
        plan_second = build_game_plan(load_tournament_config(cfg_file), game_index=0)
        assert plan_first.assignment == plan_second.assignment

    def test_when_different_seeds_then_assignments_differ_across_game_indices(self, tmp_path):
        """
        Two configs with different seeds must produce at least one differing assignment
        over 20 game indices, confirming that LLM sampling is isolated and the seed
        is the sole source of assignment randomness.
        """
        from training.tournament import build_game_plan, load_tournament_config

        cfg_a_file = tmp_path / "a.yaml"
        cfg_a_file.write_text(yaml.dump(_base_config_data(seed=1)))
        cfg_b_file = tmp_path / "b.yaml"
        cfg_b_file.write_text(yaml.dump(_base_config_data(seed=999)))

        cfg_a = load_tournament_config(cfg_a_file)
        cfg_b = load_tournament_config(cfg_b_file)

        frozen_a = [frozenset(build_game_plan(cfg_a, i).assignment.items()) for i in range(20)]
        frozen_b = [frozenset(build_game_plan(cfg_b, i).assignment.items()) for i in range(20)]
        assert frozen_a != frozen_b

    def test_when_run_tournament_twice_with_same_seed_then_ledger_game_indices_match(
        self, tmp_path
    ):
        """
        Running run_tournament twice with the same config seed and disjoint ledger paths
        must produce the same set of game_index values (all of them, in the same domain).
        """
        from training.tournament import run_tournament

        (tmp_path / "run_a").mkdir(exist_ok=True)
        (tmp_path / "run_b").mkdir(exist_ok=True)
        cfg_a = _load_cfg(tmp_path / "run_a", seed=55, n_games=3)
        cfg_b = _load_cfg(tmp_path / "run_b", seed=55, n_games=3)
        ledger_a = tmp_path / "run_a" / "games.jsonl"
        ledger_b = tmp_path / "run_b" / "games.jsonl"

        assignments_a: dict[int, frozenset] = {}
        assignments_b: dict[int, frozenset] = {}

        def play_a(plan):
            assignments_a[plan.game_index] = frozenset(plan.assignment.items())
            return _make_ledger_record(game_index=plan.game_index)

        def play_b(plan):
            assignments_b[plan.game_index] = frozenset(plan.assignment.items())
            return _make_ledger_record(game_index=plan.game_index)

        run_tournament(cfg_a, ledger_a, play_fn=play_a)
        run_tournament(cfg_b, ledger_b, play_fn=play_b)

        for idx in range(3):
            assert assignments_a[idx] == assignments_b[idx], (
                f"game_index={idx}: assignments differ between runs with the same seed"
            )

    # ── Hypothesis: determinism invariant for any game_index ─────────────────

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(game_index=st.integers(min_value=0, max_value=999))
    def test_when_any_game_index_then_assignment_is_identical_on_two_calls(
        self, game_index, tmp_path
    ):
        """Invariant: build_game_plan(cfg, i).assignment is deterministic for any i in [0,999]."""
        from training.tournament import build_game_plan, load_tournament_config

        cfg_file = _write_config(tmp_path, seed=17)
        cfg = load_tournament_config(cfg_file)
        plan_a = build_game_plan(cfg, game_index=game_index)
        plan_b = build_game_plan(cfg, game_index=game_index)
        assert plan_a.assignment == plan_b.assignment


# ─────────────────────────────────────────────────────────────────────────────
# TestRLTrainPathPreserved
# Criterion: the existing RL train path is unchanged and still passes its tests.
# ─────────────────────────────────────────────────────────────────────────────


def test_when_tournament_imported_then_actor_critic_is_still_importable():
    """Importing training.tournament must not break src.models.actor_critic.ActorCritic."""
    import src.models.actor_critic  # noqa: F401
    import training.tournament  # noqa: F401

    assert hasattr(src.models.actor_critic, "ActorCritic")


def test_when_tournament_imported_then_ppo_trainer_is_still_importable():
    """Importing training.tournament must not break src.models.ppo.PPOTrainer."""
    import src.models.ppo  # noqa: F401
    import training.tournament  # noqa: F401

    assert hasattr(src.models.ppo, "PPOTrainer")


def test_when_tournament_imported_then_train_config_is_still_loadable():
    """Importing training.tournament must not break src.config.TrainingConfig."""
    import training.tournament  # noqa: F401
    from src.config import TrainingConfig  # noqa: F401

    assert TrainingConfig is not None


def test_when_tournament_imported_then_default_yaml_is_still_valid():
    """config/default.yaml must remain loadable via load_config after tournament addition."""
    import training.tournament  # noqa: F401
    from src.config import load_config

    cfg = load_config("config/default.yaml")
    assert cfg is not None


def test_when_tournament_imported_then_cli_app_is_still_importable():
    """risiko_rl.cli must remain importable and expose its entry point after tournament addition."""
    import risiko_rl.cli as cli  # noqa: F401
    import training.tournament  # noqa: F401

    assert hasattr(cli, "app") or hasattr(cli, "main")


def test_when_tournament_imported_then_checkpoint_module_is_intact():
    """src.checkpoint must remain importable; tournament must not shadow or break it."""
    import src.checkpoint  # noqa: F401
    import training.tournament  # noqa: F401

    assert hasattr(src.checkpoint, "save_checkpoint")
    assert hasattr(src.checkpoint, "load_checkpoint")
