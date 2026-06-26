"""
End-to-end smoke and resumability tests for ``risiko-rl tournament --llm-only``.

Strategy
--------
* ``monkeypatch.chdir(tmp_path)`` — resolves ``results/tournament/`` under the
  temp dir so no artefacts land in the real repo tree.
* Patch ``training.tournament.load_tournament_config`` — returns a tiny
  TournamentConfig (n_games=2, max_concurrency=1) so tests never read the
  real config/tournament.yaml and always control game count.
* Patch ``training.tournament._play_one_game`` — returns a canned LedgerRecord
  without any real game play or LLM calls.  ``run_tournament``'s internal
  lambda references ``_play_one_game`` as a module global, so the patch takes
  effect exactly as described in the issue comment.
* Pass ``--skip-preflight`` — the CLI skips the Ollama model-availability probe
  so no HTTP goes to the Ollama server.

Tests are NOT marked ``integration`` and run under ``pytest -m "not integration"``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from risiko_rl.cli import app
from training.tournament import (
    GamePlan,
    LedgerRecord,
    TournamentConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_GAMES = 2
_SEED = 42

_STRATEGIES = (
    "australia_lock",
    "south_america_lock",
    "aggressive_blitz",
    "turtle_defensive",
    "card_cycle_hunter",
    "diplomat_coalition",
)
_MODELS = ("m1", "m2", "m3", "m4", "m5", "m6")

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def cli():
    return CliRunner()


def _make_config(n_games: int = _N_GAMES) -> TournamentConfig:
    """Return a minimal TournamentConfig for smoke tests."""
    return TournamentConfig(
        name="smoke_test",
        seed=_SEED,
        n_games=n_games,
        max_concurrency=1,
        n_rounds=1,
        strategies=_STRATEGIES,
        models=_MODELS,
        temperature=0.7,
        top_p=0.9,
    )


def _canned_record(plan: GamePlan, config: object = None, book: object = None) -> LedgerRecord:
    """Return a valid LedgerRecord without real game play or LLM calls."""
    return LedgerRecord(
        game_index=plan.game_index,
        seed=plan.seed,
        winner=0,
        winner_strategy=_STRATEGIES[0],
        winner_model=_MODELS[0],
        n_turns=10,
        elimination_order=[5, 4, 3, 2, 1],
        seat_strategies={str(i): _STRATEGIES[i] for i in range(6)},
        seat_models={str(i): _MODELS[i] for i in range(6)},
        assignment=dict(zip(_STRATEGIES, _MODELS, strict=True)),
        card_trade_turns=[],
        llm_call_count=5,
    )


def _invoke_tournament(
    cli: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_id: str,
    *,
    n_games: int = _N_GAMES,
    seed: int = _SEED,
) -> object:
    """Run the tournament CLI with all external collaborators stubbed out."""
    monkeypatch.chdir(tmp_path)
    with (
        patch("training.tournament.load_tournament_config", return_value=_make_config(n_games)),
        patch("training.tournament._play_one_game", side_effect=_canned_record),
    ):
        return cli.invoke(
            app,
            [
                "tournament",
                "--llm-only",
                "--max-games",
                str(n_games),
                "--run-id",
                run_id,
                "--seed",
                str(seed),
                "--skip-preflight",
            ],
            catch_exceptions=False,
        )


def _run_dir(tmp_path: Path, run_id: str) -> Path:
    return tmp_path / "results" / "tournament" / run_id


def _ledger_records(tmp_path: Path, run_id: str) -> list[dict]:
    ledger = _run_dir(tmp_path, run_id) / "games.jsonl"
    return [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Smoke: exit 0 and all required output artefacts produced
# ---------------------------------------------------------------------------


class TestTournamentCLISmoke:
    """Criterion: exit 0, games.jsonl + leaderboard.json + report.md produced."""

    def test_when_tournament_invoked_with_llm_only_then_exit_code_is_0(
        self, cli: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = _invoke_tournament(cli, monkeypatch, tmp_path, run_id="smoke-exit")
        assert result.exit_code == 0, result.output

    def test_when_tournament_completes_then_games_jsonl_exists(
        self, cli: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _invoke_tournament(cli, monkeypatch, tmp_path, run_id="smoke-jsonl")
        assert (_run_dir(tmp_path, "smoke-jsonl") / "games.jsonl").exists()

    def test_when_tournament_completes_then_leaderboard_json_exists(
        self, cli: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _invoke_tournament(cli, monkeypatch, tmp_path, run_id="smoke-lb")
        assert (_run_dir(tmp_path, "smoke-lb") / "leaderboard.json").exists()

    def test_when_tournament_completes_then_report_md_exists(
        self, cli: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _invoke_tournament(cli, monkeypatch, tmp_path, run_id="smoke-rpt")
        assert (_run_dir(tmp_path, "smoke-rpt") / "report.md").exists()

    def test_when_n_games_specified_then_ledger_has_exactly_n_records(
        self, cli: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _invoke_tournament(cli, monkeypatch, tmp_path, run_id="smoke-count", n_games=_N_GAMES)
        assert len(_ledger_records(tmp_path, "smoke-count")) == _N_GAMES


# ---------------------------------------------------------------------------
# No real LLM / network calls
# ---------------------------------------------------------------------------


class TestNoRealLLMCalls:
    """Criterion: no real HTTP reaches the Ollama server."""

    def test_when_play_is_patched_then_no_real_http_escapes(
        self, cli: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """
        With _play_one_game patched and --skip-preflight set, no Ollama HTTP call
        should be attempted.  We confirm this by additionally blocking all httpx
        requests — the command still exits 0, proving no real HTTP went out.
        """
        import httpx  # noqa: PLC0415

        def _fail(*_a: object, **_kw: object) -> None:
            raise AssertionError("Real HTTP call reached httpx — LLM layer was not patched")

        monkeypatch.chdir(tmp_path)
        with (
            patch("training.tournament.load_tournament_config", return_value=_make_config()),
            patch("training.tournament._play_one_game", side_effect=_canned_record),
            patch.object(httpx.Client, "request", side_effect=_fail),
        ):
            result = cli.invoke(
                app,
                [
                    "tournament",
                    "--llm-only",
                    "--max-games",
                    str(_N_GAMES),
                    "--run-id",
                    "no-http",
                    "--seed",
                    str(_SEED),
                    "--skip-preflight",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Resumability: same --run-id skips completed games
# ---------------------------------------------------------------------------


class TestTournamentCLIResumability:
    """Criterion: re-invoking with the same --run-id adds 0 new records."""

    def test_when_resumed_with_same_run_id_then_record_count_is_unchanged(
        self, cli: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Second invocation must complete 0 new games; ledger stays at N records."""
        run_id = "resume-count"
        _invoke_tournament(cli, monkeypatch, tmp_path, run_id=run_id, n_games=_N_GAMES)
        assert len(_ledger_records(tmp_path, run_id)) == _N_GAMES

        _invoke_tournament(cli, monkeypatch, tmp_path, run_id=run_id, n_games=_N_GAMES)
        records_after_second = _ledger_records(tmp_path, run_id)
        assert len(records_after_second) == _N_GAMES, (
            "Second run must not append new records to an already-complete ledger"
        )

    def test_when_resumed_with_same_run_id_then_game_indices_have_no_duplicates(
        self, cli: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_id = "resume-dedup"
        _invoke_tournament(cli, monkeypatch, tmp_path, run_id=run_id, n_games=_N_GAMES)
        _invoke_tournament(cli, monkeypatch, tmp_path, run_id=run_id, n_games=_N_GAMES)

        indices = [r["game_index"] for r in _ledger_records(tmp_path, run_id)]
        assert len(indices) == len(set(indices)), (
            f"Duplicate game_index values in ledger after resume: {indices}"
        )


# ---------------------------------------------------------------------------
# Output logging: seed, model roster, cumulative LLM-call total
# ---------------------------------------------------------------------------


class TestTournamentOutputLogging:
    """Criterion: seed + model roster + cumulative LLM-call total in result.output.

    Typer's CliRunner mixes stderr (where setup_logging() routes INFO messages)
    into result.output by default, so log lines are assertable alongside
    the typer.echo lines.
    """

    def test_when_tournament_runs_then_seed_appears_in_output(
        self, cli: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = _invoke_tournament(cli, monkeypatch, tmp_path, run_id="log-seed", seed=_SEED)
        assert str(_SEED) in result.output, (
            f"Seed {_SEED!r} not found in output:\n{result.output[:600]}"
        )

    def test_when_tournament_runs_then_at_least_one_model_appears_in_output(
        self, cli: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """log_run_header logs the full roster; we assert ≥1 model name is present."""
        result = _invoke_tournament(cli, monkeypatch, tmp_path, run_id="log-roster")
        assert any(m in result.output for m in _MODELS), (
            f"No model name found in output:\n{result.output[:600]}"
        )

    def test_when_tournament_runs_then_llm_call_total_appears_in_output(
        self, cli: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """_echo_tournament_summary prints 'total LLM calls: N' to stdout."""
        result = _invoke_tournament(cli, monkeypatch, tmp_path, run_id="log-calls")
        pattern = re.compile(r"llm.{0,30}\d+|\d+.{0,30}llm", re.IGNORECASE)
        assert pattern.search(result.output), (
            f"LLM call count not found in output:\n{result.output[:600]}"
        )


# ---------------------------------------------------------------------------
# RL train path unchanged (CLI-level guard)
# ---------------------------------------------------------------------------


class TestRlTrainPathUnchangedCLI:
    """Criterion: the RL train command still exists and responds to --help."""

    def test_when_train_help_invoked_then_exit_code_is_0(self, cli: CliRunner) -> None:
        """--help is always safe and exits 0 for a valid Typer/Click command."""
        result = cli.invoke(app, ["train", "--help"])
        assert result.exit_code == 0, (
            f"`train --help` exited {result.exit_code}:\n{result.output[:400]}"
        )


# ---------------------------------------------------------------------------
# Property-based: game_index uniqueness is an ordering invariant for any N
# ---------------------------------------------------------------------------


@given(n_games=st.integers(min_value=1, max_value=3))
@settings(max_examples=3, deadline=None)
def test_when_any_n_games_run_then_game_indices_are_unique(n_games: int) -> None:
    """
    Ordering / uniqueness invariant: for any N in [1, 3] the game_index
    values in games.jsonl are N distinct non-negative integers.

    Derived from the resumability criterion: 'ledger contains exactly N records
    with no duplicate game_index'.  Hypothesis checks multiple N values so a
    single hand-picked example cannot hide an off-by-one or duplication bug.
    """
    run_id = f"prop-n{n_games}"
    orig_cwd = os.getcwd()
    cli = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch("training.tournament.load_tournament_config", return_value=_make_config(n_games)),
            patch("training.tournament._play_one_game", side_effect=_canned_record),
        ):
            try:
                os.chdir(tmpdir)
                result = cli.invoke(
                    app,
                    [
                        "tournament",
                        "--llm-only",
                        "--max-games",
                        str(n_games),
                        "--run-id",
                        run_id,
                        "--seed",
                        str(_SEED),
                        "--skip-preflight",
                    ],
                    catch_exceptions=False,
                )
            finally:
                os.chdir(orig_cwd)

        assert result.exit_code == 0, result.output

        ledger = Path(tmpdir) / "results" / "tournament" / run_id / "games.jsonl"
        records = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        assert len(records) == n_games, f"Expected {n_games} records, got {len(records)}"
        indices = [r["game_index"] for r in records]
        assert len(indices) == len(set(indices)), (
            f"Duplicate game_index values for n_games={n_games}: {indices}"
        )
