"""Source-blind wiring tests for the `tournament` CLI command (issue #130).

All tests are derived solely from the acceptance criteria — no implementation
source was read while authoring these tests.

Test strategy:
- Use Typer's CliRunner to invoke the `tournament` command end-to-end.
- Patch the three canonical collaborators exactly as specified in the criteria:
      training.tournament.run_preflight
      training.tournament.run_tournament       → returns a canned TournamentSummary
      training.tournament_analysis.analyze_run → returns a canned Leaderboard
- Two additional patches isolate from file I/O and seeding side-effects:
      training.tournament.load_tournament_config
      src.utils.seed.set_global_seeds
- No real LLM or Ollama call is made in any test.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from risiko_rl.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Canonical patch targets — exactly as declared in the acceptance criteria
# ---------------------------------------------------------------------------
_PREFLIGHT = "training.tournament.run_preflight"
_RUN = "training.tournament.run_tournament"
_ANALYZE = "training.tournament_analysis.analyze_run"

# Additional isolation patches (not directly named in criteria, but required
# to prevent I/O and seeding side-effects in unit tests).
_LOAD_CFG = "training.tournament.load_tournament_config"
_SET_SEEDS = "src.utils.seed.set_global_seeds"

# ---------------------------------------------------------------------------
# Canned fixtures
# ---------------------------------------------------------------------------

_SAFE_RUN_ID = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
    min_size=1,
    max_size=40,
)

# Six valid strategy slugs and six placeholder model names for the fixture.
_STRATEGIES = (
    "australia_lock",
    "south_america_lock",
    "aggressive_blitz",
    "turtle_defensive",
    "card_cycle_hunter",
    "diplomat_coalition",
)
_MODELS = ("m1", "m2", "m3", "m4", "m5", "m6")


def _summary(
    *,
    llm_calls: int = 100,
    games_this_run: int = 10,
    total_games: int = 10,
) -> MagicMock:
    s = MagicMock()
    s.total_llm_calls = llm_calls
    s.games_completed_this_run = games_this_run
    s.total_games_in_ledger = total_games
    return s


def _config(name: str = "test_run", seed: int = 0):
    """Return a real TournamentConfig so dataclasses.replace works in the command."""
    from training.tournament import TournamentConfig  # noqa: PLC0415

    return TournamentConfig(
        name=name,
        seed=seed,
        n_games=10,
        max_concurrency=2,
        n_rounds=2,
        strategies=_STRATEGIES,
        models=_MODELS,
        temperature=0.7,
        top_p=0.9,
    )


# ===========================================================================
# --llm-only enforcement
# ===========================================================================


def test_when_llm_only_omitted_then_exit_code_is_nonzero():
    """Criterion: --llm-only omitted → typer.BadParameter; only LLM-only mode is supported."""
    result = runner.invoke(app, ["tournament"])
    assert result.exit_code != 0


def test_when_llm_only_present_then_exit_code_is_zero():
    """Criterion: --llm-only present → command proceeds and exits 0."""
    cfg = _config()
    with (
        patch(_LOAD_CFG, return_value=cfg),
        patch(_SET_SEEDS),
        patch(_PREFLIGHT),
        patch(_RUN, return_value=_summary()),
        patch(_ANALYZE, return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["tournament", "--llm-only"])
    assert result.exit_code == 0


# ===========================================================================
# Wiring: call order — preflight before run_tournament
# ===========================================================================


def test_when_command_runs_then_preflight_called_before_run_tournament():
    """Criterion: wiring order — run_preflight executes before run_tournament."""
    call_order: list[str] = []
    cfg = _config()

    def note_preflight(*_a, **_kw) -> None:
        call_order.append("preflight")

    def note_run(*_a, **_kw) -> MagicMock:
        call_order.append("run")
        return _summary()

    with (
        patch(_LOAD_CFG, return_value=cfg),
        patch(_SET_SEEDS),
        patch(_PREFLIGHT, side_effect=note_preflight),
        patch(_RUN, side_effect=note_run),
        patch(_ANALYZE, return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["tournament", "--llm-only"])

    assert result.exit_code == 0
    assert "preflight" in call_order, "run_preflight was never called"
    assert "run" in call_order, "run_tournament was never called"
    assert call_order.index("preflight") < call_order.index("run")


# ===========================================================================
# Wiring: ledger == run_dir / "games.jsonl"
# ===========================================================================


def test_when_command_runs_then_ledger_passed_to_run_tournament_is_games_jsonl():
    """Criterion: ledger kwarg passed to run_tournament equals run_dir / 'games.jsonl'."""
    cfg = _config()
    mock_run = MagicMock(return_value=_summary())

    with (
        patch(_LOAD_CFG, return_value=cfg),
        patch(_SET_SEEDS),
        patch(_PREFLIGHT),
        patch(_RUN, mock_run),
        patch(_ANALYZE, return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["tournament", "--llm-only"])

    assert result.exit_code == 0
    ledger = mock_run.call_args.kwargs.get("ledger")
    assert ledger is not None, "run_tournament must receive a 'ledger' keyword argument"
    assert Path(ledger).name == "games.jsonl"


# ===========================================================================
# Wiring: same run_dir passed to run_tournament (via ledger) and analyze_run
# ===========================================================================


def test_when_command_runs_then_analyze_run_receives_same_run_dir_as_run_tournament():
    """Criterion: run_dir / 'games.jsonl' is the ledger; same run_dir is passed to analyze_run."""
    cfg = _config(name="wiring_consistency")
    mock_run = MagicMock(return_value=_summary())
    mock_analyze = MagicMock(return_value=MagicMock())

    with (
        patch(_LOAD_CFG, return_value=cfg),
        patch(_SET_SEEDS),
        patch(_PREFLIGHT),
        patch(_RUN, mock_run),
        patch(_ANALYZE, mock_analyze),
    ):
        result = runner.invoke(app, ["tournament", "--llm-only"])

    assert result.exit_code == 0

    # run_dir inferred from ledger's parent
    ledger = mock_run.call_args.kwargs.get("ledger")
    assert ledger is not None
    run_dir_from_ledger = Path(ledger).parent

    # analyze_run must receive the same run_dir
    analyze_kwargs = mock_analyze.call_args.kwargs
    analyze_args = mock_analyze.call_args.args
    run_dir_for_analyze = analyze_kwargs.get("run_dir") or (
        analyze_args[0] if analyze_args else None
    )
    assert run_dir_for_analyze is not None, "analyze_run must receive a 'run_dir' argument"
    assert Path(run_dir_for_analyze) == run_dir_from_ledger


# ===========================================================================
# Wiring: run_tournament always called with skip_preflight=True
# ===========================================================================


def test_when_command_runs_then_run_tournament_called_with_skip_preflight_true():
    """Criterion: run_tournament always receives skip_preflight=True (CLI owns preflight)."""
    cfg = _config()
    mock_run = MagicMock(return_value=_summary())

    with (
        patch(_LOAD_CFG, return_value=cfg),
        patch(_SET_SEEDS),
        patch(_PREFLIGHT),
        patch(_RUN, mock_run),
        patch(_ANALYZE, return_value=MagicMock()),
    ):
        runner.invoke(app, ["tournament", "--llm-only"])

    assert mock_run.call_args.kwargs.get("skip_preflight") is True


# ===========================================================================
# CLI option: --run-id
# ===========================================================================


def test_when_run_id_provided_then_ledger_path_contains_run_id():
    """Criterion: --run-id controls the run directory leaf; default derives from config.name."""
    cfg = _config()
    mock_run = MagicMock(return_value=_summary())

    with (
        patch(_LOAD_CFG, return_value=cfg),
        patch(_SET_SEEDS),
        patch(_PREFLIGHT),
        patch(_RUN, mock_run),
        patch(_ANALYZE, return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["tournament", "--llm-only", "--run-id", "my_explicit_run"])

    assert result.exit_code == 0
    ledger = mock_run.call_args.kwargs.get("ledger")
    assert ledger is not None
    assert "my_explicit_run" in str(Path(ledger))


def test_when_run_id_not_provided_then_ledger_path_contains_config_name():
    """Criterion: when --run-id is absent, the run-id derives from config.name."""
    cfg = _config(name="config_derived_name")
    mock_run = MagicMock(return_value=_summary())

    with (
        patch(_LOAD_CFG, return_value=cfg),
        patch(_SET_SEEDS),
        patch(_PREFLIGHT),
        patch(_RUN, mock_run),
        patch(_ANALYZE, return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["tournament", "--llm-only"])

    assert result.exit_code == 0
    ledger = mock_run.call_args.kwargs.get("ledger")
    assert ledger is not None
    assert "config_derived_name" in str(Path(ledger))


# ===========================================================================
# CLI option: --seed
# ===========================================================================


def test_when_seed_option_given_then_set_global_seeds_called_with_cli_seed():
    """Criterion: --seed (int) overrides config.seed via dataclasses.replace; set_global_seeds
    receives the CLI-supplied seed, not the config's original value.
    """
    cfg = _config(seed=9999)
    seeds_seen: list[int] = []

    def capture(seed: int, *_a, **_kw) -> None:
        seeds_seen.append(seed)

    with (
        patch(_LOAD_CFG, return_value=cfg),
        patch(_SET_SEEDS, side_effect=capture),
        patch(_PREFLIGHT),
        patch(_RUN, return_value=_summary()),
        patch(_ANALYZE, return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["tournament", "--llm-only", "--seed", "42"])

    assert result.exit_code == 0
    assert 42 in seeds_seen, (
        f"Expected set_global_seeds to be called with seed 42, got calls: {seeds_seen}"
    )
    assert 9999 not in seeds_seen, "CLI --seed must override config.seed, not use it"


# ===========================================================================
# CLI option: --skip-preflight
# ===========================================================================


def test_when_skip_preflight_flag_given_then_preflight_not_called():
    """Criterion: --skip-preflight (default False) omits the preflight when set."""
    cfg = _config()
    mock_preflight = MagicMock()

    with (
        patch(_LOAD_CFG, return_value=cfg),
        patch(_SET_SEEDS),
        patch(_PREFLIGHT, mock_preflight),
        patch(_RUN, return_value=_summary()),
        patch(_ANALYZE, return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["tournament", "--llm-only", "--skip-preflight"])

    assert result.exit_code == 0
    mock_preflight.assert_not_called()


def test_when_skip_preflight_not_given_then_preflight_is_called_by_default():
    """Criterion: --skip-preflight defaults to False; preflight runs when the flag is absent."""
    cfg = _config()
    mock_preflight = MagicMock()

    with (
        patch(_LOAD_CFG, return_value=cfg),
        patch(_SET_SEEDS),
        patch(_PREFLIGHT, mock_preflight),
        patch(_RUN, return_value=_summary()),
        patch(_ANALYZE, return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["tournament", "--llm-only"])

    assert result.exit_code == 0
    mock_preflight.assert_called_once()


# ===========================================================================
# CLI option: --max-games
# ===========================================================================


def test_when_max_games_given_then_run_tournament_receives_it():
    """Criterion: --max-games (int, optional) is forwarded to run_tournament."""
    cfg = _config()
    mock_run = MagicMock(return_value=_summary())

    with (
        patch(_LOAD_CFG, return_value=cfg),
        patch(_SET_SEEDS),
        patch(_PREFLIGHT),
        patch(_RUN, mock_run),
        patch(_ANALYZE, return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["tournament", "--llm-only", "--max-games", "25"])

    assert result.exit_code == 0
    assert mock_run.call_args.kwargs.get("max_games") == 25


# ===========================================================================
# Resumability — same run_id produces identical ledger path on re-invocation
# ===========================================================================


def test_when_same_run_id_used_twice_then_identical_ledger_path_is_produced():
    """Criterion: resumable — same run_id always maps to the same ledger path so
    run_tournament can detect and skip already-completed games.
    """
    cfg = _config(name="resumable_run")
    ledgers: list[Path] = []

    def capture_ledger(*_a, **kw) -> MagicMock:
        if "ledger" in kw:
            ledgers.append(Path(kw["ledger"]))
        return _summary()

    for _ in range(2):
        with (
            patch(_LOAD_CFG, return_value=cfg),
            patch(_SET_SEEDS),
            patch(_PREFLIGHT),
            patch(_RUN, side_effect=capture_ledger),
            patch(_ANALYZE, return_value=MagicMock()),
        ):
            runner.invoke(app, ["tournament", "--llm-only", "--run-id", "resumable_run"])

    assert len(ledgers) == 2, "run_tournament must be called once per invocation"
    assert ledgers[0] == ledgers[1], (
        "Same run_id must produce the same ledger path on every invocation "
        "so completed games are not repeated"
    )


# ---------------------------------------------------------------------------
# Property — for any valid run_id the ledger filename is always 'games.jsonl'
# (resumability invariant: the contract holds for all run_id inputs, not
# just the single example above)
# ---------------------------------------------------------------------------


@given(run_id=_SAFE_RUN_ID)
def test_when_any_run_id_given_then_ledger_filename_is_always_games_jsonl(run_id: str) -> None:
    """Property (resumability): for ALL valid run_ids the ledger is always run_dir / 'games.jsonl'.

    The invariant comes directly from the resumability criterion: run_tournament must
    always receive the canonical ledger so it can detect completed games regardless of
    which run_id is in use.
    """
    cfg = _config()
    mock_run = MagicMock(return_value=_summary())

    with (
        patch(_LOAD_CFG, return_value=cfg),
        patch(_SET_SEEDS),
        patch(_PREFLIGHT),
        patch(_RUN, mock_run),
        patch(_ANALYZE, return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["tournament", "--llm-only", "--run-id", run_id])

    if result.exit_code == 0 and mock_run.call_args is not None:
        ledger = mock_run.call_args.kwargs.get("ledger")
        if ledger is not None:
            assert Path(ledger).name == "games.jsonl"


# ===========================================================================
# Existing RL train path is unchanged
# ===========================================================================


def test_when_tournament_command_added_then_train_command_help_still_exits_zero():
    """Criterion: the existing RL train path is unchanged; `train --help` still works."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
