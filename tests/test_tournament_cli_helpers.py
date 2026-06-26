"""Source-blind example tests for training/tournament_cli.py helpers (issue #129).

All tests are derived solely from the acceptance criteria — no implementation
source was read while authoring these tests.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from hypothesis import given
from hypothesis import strategies as st

from training.tournament_cli import (
    build_run_id,
    ledger_path,
    log_run_header,
    log_run_summary,
    tournament_run_dir,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAFE_TEXT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
    min_size=1,
    max_size=40,
)


def _make_config(
    name: str = "test_tournament",
    models: tuple[str, ...] = ("m1", "m2", "m3", "m4", "m5", "m6"),
) -> MagicMock:
    cfg = MagicMock()
    cfg.name = name
    cfg.models = models
    return cfg


def _make_summary(
    *,
    total_llm_calls: int = 120,
    games_completed_this_run: int = 5,
    total_games_in_ledger: int = 10,
) -> MagicMock:
    s = MagicMock()
    s.total_llm_calls = total_llm_calls
    s.games_completed_this_run = games_completed_this_run
    s.total_games_in_ledger = total_games_in_ledger
    return s


def _all_logged_text(log: MagicMock) -> str:
    """Flatten every positional and keyword argument from all mock calls to one string."""
    parts: list[str] = []
    for call in log.mock_calls:
        for arg in call.args:
            parts.append(str(arg))
        for val in call.kwargs.values():
            parts.append(str(val))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# build_run_id — passthrough when run_id is given
# ---------------------------------------------------------------------------


def test_when_run_id_given_then_build_run_id_returns_it_verbatim():
    """Criterion: returns run_id verbatim when given."""
    result = build_run_id("my_tournament", "custom_run_42")
    assert result == "custom_run_42"


def test_when_run_id_given_and_now_injected_then_run_id_wins_over_timestamp():
    """Criterion: an explicit run_id is returned unchanged even when now is injected."""
    now = datetime(2025, 1, 15, 10, 30, 0)
    result = build_run_id("my_tournament", "explicit_id", now=now)
    assert result == "explicit_id"


@given(run_id=_SAFE_TEXT)
def test_when_any_non_none_run_id_given_then_build_run_id_returns_it_verbatim(run_id: str) -> None:
    """Property: passthrough holds for ANY non-None run_id value."""
    result = build_run_id("any_config", run_id)
    assert result == run_id


# ---------------------------------------------------------------------------
# build_run_id — deterministic fallback with injected now
# ---------------------------------------------------------------------------


def test_when_run_id_is_none_and_now_injected_then_result_contains_config_name():
    """Criterion: deterministic id includes config_name when run_id is None."""
    now = datetime(2025, 1, 15, 10, 30, 0)
    result = build_run_id("llm_strategy_tournament", None, now=now)
    assert "llm_strategy_tournament" in result


def test_when_run_id_is_none_and_now_injected_then_result_contains_formatted_timestamp():
    """Criterion: deterministic id contains the timestamp in YYYYMMDD_HHMMSS format."""
    now = datetime(2025, 1, 15, 10, 30, 0)
    result = build_run_id("my_tournament", None, now=now)
    assert "20250115_103000" in result


def test_when_same_now_injected_twice_then_build_run_id_is_idempotent():
    """Criterion: injectable now makes tests deterministic — same inputs yield same output."""
    now = datetime(2024, 6, 15, 8, 0, 0)
    assert build_run_id("tournament", None, now=now) == build_run_id("tournament", None, now=now)


def test_when_different_now_injected_then_build_run_id_returns_different_values():
    """Criterion: different timestamps produce different run-ids."""
    now_a = datetime(2024, 6, 15, 8, 0, 0)
    now_b = datetime(2024, 6, 15, 9, 0, 0)
    assert build_run_id("tournament", None, now=now_a) != build_run_id(
        "tournament", None, now=now_b
    )


# ---------------------------------------------------------------------------
# tournament_run_dir — example tests
# ---------------------------------------------------------------------------


def test_when_run_id_given_then_tournament_run_dir_uses_default_base():
    """Criterion: tournament_run_dir(run_id) returns results/tournament/<run_id>."""
    result = tournament_run_dir("my_run")
    assert result == Path("results/tournament") / "my_run"


def test_when_run_id_and_custom_base_given_then_tournament_run_dir_returns_base_slash_run_id():
    """Criterion: tournament_run_dir(run_id, base=X) returns X / run_id."""
    custom_base = Path("/tmp/custom_base")
    result = tournament_run_dir("my_run", base=custom_base)
    assert result == custom_base / "my_run"


def test_when_run_id_given_then_tournament_run_dir_leaf_is_run_id():
    """Criterion: the terminal component of the returned path is run_id."""
    result = tournament_run_dir("unique_run_name")
    assert result.name == "unique_run_name"


# ---------------------------------------------------------------------------
# tournament_run_dir — property-based
# ---------------------------------------------------------------------------


@given(run_id=_SAFE_TEXT)
def test_when_any_run_id_then_tournament_run_dir_leaf_is_always_run_id(run_id: str) -> None:
    """Property: the last path component is always run_id regardless of input."""
    assert tournament_run_dir(run_id).name == run_id


@given(run_id=_SAFE_TEXT)
def test_when_any_run_id_and_default_base_then_result_equals_base_slash_run_id(run_id: str) -> None:
    """Property: result == Path('results/tournament') / run_id for all run_id values."""
    assert tournament_run_dir(run_id) == Path("results/tournament") / run_id


@given(
    run_id=_SAFE_TEXT,
    base_parts=st.lists(_SAFE_TEXT, min_size=1, max_size=4),
)
def test_when_any_run_id_and_custom_base_then_result_equals_base_slash_run_id(
    run_id: str,
    base_parts: list[str],
) -> None:
    """Property: result == base / run_id for any base and run_id."""
    base = Path("/root") / Path(*base_parts)
    assert tournament_run_dir(run_id, base=base) == base / run_id


# ---------------------------------------------------------------------------
# ledger_path — example tests
# ---------------------------------------------------------------------------


def test_when_run_dir_given_then_ledger_path_returns_games_jsonl_under_run_dir():
    """Criterion: ledger_path(run_dir) returns run_dir / 'games.jsonl'."""
    run_dir = Path("/some/run/dir")
    assert ledger_path(run_dir) == run_dir / "games.jsonl"


def test_when_run_dir_given_then_ledger_path_filename_is_games_jsonl():
    """Criterion: the file name is exactly 'games.jsonl'."""
    assert ledger_path(Path("/results/tournament/my_run")).name == "games.jsonl"


def test_when_run_dir_given_then_ledger_path_parent_equals_run_dir():
    """Criterion: ledger is a direct child of run_dir (not nested further)."""
    run_dir = Path("/results/tournament/my_run")
    assert ledger_path(run_dir).parent == run_dir


# ---------------------------------------------------------------------------
# ledger_path — property-based
# ---------------------------------------------------------------------------


@given(
    parts=st.lists(_SAFE_TEXT, min_size=1, max_size=5),
)
def test_when_any_run_dir_then_ledger_path_is_always_run_dir_slash_games_jsonl(
    parts: list[str],
) -> None:
    """Property: ledger_path always returns run_dir / 'games.jsonl' for any run_dir."""
    run_dir = Path("/base") / Path(*parts)
    assert ledger_path(run_dir) == run_dir / "games.jsonl"


# ---------------------------------------------------------------------------
# log_run_header — seed, model roster, run_dir are all logged
# ---------------------------------------------------------------------------


def test_when_log_run_header_called_then_seed_value_appears_in_log():
    """Criterion: log_run_header logs the seed."""
    log = MagicMock()
    config = _make_config(models=("ma", "mb", "mc", "md", "me", "mf"))
    log_run_header(log, config, seed=99, run_dir=Path("results/tournament/r"))
    assert "99" in _all_logged_text(log)


def test_when_log_run_header_called_then_every_model_name_appears_in_log():
    """Criterion: log_run_header logs the full model roster."""
    log = MagicMock()
    models = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
    config = _make_config(models=models)
    log_run_header(log, config, seed=0, run_dir=Path("results/tournament/r"))
    text = _all_logged_text(log)
    for model in models:
        assert model in text, f"Model '{model}' missing from logged output"


def test_when_log_run_header_called_then_run_dir_appears_in_log():
    """Criterion: log_run_header logs the run_dir."""
    log = MagicMock()
    config = _make_config()
    run_dir = Path("results/tournament/unique_run_xyz_123")
    log_run_header(log, config, seed=7, run_dir=run_dir)
    assert "unique_run_xyz_123" in _all_logged_text(log)


def test_when_log_run_header_called_then_log_receives_at_least_one_call():
    """Criterion: log_run_header emits at least one log call."""
    log = MagicMock()
    config = _make_config()
    log_run_header(log, config, seed=42, run_dir=Path("results/tournament/r"))
    assert len(log.mock_calls) > 0


# ---------------------------------------------------------------------------
# log_run_summary — total_llm_calls, games_completed_this_run, total_games_in_ledger
# ---------------------------------------------------------------------------


def test_when_log_run_summary_called_then_total_llm_calls_appears_in_log():
    """Criterion: log_run_summary logs total_llm_calls."""
    log = MagicMock()
    summary = _make_summary(
        total_llm_calls=9001, games_completed_this_run=3, total_games_in_ledger=5
    )
    log_run_summary(log, summary)
    assert "9001" in _all_logged_text(log)


def test_when_log_run_summary_called_then_games_completed_this_run_appears_in_log():
    """Criterion: log_run_summary logs games_completed_this_run."""
    log = MagicMock()
    summary = _make_summary(
        total_llm_calls=10, games_completed_this_run=7777, total_games_in_ledger=20
    )
    log_run_summary(log, summary)
    assert "7777" in _all_logged_text(log)


def test_when_log_run_summary_called_then_total_games_in_ledger_appears_in_log():
    """Criterion: log_run_summary logs total_games_in_ledger."""
    log = MagicMock()
    summary = _make_summary(
        total_llm_calls=10, games_completed_this_run=5, total_games_in_ledger=8888
    )
    log_run_summary(log, summary)
    assert "8888" in _all_logged_text(log)


def test_when_log_run_summary_called_then_log_receives_at_least_one_call():
    """Criterion: log_run_summary emits at least one log call."""
    log = MagicMock()
    log_run_summary(log, _make_summary())
    assert len(log.mock_calls) > 0
