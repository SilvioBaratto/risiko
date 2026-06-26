"""
Tests for issue #119 — preflight model verification: abort before any game.

Source-blind: authored against acceptance criteria only. Red phase of TDD.

Assumed interface (inferred from spec, not from reading src/):
  training.tournament.run_preflight(required_models)
      Verifies models are available via list_ollama_models.
      Raises if any are missing (naming them); logs WARNING and proceeds
      if list_ollama_models returns None.
  training.tournament.TournamentRunner(run_dir, max_concurrency)
      Parallel, resumable tournament driver. Exposes .max_concurrency.
  training.tournament.load_ledger(path) -> set[int]
      Parses a games.jsonl ledger and returns the set of completed game indices.
  training.tournament.list_ollama_models
      The local reference (imported from src.agents.ollama_client) that we
      patch to control which models appear "available".
"""

import json
import logging
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Patch target: list_ollama_models as imported inside the tournament module.
_LIST_FN = "training.tournament.list_ollama_models"

# Six-model roster defined in spec (config/tournament.yaml).
FULL_ROSTER = [
    "glm-5.2:cloud",
    "minimax-m3:cloud",
    "glm-5.1:cloud",
    "kimi-k2.6:cloud",
    "deepseek-v4-flash:cloud",
    "qwen3.5:cloud",
]


# ══════════════════════════════════════════════════════════════════════════════
# Scope criterion — superset passes silently
# "With list_ollama_models patched to return a superset, preflight passes
#  silently"
# ══════════════════════════════════════════════════════════════════════════════


def test_when_available_models_are_superset_then_preflight_passes_silently(caplog):
    """
    With list_ollama_models returning a superset of required models, run_preflight
    must not raise and must emit no WARNING-or-above log records.
    """
    from training.tournament import run_preflight

    superset = FULL_ROSTER + ["bonus-model:cloud"]
    with patch(_LIST_FN, return_value=superset), caplog.at_level(logging.WARNING):
        run_preflight(FULL_ROSTER)  # must not raise

    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_when_available_models_exactly_match_required_then_preflight_passes_silently(
    caplog,
):
    """
    Edge case: exact match (not just a superset) also passes silently.
    """
    from training.tournament import run_preflight

    with patch(_LIST_FN, return_value=list(FULL_ROSTER)), caplog.at_level(logging.WARNING):
        run_preflight(list(FULL_ROSTER))  # must not raise

    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


# ══════════════════════════════════════════════════════════════════════════════
# Scope criterion — missing model raises before any game, naming the model(s)
# "With a model omitted, preflight raises before any game, naming the missing
#  model(s), message format consistent with the CLI's"
# ══════════════════════════════════════════════════════════════════════════════


def test_when_one_model_is_missing_then_preflight_raises():
    """
    With one model absent from the available list, run_preflight must raise.
    """
    from training.tournament import run_preflight

    available = [m for m in FULL_ROSTER if m != "minimax-m3:cloud"]
    with patch(_LIST_FN, return_value=available), pytest.raises(RuntimeError):
        run_preflight(FULL_ROSTER)


def test_when_one_model_is_missing_then_error_names_that_model():
    """
    The raised exception's string representation must contain the missing model
    name (consistent with the CLI's error-message format).
    """
    from training.tournament import run_preflight

    available = [m for m in FULL_ROSTER if m != "minimax-m3:cloud"]
    with patch(_LIST_FN, return_value=available), pytest.raises(RuntimeError) as exc_info:
        run_preflight(FULL_ROSTER)

    assert "minimax-m3:cloud" in str(exc_info.value)


def test_when_multiple_models_are_missing_then_all_are_named_in_the_error():
    """
    All missing models must appear in the error, not just the first one.
    """
    from training.tournament import run_preflight

    absent = {"minimax-m3:cloud", "deepseek-v4-flash:cloud"}
    available = [m for m in FULL_ROSTER if m not in absent]
    with patch(_LIST_FN, return_value=available), pytest.raises(RuntimeError) as exc_info:
        run_preflight(FULL_ROSTER)

    error = str(exc_info.value)
    for model in absent:
        assert model in error, f"{model!r} not named in error: {error!r}"


def test_when_model_is_missing_then_no_game_is_dispatched_before_the_raise():
    """
    Criterion: preflight raises **before any game** starts.
    We intercept the game-dispatch primitive and confirm zero calls were made
    before the exception propagated.
    """
    from training.tournament import run_preflight

    available = [m for m in FULL_ROSTER if m != "minimax-m3:cloud"]
    game_calls: list[int] = []

    def spy_game(*args, **kwargs):
        game_calls.append(1)

    with (
        patch(_LIST_FN, return_value=available),
        patch("training.tournament.run_game", side_effect=spy_game),
        pytest.raises(RuntimeError),
    ):
        run_preflight(FULL_ROSTER)

    assert game_calls == [], "run_game must not be called when preflight raises"


# ══════════════════════════════════════════════════════════════════════════════
# Scope criterion — None from list_ollama_models → warn and proceed (no raise)
# "With list_ollama_models returning None, preflight logs a warning and
#  proceeds (no raise)"
# ══════════════════════════════════════════════════════════════════════════════


def test_when_list_ollama_models_returns_none_then_preflight_does_not_raise():
    """
    None return value (unavailable / unreachable model registry) must not abort
    the run — preflight proceeds.
    """
    from training.tournament import run_preflight

    with patch(_LIST_FN, return_value=None):
        run_preflight(FULL_ROSTER)  # must not raise


def test_when_list_ollama_models_returns_none_then_a_warning_is_logged(caplog):
    """
    Proceeding without model verification is dangerous; at least one WARNING must
    be emitted so operators see it.
    """
    from training.tournament import run_preflight

    with patch(_LIST_FN, return_value=None), caplog.at_level(logging.WARNING):
        run_preflight(FULL_ROSTER)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# Property — any non-empty subset of missing models always triggers the error
# Invariant class: "always-raises-for-invalid-input" over the roster domain.
# ══════════════════════════════════════════════════════════════════════════════


@settings(max_examples=60)
@given(
    missing=st.lists(
        st.sampled_from(FULL_ROSTER),
        min_size=1,
        max_size=len(FULL_ROSTER) - 1,
        unique=True,
    )
)
def test_when_any_model_subset_is_absent_then_preflight_always_raises_naming_each(
    missing,
):
    """
    Invariant: for every non-empty subset of absent models, run_preflight must
    (a) raise an exception and (b) name every absent model in the error message.

    Derived from: "preflight raises naming the missing model(s)" — the 'any'
    quantifier over the roster implies an invariant, not just a single example.
    """
    from training.tournament import run_preflight

    available = [m for m in FULL_ROSTER if m not in missing]
    with patch(_LIST_FN, return_value=available), pytest.raises(RuntimeError) as exc_info:
        run_preflight(FULL_ROSTER)

    error = str(exc_info.value)
    for model in missing:
        assert model in error, f"missing model {model!r} absent from error: {error!r}"


# ══════════════════════════════════════════════════════════════════════════════
# Global criterion — existing RL train path is unchanged
# "The existing RL train path is unchanged and still passes its tests."
# ══════════════════════════════════════════════════════════════════════════════


def test_when_self_play_module_is_imported_then_it_remains_importable():
    """
    training.self_play must still exist and be importable (not removed / renamed).
    """
    import training.self_play  # noqa: F401


def test_when_self_play_trainer_class_is_accessed_then_it_still_exists():
    """
    SelfPlayTrainer must remain accessible in training.self_play.
    """
    from training.self_play import SelfPlayTrainer  # noqa: F401


def test_when_evaluate_module_is_imported_then_it_remains_importable():
    """
    training.evaluate must still exist (RL evaluation path intact).
    """
    import training.evaluate  # noqa: F401


# ══════════════════════════════════════════════════════════════════════════════
# Global criterion — configurable max_concurrency
# "Games execute concurrently (configurable max_concurrency)"
# ══════════════════════════════════════════════════════════════════════════════


def test_when_max_concurrency_is_provided_then_tournament_runner_accepts_it(tmp_path):
    """
    TournamentRunner must accept a max_concurrency keyword argument.
    """
    from training.tournament import TournamentRunner

    runner = TournamentRunner(run_dir=str(tmp_path), max_concurrency=3)
    assert runner is not None


def test_when_max_concurrency_is_set_then_runner_exposes_it(tmp_path):
    """
    The configured value must be readable back from the runner instance so
    callers (and tests) can inspect the active concurrency level.
    """
    from training.tournament import TournamentRunner

    runner = TournamentRunner(run_dir=str(tmp_path), max_concurrency=5)
    assert runner.max_concurrency == 5


# ══════════════════════════════════════════════════════════════════════════════
# Global criterion — resumable ledger skips completed games
# "Re-invoking with the same run id skips completed games (ledger-based)"
# ══════════════════════════════════════════════════════════════════════════════


def test_when_game_indices_are_in_ledger_then_load_ledger_returns_them(tmp_path):
    """
    load_ledger must parse a games.jsonl file and return the set of completed
    game indices, so the tournament driver can skip those games on resume.
    """
    from training.tournament import load_ledger

    ledger_file = tmp_path / "games.jsonl"
    ledger_file.write_text(
        json.dumps({"game_index": 0, "winner": "model-a"})
        + "\n"
        + json.dumps({"game_index": 2, "winner": "model-b"})
        + "\n"
    )

    completed = load_ledger(str(ledger_file))

    assert 0 in completed
    assert 2 in completed


def test_when_game_index_is_not_in_ledger_then_it_is_not_in_completed_set(tmp_path):
    """
    Indices absent from the ledger must not appear in the completed set —
    they should run on the next invocation.
    """
    from training.tournament import load_ledger

    ledger_file = tmp_path / "games.jsonl"
    ledger_file.write_text(json.dumps({"game_index": 0, "winner": "model-a"}) + "\n")

    completed = load_ledger(str(ledger_file))

    assert 1 not in completed


def test_when_ledger_file_is_empty_then_completed_set_is_empty(tmp_path):
    """
    A fresh run (empty ledger) must report zero completed games so nothing is
    skipped and all N games are executed.
    """
    from training.tournament import load_ledger

    ledger_file = tmp_path / "games.jsonl"
    ledger_file.write_text("")

    completed = load_ledger(str(ledger_file))

    assert len(completed) == 0
