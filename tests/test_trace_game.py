"""Tests for the single-game tracer.

The property that matters is the one a crash exposes: a traced game runs for tens of
minutes of paid LLM calls, so the trace must reach disk *while the game is running*,
not only when it returns.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import visualization.trace_game as tg

CONFIG = Path("config/tournament.yaml")

pytestmark = pytest.mark.skipif(not CONFIG.is_file(), reason="tournament config not on disk")


@pytest.fixture
def mocked_llm():
    """Answer every LLM call locally: action index 0, no negotiation."""
    with (
        patch("src.agents.ollama_client.call_ollama_for_action_index", return_value=0),
        patch("src.agents.ollama_client.call_ollama_for_negotiation", return_value=None),
    ):
        yield


def test_trace_records_calls_board_and_result(mocked_llm, tmp_path: Path) -> None:
    trace = tg.trace_one_game(CONFIG, game_index=0, max_turns=4)

    assert trace["n_snapshots"] == len(trace["board_snapshots"]) > 0
    assert trace["result"]["game_index"] == 0
    assert len(trace["assignment"]) == 6
    json.dumps(trace)  # must be serialisable end to end


def test_think_raises_the_action_timeout(mocked_llm) -> None:
    """A deliberating model needs longer than the tournament's per-action cap.

    Without this, the watchdog scores deliberation as a timeout and plays a random move
    — the one move a trace exists to avoid capturing.
    """
    plain = tg.trace_one_game(CONFIG, game_index=0, max_turns=2)
    thinking = tg.trace_one_game(CONFIG, game_index=0, max_turns=2, think=True)

    assert thinking["think"] is True
    assert thinking["action_timeout"] == tg.THINK_ACTION_TIMEOUT
    assert thinking["action_timeout"] > plain["action_timeout"]


def test_explicit_action_timeout_wins_over_the_think_default(mocked_llm) -> None:
    trace = tg.trace_one_game(CONFIG, game_index=0, max_turns=2, think=True, action_timeout=42)
    assert trace["action_timeout"] == 42.0


def test_progress_file_is_written_and_is_marked_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flush on a timer, so a crash never throws away an hour of paid LLM calls.

    The LLM is mocked, so the game would otherwise finish in ~8ms — faster than any
    flush tick. Each move sleeps a little to give the flusher a game to catch.
    """
    monkeypatch.setattr(tg, "PROGRESS_FLUSH_SECONDS", 0.005)
    progress = tmp_path / "partial.json"

    def slow_move(*args, **kwargs) -> int:
        time.sleep(0.005)
        return 0

    with (
        patch("src.agents.ollama_client.call_ollama_for_action_index", side_effect=slow_move),
        patch("src.agents.ollama_client.call_ollama_for_negotiation", return_value=None),
    ):
        trace = tg.trace_one_game(CONFIG, game_index=0, max_turns=6, progress_path=progress)

    assert progress.is_file(), "no partial trace was ever flushed"
    partial = json.loads(progress.read_text())  # must be valid JSON, never half-written
    assert partial["partial"] is True
    assert partial["result"] is None
    assert partial["n_snapshots"] <= trace["n_snapshots"]
    # The finished trace is not marked partial — that is how a reader tells them apart.
    assert "partial" not in trace


def test_progress_file_survives_an_aborted_game(tmp_path: Path) -> None:
    """If the game blows up mid-way, whatever was already paid for stays on disk."""
    progress = tmp_path / "partial.json"

    with (
        patch("src.agents.ollama_client.call_ollama_for_action_index", return_value=0),
        patch("src.agents.ollama_client.call_ollama_for_negotiation", return_value=None),
        patch(
            "visualization.trace_game._play_one_game",
            side_effect=RuntimeError("simulated crash"),
        ),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        tg.trace_one_game(CONFIG, game_index=0, max_turns=4, progress_path=progress)

    assert progress.is_file()
    assert json.loads(progress.read_text())["partial"] is True


def test_export_gif_without_snapshots_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no board snapshots"):
        tg.export_gif({"board_snapshots": []}, tmp_path / "x.gif")


def test_export_gif_writes_a_playable_animation(mocked_llm, tmp_path: Path) -> None:
    trace = tg.trace_one_game(CONFIG, game_index=0, max_turns=4)
    gif = tg.export_gif(trace, tmp_path / "partita.gif", every=5)

    assert gif.is_file()
    assert gif.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
