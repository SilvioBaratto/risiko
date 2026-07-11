"""Tests for the opt-in board-snapshot sink used by the game replay.

The contract that matters: tracing is off by default and costs nothing, and when it is
on it records a JSON-serialisable snapshot per env step without changing the game.
"""

from __future__ import annotations

import json

import src.multi_agent as ma
from src.agents.random_agent import RandomAgent
from src.env import RisikoEnv
from src.multi_agent import MultiAgentRunner
from src.utils.constants import NUM_TERRITORIES
from src.utils.seed import set_global_seeds


def _runner(max_turns: int = 12) -> MultiAgentRunner:
    return MultiAgentRunner(
        RisikoEnv(n_players=6),
        [RandomAgent() for _ in range(6)],
        max_turns=max_turns,
    )


def test_tracing_is_off_by_default() -> None:
    _runner().run_game(seed=3)
    assert ma.stop_board_trace() is None


def test_trace_records_one_snapshot_per_step_plus_the_initial_board() -> None:
    ma.start_board_trace()
    _runner().run_game(seed=3)
    snapshots = ma.stop_board_trace()

    assert snapshots is not None
    assert len(snapshots) >= 2
    assert [s["step"] for s in snapshots] == list(range(len(snapshots)))
    assert snapshots[0]["action"] is None, "the opening board precedes any action"
    assert all(s["action"] is not None for s in snapshots[1:])


def test_snapshots_are_json_serialisable_and_shaped_like_the_board() -> None:
    ma.start_board_trace()
    _runner().run_game(seed=5)
    snapshots = ma.stop_board_trace() or []

    for snapshot in snapshots:
        assert len(snapshot["territory_owner"]) == NUM_TERRITORIES
        assert len(snapshot["armies"]) == NUM_TERRITORIES
        # The "at least one army" invariant holds between turns, not between steps: a
        # just-captured territory sits at 0 until the capture-move step fills it.
        assert all(army >= 0 for army in snapshot["armies"])
        assert all(0 <= owner < 6 for owner in snapshot["territory_owner"])
        assert 0 <= snapshot["current_player"] < 6
    json.dumps(snapshots)  # numpy ints would raise here


def test_snapshots_agree_with_the_game_result() -> None:
    """The trace must describe the game that was actually played, not a parallel one.

    (Two runs of the same seed cannot be compared: ``RandomAgent`` samples from the
    action space, whose RNG ``reset(seed=...)`` does not reseed.)
    """
    set_global_seeds(11)
    ma.start_board_trace()
    result = _runner().run_game(seed=11)
    snapshots = ma.stop_board_trace() or []

    final = snapshots[-1]
    counts = [final["territory_owner"].count(player) for player in range(6)]
    assert counts == list(result.territory_history[-1])
    assert sum(counts) == NUM_TERRITORIES
