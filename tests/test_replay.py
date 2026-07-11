"""Tests for the game replay renderer.

The renderer exists because the old one was unreadable: seats were called "Player 0",
so a viewer could not tell which strategy was winning. These pin the properties that
fix that — a seat wears its strategy's colour, and the final board always makes it into
the animation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from visualization import replay
from visualization.theme import STRATEGY_COLORS

_STRATEGIES = [
    "diplomat_coalition",
    "card_cycle_hunter",
    "aggressive_blitz",
    "australia_lock",
    "south_america_lock",
    "turtle_defensive",
]


def _snapshot(step: int, turn: int) -> dict:
    """A board where seat i owns territories i, i+6, i+12, ... — 7 each.

    The army counts shift with *step* so that no two snapshots render to the same image:
    PIL silently drops a frame identical to the one before it, which would make a frame
    count assertion measure the fixture rather than the code.
    """
    return {
        "step": step,
        "turn": turn,
        "current_player": turn % 6,
        "territory_owner": [t % 6 for t in range(42)],
        "armies": [1 + ((t + step) % 5) for t in range(42)],
        "action": {"action_type": 2, "param_a": 0, "param_b": 1, "param_c": 1, "param_d": 0},
    }


def _trace(n_snapshots: int = 10) -> dict:
    return {
        "result": {
            "seat_strategies": {str(i): s for i, s in enumerate(_STRATEGIES)},
            "seat_models": {str(i): "gemma4:cloud" for i in range(6)},
        },
        "board_snapshots": [_snapshot(i, i // 2) for i in range(n_snapshots)],
    }


def test_seats_carry_their_strategy_colour() -> None:
    """Colour follows the entity: a strategy has one hue across every figure."""
    seats = replay.seats_from_trace(_trace())

    assert len(seats) == 6
    for seat in seats.values():
        assert seat.color == STRATEGY_COLORS[seat.strategy]
    assert len({seat.color for seat in seats.values()}) == 6, "two seats share a hue"


def test_seats_are_named_by_strategy_not_by_number() -> None:
    seats = replay.seats_from_trace(_trace())
    labels = {seat.label for seat in seats.values()}

    assert "Diplomazia" in labels
    assert not any(label.startswith("Player") for label in labels)


def test_partial_trace_has_no_seats_and_says_so() -> None:
    with pytest.raises(ValueError, match="partial"):
        replay.seats_from_trace({"partial": True, "result": None, "board_snapshots": []})


def test_render_frame_produces_a_figure() -> None:
    trace = _trace()
    seats = replay.seats_from_trace(trace)

    figure = replay.render_frame(trace["board_snapshots"][3], seats)

    assert figure.get_size_inches()[0] > 0
    figure.clf()


def test_export_gif_always_keeps_the_final_board(tmp_path: Path) -> None:
    """A stride that skips the last snapshot would end the replay mid-move."""
    trace = _trace(n_snapshots=10)  # strides of 4 → 0, 4, 8 … and 9 must still appear

    gif = replay.export_gif(trace, tmp_path / "replay.gif", every=4)

    from PIL import Image

    with Image.open(gif) as image:
        assert image.n_frames == 4  # 0, 4, 8, plus the final 9
    assert gif.read_bytes()[:6] in (b"GIF87a", b"GIF89a")


def test_export_gif_without_snapshots_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no board snapshots"):
        replay.export_gif({"board_snapshots": []}, tmp_path / "x.gif")
