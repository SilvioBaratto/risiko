"""Render a traced game into a readable replay.

Single Responsibility: turn board snapshots into frames a viewer can actually read.
No data loading, no LLM, no styling constants of its own — those live in ``data.py``,
``trace_game.py`` and ``theme.py``.

What the old renderer got wrong, and why each is fixed here:

* it labelled the seats ``Player 0..5``, which says nothing — a viewer cannot tell who
  is who, let alone which strategy is winning. Seats are now named by their **strategy**
  (and the model playing it), and **a seat wears its strategy's colour** — the same hue
  it has in every other figure in this repo.
* it shaded continents in six saturated colours *and* coloured the territories by owner,
  so two colour systems fought for the same pixels. Continents are now neutral structure
  (light grey, named); colour means ownership and nothing else.
* territories were small dots with unreadable numbers. They are now sized by army count
  with the number legible inside.

Nothing here is Risiko-specific beyond the board: hand it any snapshot list.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402
from PIL import Image  # noqa: E402

from src.utils.constants import ADJACENCY, CONTINENTS, NUM_TERRITORIES  # noqa: E402
from visualization import map_layout  # noqa: E402
from visualization.theme import (  # noqa: E402
    CONTINENT_LABELS,
    INK,
    INK_MUTED,
    STRATEGY_COLORS,
    model_label,
    strategy_label,
)

__all__ = ["Seat", "seats_from_trace", "render_frame", "export_gif"]

FRAME_SIZE_PX: tuple[int, int] = (1600, 900)
FRAME_DPI: int = 100

# action_type in the env's action dict (see src/env.py phases).
ACTION_NAMES: dict[int, str] = {
    0: "gioca un tris",
    1: "rinforza",
    2: "attacca",
    3: "sposta le armate conquistate",
    4: "fortifica",
    5: "passa",
}


class Seat:
    """Who is sitting in a seat: the strategy it plays, and the model playing it."""

    def __init__(self, seat: int, strategy: str, model: str) -> None:
        """Bind a seat number to its strategy slug and model tag."""
        self.seat = seat
        self.strategy = strategy
        self.model = model
        self.color = STRATEGY_COLORS.get(strategy, INK_MUTED)
        self.label = strategy_label(strategy)
        self.model_label = model_label(model)


def seats_from_trace(trace: dict[str, Any]) -> dict[int, Seat]:
    """Read the seat → (strategy, model) assignment out of a game trace.

    Args:
        trace: Output of ``trace_game.trace_one_game`` (or its partial dump — but a
            partial trace has no ``result``, so it carries no assignment yet).

    Returns:
        Seat number → :class:`Seat`.

    Raises:
        ValueError: If the trace has no finished game record to read seats from.
    """
    result = trace.get("result")
    if not result or not result.get("seat_strategies"):
        raise ValueError("trace has no seat assignment — is it a partial dump?")

    strategies = result["seat_strategies"]
    models = result["seat_models"]
    # The ledger keys these by string seat id.
    return {
        int(seat): Seat(int(seat), strategy, models[seat]) for seat, strategy in strategies.items()
    }


def render_frame(snapshot: dict[str, Any], seats: dict[int, Seat]) -> Figure:
    """Draw one board state: who holds what, with how many armies, and whose turn it is.

    Args:
        snapshot: One entry of ``trace["board_snapshots"]``.
        seats: Output of :func:`seats_from_trace`.

    Returns:
        The figure. The caller closes it.
    """
    fig, ax = plt.subplots(
        figsize=(FRAME_SIZE_PX[0] / FRAME_DPI, FRAME_SIZE_PX[1] / FRAME_DPI),
        dpi=FRAME_DPI,
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    owner = snapshot["territory_owner"]
    armies = snapshot["armies"]
    coords = map_layout.get_layout()

    _draw_continents(ax)
    _draw_borders(ax, coords)
    _draw_territories(ax, coords, owner, armies, seats)
    _draw_scoreboard(ax, owner, armies, seats, int(snapshot["current_player"]))
    _draw_caption(ax, snapshot, seats)

    ax.set_xlim(-0.04, 1.52)
    ax.set_ylim(0.08, 1.10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    return fig


def _draw_continents(ax: Axes) -> None:
    """Shade the continents as neutral structure and name them.

    Deliberately grey: colour in this figure means *ownership*. A second colour system
    for continents would compete with it for the viewer's attention.
    """
    for continent in CONTINENTS:
        hull = map_layout.continent_hull(continent, pad=0.05)
        ax.add_patch(
            Polygon(
                hull,
                closed=True,
                facecolor="#F2F2F2",
                edgecolor="#DDDDDD",
                linewidth=1.2,
                zorder=0,
            )
        )
        lx, ly = map_layout.CONTINENT_LABEL_POS[continent]
        ax.text(
            lx,
            ly,
            CONTINENT_LABELS[continent].upper(),
            ha="center",
            va="center",
            fontsize=12,
            color="#B0B0B0",
            fontweight="bold",
            zorder=1,
        )


def _draw_borders(ax: Axes, coords: dict[int, tuple[float, float]]) -> None:
    """Draw every adjacency once; dash the one that crosses the date line."""
    drawn: set[tuple[int, int]] = set()
    for src, neighbours in ADJACENCY.items():
        for dst in neighbours:
            edge = (min(src, dst), max(src, dst))
            if edge in drawn:
                continue
            drawn.add(edge)
            x0, y0 = coords[edge[0]]
            x1, y1 = coords[edge[1]]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="#CFCFCF",
                linewidth=1.0,
                linestyle="--" if edge in map_layout.WRAP_EDGES else "-",
                zorder=2,
            )


def _draw_territories(
    ax: Axes,
    coords: dict[int, tuple[float, float]],
    owner: list[int],
    armies: list[int],
    seats: dict[int, Seat],
) -> None:
    """One disc per territory: colour = who owns it, size and number = how many armies."""
    for territory in range(NUM_TERRITORIES):
        x, y = coords[territory]
        held = int(owner[territory])
        count = int(armies[territory])
        seat = seats.get(held)
        color = seat.color if seat else INK_MUTED

        ax.scatter(
            x,
            y,
            s=340 + 70 * min(count, 12),
            color=color,
            edgecolor="white",
            linewidth=2.0,
            zorder=4,
        )
        text = ax.text(
            x,
            y,
            str(count),
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
            zorder=5,
        )
        text.set_path_effects([pe.withStroke(linewidth=1.8, foreground=color)])


def _draw_scoreboard(
    ax: Axes,
    owner: list[int],
    armies: list[int],
    seats: dict[int, Seat],
    current: int,
) -> None:
    """List the six players by strategy, with their live territory and army counts.

    Sorted by territories held, so the leader is always on top — the coalition beat of
    the whole project is "who is winning right now", and a fixed seat order hides it.
    """
    standing = []
    for seat_id, seat in seats.items():
        held = [t for t in range(NUM_TERRITORIES) if int(owner[t]) == seat_id]
        standing.append((len(held), sum(int(armies[t]) for t in held), seat))
    standing.sort(key=lambda row: (-row[0], -row[1]))

    top, step = 0.98, 0.135
    for row, (territories, army_total, seat) in enumerate(standing):
        y = top - row * step
        active = seat.seat == current

        if active:
            ax.add_patch(
                FancyBboxPatch(
                    (1.03, y - 0.052),
                    0.47,
                    0.108,
                    boxstyle="round,pad=0.008",
                    facecolor="#F4F4F4",
                    edgecolor=seat.color,
                    linewidth=2.0,
                    zorder=3,
                )
            )
        ax.scatter(1.07, y + 0.012, s=260, color=seat.color, zorder=4)
        ax.text(
            1.12,
            y + 0.028,
            seat.label,
            fontsize=15,
            color=INK,
            fontweight="bold",
            va="center",
            zorder=4,
        )
        ax.text(
            1.12,
            y - 0.004,
            seat.model_label,
            fontsize=11,
            color=INK_MUTED,
            va="center",
            zorder=4,
        )
        ax.text(
            1.12,
            y - 0.034,
            f"{territories} territori · {army_total} armate",
            fontsize=12,
            color=INK if active else INK_MUTED,
            va="center",
            zorder=4,
        )


def _draw_caption(ax: Axes, snapshot: dict[str, Any], seats: dict[int, Seat]) -> None:
    """Turn number, whose turn it is, and what they just did — in plain Italian."""
    current = int(snapshot["current_player"])
    seat = seats.get(current)
    ax.text(
        -0.02,
        1.085,
        f"Turno {int(snapshot['turn'])}",
        fontsize=25,
        color=INK,
        fontweight="bold",
        va="center",
    )
    if seat is None:
        return

    action = snapshot.get("action")
    verb = ACTION_NAMES.get(int(action["action_type"]), "gioca") if action else "inizia la partita"
    # Its own line: a two-digit turn number would otherwise run into the caption.
    ax.text(
        -0.02,
        1.035,
        f"{seat.label} ({seat.model_label}) {verb}",
        fontsize=15,
        color=seat.color,
        va="center",
        fontweight="bold",
    )


def export_gif(
    trace: dict[str, Any],
    path: Path,
    every: int = 3,
    duration_ms: int = 900,
) -> Path:
    """Replay a traced game into an animated GIF.

    Args:
        trace: A finished trace from ``trace_game.trace_one_game``.
        path: Destination ``.gif``.
        every: Keep one snapshot out of *every* — a game is hundreds of env steps and a
            frame per step is neither watchable nor small.
        duration_ms: How long each frame is held.

    Returns:
        The written path.

    Raises:
        ValueError: If the trace carries no board snapshots.
    """
    snapshots = trace.get("board_snapshots") or []
    if not snapshots:
        raise ValueError("trace has no board snapshots — was it produced before tracing existed?")

    seats = seats_from_trace(trace)
    # Always keep the final board: the last step is the one that decides the game, and
    # a stride that misses it ends the replay mid-move.
    chosen = list(snapshots[::every])
    if chosen[-1] is not snapshots[-1]:
        chosen.append(snapshots[-1])

    frames = [_to_image(snapshot, seats) for snapshot in chosen]
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return path


def _to_image(snapshot: dict[str, Any], seats: dict[int, Seat]) -> Image.Image:
    """Rasterise one frame."""
    fig = render_frame(snapshot, seats)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=FRAME_DPI, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")
