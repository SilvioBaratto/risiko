"""ASCII, matplotlib, and PIL animated-GIF renderer for Risiko game replays."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PIL import Image

from src.utils.constants import ADJACENCY, CONTINENTS, NUM_TERRITORIES, TERRITORY_NAMES

matplotlib.use("Agg")

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_PLAYER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]

_CONTINENT_BG: dict[str, str] = {
    "North America": "#e8f4f8",
    "South America": "#e8f8e8",
    "Europe": "#f8f0e8",
    "Africa": "#f8e8e8",
    "Asia": "#f0e8f8",
    "Australia": "#f8f8e8",
}

# Pre-computed spring layout for stable visualisation
_GRAPH_LAYOUT: dict[int, tuple[float, float]] | None = None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def render_ascii(state: dict[str, Any]) -> str:
    """Return an ASCII representation of the board.

    Args:
        state: Dict with keys ``territory_owner`` (np.ndarray, int32),
            ``armies`` (np.ndarray, int32).
    """
    owner = state["territory_owner"]
    armies = state["armies"]
    lines: list[str] = []
    for continent, tids in CONTINENTS.items():
        lines.append(f"{continent}")
        lines.append("-" * len(continent))
        for tid in tids:
            name = TERRITORY_NAMES[tid]
            player = int(owner[tid])
            count = int(armies[tid])
            lines.append(f"  {name:25s}  P{player}  {count} armies")
        lines.append("")
    return "\n".join(lines)


def render_matplotlib(state: dict[str, Any]) -> Figure:
    """Return a matplotlib Figure showing the board as a network graph.

    Territories are nodes coloured by owner and sized by army count.
    Continent backgrounds are lightly shaded.
    """
    owner = state["territory_owner"]
    armies = state["armies"]
    layout = _get_layout()
    fig, ax = plt.subplots(figsize=(14, 10))
    _draw_continent_backgrounds(ax, layout)
    _draw_edges(ax, layout)
    _draw_nodes(ax, layout, owner, armies)
    _draw_legend(ax, owner)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


class ReplayExporter:
    """Record board states and export to an animated GIF."""

    def __init__(self) -> None:
        """Initialise an empty exporter."""
        self._frames: list[dict[str, Any]] = []

    def add_frame(self, state: dict[str, Any]) -> None:
        """Store a deep copy of *state*."""
        self._frames.append(_copy_state(state))

    def export_gif(self, path: Path, duration_ms: int = 500) -> None:
        """Export all recorded frames as an animated GIF.

        Args:
            path: Destination file path.
            duration_ms: Delay between frames in milliseconds.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        images = [_state_to_image(f) for f in self._frames]
        if not images:
            raise ValueError("No frames to export")
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )

    def __len__(self) -> int:
        """Return the number of recorded frames."""
        return len(self._frames)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _get_layout() -> dict[int, tuple[float, float]]:
    """Return a cached spring layout for the 42-territory graph."""
    global _GRAPH_LAYOUT
    if _GRAPH_LAYOUT is None:
        g = nx.Graph()
        g.add_nodes_from(range(NUM_TERRITORIES))
        for src, neighbors in ADJACENCY.items():
            for dst in neighbors:
                g.add_edge(src, dst)
        layout = nx.spring_layout(g, seed=42, iterations=200)
        _GRAPH_LAYOUT = {k: (float(v[0]), float(v[1])) for k, v in layout.items()}
    assert _GRAPH_LAYOUT is not None
    return _GRAPH_LAYOUT


def _draw_continent_backgrounds(
    ax: matplotlib.axes.Axes,
    layout: dict[int, tuple[float, float]],
) -> None:
    """Add lightly shaded background patches per continent."""
    for continent, tids in CONTINENTS.items():
        xs = [layout[tid][0] for tid in tids]
        ys = [layout[tid][1] for tid in tids]
        color = _CONTINENT_BG.get(continent, "#ffffff")
        ax.fill(
            xs + [sum(xs) / len(xs)],
            ys + [sum(ys) / len(ys)],
            color=color,
            alpha=0.3,
            zorder=0,
        )


def _draw_edges(
    ax: matplotlib.axes.Axes,
    layout: dict[int, tuple[float, float]],
) -> None:
    """Draw adjacency edges between territories."""
    drawn: set[tuple[int, int]] = set()
    for src, neighbors in ADJACENCY.items():
        for dst in neighbors:
            edge = (min(src, dst), max(src, dst))
            if edge in drawn:
                continue
            drawn.add(edge)
            x1, y1 = layout[src]
            x2, y2 = layout[dst]
            ax.plot([x1, x2], [y1, y2], color="#cccccc", linewidth=0.5, zorder=1)


def _draw_nodes(
    ax: matplotlib.axes.Axes,
    layout: dict[int, tuple[float, float]],
    owner: np.ndarray,
    armies: np.ndarray,
) -> None:
    """Draw territory nodes coloured by owner and sized by armies."""
    for tid in range(NUM_TERRITORIES):
        x, y = layout[tid]
        player = int(owner[tid])
        count = max(1, int(armies[tid]))
        color = _PLAYER_COLORS[player % len(_PLAYER_COLORS)]
        size = 100 + count * 30
        ax.scatter(x, y, s=size, c=color, edgecolors="black", linewidths=1, zorder=2)
        ax.text(
            x,
            y,
            str(count),
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold",
            zorder=3,
        )


def _draw_legend(ax: matplotlib.axes.Axes, owner: np.ndarray) -> None:
    """Add a legend for active players."""
    active = sorted({int(p) for p in owner})
    patches = [
        Patch(facecolor=_PLAYER_COLORS[p % len(_PLAYER_COLORS)], label=f"Player {p}")
        for p in active
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)


def _copy_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of a board state dict."""
    copied: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


def _state_to_image(state: dict[str, Any]) -> Image.Image:
    """Render a state to a PIL Image."""
    fig = render_matplotlib(state)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
