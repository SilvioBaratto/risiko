"""Shared figure style for every Risiko chart.

Single Responsibility: geometry, palette, and Italian labels. Nothing that reads
data or draws a specific chart belongs here.

Conventions (mirrored from the Monopoly repo so the two videos look like one series):
    * exactly 1080p at dpi 150
    * light theme, white background
    * colorblind-safe Wong (2011) palette, validated with the design-system checker
    * figure text is ITALIAN (it is on screen), code and API are ENGLISH

The palette validator flags amber / sky / purple as below 3:1 contrast against a
white surface. That is legal only with "relief": every chart in plots.py therefore
carries visible labels (values on bars, direct labels on points, text in cells).
Do not drop those labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

__all__ = [
    "FIGURE_SIZE_PX",
    "DPI",
    "PALETTE",
    "STRATEGY_COLORS",
    "STRATEGY_ORDER",
    "STRATEGY_LABELS",
    "MODEL_LABELS",
    "CONTINENT_LABELS",
    "SEQUENTIAL_CMAP",
    "RANDOM_BASELINE",
    "GRID_COLOR",
    "INK",
    "INK_MUTED",
    "create_figure",
    "save_figure",
    "style_axes",
    "strategy_label",
    "model_label",
]

# ── Geometry ─────────────────────────────────────────────────────────────────
FIGURE_SIZE_PX: tuple[int, int] = (1920, 1080)
DPI: int = 150

# ── Colour ───────────────────────────────────────────────────────────────────
# Wong (2011) colourblind-safe hues. Validated: lightness band PASS, chroma PASS,
# worst adjacent CVD separation ΔE 18.3 (deutan) — well above the 12 target.
PALETTE: dict[str, str] = {
    "blue": "#0072B2",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "amber": "#E69F00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
}

# Colour follows the entity, never its rank: a strategy keeps its hue in EVERY figure.
STRATEGY_COLORS: dict[str, str] = {
    "diplomat_coalition": PALETTE["blue"],
    "card_cycle_hunter": PALETTE["green"],
    "aggressive_blitz": PALETTE["vermillion"],
    "australia_lock": PALETTE["amber"],
    "south_america_lock": PALETTE["sky"],
    "turtle_defensive": PALETTE["purple"],
}

# Fixed order (by 100-game win rate), used whenever an ordering is not data-driven.
STRATEGY_ORDER: tuple[str, ...] = (
    "diplomat_coalition",
    "card_cycle_hunter",
    "aggressive_blitz",
    "australia_lock",
    "south_america_lock",
    "turtle_defensive",
)

SEQUENTIAL_CMAP: str = "Blues"  # magnitude ⇒ one hue, light→dark. Never a rainbow.

GRID_COLOR: str = "#DDDDDD"
INK: str = "#222222"
INK_MUTED: str = "#666666"

# ── Semantics ────────────────────────────────────────────────────────────────
RANDOM_BASELINE: float = 1 / 6  # 6 players ⇒ 16.7% by chance

# ── Italian on-screen labels ─────────────────────────────────────────────────
STRATEGY_LABELS: dict[str, str] = {
    "diplomat_coalition": "Diplomazia",
    "card_cycle_hunter": "Carte",
    "aggressive_blitz": "Aggressione",
    "australia_lock": "Australia",
    "south_america_lock": "Sud America",
    "turtle_defensive": "Difesa",
}

MODEL_LABELS: dict[str, str] = {
    "gemma4:31b-cloud": "gemma4 31b",
    "gemma4:cloud": "gemma4",
    "qwen3.5:cloud": "qwen3.5",
    "kimi-k2.6:cloud": "kimi k2.6",
    "nemotron-3-super:cloud": "nemotron 3",
    "deepseek-v4-flash:cloud": "deepseek v4",
}

CONTINENT_LABELS: dict[str, str] = {
    "North America": "Nord America",
    "South America": "Sud America",
    "Europe": "Europa",
    "Africa": "Africa",
    "Asia": "Asia",
    "Australia": "Oceania",
}


def strategy_label(slug: str) -> str:
    """Return the Italian on-screen name for a strategy slug."""
    return STRATEGY_LABELS.get(slug, slug)


def model_label(slug: str) -> str:
    """Return the short on-screen name for a model tag."""
    return MODEL_LABELS.get(slug, slug)


# ── Figure lifecycle ─────────────────────────────────────────────────────────
def create_figure() -> tuple[Figure, Axes]:
    """Return a blank 1920x1080 figure and its axes."""
    width_in = FIGURE_SIZE_PX[0] / DPI
    height_in = FIGURE_SIZE_PX[1] / DPI
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def style_axes(ax: Axes, *, xgrid: bool = False, ygrid: bool = False) -> None:
    """Apply the recessive grid/spine treatment shared by every chart."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_MUTED, labelsize=13, length=0)
    if xgrid:
        ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    if ygrid:
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Write the figure to *path* on a white background and close it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out
