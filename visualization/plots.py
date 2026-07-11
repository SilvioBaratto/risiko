"""Every chart for the Risiko video, drawn from the 100-game tournament.

Single Responsibility: turn loaded results into figures. Nothing that reads files or
defines style belongs here — see ``data.py`` and ``theme.py``.

Each public function follows the same shape: ``create_figure()`` → private ``_draw_*``
helpers → ``save_figure()``. Labels on screen are Italian; the API is English.

Design rules being honoured (see the design-system checks):
    * a strategy keeps the same hue in every figure (colour follows the entity)
    * magnitude is encoded with a single sequential hue, never a rainbow
    * no dual-axis chart anywhere: betrayals vs win rate is a scatter, not two scales
    * every mark carries a visible label — the palette validator requires that relief
      for the lower-contrast hues
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Polygon, Rectangle

from src.utils.constants import (
    ADJACENCY,
    CONTINENT_BONUSES,
    CONTINENTS,
    TERRITORY_NAMES,
)
from visualization import data as vdata
from visualization import map_layout
from visualization.theme import (
    CONTINENT_LABELS,
    INK,
    INK_MUTED,
    PALETTE,
    RANDOM_BASELINE,
    SEQUENTIAL_CMAP,
    STRATEGY_COLORS,
    create_figure,
    model_label,
    save_figure,
    strategy_label,
    style_axes,
)

__all__ = [
    "plot_strategy_win_rates",
    "plot_betrayals_vs_winrate",
    "plot_strategy_model_matrix",
    "plot_model_win_rates",
    "plot_convergence",
    "plot_mean_placement",
    "plot_world_map",
    "plot_all",
]

FIGURES_DIR: Path = Path(__file__).resolve().parents[1] / "figures"


def _pct(value: float, decimals: int = 0) -> str:
    """Format a rate as an Italian percentage (comma decimal separator)."""
    return f"{value * 100:.{decimals}f}".replace(".", ",") + "%"


def _ink_on(fill: str) -> str:
    """Pick a readable text colour for a label sitting inside a coloured bar.

    White on amber or sky is barely legible — the palette validator flags exactly those
    hues for low contrast. Dark ink goes on the light fills, white on the dark ones.
    """
    r, g, b = (int(fill[i : i + 2], 16) / 255 for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if luminance > 0.45 else "white"


# ── 1. The headline chart ────────────────────────────────────────────────────
def plot_strategy_win_rates(leaderboard: dict, output: Path) -> Path:
    """Win rate per strategy with Wilson intervals, against the random baseline.

    This is the figure the script leans on: 27 / 25 / 19 / 14 / 11 / 4 against the
    16.7% a random player would score.

    Args:
        leaderboard: Output of ``data.load_leaderboard``.
        output: Destination PNG path.

    Returns:
        The written path.
    """
    stats = sorted(leaderboard["strategies"], key=lambda s: s["win_rate"])
    fig, ax = create_figure()

    labels = [strategy_label(s["strategy"]) for s in stats]
    rates = [s["win_rate"] for s in stats]
    colors = [STRATEGY_COLORS[s["strategy"]] for s in stats]
    lower = [s["win_rate"] - s["ci_low"] for s in stats]
    upper = [s["ci_high"] - s["win_rate"] for s in stats]

    y = np.arange(len(stats))
    ax.barh(y, rates, color=colors, height=0.62, zorder=3)
    ax.errorbar(
        rates,
        y,
        xerr=[lower, upper],
        fmt="none",
        ecolor=INK_MUTED,
        elinewidth=2,
        capsize=6,
        zorder=4,
    )

    # Values live in a fixed column clear of every bar and interval, so no label
    # ever collides with the baseline rule.
    label_x = 0.415
    for yi, (rate, stat) in enumerate(zip(rates, stats, strict=True)):
        ax.text(
            label_x,
            yi,
            f"{_pct(rate)}  ({stat['wins']}/{stat['games']})",
            va="center",
            fontsize=15,
            color=INK,
        )

    ax.axvline(RANDOM_BASELINE, color=INK, linestyle="--", linewidth=2, zorder=5)
    ax.text(
        RANDOM_BASELINE + 0.007,
        len(stats) - 0.45,
        "giocare a caso · 16,7%",
        fontsize=14,
        color=INK,
        ha="left",
        va="bottom",
    )

    ax.set_yticks(y, labels, fontsize=17, color=INK)
    ax.set_ylim(-0.65, len(stats) - 0.05)
    ax.set_xlim(0, 0.52)
    ax.set_xticks(np.arange(0, 0.51, 0.1))
    ax.set_xticklabels([_pct(v) for v in np.arange(0, 0.51, 0.1)])
    ax.set_xlabel("Partite vinte su 100", fontsize=15, color=INK_MUTED, labelpad=10)
    ax.set_title(
        "Quale strategia vince a Risiko",
        fontsize=25,
        color=INK,
        pad=40,
        loc="left",
        fontweight="bold",
    )
    ax.text(
        0,
        1.015,
        "100 partite, 6 modelli, strategie assegnate a caso a ogni partita"
        " — barre = intervallo di confidenza 95%",
        transform=ax.transAxes,
        fontsize=13,
        color=INK_MUTED,
    )
    style_axes(ax, xgrid=True)
    return save_figure(fig, output)


# ── 2. Betrayal does not pay ─────────────────────────────────────────────────
def plot_betrayals_vs_winrate(leaderboard: dict, output: Path) -> Path:
    """Scatter betrayals against win rate — one point per strategy, directly labelled.

    Deliberately a scatter and not two bar scales: a dual axis would invent a
    relationship between the two measures.

    Args:
        leaderboard: Output of ``data.load_leaderboard``.
        output: Destination PNG path.

    Returns:
        The written path.
    """
    stats = leaderboard["strategies"]
    fig, ax = create_figure()

    # Five of the six points sit in open space; Australia and South America are close
    # enough to each other (and to the baseline rule) that a centred label lands on
    # top of a neighbour. Those two get pushed sideways.
    offsets: dict[str, tuple[tuple[int, int], str]] = {
        "australia_lock": ((30, -8), "left"),
        "south_america_lock": ((-30, -8), "right"),
        "turtle_defensive": ((22, 14), "left"),
    }
    for stat in stats:
        slug = stat["strategy"]
        x, y = stat["betrayals"], stat["win_rate"]
        xytext, ha = offsets.get(slug, ((0, 26), "center"))
        ax.scatter(
            x,
            y,
            s=520,
            color=STRATEGY_COLORS[slug],
            edgecolor="white",
            linewidth=2.5,
            zorder=3,
        )
        ax.annotate(
            f"{strategy_label(slug)}\n{stat['betrayals']} tradimenti · {_pct(y)}",
            (x, y),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            fontsize=14,
            color=INK,
        )

    ax.axhline(RANDOM_BASELINE, color=INK, linestyle="--", linewidth=2, zorder=2)
    ax.text(
        -20,
        RANDOM_BASELINE + 0.008,
        "giocare a caso",
        fontsize=13,
        color=INK,
        ha="left",
    )

    ax.set_xlim(-40, 980)
    ax.set_ylim(0, 0.34)
    ax.set_yticks(np.arange(0, 0.31, 0.1))
    ax.set_yticklabels([_pct(v) for v in np.arange(0, 0.31, 0.1)])
    ax.set_xlabel("Tradimenti commessi in 100 partite", fontsize=15, color=INK_MUTED, labelpad=10)
    ax.set_ylabel("Partite vinte", fontsize=15, color=INK_MUTED, labelpad=10)
    ax.set_title(
        "Tradire non paga",
        fontsize=25,
        color=INK,
        pad=40,
        loc="left",
        fontweight="bold",
    )
    ax.text(
        0,
        1.015,
        "chi attacca sempre tradisce 835 volte e vince meno di chi si allea",
        transform=ax.transAxes,
        fontsize=13,
        color=INK_MUTED,
    )
    style_axes(ax, xgrid=True, ygrid=True)
    return save_figure(fig, output)


# ── 3. The methodological control ────────────────────────────────────────────
def plot_strategy_model_matrix(leaderboard: dict, output: Path) -> Path:
    """Heat-map the 6x6 strategy x model win rates.

    Shows the control at a glance: every strategy was played by every model. Also
    exposes the two anomalies — the all-zero ``deepseek`` column and the very hot
    Diplomacy x gemma4 cell.

    Args:
        leaderboard: Output of ``data.load_leaderboard``.
        output: Destination PNG path.

    Returns:
        The written path.
    """
    rates, wins, games = vdata.strategy_matrix(leaderboard)
    strategies = [s["strategy"] for s in leaderboard["strategies"]]
    models = [m["model"] for m in leaderboard["models"]]
    rates = rates.reindex(index=strategies, columns=models)
    wins = wins.reindex(index=strategies, columns=models)
    games = games.reindex(index=strategies, columns=models)

    fig, ax = create_figure()
    image = ax.imshow(rates.to_numpy(), cmap=SEQUENTIAL_CMAP, vmin=0, vmax=0.75, aspect="auto")

    for i in range(len(strategies)):
        for j in range(len(models)):
            rate = rates.iat[i, j]
            ink = "white" if rate > 0.45 else INK
            ax.text(
                j,
                i,
                f"{_pct(rate)}\n{int(wins.iat[i, j])}/{int(games.iat[i, j])}",
                ha="center",
                va="center",
                fontsize=14,
                color=ink,
            )

    ax.set_xticks(range(len(models)), [model_label(m) for m in models], fontsize=14, color=INK)
    ax.set_yticks(
        range(len(strategies)),
        [strategy_label(s) for s in strategies],
        fontsize=16,
        color=INK,
    )
    ax.set_title(
        "Ogni strategia giocata da ogni modello",
        fontsize=25,
        color=INK,
        pad=40,
        loc="left",
        fontweight="bold",
    )
    ax.text(
        0,
        1.03,
        "le strategie ruotano a caso sui modelli: così la forza del modello si annulla",
        transform=ax.transAxes,
        fontsize=13,
        color=INK_MUTED,
    )
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("Partite vinte", fontsize=13, color=INK_MUTED)
    ticks = [0.0, 0.2, 0.4, 0.6]
    bar.set_ticks(ticks)
    bar.set_ticklabels([_pct(v) for v in ticks])
    bar.ax.tick_params(labelsize=11, colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return save_figure(fig, output)


# ── 4. Model strength (with the caveat) ──────────────────────────────────────
def plot_model_win_rates(leaderboard: dict, output: Path) -> Path:
    """Win rate per model, with the ``think=False`` caveat spelled out on the figure.

    Args:
        leaderboard: Output of ``data.load_leaderboard``.
        output: Destination PNG path.

    Returns:
        The written path.
    """
    stats = sorted(leaderboard["models"], key=lambda m: m["win_rate"])
    fig, ax = create_figure()

    labels = [model_label(m["model"]) for m in stats]
    rates = [m["win_rate"] for m in stats]
    y = np.arange(len(stats))

    ax.barh(y, rates, color=PALETTE["blue"], height=0.62, zorder=3)
    for yi, stat in enumerate(stats):
        ax.text(
            stat["win_rate"] + 0.008,
            yi,
            f"{_pct(stat['win_rate'])}  ({stat['wins']}/{stat['games']})",
            va="center",
            fontsize=15,
            color=INK,
        )

    ax.set_yticks(y, labels, fontsize=17, color=INK)
    ax.set_xlim(0, 0.52)
    ax.set_xticks(np.arange(0, 0.51, 0.1))
    ax.set_xticklabels([_pct(v) for v in np.arange(0, 0.51, 0.1)])
    ax.set_xlabel("Partite vinte su 100", fontsize=15, color=INK_MUTED, labelpad=10)
    ax.set_title(
        "Quale modello gioca meglio",
        fontsize=25,
        color=INK,
        pad=40,
        loc="left",
        fontweight="bold",
    )
    ax.text(
        0,
        1.015,
        "attenzione: il ragionamento è disattivato, e questo favorisce i modelli "
        "istruiti a rispondere subito e penalizza quelli che ragionano",
        transform=ax.transAxes,
        fontsize=13,
        color=INK_MUTED,
    )
    style_axes(ax, xgrid=True)
    return save_figure(fig, output)


# ── 5. The answer stabilising ────────────────────────────────────────────────
def plot_convergence(games: pd.DataFrame, output: Path) -> Path:
    """Running win rate of each strategy across the 100 games.

    Visual proof that a short run would have lied: the lead changes hands early and
    only settles well past game 30.

    Args:
        games: Output of ``data.load_games``.
        output: Destination PNG path.

    Returns:
        The written path.
    """
    strategies = list(STRATEGY_COLORS)
    # The first games are noise, not signal: one win out of three reads as 33%, and the
    # curve would shoot off the top of the axis and squash the band that matters.
    first_game = 10
    curves = vdata.cumulative_win_rate(games, strategies).loc[first_game:]
    fig, ax = create_figure()

    for slug in strategies:
        ax.plot(
            curves.index,
            curves[slug],
            color=STRATEGY_COLORS[slug],
            linewidth=2.4,
            zorder=3,
        )
        ax.text(
            curves.index[-1] + 1.5,
            curves[slug].iloc[-1],
            f"{strategy_label(slug)} {_pct(curves[slug].iloc[-1])}",
            fontsize=14,
            color=STRATEGY_COLORS[slug],
            va="center",
        )

    ax.axhline(RANDOM_BASELINE, color=INK, linestyle="--", linewidth=2, zorder=2)
    caption = ax.text(
        first_game + 0.5,
        RANDOM_BASELINE + 0.009,
        "giocare a caso · 16,7%",
        fontsize=13,
        color=INK,
        zorder=4,
    )
    caption.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    ax.set_xlim(first_game, 128)
    ax.set_ylim(0, 0.52)
    ax.set_yticks(np.arange(0, 0.51, 0.1))
    ax.set_yticklabels([_pct(v) for v in np.arange(0, 0.51, 0.1)])
    ax.set_xticks([10, 25, 50, 75, 100])
    ax.set_xlabel("Partite giocate", fontsize=15, color=INK_MUTED, labelpad=10)
    ax.set_ylabel("Partite vinte (cumulato)", fontsize=15, color=INK_MUTED, labelpad=10)
    ax.set_title(
        "Perché servivano cento partite",
        fontsize=25,
        color=INK,
        pad=40,
        loc="left",
        fontweight="bold",
    )
    ax.text(
        0,
        1.015,
        "dopo venti partite la diplomazia sembrava vincerne il 35%; dopo cento, il 27%",
        transform=ax.transAxes,
        fontsize=13,
        color=INK_MUTED,
    )
    style_axes(ax, ygrid=True)
    return save_figure(fig, output)


# ── 6. Wins are not everything ───────────────────────────────────────────────
def plot_mean_placement(leaderboard: dict, output: Path) -> Path:
    """Average finishing position per strategy (1 = best of six).

    The real tension of the run: Carte finishes higher on average (2.66) than
    Diplomazia (2.99) despite winning fewer games.

    Args:
        leaderboard: Output of ``data.load_leaderboard``.
        output: Destination PNG path.

    Returns:
        The written path.
    """
    stats = sorted(leaderboard["strategies"], key=lambda s: -s["mean_placement"])
    fig, ax = create_figure()

    y = np.arange(len(stats))
    values = [s["mean_placement"] for s in stats]
    colors = [STRATEGY_COLORS[s["strategy"]] for s in stats]

    ax.barh(y, values, color=colors, height=0.62, zorder=3)
    for yi, (value, fill) in enumerate(zip(values, colors, strict=True)):
        ax.text(
            value - 0.08,
            yi,
            f"{value:.2f}".replace(".", ","),
            va="center",
            ha="right",
            fontsize=15,
            color=_ink_on(fill),
            fontweight="bold",
        )

    ax.set_yticks(y, [strategy_label(s["strategy"]) for s in stats], fontsize=17, color=INK)
    ax.set_xlim(0, 6)
    ax.set_xticks(range(1, 7))
    ax.set_xlabel(
        "Posizione media a fine partita (1 = primo su sei)",
        fontsize=15,
        color=INK_MUTED,
        labelpad=10,
    )
    ax.set_title(
        "Chi arriva più in alto, anche quando non vince",
        fontsize=25,
        color=INK,
        pad=40,
        loc="left",
        fontweight="bold",
    )
    ax.text(
        0,
        1.015,
        "le Carte chiudono più in alto della Diplomazia pur vincendo meno:"
        " barre più corte sono migliori",
        transform=ax.transAxes,
        fontsize=13,
        color=INK_MUTED,
    )
    style_axes(ax, xgrid=True)
    return save_figure(fig, output)


# ── 7. The board itself ──────────────────────────────────────────────────────
def _external_borders(continent: str) -> int:
    """Count the adjacencies that leave a continent."""
    inside = set(CONTINENTS[continent])
    return sum(1 for t in inside for n in ADJACENCY[t] if n not in inside)


def _draw_map_edges(ax: Axes, coords: dict[int, tuple[float, float]]) -> None:
    """Draw every adjacency once; the date-line link is dashed so it reads as a wrap."""
    drawn: set[tuple[int, int]] = set()
    for territory, neighbours in ADJACENCY.items():
        for neighbour in neighbours:
            edge = (min(territory, neighbour), max(territory, neighbour))
            if edge in drawn:
                continue
            drawn.add(edge)
            x0, y0 = coords[edge[0]]
            x1, y1 = coords[edge[1]]
            wrap = edge in map_layout.WRAP_EDGES
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="#BBBBBB",
                linewidth=1.0,
                linestyle="--" if wrap else "-",
                zorder=2,
            )


def _draw_continents(ax: Axes, coords: dict[int, tuple[float, float]]) -> None:
    """Shade each continent and caption it with its bonus and border count."""
    fills = {
        "North America": PALETTE["sky"],
        "South America": PALETTE["green"],
        "Europe": PALETTE["purple"],
        "Africa": PALETTE["amber"],
        "Asia": PALETTE["vermillion"],
        "Australia": PALETTE["blue"],
    }
    for continent, color in fills.items():
        hull = map_layout.continent_hull(continent)
        ax.add_patch(Polygon(hull, closed=True, facecolor=color, alpha=0.13, zorder=1))

    # Territory dots first, captions on top, so a caption never hides behind a node.
    for territory, (x, y) in coords.items():
        ax.scatter(x, y, s=70, color="white", edgecolor=INK_MUTED, linewidth=1.1, zorder=4)
        below = territory in map_layout.LABEL_BELOW
        label = ax.text(
            x,
            y - 0.024 if below else y + 0.021,
            TERRITORY_NAMES[territory],
            ha="center",
            va="top" if below else "baseline",
            fontsize=7,
            color=INK_MUTED,
            zorder=5,
        )
        # A white halo keeps the 42 names legible where they crowd each other.
        label.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

    _draw_map_legend(ax, fills)


def _draw_map_legend(ax: Axes, fills: dict[str, str]) -> None:
    """List the continents beside the map: bonus and how many borders each exposes.

    Kept off the board on purpose — captions placed on the landmasses collided with
    each other and with the 42 territory names.
    """
    ordered = sorted(fills, key=lambda c: _external_borders(c))
    top = 0.93
    step = 0.105
    for row, continent in enumerate(ordered):
        y = top - row * step
        color = fills[continent]
        borders = _external_borders(continent)
        crossing = "confine" if borders == 1 else "confini"
        ax.add_patch(
            Rectangle(
                (1.06, y - 0.022),
                0.035,
                0.044,
                facecolor=color,
                edgecolor="none",
                zorder=6,
            )
        )
        ax.text(
            1.11,
            y + 0.012,
            CONTINENT_LABELS[continent].upper(),
            fontsize=13,
            color=INK,
            fontweight="bold",
            va="center",
            zorder=6,
        )
        ax.text(
            1.11,
            y - 0.022,
            f"+{CONTINENT_BONUSES[continent]} armate · {borders} {crossing}",
            fontsize=12,
            color=INK_MUTED,
            va="center",
            zorder=6,
        )


def plot_world_map(output: Path) -> Path:
    """Draw the board: continents, their bonuses, and how many borders they expose.

    Backs the two claims the script makes about the map — Australia has a single
    border, South America two.

    Args:
        output: Destination PNG path.

    Returns:
        The written path.
    """
    coords = map_layout.get_layout()
    fig, ax = create_figure()

    _draw_map_edges(ax, coords)
    _draw_continents(ax, coords)

    ax.set_xlim(-0.02, 1.42)  # the strip past x=1 holds the continent legend
    ax.set_ylim(0.10, 1.03)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "La mappa decide la strategia",
        fontsize=25,
        color=INK,
        pad=40,
        loc="left",
        fontweight="bold",
    )
    ax.text(
        0,
        1.0,
        "ogni continente dà un bonus di armate, ma quanto sia difendibile"
        " dipende da quanti confini espone",
        transform=ax.transAxes,
        fontsize=13,
        color=INK_MUTED,
    )
    fig.patch.set_facecolor("white")
    return save_figure(fig, output)


# ── Orchestrator ─────────────────────────────────────────────────────────────
def plot_all(run: str = vdata.DEFAULT_RUN, figures_dir: Path | None = None) -> list[Path]:
    """Regenerate every figure for the video from a tournament run.

    Args:
        run: Run directory under ``results/tournament/``.
        figures_dir: Output directory; defaults to ``figures/`` at the repo root.

    Returns:
        The paths written, in narrative order.
    """
    out = figures_dir or FIGURES_DIR
    leaderboard = vdata.load_leaderboard(run)
    games = vdata.load_games(run)

    return [
        plot_world_map(out / "mappa_risiko.png"),
        plot_strategy_win_rates(leaderboard, out / "vittorie_per_strategia.png"),
        plot_betrayals_vs_winrate(leaderboard, out / "tradimenti_vs_vittorie.png"),
        plot_mean_placement(leaderboard, out / "piazzamento_medio.png"),
        plot_convergence(games, out / "convergenza.png"),
        plot_strategy_model_matrix(leaderboard, out / "matrice_strategia_modello.png"),
        plot_model_win_rates(leaderboard, out / "vittorie_per_modello.png"),
    ]
