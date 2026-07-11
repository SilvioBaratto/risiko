"""Narrative script: Come vincere al Risiko — the 100-game tournament, section by section.

Walks the beats of ``script.md`` and regenerates every figure the video uses:

    python scripts/come_vincere_al_risiko.py

Sections:
    section_01_setup()        — the six strategies, the six models, the map
    section_02_defence()      — turtling loses to random
    section_03_aggression()   — attacking always, and betraying always
    section_04_continents()   — Australia and South America
    section_05_diplomacy()    — the winner
    section_06_cards()        — the co-champion, and why placement matters
    section_07_control()      — every strategy played by every model
    section_08_convergence()  — why a hundred games and not twenty
    section_09_conclusion()   — the combination

Reads ``results/tournament/<run>/`` and writes ``figures/``. No network, no Ollama call,
no simulation: the tournament already ran. Point it at another run with

    RISIKO_RUN=pilot30 python scripts/come_vincere_al_risiko.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _narrative_text import TEXTS, header  # noqa: E402

from visualization import data as vdata  # noqa: E402
from visualization import plots  # noqa: E402
from visualization.theme import RANDOM_BASELINE  # noqa: E402

RUN: str = os.environ.get("RISIKO_RUN", vdata.DEFAULT_RUN)
FIGURES_DIR: Path = Path(os.environ.get("RISIKO_FIGURES", str(_PROJECT_ROOT / "figures")))


def _pct(value: float) -> str:
    """Format a rate the way the voice-over says it: ``27%``."""
    return f"{value * 100:.0f}%"


def _rate(leaderboard: dict, slug: str) -> float:
    """Return the win rate of one strategy."""
    return _stat(leaderboard, slug)["win_rate"]


def _stat(leaderboard: dict, slug: str) -> dict:
    """Return the full leaderboard row for one strategy.

    Raises:
        KeyError: If the run never played that strategy.
    """
    for row in leaderboard["strategies"]:
        if row["strategy"] == slug:
            return row
    raise KeyError(f"strategy not in run {RUN!r}: {slug}")


# ── Sections ─────────────────────────────────────────────────────────────────
def section_01_setup(leaderboard: dict) -> None:
    """The premise, the six strategies, and the map they are played on."""
    print(header(1, "Il setup: sei strategie, sei modelli"))
    print(TEXTS["setup"].format(n_games=leaderboard["n_games"]))
    print()
    print(TEXTS["map"])
    print("  →", plots.plot_world_map(FIGURES_DIR / "mappa_risiko.png"))


def section_02_defence(leaderboard: dict) -> None:
    """Turtling: the strategy that loses to a coin flip."""
    print(header(2, "La difesa perde contro il caso"))
    print(
        TEXTS["defence"].format(
            turtle=_pct(_rate(leaderboard, "turtle_defensive")),
            baseline=_pct(RANDOM_BASELINE),
        )
    )


def section_03_aggression(leaderboard: dict) -> None:
    """Attacking always — and the betrayals it costs."""
    blitz = _stat(leaderboard, "aggressive_blitz")
    print(header(3, "Attaccare sempre, tradire sempre"))
    print(
        TEXTS["aggression"].format(
            blitz=_pct(blitz["win_rate"]),
            betrayals=blitz["betrayals"],
        )
    )
    figure = plots.plot_betrayals_vs_winrate(
        leaderboard, FIGURES_DIR / "tradimenti_vs_vittorie.png"
    )
    print("  →", figure)


def section_04_continents(leaderboard: dict) -> None:
    """The two continent locks everyone recommends."""
    print(header(4, "Australia e Sud America"))
    print(
        TEXTS["continents"].format(
            australia=_pct(_rate(leaderboard, "australia_lock")),
            south_america=_pct(_rate(leaderboard, "south_america_lock")),
        )
    )


def section_05_diplomacy(leaderboard: dict) -> None:
    """The winner, and the headline chart the whole video leans on."""
    print(header(5, "Vince la diplomazia"))
    print(TEXTS["diplomacy"].format(diplomacy=_pct(_rate(leaderboard, "diplomat_coalition"))))
    figure = plots.plot_strategy_win_rates(leaderboard, FIGURES_DIR / "vittorie_per_strategia.png")
    print("  →", figure)


def section_06_cards(leaderboard: dict) -> None:
    """The card cycle: fewer wins, better finishes."""
    cards = _stat(leaderboard, "card_cycle_hunter")
    diplomacy = _stat(leaderboard, "diplomat_coalition")
    print(header(6, "Le carte, quasi a pari merito"))
    print(
        TEXTS["cards"].format(
            cards=_pct(cards["win_rate"]),
            cards_place=f"{cards['mean_placement']:.2f}".replace(".", ","),
            diplomacy_place=f"{diplomacy['mean_placement']:.2f}".replace(".", ","),
        )
    )
    print("  →", plots.plot_mean_placement(leaderboard, FIGURES_DIR / "piazzamento_medio.png"))


def section_07_control(leaderboard: dict) -> None:
    """The methodological control — and the caveat on comparing models."""
    print(header(7, "Ogni strategia giocata da ogni modello"))
    print(TEXTS["control"])
    matrix = plots.plot_strategy_model_matrix(
        leaderboard, FIGURES_DIR / "matrice_strategia_modello.png"
    )
    print("  →", matrix)
    print("  →", plots.plot_model_win_rates(leaderboard, FIGURES_DIR / "vittorie_per_modello.png"))


def section_08_convergence(leaderboard: dict, games) -> None:  # noqa: ANN001 - pandas frame
    """Why the run needed a hundred games."""
    print(header(8, "Perché servivano cento partite"))
    print(
        TEXTS["convergence"].format(
            n_games=leaderboard["n_games"],
            diplomacy=_pct(_rate(leaderboard, "diplomat_coalition")),
        )
    )
    print("  →", plots.plot_convergence(games, FIGURES_DIR / "convergenza.png"))


def section_09_conclusion() -> None:
    """The combination the data points at."""
    print(header(9, "La combinazione"))
    print(TEXTS["conclusion"])


def main() -> None:
    """Run every section in order, writing the figures to ``figures/``."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard = vdata.load_leaderboard(RUN)
    games = vdata.load_games(RUN)

    section_01_setup(leaderboard)
    section_02_defence(leaderboard)
    section_03_aggression(leaderboard)
    section_04_continents(leaderboard)
    section_05_diplomacy(leaderboard)
    section_06_cards(leaderboard)
    section_07_control(leaderboard)
    section_08_convergence(leaderboard, games)
    section_09_conclusion()

    print(f"\nFigure scritte in {FIGURES_DIR}\n")


if __name__ == "__main__":
    main()
