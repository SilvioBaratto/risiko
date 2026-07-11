"""Italian voice-over strings for the Risiko narrative script.

Single Responsibility: owns every human-readable Italian string used by
``come_vincere_al_risiko.py``. No computation and no plotting live here.

The sections follow the beats of ``script.md`` one to one, so the console output can
be read as the voice-over while the matching figure is on screen. Numbers are left as
``{placeholders}`` and filled from the tournament ledger — nothing is hardcoded twice.
"""

from __future__ import annotations

__all__ = ["TEXTS", "header"]

_RULE = "=" * 78


def header(number: int, title: str) -> str:
    """Return a section banner for the console."""
    return f"\n{_RULE}\n  SEZIONE {number} — {title}\n{_RULE}"


TEXTS: dict[str, str] = {
    # ── 1. Setup ──────────────────────────────────────────────────────────────
    "setup": (
        "La mia ragazza mi ha battuto a Risiko, quindi ho pensato a come batterla.\n"
        "\n"
        "Ho fatto giocare sei intelligenze artificiali una contro l'altra, per {n_games}\n"
        "partite, con sei strategie diverse:\n"
        "\n"
        "  - attaccare sempre\n"
        "  - fortificarsi e difendere fino alla fine\n"
        "  - chiudere l'Australia e vivere di rendita\n"
        "  - chiudere il Sud America e allargarsi da lì\n"
        "  - garantire almeno una conquista a turno per pescare carte e calare tris\n"
        "  - allearsi e fare coalizione contro chi è in testa\n"
        "\n"
        "A ogni partita ho assegnato le strategie ai modelli in modo casuale, così la\n"
        "forza del singolo modello si annulla e resta solo il merito della strategia."
    ),
    "map": (
        "La mappa decide la strategia: ogni continente dà un bonus di armate, ma quanto\n"
        "sia difendibile dipende da quanti confini espone.\n"
    ),
    # ── 2. Defence ────────────────────────────────────────────────────────────
    "defence": (
        "Chi si fortifica e aspetta vince il {turtle} delle partite.\n"
        "Giocare a caso ne vincerebbe il {baseline}."
    ),
    # ── 3. Aggression ─────────────────────────────────────────────────────────
    "aggression": (
        "Chi attacca sempre vince il {blitz}, e tradisce {betrayals} volte, più di\n"
        "chiunque altro. Se tradisci tutti, prima o poi si alleano contro di te."
    ),
    # ── 4. Continents ─────────────────────────────────────────────────────────
    "continents": (
        "Chiudere l'Australia, il consiglio più ripetuto perché ha un solo confine da\n"
        "difendere, vince il {australia} delle partite. Come giocare a caso.\n"
        "E il Sud America fa ancora peggio: {south_america}."
    ),
    # ── 5. Diplomacy ──────────────────────────────────────────────────────────
    "diplomacy": (
        "A vincere è la diplomazia: allearsi e accerchiare chi è in testa.\n"
        "{diplomacy} delle partite, quasi il doppio dell'Australia."
    ),
    # ── 6. Cards ──────────────────────────────────────────────────────────────
    "cards": (
        "Al secondo posto conquistare almeno un territorio a ogni turno per pescare una\n"
        "carta, e calare tris appena possibile per trasformarli in armate: {cards}.\n"
        "\n"
        "Vince meno della diplomazia, ma chiude più in alto: piazzamento medio {cards_place}\n"
        "contro {diplomacy_place}. Le due strategie sono appaiate."
    ),
    # ── 7. The control ────────────────────────────────────────────────────────
    "control": (
        "Il controllo metodologico: ogni strategia è stata giocata da ogni modello.\n"
        "Se una strategia vince, non è perché le è toccato il modello migliore.\n"
        "\n"
        "Il confronto fra i modelli, invece, non è pulito: girano con il ragionamento\n"
        "disattivato, il che favorisce i modelli istruiti a rispondere subito e penalizza\n"
        "quelli che ragionano. Serve a leggere la matrice, non a classificare i modelli."
    ),
    # ── 8. Convergence ────────────────────────────────────────────────────────
    "convergence": (
        "Perché {n_games} partite e non venti: dopo venti partite la diplomazia sembrava\n"
        "vincerne il 35%. Dopo {n_games}, il {diplomacy}. Il primato cambia mano più volte\n"
        "prima di stabilizzarsi."
    ),
    # ── 9. Conclusion ─────────────────────────────────────────────────────────
    "conclusion": (
        "La combinazione ottimale: allearsi subito contro chi è in testa, garantire una\n"
        "conquista per turno per accumulare carte, e tradire nel momento in cui i tris ti\n"
        "danno abbastanza armate per chiudere la partita.\n"
        "\n"
        "La mia ragazza mi batterà lo stesso perché è molto più brava di me a stringere\n"
        "alleanze, ma scrivi nei commenti come migliorare questo algoritmo."
    ),
}
