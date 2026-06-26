"""Strategy catalog for Risiko RL — pure data module, no I/O or LLM calls.

Defines the six research-grounded Risiko strategies, each with a board-play
hint, a diplomatic posture, and helpers to build a ``PlayerConfig`` from a
strategy. This is the single source of truth consumed by Cycle 8's tournament
driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from src.agents.player_config import DEFAULT_MODEL, PlayerConfig

__all__ = [
    "Posture",
    "Strategy",
    "STRATEGY_CATALOG",
    "CATALOG_BY_NAME",
    "POSTURE_LEVERS",
    "compose_strategy_hint",
    "to_player_config",
    "get_strategy",
]


class Posture(StrEnum):
    """Diplomatic posture — serialisable as a plain string (StrEnum)."""

    LONE_WOLF = "lone-wolf"
    DEFENSIVE_PACT = "defensive-pact"
    BETRAYER = "betrayer"
    ALLIANCE_HEAVY = "alliance-heavy"
    OPPORTUNIST = "opportunist"
    COALITION = "coalition"


@dataclass(frozen=True)
class Strategy:
    """Immutable descriptor for a single Risiko play-style.

    Attributes:
        name: Unique slug used as a lookup key.
        strategy_hint: Board-play instruction injected into the action prompt.
        posture: Diplomatic stance (drives trust/vengefulness levers).
        posture_directive: One-line social instruction composed into strategy_hint.
    """

    name: str
    strategy_hint: str
    posture: Posture
    posture_directive: str


# ---------------------------------------------------------------------------
# Posture → (trust_propensity, vengefulness) levers
# ---------------------------------------------------------------------------

POSTURE_LEVERS: dict[Posture, tuple[float, float]] = {
    Posture.LONE_WOLF: (0.1, 0.3),
    Posture.DEFENSIVE_PACT: (0.7, 0.2),
    Posture.BETRAYER: (0.2, 0.9),
    Posture.ALLIANCE_HEAVY: (0.9, 0.1),
    Posture.OPPORTUNIST: (0.4, 0.6),
    Posture.COALITION: (0.8, 0.4),
}

# ---------------------------------------------------------------------------
# Six-strategy catalog
# ---------------------------------------------------------------------------

STRATEGY_CATALOG: tuple[Strategy, ...] = (
    Strategy(
        name="australia_lock",
        strategy_hint=(
            "Secure Australia first: take all four territories, then expand "
            "north through Indonesia. Defend the single northern chokepoint "
            "aggressively and only strike outside when your continent bonus is safe."
        ),
        posture=Posture.LONE_WOLF,
        posture_directive="Act alone; refuse alliances and treat every neighbour as a threat.",
    ),
    Strategy(
        name="south_america_lock",
        strategy_hint=(
            "Lock down South America early: hold all four territories for the "
            "+2 bonus, fortify Venezuela and Brazil borders, then push into "
            "Africa or North America only after the continent is secure."
        ),
        posture=Posture.DEFENSIVE_PACT,
        posture_directive="Propose non-aggression pacts with immediate neighbours to buy time.",
    ),
    Strategy(
        name="aggressive_blitz",
        strategy_hint=(
            "Attack relentlessly from turn 1: target the weakest player each "
            "round to collect their cards, chain eliminations to snowball army "
            "count, and never end a turn without attacking at least once."
        ),
        posture=Posture.BETRAYER,
        posture_directive="Feign alliance, then betray at the moment of maximum gain.",
    ),
    Strategy(
        name="turtle_defensive",
        strategy_hint=(
            "Minimise exposure: fortify interior territories, cede contested "
            "border regions rather than overcommitting armies, and wait for "
            "opponents to exhaust each other before expanding."
        ),
        posture=Posture.ALLIANCE_HEAVY,
        posture_directive="Seek and honour as many alliances as possible to avoid early conflict.",
    ),
    Strategy(
        name="card_cycle_hunter",
        strategy_hint=(
            "Prioritise card collection above all else: always attack at least "
            "one territory per turn, hunt down players who are about to trade "
            "cards, and trade sets at the highest possible value."
        ),
        posture=Posture.OPPORTUNIST,
        posture_directive=(
            "Switch sides opportunistically; ally with whoever weakens your main rival."
        ),
    ),
    Strategy(
        name="diplomat_coalition",
        strategy_hint=(
            "Build a dominant coalition: share intelligence, coordinate attacks "
            "on the leader with allies, and leverage diplomacy to avoid "
            "two-front wars while steadily accumulating territory."
        ),
        posture=Posture.COALITION,
        posture_directive="Invest in coalition-building; punish defection and reward loyalty.",
    ),
)

CATALOG_BY_NAME: dict[str, Strategy] = {s.name: s for s in STRATEGY_CATALOG}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compose_strategy_hint(strategy: Strategy | str) -> str:
    """Return the full strategy hint: board play joined with posture directive."""
    if isinstance(strategy, str):
        strategy = CATALOG_BY_NAME[strategy]
    return f"{strategy.strategy_hint} {strategy.posture_directive}"


def to_player_config(
    strategy: Strategy,
    player_id: int,
    *,
    temperature: float,
    top_p: float,
    model: str = DEFAULT_MODEL,
) -> PlayerConfig:
    """Build a frozen ``PlayerConfig`` from a strategy.

    ``strategy_hint`` is the composed board+posture string.
    ``trust_propensity`` and ``vengefulness`` come from ``POSTURE_LEVERS``.
    ``PlayerConfig`` is not modified.
    """
    trust, vengefulness = POSTURE_LEVERS[strategy.posture]
    return PlayerConfig(
        player_id=player_id,
        temperature=temperature,
        top_p=top_p,
        strategy_hint=compose_strategy_hint(strategy),
        model=model,
        trust_propensity=trust,
        vengefulness=vengefulness,
    )


def get_strategy(name: str) -> Strategy:
    """Return the ``Strategy`` for *name*; raise ``ValueError`` if not found."""
    if name not in CATALOG_BY_NAME:
        raise ValueError(f"Unknown strategy name {name!r}. Valid names: {sorted(CATALOG_BY_NAME)}")
    return CATALOG_BY_NAME[name]
