"""Hand-authored 2D layout of the 42 Risiko territories.

Single Responsibility: where each territory sits on screen. No drawing, no data.

Why this exists: the repo had no coordinates at all, so ``render_game`` fell back to
``networkx.spring_layout`` and produced a force-directed blob that looks nothing like
the board. These coordinates place every territory roughly where it belongs on the
world map (x = west→east, y = south→north, both normalised to [0, 1]), which turns the
same renderer into a recognisable map.

Territory ids and names come from ``src.utils.constants.TERRITORY_NAMES`` — ids are
grouped contiguously by continent (NA 0-8, SA 9-12, EU 13-19, AF 20-25, AS 26-37,
AU 38-41) and alphabetical within each continent.

One quirk of the real board: Alaska (0) and Kamchatka (31) are adjacent across the
date line. On a flat map that edge spans the whole width; callers that draw edges
should treat ``WRAP_EDGES`` specially (e.g. dashed) instead of drawing a straight
line through Europe.
"""

from __future__ import annotations

from src.utils.constants import CONTINENTS, NUM_TERRITORIES, TERRITORY_NAMES

__all__ = [
    "TERRITORY_COORDS",
    "WRAP_EDGES",
    "LABEL_BELOW",
    "CONTINENT_LABEL_POS",
    "get_layout",
    "continent_hull",
    "continent_centroid",
]

# (x, y) in [0, 1]: x runs west→east, y runs south→north.
TERRITORY_COORDS: dict[int, tuple[float, float]] = {
    # ── North America (0-8) ──
    0: (0.05, 0.88),  # Alaska
    1: (0.13, 0.78),  # Alberta
    2: (0.15, 0.55),  # Central America
    3: (0.22, 0.66),  # Eastern United States
    4: (0.29, 0.93),  # Greenland
    5: (0.15, 0.88),  # Northwest Territory
    6: (0.20, 0.78),  # Ontario
    7: (0.27, 0.78),  # Quebec
    8: (0.12, 0.66),  # Western United States
    # ── South America (9-12) ──
    9: (0.23, 0.21),  # Argentina
    10: (0.28, 0.35),  # Brazil
    11: (0.20, 0.33),  # Peru
    12: (0.20, 0.46),  # Venezuela
    # ── Europe (13-19) ──
    13: (0.38, 0.75),  # Great Britain
    14: (0.40, 0.87),  # Iceland
    15: (0.47, 0.74),  # Northern Europe
    16: (0.48, 0.89),  # Scandinavia
    17: (0.48, 0.63),  # Southern Europe
    18: (0.56, 0.80),  # Ukraine
    19: (0.39, 0.63),  # Western Europe
    # ── Africa (20-25) ──
    20: (0.49, 0.33),  # Congo
    21: (0.56, 0.40),  # East Africa
    22: (0.51, 0.50),  # Egypt
    23: (0.60, 0.23),  # Madagascar
    24: (0.42, 0.47),  # North Africa
    25: (0.50, 0.21),  # South Africa
    # ── Asia (26-37) ──
    26: (0.63, 0.69),  # Afghanistan
    27: (0.76, 0.61),  # China
    28: (0.69, 0.52),  # India
    29: (0.77, 0.79),  # Irkutsk
    30: (0.91, 0.70),  # Japan
    31: (0.89, 0.88),  # Kamchatka
    32: (0.58, 0.55),  # Middle East
    33: (0.79, 0.70),  # Mongolia
    34: (0.79, 0.51),  # Siam
    35: (0.71, 0.87),  # Siberia
    36: (0.64, 0.82),  # Ural
    37: (0.80, 0.90),  # Yakutsk
    # ── Australia (38-41) ──
    38: (0.91, 0.22),  # Eastern Australia
    39: (0.79, 0.35),  # Indonesia
    40: (0.90, 0.37),  # New Guinea
    41: (0.81, 0.20),  # Western Australia
}

# Adjacencies that cross the date line: a straight segment would cut the map in half.
WRAP_EDGES: frozenset[tuple[int, int]] = frozenset({(0, 31)})

# Where to anchor a continent's name. NOT the centroid: the centroid of a continent is
# the most crowded point in it, so a caption there lands under the territory discs. These
# sit in the empty margin beside each landmass.
CONTINENT_LABEL_POS: dict[str, tuple[float, float]] = {
    "North America": (0.03, 0.97),
    "South America": (0.10, 0.24),
    "Europe": (0.44, 0.98),
    "Africa": (0.54, 0.12),
    "Asia": (0.97, 0.97),
    "Australia": (0.95, 0.12),
}

# Territory names sit above their dot by default. These five sit below instead: their
# neighbour shares a latitude and the two long names would collide (e.g. "Western United
# States" running straight into "Eastern United States").
LABEL_BELOW: frozenset[int] = frozenset(
    {
        0,  # Alaska, level with Northwest Territory
        8,  # Western United States, level with Eastern United States
        13,  # Great Britain, level with Northern Europe
        19,  # Western Europe, level with Southern Europe
        41,  # Western Australia, level with Eastern Australia
    }
)


def get_layout() -> dict[int, tuple[float, float]]:
    """Return the territory-id → (x, y) map used to draw the board.

    Drop-in replacement for the old spring-layout lookup in ``render_game``.

    Returns:
        A copy of the coordinate table, one entry per territory.
    """
    return dict(TERRITORY_COORDS)


def continent_centroid(continent: str) -> tuple[float, float]:
    """Return the mean (x, y) of a continent's territories.

    Args:
        continent: Continent name as used in ``CONTINENTS``.

    Returns:
        The centroid coordinate.

    Raises:
        KeyError: If *continent* is unknown.
    """
    points = [TERRITORY_COORDS[t] for t in CONTINENTS[continent]]
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def continent_hull(continent: str, pad: float = 0.045) -> list[tuple[float, float]]:
    """Return a padded convex hull around a continent, for background shading.

    The old renderer shaded continents by filling raw spring-layout points, which
    self-intersected. A convex hull pushed outward from the centroid gives a clean,
    non-degenerate polygon.

    Args:
        continent: Continent name as used in ``CONTINENTS``.
        pad: How far to push each hull vertex away from the centroid.

    Returns:
        Hull vertices in counter-clockwise order.

    Raises:
        KeyError: If *continent* is unknown.
    """
    points = sorted({TERRITORY_COORDS[t] for t in CONTINENTS[continent]})
    hull = _convex_hull(points)
    cx, cy = continent_centroid(continent)
    return [_push_out(x, y, cx, cy, pad) for x, y in hull]


def _push_out(x: float, y: float, cx: float, cy: float, pad: float) -> tuple[float, float]:
    """Move a point away from a centroid by *pad*, guarding the degenerate case."""
    dx, dy = x - cx, y - cy
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 1e-9:
        return (x, y + pad)
    return (x + pad * dx / dist, y + pad * dy / dist)


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the convex hull of *points* (Andrew's monotone chain).

    Kept dependency-free on purpose: scipy is not a dependency of this repo.
    """
    if len(points) <= 2:
        return list(points)

    def cross(o, a, b) -> float:  # noqa: ANN001 - local numeric helper
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _validate() -> None:
    """Fail loudly at import time if the table drifts from the board definition."""
    if len(TERRITORY_COORDS) != NUM_TERRITORIES:
        raise ValueError(f"expected {NUM_TERRITORIES} coordinates, got {len(TERRITORY_COORDS)}")
    missing = set(range(NUM_TERRITORIES)) - set(TERRITORY_COORDS)
    if missing:
        names = ", ".join(TERRITORY_NAMES[t] for t in sorted(missing))
        raise ValueError(f"missing coordinates for: {names}")


_validate()
