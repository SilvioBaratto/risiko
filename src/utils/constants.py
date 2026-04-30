"""Immutable game data for the Risiko (Risk) board.

This module contains no game logic — only the board topology,
continent definitions, card deck composition, and static lookup helpers.
"""

from __future__ import annotations

NUM_TERRITORIES = 42
NUM_WILD_CARDS = 2
TRADE_VALUES_INCREMENT = 5

TERRITORY_NAMES: list[str] = [
    # North America (0–8)
    "Alaska",
    "Alberta",
    "Central America",
    "Eastern United States",
    "Greenland",
    "Northwest Territory",
    "Ontario",
    "Quebec",
    "Western United States",
    # South America (9–12)
    "Argentina",
    "Brazil",
    "Peru",
    "Venezuela",
    # Europe (13–19)
    "Great Britain",
    "Iceland",
    "Northern Europe",
    "Scandinavia",
    "Southern Europe",
    "Ukraine",
    "Western Europe",
    # Africa (20–25)
    "Congo",
    "East Africa",
    "Egypt",
    "Madagascar",
    "North Africa",
    "South Africa",
    # Asia (26–37)
    "Afghanistan",
    "China",
    "India",
    "Irkutsk",
    "Japan",
    "Kamchatka",
    "Middle East",
    "Mongolia",
    "Siam",
    "Siberia",
    "Ural",
    "Yakutsk",
    # Australia (38–41)
    "Eastern Australia",
    "Indonesia",
    "New Guinea",
    "Western Australia",
]

CONTINENTS: dict[str, list[int]] = {
    "North America": list(range(0, 9)),
    "South America": list(range(9, 13)),
    "Europe": list(range(13, 20)),
    "Africa": list(range(20, 26)),
    "Asia": list(range(26, 38)),
    "Australia": list(range(38, 42)),
}

CONTINENT_BONUSES: dict[str, int] = {
    "North America": 5,
    "South America": 2,
    "Europe": 5,
    "Africa": 3,
    "Asia": 7,
    "Australia": 2,
}

ADJACENCY: dict[int, list[int]] = {
    0: [1, 5, 31],  # Alaska
    1: [0, 5, 6, 8],  # Alberta
    2: [3, 8, 12],  # Central America
    3: [2, 6, 7, 8],  # Eastern United States
    4: [5, 6, 7, 14],  # Greenland
    5: [0, 1, 4, 6],  # Northwest Territory
    6: [1, 3, 4, 5, 7, 8],  # Ontario
    7: [3, 4, 6],  # Quebec
    8: [1, 2, 3, 6],  # Western United States
    9: [10, 11],  # Argentina
    10: [9, 11, 12, 24],  # Brazil
    11: [9, 10, 12],  # Peru
    12: [2, 10, 11],  # Venezuela
    13: [14, 15, 16, 19],  # Great Britain
    14: [4, 13, 16],  # Iceland
    15: [13, 16, 17, 18, 19],  # Northern Europe
    16: [13, 14, 15, 18],  # Scandinavia
    17: [15, 18, 19, 22, 24, 32],  # Southern Europe
    18: [15, 16, 17, 26, 32],  # Ukraine
    19: [13, 15, 17, 24],  # Western Europe
    20: [21, 24, 25],  # Congo
    21: [20, 22, 23, 24, 25, 32],  # East Africa
    22: [17, 21, 24, 32],  # Egypt
    23: [21, 25],  # Madagascar
    24: [10, 17, 19, 20, 21, 22],  # North Africa
    25: [20, 21, 23],  # South Africa
    26: [18, 27, 28, 32, 36],  # Afghanistan
    27: [26, 28, 29, 33, 34, 35, 36],  # China
    28: [26, 27, 32, 34],  # India
    29: [27, 31, 33, 35, 37],  # Irkutsk
    30: [31, 33],  # Japan
    31: [0, 29, 30, 33, 37],  # Kamchatka
    32: [17, 18, 21, 22, 26, 28],  # Middle East
    33: [27, 29, 30, 31, 35],  # Mongolia
    34: [27, 28, 39],  # Siam
    35: [27, 29, 33, 36, 37],  # Siberia
    36: [26, 27, 35],  # Ural
    37: [29, 31, 35],  # Yakutsk
    38: [40, 41],  # Eastern Australia
    39: [34, 40, 41],  # Indonesia
    40: [38, 39, 41],  # New Guinea
    41: [38, 39, 40],  # Western Australia
}

# 42 territory cards (0=infantry, 1=cavalry, 2=artillery) + 2 wild (3)
# Even distribution: 14 each type, then 2 wild
CARD_SYMBOLS: dict[int, int] = {
    cid: (0 if cid < 14 else 1 if cid < 28 else 2 if cid < 42 else 3) for cid in range(44)
}

TRADE_VALUES: list[int] = [4, 6, 8, 10, 12, 15]

# Classic starting armies by player count
STARTING_ARMIES: dict[int, int] = {
    2: 40,
    3: 35,
    4: 30,
    5: 25,
    6: 20,
}

# Reverse lookup: territory_id -> continent name
_TERRITORY_TO_CONTINENT: dict[int, str] = {
    tid: name for name, ids in CONTINENTS.items() for tid in ids
}


def get_trade_value(trade_count: int) -> int:
    """Return the number of armies gained for the nth trade.

    Args:
        trade_count: Zero-based trade index (0 = first trade).

    Returns:
        Armies received for this trade.
    """
    if trade_count < len(TRADE_VALUES):
        return TRADE_VALUES[trade_count]
    return TRADE_VALUES[-1] + TRADE_VALUES_INCREMENT * (trade_count - len(TRADE_VALUES) + 1)


def territory_to_continent(territory_id: int) -> str | None:
    """Return the continent name for a territory, or None if invalid."""
    return _TERRITORY_TO_CONTINENT.get(territory_id)


ACTION_DIMS: dict[str, int] = {
    "action_type": 6,
    "param_a": 42,
    "param_b": 42,
    "param_c": 43,
    "param_d": 43,
}
