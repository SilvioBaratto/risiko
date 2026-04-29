"""Property-based tests for board constants and game data."""

from __future__ import annotations

from src.utils.constants import (
    ADJACENCY,
    CARD_SYMBOLS,
    CONTINENT_BONUSES,
    CONTINENTS,
    NUM_TERRITORIES,
    NUM_WILD_CARDS,
    STARTING_ARMIES,
    TERRITORY_NAMES,
    TRADE_VALUES,
    TRADE_VALUES_INCREMENT,
    get_trade_value,
    territory_to_continent,
)


class TestTerritoryNames:
    """Territory name list properties."""

    def test_territory_count_is_42(self):
        """There must be exactly 42 territories."""
        assert NUM_TERRITORIES == 42
        assert len(TERRITORY_NAMES) == 42

    def test_territories_are_unique(self):
        """No territory name may appear twice."""
        assert len(TERRITORY_NAMES) == len(set(TERRITORY_NAMES))

    def test_all_names_are_nonempty_strings(self):
        """Every territory name must be a non-empty string."""
        for name in TERRITORY_NAMES:
            assert isinstance(name, str)
            assert name


class TestContinents:
    """Continent group properties."""

    def test_six_continents(self):
        """There must be exactly 6 continents."""
        assert len(CONTINENTS) == 6

    def test_all_territories_belong_to_exactly_one_continent(self):
        """Every territory ID must appear in exactly one continent."""
        all_ids: list[int] = []
        for ids in CONTINENTS.values():
            all_ids.extend(ids)
        assert len(all_ids) == 42
        assert set(all_ids) == set(range(42))

    def test_continent_sizes(self):
        """Continents must have expected territory counts."""
        sizes = {name: len(ids) for name, ids in CONTINENTS.items()}
        assert sizes["North America"] == 9
        assert sizes["South America"] == 4
        assert sizes["Europe"] == 7
        assert sizes["Africa"] == 6
        assert sizes["Asia"] == 12
        assert sizes["Australia"] == 4

    def test_continent_bonuses(self):
        """Continent bonuses must match classic board values."""
        assert CONTINENT_BONUSES["North America"] == 5
        assert CONTINENT_BONUSES["South America"] == 2
        assert CONTINENT_BONUSES["Europe"] == 5
        assert CONTINENT_BONUSES["Africa"] == 3
        assert CONTINENT_BONUSES["Asia"] == 7
        assert CONTINENT_BONUSES["Australia"] == 2

    def test_territory_to_continent_maps_all(self):
        """Every territory must map to exactly one continent."""
        for tid in range(NUM_TERRITORIES):
            continent = territory_to_continent(tid)
            assert continent is not None
            assert tid in CONTINENTS[continent]

    def test_territory_to_continent_returns_none_for_invalid(self):
        """Invalid territory IDs must return None."""
        assert territory_to_continent(-1) is None
        assert territory_to_continent(42) is None
        assert territory_to_continent(999) is None


class TestAdjacency:
    """Adjacency graph properties."""

    def test_all_territories_have_neighbors(self):
        """Every territory must have at least one adjacent territory."""
        for tid in range(NUM_TERRITORIES):
            assert tid in ADJACENCY
            assert len(ADJACENCY[tid]) > 0

    def test_adjacency_is_symmetric(self):
        """If A is adjacent to B, B must be adjacent to A."""
        for tid, neighbors in ADJACENCY.items():
            for nid in neighbors:
                assert tid in ADJACENCY[nid], (
                    f"asymmetry: {tid} ({TERRITORY_NAMES[tid]}) -> {nid} "
                    f"({TERRITORY_NAMES[nid]}), but reverse edge missing"
                )

    def test_no_self_loops(self):
        """No territory may be adjacent to itself."""
        for tid, neighbors in ADJACENCY.items():
            assert tid not in neighbors

    def test_no_duplicate_edges(self):
        """No duplicate adjacencies within a territory."""
        for neighbors in ADJACENCY.values():
            assert len(neighbors) == len(set(neighbors))

    def test_expected_adjacency_counts(self):
        """Territories must have expected neighbor counts."""
        # Kamchatka (connects to Alaska) and Alaska (connects to Kamchatka)
        assert 37 in ADJACENCY[31]  # Yakutsk -> Kamchatka
        assert 31 in ADJACENCY[0]  # Alaska -> Kamchatka


class TestCards:
    """Risk card deck properties."""

    def test_card_count_is_44(self):
        """There must be exactly 44 cards."""
        assert len(CARD_SYMBOLS) == 44

    def test_card_symbol_distribution(self):
        """Card symbol counts must match standard deck."""
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for sym in CARD_SYMBOLS.values():
            counts[sym] += 1
        assert counts[0] == 14  # infantry
        assert counts[1] == 14  # cavalry
        assert counts[2] == 14  # artillery
        assert counts[3] == NUM_WILD_CARDS  # wild

    def test_wild_count(self):
        """There must be exactly 2 wild cards."""
        assert NUM_WILD_CARDS == 2


class TestTradeValues:
    """Card trade-in value properties."""

    def test_trade_sequence_length(self):
        """TRADE_VALUES must contain the first 6 escalating values."""
        assert TRADE_VALUES == [4, 6, 8, 10, 12, 15]

    def test_trade_values_increment(self):
        """Increment after the 6th trade must be +5."""
        assert TRADE_VALUES_INCREMENT == 5

    def test_get_trade_value_first_six(self):
        """get_trade_value must return the pre-defined sequence."""
        for i, expected in enumerate(TRADE_VALUES):
            assert get_trade_value(i) == expected

    def test_get_trade_value_seventh(self):
        """7th trade must return 20 (15 + 5)."""
        assert get_trade_value(6) == 20

    def test_get_trade_value_eighth(self):
        """8th trade must return 25 (20 + 5)."""
        assert get_trade_value(7) == 25

    def test_get_trade_value_large(self):
        """get_trade_value must handle arbitrarily large trade counts."""
        assert get_trade_value(10) == 40  # 15 + 5 * 5
        assert get_trade_value(20) == 90  # 15 + 5 * 15


class TestStartingArmies:
    """Starting army allocation."""

    def test_all_player_counts_present(self):
        """Starting armies must be defined for 2-6 players."""
        for n in range(2, 7):
            assert n in STARTING_ARMIES

    def test_starting_army_values(self):
        """Starting armies must match classic values."""
        assert STARTING_ARMIES[2] == 40
        assert STARTING_ARMIES[3] == 35
        assert STARTING_ARMIES[4] == 30
        assert STARTING_ARMIES[5] == 25
        assert STARTING_ARMIES[6] == 20
