"""Tests for the hand-authored board layout.

The coordinates are typed by hand, so the invariants that matter are the ones a typo
would break: one coordinate per territory, all in frame, no two territories stacked on
each other, and each continent forming a coherent blob rather than being scattered.
"""

from __future__ import annotations

import math

import pytest

from src.utils.constants import CONTINENTS, NUM_TERRITORIES, TERRITORY_NAMES
from visualization import map_layout


def test_every_territory_has_a_coordinate() -> None:
    layout = map_layout.get_layout()
    assert len(layout) == NUM_TERRITORIES
    assert set(layout) == set(range(NUM_TERRITORIES))


def test_coordinates_are_inside_the_frame() -> None:
    for territory, (x, y) in map_layout.get_layout().items():
        assert 0.0 <= x <= 1.0, TERRITORY_NAMES[territory]
        assert 0.0 <= y <= 1.0, TERRITORY_NAMES[territory]


def test_no_two_territories_sit_on_top_of_each_other() -> None:
    layout = map_layout.get_layout()
    for a in range(NUM_TERRITORIES):
        for b in range(a + 1, NUM_TERRITORIES):
            distance = math.dist(layout[a], layout[b])
            assert distance > 0.03, f"{TERRITORY_NAMES[a]} overlaps {TERRITORY_NAMES[b]}"


def test_get_layout_returns_a_copy() -> None:
    layout = map_layout.get_layout()
    layout[0] = (0.0, 0.0)
    assert map_layout.get_layout()[0] != (0.0, 0.0)


@pytest.mark.parametrize("continent", sorted(CONTINENTS))
def test_continent_is_spatially_coherent(continent: str) -> None:
    """Every territory of a continent sits closer to its own centroid than 0.35."""
    layout = map_layout.get_layout()
    centroid = map_layout.continent_centroid(continent)
    for territory in CONTINENTS[continent]:
        spread = math.dist(layout[territory], centroid)
        assert spread < 0.35, f"{TERRITORY_NAMES[territory]} is far from {continent}"


@pytest.mark.parametrize("continent", sorted(CONTINENTS))
def test_continent_hull_is_a_polygon(continent: str) -> None:
    hull = map_layout.continent_hull(continent)
    assert len(hull) >= 3
    assert len(hull) == len(set(hull)), "hull repeats a vertex"


def test_wrap_edges_are_real_adjacencies() -> None:
    from src.utils.constants import ADJACENCY

    for a, b in map_layout.WRAP_EDGES:
        assert b in ADJACENCY[a], f"{TERRITORY_NAMES[a]} is not adjacent to {TERRITORY_NAMES[b]}"


def test_label_below_ids_are_valid_territories() -> None:
    assert set(range(NUM_TERRITORIES)) >= map_layout.LABEL_BELOW
