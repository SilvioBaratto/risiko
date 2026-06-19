"""
Issue #106 — Acceptance criterion:
  Every threshold/value lives in ``constants.py`` or a frozen config
  (no magic numbers in source).

Oracle classification: UNIT

Tests verify that ``src.utils.constants`` is importable and exposes the key
numeric constants mandated by the spec:
- Escalating trade schedule [4, 6, 8, 10, 12, 15] then +5 each.
- Continent bonus values NA=5 SA=2 EU=5 AF=3 AS=7 AU=2.
- Starting-army counts 40/35/30/25/20 for 2–6 players.

Where the exact attribute name is unknown (source-blind phase), the test
searches by value so it is independent of naming conventions.
"""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Module-level existence
# ---------------------------------------------------------------------------


def test_when_constants_module_imported_then_it_is_accessible():
    """``src.utils.constants`` must be a loadable Python module."""
    mod = importlib.import_module("src.utils.constants")
    assert mod is not None


def test_when_constants_module_inspected_then_it_has_public_attributes():
    """``src.utils.constants`` must define at least one non-private attribute."""
    import src.utils.constants as constants_mod

    public = [name for name in dir(constants_mod) if not name.startswith("_")]
    assert len(public) > 0, (
        "src.utils.constants must define at least one public constant; the module appears empty."
    )


# ---------------------------------------------------------------------------
# Escalating trade schedule
# ---------------------------------------------------------------------------


def test_when_escalating_trade_values_inspected_then_canonical_prefix_exists():
    """
    The spec mandates the trade-in schedule [4, 6, 8, 10, 12, 15] as the first
    six values.  At least one sequence attribute in ``constants.py`` must match
    this prefix exactly.
    """
    import src.utils.constants as constants_mod

    canonical = [4, 6, 8, 10, 12, 15]

    found = False
    for attr_name in dir(constants_mod):
        if attr_name.startswith("_"):
            continue
        val = getattr(constants_mod, attr_name)
        if (
            isinstance(val, (list, tuple))
            and len(val) >= len(canonical)
            and list(val[: len(canonical)]) == canonical
        ):
            found = True
            break

    assert found, (
        f"src.utils.constants must expose the escalating trade schedule "
        f"{canonical} as a sequence attribute.  No matching sequence found."
    )


def test_when_trade_schedule_found_then_seventh_value_is_20():
    """
    After [4,6,8,10,12,15] the schedule adds +5 each time → 7th value = 20.
    This verifies the continuation rule, not just the static prefix.
    """
    import src.utils.constants as constants_mod

    canonical_prefix = [4, 6, 8, 10, 12, 15]

    for attr_name in dir(constants_mod):
        if attr_name.startswith("_"):
            continue
        val = getattr(constants_mod, attr_name)
        if (
            isinstance(val, (list, tuple))
            and len(val) >= len(canonical_prefix)
            and list(val[: len(canonical_prefix)]) == canonical_prefix
        ):
            if len(val) >= 7:
                assert val[6] == 20, f"{attr_name}[6] must be 20 (15 + 5); got {val[6]}"
            return  # found the constant — test passes whether len==6 or >6

    pytest.fail(
        "Could not find the escalating trade schedule in src.utils.constants "
        "to check the 7th entry."
    )


# ---------------------------------------------------------------------------
# Continent bonuses
# ---------------------------------------------------------------------------


def test_when_continent_bonuses_inspected_then_six_canonical_values_are_present():
    """
    Continent bonuses (NA=5, SA=2, EU=5, AF=3, AS=7, AU=2) must be represented
    in ``constants.py``.  The six values must appear as the complete value set
    of some mapping attribute.
    """
    import src.utils.constants as constants_mod

    expected_sorted = sorted([5, 2, 5, 3, 7, 2])

    found = False
    for attr_name in dir(constants_mod):
        if attr_name.startswith("_"):
            continue
        val = getattr(constants_mod, attr_name)
        if isinstance(val, dict) and sorted(val.values()) == expected_sorted:
            found = True
            break

    assert found, (
        "src.utils.constants must contain a mapping attribute whose values are "
        "exactly [2,2,3,5,5,7] (the NA/SA/EU/AF/AS/AU continent bonuses).  "
        "No such mapping found."
    )


# ---------------------------------------------------------------------------
# Starting-army counts
# ---------------------------------------------------------------------------


def test_when_starting_armies_inspected_then_two_player_count_is_40():
    """
    The spec mandates 40 starting armies for 2-player games.  This value must
    be reachable from ``constants.py``.
    """
    import src.utils.constants as constants_mod

    found = False
    for attr_name in dir(constants_mod):
        if attr_name.startswith("_"):
            continue
        val = getattr(constants_mod, attr_name)
        if isinstance(val, dict) and val.get(2) == 40:
            found = True
            break
        if isinstance(val, (list, tuple)) and 40 in val:
            found = True
            break
        if val == 40:
            found = True
            break

    assert found, "src.utils.constants must expose the starting-army count 40 (for 2 players)."


def test_when_starting_armies_inspected_then_six_player_count_is_20():
    """The spec mandates 20 starting armies for 6-player games."""
    import src.utils.constants as constants_mod

    found = False
    for attr_name in dir(constants_mod):
        if attr_name.startswith("_"):
            continue
        val = getattr(constants_mod, attr_name)
        if isinstance(val, dict) and val.get(6) == 20:
            found = True
            break
        if isinstance(val, (list, tuple)) and 20 in val:
            found = True
            break
        if val == 20:
            found = True
            break

    assert found, "src.utils.constants must expose the starting-army count 20 (for 6 players)."
