"""Canonical tests for aggregate_diplomacy / with_diplomacy — issue #127.

Imports directly from training.tournament_stats (the canonical location).
Uses actual catalog strategy slugs from src/agents/strategies.py.
"""

from __future__ import annotations

from training.tournament_stats import (
    StrategyStat,
    aggregate_diplomacy,
    with_diplomacy,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SEAT_STRATS: dict[str, str] = {
    "0": "australia_lock",
    "1": "south_america_lock",
    "2": "aggressive_blitz",
    "3": "turtle_defensive",
    "4": "card_cycle_hunter",
    "5": "diplomat_coalition",
}
_ALL_CATALOG_STRATS: list[str] = list(_SEAT_STRATS.values())


def _rec(**overrides: object) -> dict:
    """Minimal ledger record for diplomacy tests (only seat_strategies required here)."""
    return {"seat_strategies": _SEAT_STRATS, **overrides}


def _make_stat(strategy: str, *, wins: int = 0, games: int = 5) -> StrategyStat:
    return StrategyStat(
        strategy=strategy,
        games=games,
        wins=wins,
        win_rate=wins / games if games else 0.0,
        ci_low=0.0,
        ci_high=1.0,
        mean_placement=0.0,
    )


# ---------------------------------------------------------------------------
# aggregate_diplomacy — no diplomacy fields present
# ---------------------------------------------------------------------------


class TestAggregateDiplomacyNoDiplomacy:
    def test_when_record_has_no_diplomacy_keys_then_no_exception_raised(self):
        aggregate_diplomacy([_rec()])

    def test_when_record_has_no_diplomacy_keys_then_all_strategies_report_zero_zero(self):
        result = aggregate_diplomacy([_rec()])
        for strategy in _ALL_CATALOG_STRATS:
            assert result[strategy] == (0, 0)

    def test_when_three_records_have_no_diplomacy_then_all_strategies_report_zero_zero(self):
        result = aggregate_diplomacy([_rec(), _rec(), _rec()])
        for strategy in _ALL_CATALOG_STRATS:
            assert result[strategy] == (0, 0)

    def test_when_no_diplomacy_then_result_keys_match_all_strategies(self):
        result = aggregate_diplomacy([_rec()])
        assert set(result.keys()) == set(_ALL_CATALOG_STRATS)


# ---------------------------------------------------------------------------
# aggregate_diplomacy — seat-keyed form ({"0": n})
# ---------------------------------------------------------------------------


class TestAggregateDiplomacySeatKeyed:
    def test_when_seat_keyed_betrayals_then_mapped_via_seat_strategies(self):
        record = _rec(betrayals={"2": 3})  # seat 2 = aggressive_blitz
        result = aggregate_diplomacy([record])
        betrayals, _ = result["aggressive_blitz"]
        assert betrayals == 3

    def test_when_seat_keyed_alliances_then_mapped_via_seat_strategies(self):
        record = _rec(alliances={"5": 4})  # seat 5 = diplomat_coalition
        result = aggregate_diplomacy([record])
        _, alliances = result["diplomat_coalition"]
        assert alliances == 4

    def test_when_partial_seat_coverage_then_uncovered_strategies_return_zero(self):
        record = _rec(betrayals={"0": 1})  # only seat 0 = australia_lock
        result = aggregate_diplomacy([record])
        for strategy in _ALL_CATALOG_STRATS:
            if strategy != "australia_lock":
                betrayals, _ = result[strategy]
                assert betrayals == 0


# ---------------------------------------------------------------------------
# aggregate_diplomacy — strategy-keyed form ({"aggressive_blitz": n})
# ---------------------------------------------------------------------------


class TestAggregateDiplomacyStrategyKeyed:
    def test_when_strategy_keyed_betrayals_then_counted_directly(self):
        record = _rec(betrayals={"aggressive_blitz": 5})
        result = aggregate_diplomacy([record])
        betrayals, _ = result["aggressive_blitz"]
        assert betrayals == 5

    def test_when_strategy_keyed_alliances_then_counted_directly(self):
        record = _rec(alliances={"diplomat_coalition": 2})
        result = aggregate_diplomacy([record])
        _, alliances = result["diplomat_coalition"]
        assert alliances == 2


# ---------------------------------------------------------------------------
# aggregate_diplomacy — summation across records
# ---------------------------------------------------------------------------


class TestAggregateDiplomacySummation:
    def test_when_two_seat_keyed_betrayal_records_then_totals_summed(self):
        records = [_rec(betrayals={"2": 1}), _rec(betrayals={"2": 2})]
        result = aggregate_diplomacy(records)
        betrayals, _ = result["aggressive_blitz"]
        assert betrayals == 3

    def test_when_two_strategy_keyed_alliance_records_then_totals_summed(self):
        records = [_rec(alliances={"australia_lock": 1}), _rec(alliances={"australia_lock": 3})]
        result = aggregate_diplomacy(records)
        _, alliances = result["australia_lock"]
        assert alliances == 4

    def test_when_seat_and_strategy_keyed_records_mixed_then_all_accumulated(self):
        records = [
            _rec(betrayals={"2": 2}),  # seat-keyed → aggressive_blitz
            _rec(betrayals={"aggressive_blitz": 1}),  # strategy-keyed
        ]
        result = aggregate_diplomacy(records)
        betrayals, _ = result["aggressive_blitz"]
        assert betrayals == 3

    def test_when_betrayals_and_alliances_present_then_dimensions_are_independent(self):
        record = _rec(betrayals={"aggressive_blitz": 2}, alliances={"aggressive_blitz": 7})
        result = aggregate_diplomacy([record])
        assert result["aggressive_blitz"] == (2, 7)


# ---------------------------------------------------------------------------
# with_diplomacy — field population and preservation
# ---------------------------------------------------------------------------


class TestWithDiplomacy:
    def test_when_records_have_betrayals_then_stat_betrayals_populated(self):
        stat = _make_stat("aggressive_blitz", wins=5, games=10)
        record = _rec(betrayals={"aggressive_blitz": 7})
        enriched = with_diplomacy((stat,), [record])
        ag = next(s for s in enriched if s.strategy == "aggressive_blitz")
        assert ag.betrayals == 7

    def test_when_records_have_alliances_then_stat_alliances_populated(self):
        stat = _make_stat("diplomat_coalition", wins=3, games=8)
        record = _rec(alliances={"diplomat_coalition": 6})
        enriched = with_diplomacy((stat,), [record])
        dip = next(s for s in enriched if s.strategy == "diplomat_coalition")
        assert dip.alliances == 6

    def test_when_seat_keyed_betrayals_then_stat_betrayals_populated(self):
        stat = _make_stat("aggressive_blitz")
        record = _rec(betrayals={"2": 4})  # seat 2 = aggressive_blitz
        enriched = with_diplomacy((stat,), [record])
        ag = next(s for s in enriched if s.strategy == "aggressive_blitz")
        assert ag.betrayals == 4

    def test_when_no_diplomacy_in_records_then_enriched_stats_have_zero_counts(self):
        stats = tuple(_make_stat(s) for s in _ALL_CATALOG_STRATS)
        enriched = with_diplomacy(stats, [_rec()])
        for s in enriched:
            assert s.betrayals == 0
            assert s.alliances == 0

    def test_when_with_diplomacy_called_then_wins_field_is_unchanged(self):
        stat = _make_stat("turtle_defensive", wins=4, games=12)
        enriched = with_diplomacy((stat,), [_rec()])
        turtle = next(s for s in enriched if s.strategy == "turtle_defensive")
        assert turtle.wins == 4

    def test_when_with_diplomacy_called_then_games_field_is_unchanged(self):
        stat = _make_stat("turtle_defensive", wins=4, games=12)
        enriched = with_diplomacy((stat,), [_rec()])
        turtle = next(s for s in enriched if s.strategy == "turtle_defensive")
        assert turtle.games == 12

    def test_when_with_diplomacy_called_then_output_length_equals_input_length(self):
        stats = tuple(_make_stat(s) for s in _ALL_CATALOG_STRATS)
        enriched = with_diplomacy(stats, [_rec()])
        assert len(enriched) == len(stats)

    def test_when_with_diplomacy_called_then_result_is_a_tuple(self):
        stats = tuple(_make_stat(s) for s in _ALL_CATALOG_STRATS)
        enriched = with_diplomacy(stats, [_rec()])
        assert isinstance(enriched, tuple)

    def test_when_with_diplomacy_called_then_strategy_names_preserved(self):
        stats = tuple(_make_stat(s) for s in _ALL_CATALOG_STRATS)
        enriched = with_diplomacy(stats, [_rec()])
        assert {s.strategy for s in enriched} == set(_ALL_CATALOG_STRATS)
