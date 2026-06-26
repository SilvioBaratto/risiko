"""Per-strategy alliance/betrayal aggregation tests — issue #127.

Source-blind example tests written from acceptance criteria only.
No implementation source was read.

Expected imports:
  training.tournament_analysis.aggregate_diplomacy
  training.tournament_analysis.with_diplomacy
  training.tournament_analysis.StrategyStat
"""

from hypothesis import given
from hypothesis import strategies as st

from training.tournament_analysis import StrategyStat, aggregate_diplomacy, with_diplomacy

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SEAT_STRATEGIES: dict[str, str] = {
    "0": "aggressive_blitz",
    "1": "turtle",
    "2": "diplomat",
    "3": "australia_lock",
    "4": "south_america_lock",
    "5": "card_cycle",
}
_ALL_STRATEGIES: list[str] = list(_SEAT_STRATEGIES.values())


def _bare_record() -> dict:
    """Game record with seat_strategies present but no diplomacy count fields."""
    return {"seat_strategies": _SEAT_STRATEGIES}


def _make_stat(strategy: str, *, wins: int = 0, games: int = 0) -> StrategyStat:
    """Construct a minimal StrategyStat for test input."""
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
# aggregate_diplomacy — example tests
# ---------------------------------------------------------------------------


class TestAggregateDiplomacyNoDiplomacyFields:
    """Criterion: Optional fields absent → every strategy reports (0, 0); no crash, no KeyError."""

    def test_when_no_diplomacy_fields_then_no_exception_raised(self):
        aggregate_diplomacy([_bare_record()])  # must not raise

    def test_when_no_diplomacy_fields_then_every_strategy_reports_zero_zero(self):
        result = aggregate_diplomacy([_bare_record()])
        for strategy in _ALL_STRATEGIES:
            assert result[strategy] == (0, 0)

    def test_when_multiple_records_with_no_diplomacy_then_every_strategy_reports_zero_zero(self):
        result = aggregate_diplomacy([_bare_record(), _bare_record(), _bare_record()])
        for strategy in _ALL_STRATEGIES:
            assert result[strategy] == (0, 0)


class TestAggregateDiplomacySeatKeyedForm:
    """Criterion: seat-keyed ({"0": n}) diplomacy dict form is tolerated."""

    def test_when_seat_keyed_betrayals_then_mapped_to_correct_strategy(self):
        record = {**_bare_record(), "betrayals": {"0": 3}}
        result = aggregate_diplomacy([record])
        betrayals, _ = result["aggressive_blitz"]
        assert betrayals == 3

    def test_when_seat_keyed_alliances_then_mapped_to_correct_strategy(self):
        record = {**_bare_record(), "alliances": {"1": 4}}
        result = aggregate_diplomacy([record])
        _, alliances = result["turtle"]
        assert alliances == 4

    def test_when_partial_seat_coverage_in_betrayals_then_uncovered_strategies_are_zero(self):
        record = {**_bare_record(), "betrayals": {"0": 1}}
        result = aggregate_diplomacy([record])
        for strategy in _ALL_STRATEGIES:
            if strategy != "aggressive_blitz":
                betrayals, _ = result[strategy]
                assert betrayals == 0


class TestAggregateDiplomacyStrategyKeyedForm:
    """Criterion: strategy-keyed ({"aggressive_blitz": n}) diplomacy dict form is tolerated."""

    def test_when_strategy_keyed_betrayals_then_counted_for_that_strategy(self):
        record = {**_bare_record(), "betrayals": {"aggressive_blitz": 5}}
        result = aggregate_diplomacy([record])
        betrayals, _ = result["aggressive_blitz"]
        assert betrayals == 5

    def test_when_strategy_keyed_alliances_then_counted_for_that_strategy(self):
        record = {**_bare_record(), "alliances": {"diplomat": 2}}
        result = aggregate_diplomacy([record])
        _, alliances = result["diplomat"]
        assert alliances == 2


class TestAggregateDiplomacySummation:
    """Criterion: per-strategy (betrayals, alliances) totals summed across records."""

    def test_when_two_seat_keyed_betrayal_records_then_totals_are_summed(self):
        records = [
            {**_bare_record(), "betrayals": {"0": 1}},
            {**_bare_record(), "betrayals": {"0": 2}},
        ]
        result = aggregate_diplomacy(records)
        betrayals, _ = result["aggressive_blitz"]
        assert betrayals == 3

    def test_when_two_strategy_keyed_alliance_records_then_totals_are_summed(self):
        records = [
            {**_bare_record(), "alliances": {"aggressive_blitz": 1}},
            {**_bare_record(), "alliances": {"aggressive_blitz": 3}},
        ]
        result = aggregate_diplomacy(records)
        _, alliances = result["aggressive_blitz"]
        assert alliances == 4

    def test_when_betrayals_and_alliances_in_same_record_then_each_dimension_correct(self):
        record = {
            **_bare_record(),
            "betrayals": {"aggressive_blitz": 2},
            "alliances": {"aggressive_blitz": 7},
        }
        result = aggregate_diplomacy([record])
        assert result["aggressive_blitz"] == (2, 7)

    def test_when_one_seat_keyed_and_one_strategy_keyed_betrayal_record_then_both_accumulated(self):
        """Criterion: both dict-key forms tolerated — mixed across records."""
        records = [
            {**_bare_record(), "betrayals": {"0": 2}},  # seat-keyed
            {**_bare_record(), "betrayals": {"aggressive_blitz": 1}},  # strategy-keyed
        ]
        result = aggregate_diplomacy(records)
        betrayals, _ = result["aggressive_blitz"]
        assert betrayals == 3

    def test_when_called_then_result_contains_key_for_every_strategy(self):
        result = aggregate_diplomacy([_bare_record()])
        assert set(result.keys()) == set(_ALL_STRATEGIES)


# ---------------------------------------------------------------------------
# with_diplomacy — example tests
# ---------------------------------------------------------------------------


class TestWithDiplomacyPopulatesFields:
    """Criterion: with_diplomacy returns StrategyStat with betrayals/alliances populated."""

    def test_when_strategy_keyed_betrayals_then_stat_betrayals_populated(self):
        stat = _make_stat("aggressive_blitz", wins=5, games=10)
        record = {**_bare_record(), "betrayals": {"aggressive_blitz": 7}}
        enriched = with_diplomacy((stat,), [record])
        ag = next(s for s in enriched if s.strategy == "aggressive_blitz")
        assert ag.betrayals == 7

    def test_when_strategy_keyed_alliances_then_stat_alliances_populated(self):
        stat = _make_stat("diplomat", wins=3, games=8)
        record = {**_bare_record(), "alliances": {"diplomat": 6}}
        enriched = with_diplomacy((stat,), [record])
        dip = next(s for s in enriched if s.strategy == "diplomat")
        assert dip.alliances == 6

    def test_when_records_have_seat_keyed_betrayals_then_returned_stat_betrayals_populated(self):
        stat = _make_stat("aggressive_blitz")
        record = {**_bare_record(), "betrayals": {"0": 4}}
        enriched = with_diplomacy((stat,), [record])
        ag = next(s for s in enriched if s.strategy == "aggressive_blitz")
        assert ag.betrayals == 4

    def test_when_no_diplomacy_in_records_then_all_enriched_stats_have_zero_counts(self):
        """Criterion: Optional fields absent → every strategy reports (0, 0)."""
        stats = tuple(_make_stat(s) for s in _ALL_STRATEGIES)
        enriched = with_diplomacy(stats, [_bare_record()])
        for s in enriched:
            assert s.betrayals == 0
            assert s.alliances == 0


class TestWithDiplomacyPreservesOtherFields:
    """Criterion: all other fields unchanged."""

    def test_when_with_diplomacy_called_then_wins_field_is_unchanged(self):
        stat = _make_stat("turtle", wins=4, games=12)
        enriched = with_diplomacy((stat,), [_bare_record()])
        turtle = next(s for s in enriched if s.strategy == "turtle")
        assert turtle.wins == 4

    def test_when_with_diplomacy_called_then_games_field_is_unchanged(self):
        stat = _make_stat("turtle", wins=4, games=12)
        enriched = with_diplomacy((stat,), [_bare_record()])
        turtle = next(s for s in enriched if s.strategy == "turtle")
        assert turtle.games == 12

    def test_when_with_diplomacy_called_then_strategy_name_is_unchanged(self):
        stats = tuple(_make_stat(s) for s in _ALL_STRATEGIES)
        enriched = with_diplomacy(stats, [_bare_record()])
        assert {s.strategy for s in enriched} == set(_ALL_STRATEGIES)

    def test_when_with_diplomacy_called_then_output_length_equals_input_length(self):
        stats = tuple(_make_stat(s) for s in _ALL_STRATEGIES)
        enriched = with_diplomacy(stats, [_bare_record()])
        assert len(enriched) == len(stats)

    def test_when_with_diplomacy_called_then_result_is_a_tuple(self):
        stats = tuple(_make_stat(s) for s in _ALL_STRATEGIES)
        enriched = with_diplomacy(stats, [_bare_record()])
        assert isinstance(enriched, tuple)


# ---------------------------------------------------------------------------
# Property-based tests — invariants derived from acceptance criteria
# ---------------------------------------------------------------------------

_records_no_diplomacy = st.lists(
    st.fixed_dictionaries({"seat_strategies": st.just(_SEAT_STRATEGIES)}),
    min_size=1,
    max_size=30,
)

_nonneg_per_record = st.lists(
    st.integers(min_value=0, max_value=200),
    min_size=0,
    max_size=30,
)


@given(_records_no_diplomacy)
def test_when_any_count_of_records_lack_diplomacy_fields_then_all_counts_are_zero(records):
    """Invariant: records without diplomacy fields always yield (0, 0) for every strategy."""
    result = aggregate_diplomacy(records)
    for strategy in _ALL_STRATEGIES:
        assert result[strategy] == (0, 0)


@given(_nonneg_per_record)
def test_when_strategy_keyed_betrayal_per_record_then_total_equals_arithmetic_sum(counts):
    """Invariant: aggregate betrayal total == sum of per-record counts (strategy-keyed)."""
    records = [{**_bare_record(), "betrayals": {"aggressive_blitz": c}} for c in counts]
    result = aggregate_diplomacy(records)
    betrayals, _ = result.get("aggressive_blitz", (0, 0))
    assert betrayals == sum(counts)


@given(_nonneg_per_record)
def test_when_seat_keyed_alliance_per_record_then_total_equals_arithmetic_sum(counts):
    """Invariant: alliance total == per-record sum (seat-keyed; seat 0 → aggressive_blitz)."""
    records = [{**_bare_record(), "alliances": {"0": c}} for c in counts]
    result = aggregate_diplomacy(records)
    _, alliances = result.get("aggressive_blitz", (0, 0))
    assert alliances == sum(counts)


@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=20),
    st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=20),
)
def test_when_betrayals_and_alliances_in_separate_records_then_each_accumulates_independently(
    betrayal_counts, alliance_counts
):
    """Invariant: betrayal and alliance dimensions accumulate independently across records."""
    b_records = [{**_bare_record(), "betrayals": {"diplomat": c}} for c in betrayal_counts]
    a_records = [{**_bare_record(), "alliances": {"diplomat": c}} for c in alliance_counts]
    result = aggregate_diplomacy(b_records + a_records)
    b, a = result.get("diplomat", (0, 0))
    assert b == sum(betrayal_counts)
    assert a == sum(alliance_counts)


@given(
    st.lists(
        st.sampled_from(_ALL_STRATEGIES),
        min_size=1,
        max_size=10,
    )
)
def test_when_aggregate_diplomacy_called_then_no_exception_raised_for_valid_records(strategies):
    """Invariant: aggregate_diplomacy never raises for valid records with seat_strategies."""
    records = [_bare_record() for _ in strategies]
    aggregate_diplomacy(records)  # must not raise
