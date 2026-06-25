"""Source-blind example tests for src/agents/strategies — Issue #114.

Tests are derived from the acceptance criteria only; no implementation source
was read when authoring these tests. Each test pins down one observable
behaviour demanded by a criterion.
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.agents.player_config import DEFAULT_MODEL, PlayerConfig
from src.agents.strategies import (
    CATALOG_BY_NAME,
    POSTURE_LEVERS,
    STRATEGY_CATALOG,
    Posture,
    Strategy,
    compose_strategy_hint,
    get_strategy,
    to_player_config,
)

# ---------------------------------------------------------------------------
# Strategy frozen dataclass
# ---------------------------------------------------------------------------


class TestStrategyDataclass:
    """Strategy is a frozen dataclass with the four required fields."""

    def test_when_strategy_accessed_name_field_is_present(self) -> None:
        """Strategy exposes a 'name' field (unique slug)."""
        strategy = STRATEGY_CATALOG[0]
        assert isinstance(strategy.name, str)

    def test_when_strategy_accessed_strategy_hint_field_is_present(self) -> None:
        """Strategy exposes a 'strategy_hint' field (board play text)."""
        strategy = STRATEGY_CATALOG[0]
        assert isinstance(strategy.strategy_hint, str)

    def test_when_strategy_accessed_posture_field_is_present(self) -> None:
        """Strategy exposes a 'posture' field (a Posture enum value)."""
        strategy = STRATEGY_CATALOG[0]
        assert isinstance(strategy.posture, Posture)

    def test_when_strategy_accessed_posture_directive_field_is_present(self) -> None:
        """Strategy exposes a 'posture_directive' field (one-line social text)."""
        strategy = STRATEGY_CATALOG[0]
        assert isinstance(strategy.posture_directive, str)

    def test_when_strategy_mutated_then_frozen_error_is_raised(self) -> None:
        """Strategy is frozen — attribute mutation raises FrozenInstanceError or AttributeError."""
        strategy = STRATEGY_CATALOG[0]
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            strategy.name = "mutated"  # type: ignore[misc]

    def test_when_strategy_class_checked_it_is_a_dataclass(self) -> None:
        """dataclasses.is_dataclass(Strategy) returns True."""
        assert dataclasses.is_dataclass(Strategy)


# ---------------------------------------------------------------------------
# Posture enum
# ---------------------------------------------------------------------------


class TestPostureEnum:
    """Posture subclasses (str, Enum) and is YAML/JSON-serialisable."""

    def test_when_posture_values_checked_each_is_an_instance_of_str(self) -> None:
        """Every Posture value is an instance of str (str-enum mixin)."""
        for posture in Posture:
            assert isinstance(posture, str)

    def test_when_posture_class_checked_it_subclasses_enum(self) -> None:
        """Posture is an Enum subclass."""
        assert issubclass(Posture, Enum)

    def test_when_posture_serialised_then_json_encodes_to_string(self) -> None:
        """json.dumps on a Posture value succeeds and decodes back to a string."""
        posture = next(iter(Posture))
        decoded = json.loads(json.dumps(posture))
        assert isinstance(decoded, str)


# ---------------------------------------------------------------------------
# STRATEGY_CATALOG and CATALOG_BY_NAME
# ---------------------------------------------------------------------------


class TestStrategyCatalog:
    """STRATEGY_CATALOG is a tuple of exactly 6 Strategy objects."""

    def test_when_catalog_type_checked_it_is_a_tuple(self) -> None:
        """STRATEGY_CATALOG is a tuple."""
        assert isinstance(STRATEGY_CATALOG, tuple)

    def test_when_catalog_counted_it_has_exactly_six_entries(self) -> None:
        """STRATEGY_CATALOG contains exactly 6 Strategy entries."""
        assert len(STRATEGY_CATALOG) == 6

    def test_when_catalog_elements_checked_all_are_strategy_instances(self) -> None:
        """Every element of STRATEGY_CATALOG is a Strategy."""
        for s in STRATEGY_CATALOG:
            assert isinstance(s, Strategy)

    def test_when_catalog_by_name_type_checked_it_is_a_dict(self) -> None:
        """CATALOG_BY_NAME is a dict."""
        assert isinstance(CATALOG_BY_NAME, dict)

    def test_when_catalog_by_name_counted_it_has_exactly_six_entries(self) -> None:
        """CATALOG_BY_NAME has exactly 6 entries."""
        assert len(CATALOG_BY_NAME) == 6

    def test_when_catalog_by_name_keys_checked_they_match_strategy_name_slugs(self) -> None:
        """CATALOG_BY_NAME keys equal the set of strategy name slugs in STRATEGY_CATALOG."""
        expected = {s.name for s in STRATEGY_CATALOG}
        assert set(CATALOG_BY_NAME.keys()) == expected

    def test_when_catalog_by_name_values_checked_each_matches_catalog_entry(self) -> None:
        """CATALOG_BY_NAME[s.name] returns the same Strategy as in the tuple."""
        for s in STRATEGY_CATALOG:
            assert CATALOG_BY_NAME[s.name] == s


# ---------------------------------------------------------------------------
# Distinctness and non-emptiness
# ---------------------------------------------------------------------------


class TestCatalogDistinctness:
    """Six distinct non-empty strategy_hints; six distinct postures; non-empty directives."""

    def test_when_strategy_hints_checked_all_are_non_empty(self) -> None:
        """Every strategy_hint is a non-empty string."""
        for s in STRATEGY_CATALOG:
            assert len(s.strategy_hint) > 0, f"Empty strategy_hint for {s.name}"

    def test_when_strategy_hints_checked_all_are_distinct(self) -> None:
        """All six strategy_hint values are distinct."""
        hints = [s.strategy_hint for s in STRATEGY_CATALOG]
        assert len(set(hints)) == 6

    def test_when_postures_checked_all_are_distinct(self) -> None:
        """All six posture values are distinct."""
        postures = [s.posture for s in STRATEGY_CATALOG]
        assert len(set(postures)) == 6

    def test_when_posture_directives_checked_all_are_non_empty(self) -> None:
        """Every posture_directive is a non-empty string."""
        for s in STRATEGY_CATALOG:
            assert len(s.posture_directive) > 0, f"Empty posture_directive for {s.name}"

    def test_when_names_checked_all_are_distinct(self) -> None:
        """All six name slugs are distinct."""
        names = [s.name for s in STRATEGY_CATALOG]
        assert len(set(names)) == 6


# ---------------------------------------------------------------------------
# POSTURE_LEVERS
# ---------------------------------------------------------------------------


class TestPostureLevers:
    """POSTURE_LEVERS maps each Posture to a distinct (trust, vengefulness) pair in [0,1]."""

    def test_when_posture_levers_type_checked_it_is_a_dict(self) -> None:
        """POSTURE_LEVERS is a dict."""
        assert isinstance(POSTURE_LEVERS, dict)

    def test_when_posture_levers_counted_it_has_exactly_six_entries(self) -> None:
        """POSTURE_LEVERS has exactly 6 entries."""
        assert len(POSTURE_LEVERS) == 6

    def test_when_posture_levers_keys_checked_all_catalog_postures_are_covered(self) -> None:
        """POSTURE_LEVERS covers all postures that appear in STRATEGY_CATALOG."""
        catalog_postures = {s.posture for s in STRATEGY_CATALOG}
        assert catalog_postures <= set(POSTURE_LEVERS.keys())

    def test_when_posture_levers_values_checked_each_is_a_two_element_sequence(self) -> None:
        """Each POSTURE_LEVERS value is a 2-element sequence (trust, vengefulness)."""
        for posture, levers in POSTURE_LEVERS.items():
            assert len(levers) == 2, f"Expected 2-tuple for posture {posture}"

    def test_when_posture_levers_values_checked_elements_are_numeric(self) -> None:
        """Each lever value is a numeric type (int or float)."""
        for posture, (trust, vengefulness) in POSTURE_LEVERS.items():
            assert isinstance(trust, (int, float)), f"trust for {posture} is not numeric"
            assert isinstance(vengefulness, (int, float)), (
                f"vengefulness for {posture} is not numeric"
            )

    def test_when_posture_levers_checked_all_trust_values_in_unit_interval(self) -> None:
        """Every trust_propensity value is in [0, 1]."""
        for posture, (trust, _) in POSTURE_LEVERS.items():
            assert 0.0 <= trust <= 1.0, f"trust for {posture} out of [0, 1]: {trust}"

    def test_when_posture_levers_checked_all_vengefulness_values_in_unit_interval(self) -> None:
        """Every vengefulness value is in [0, 1]."""
        for posture, (_, vengefulness) in POSTURE_LEVERS.items():
            assert 0.0 <= vengefulness <= 1.0, (
                f"vengefulness for {posture} out of [0, 1]: {vengefulness}"
            )

    def test_when_posture_levers_pairs_checked_all_are_distinct(self) -> None:
        """The six (trust, vengefulness) pairs are all distinct."""
        pairs = [tuple(v) for v in POSTURE_LEVERS.values()]
        assert len(set(pairs)) == 6


# ---------------------------------------------------------------------------
# compose_strategy_hint
# ---------------------------------------------------------------------------


class TestComposeStrategyHint:
    """compose_strategy_hint returns a string containing both substrings."""

    def test_when_composed_then_result_is_a_string(self) -> None:
        """compose_strategy_hint returns a str."""
        strategy = STRATEGY_CATALOG[0]
        assert isinstance(compose_strategy_hint(strategy), str)

    def test_when_composed_then_result_contains_strategy_hint(self) -> None:
        """compose_strategy_hint result contains the strategy's strategy_hint as a substring."""
        strategy = STRATEGY_CATALOG[0]
        assert strategy.strategy_hint in compose_strategy_hint(strategy)

    def test_when_composed_then_result_contains_posture_directive(self) -> None:
        """compose_strategy_hint result contains the strategy's posture_directive as a substring."""
        strategy = STRATEGY_CATALOG[0]
        assert strategy.posture_directive in compose_strategy_hint(strategy)

    @given(st.integers(min_value=0, max_value=5))
    def test_when_composed_for_any_catalog_index_both_substrings_appear(self, index: int) -> None:
        """Invariant: for every catalog strategy, both hint and directive appear in the result."""
        strategy = STRATEGY_CATALOG[index]
        result = compose_strategy_hint(strategy)
        assert strategy.strategy_hint in result, (
            f"strategy_hint missing from composed result for {strategy.name}"
        )
        assert strategy.posture_directive in result, (
            f"posture_directive missing from composed result for {strategy.name}"
        )


# ---------------------------------------------------------------------------
# to_player_config
# ---------------------------------------------------------------------------


class TestToPlayerConfig:
    """to_player_config builds a correct frozen PlayerConfig from a Strategy."""

    def test_when_to_player_config_called_then_result_is_player_config(self) -> None:
        """to_player_config returns a PlayerConfig instance."""
        strategy = STRATEGY_CATALOG[0]
        result = to_player_config(strategy, player_id=0, temperature=0.5, top_p=0.9)
        assert isinstance(result, PlayerConfig)

    def test_when_to_player_config_called_then_strategy_hint_matches_composed(self) -> None:
        """PlayerConfig.strategy_hint equals compose_strategy_hint(strategy)."""
        strategy = STRATEGY_CATALOG[0]
        result = to_player_config(strategy, player_id=0, temperature=0.5, top_p=0.9)
        assert result.strategy_hint == compose_strategy_hint(strategy)

    def test_when_to_player_config_called_then_player_id_is_passed_through(self) -> None:
        """PlayerConfig.player_id equals the provided player_id argument."""
        strategy = STRATEGY_CATALOG[0]
        result = to_player_config(strategy, player_id=3, temperature=0.5, top_p=0.9)
        assert result.player_id == 3

    def test_when_to_player_config_called_then_temperature_is_passed_through(self) -> None:
        """PlayerConfig.temperature equals the provided temperature argument."""
        strategy = STRATEGY_CATALOG[0]
        result = to_player_config(strategy, player_id=0, temperature=0.7, top_p=0.9)
        assert result.temperature == pytest.approx(0.7)

    def test_when_to_player_config_called_then_top_p_is_passed_through(self) -> None:
        """PlayerConfig.top_p equals the provided top_p argument."""
        strategy = STRATEGY_CATALOG[0]
        result = to_player_config(strategy, player_id=0, temperature=0.5, top_p=0.85)
        assert result.top_p == pytest.approx(0.85)

    def test_when_to_player_config_called_without_model_then_default_model_is_used(
        self,
    ) -> None:
        """When model is omitted, PlayerConfig.model equals DEFAULT_MODEL."""
        strategy = STRATEGY_CATALOG[0]
        result = to_player_config(strategy, player_id=0, temperature=0.5, top_p=0.9)
        assert result.model == DEFAULT_MODEL

    def test_when_to_player_config_called_with_model_then_model_is_passed_through(
        self,
    ) -> None:
        """When model is provided, PlayerConfig.model equals the provided model."""
        strategy = STRATEGY_CATALOG[0]
        result = to_player_config(
            strategy, player_id=0, temperature=0.5, top_p=0.9, model="test-model:cloud"
        )
        assert result.model == "test-model:cloud"

    def test_when_to_player_config_called_then_trust_propensity_from_posture_levers(
        self,
    ) -> None:
        """PlayerConfig.trust_propensity equals POSTURE_LEVERS[strategy.posture][0]."""
        strategy = STRATEGY_CATALOG[0]
        expected_trust, _ = POSTURE_LEVERS[strategy.posture]
        result = to_player_config(strategy, player_id=0, temperature=0.5, top_p=0.9)
        assert result.trust_propensity == pytest.approx(expected_trust)

    def test_when_to_player_config_called_then_vengefulness_from_posture_levers(
        self,
    ) -> None:
        """PlayerConfig.vengefulness equals POSTURE_LEVERS[strategy.posture][1]."""
        strategy = STRATEGY_CATALOG[0]
        _, expected_vengefulness = POSTURE_LEVERS[strategy.posture]
        result = to_player_config(strategy, player_id=0, temperature=0.5, top_p=0.9)
        assert result.vengefulness == pytest.approx(expected_vengefulness)

    def test_when_player_config_returned_then_it_is_frozen(self) -> None:
        """PlayerConfig returned by to_player_config is frozen (mutation raises)."""
        strategy = STRATEGY_CATALOG[0]
        result = to_player_config(strategy, player_id=0, temperature=0.5, top_p=0.9)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.player_id = 99  # type: ignore[misc]

    def test_when_to_player_config_called_for_all_catalog_strategies_then_levers_match(
        self,
    ) -> None:
        """to_player_config correctly maps posture→levers for every catalog strategy."""
        for strategy in STRATEGY_CATALOG:
            expected_trust, expected_vengefulness = POSTURE_LEVERS[strategy.posture]
            result = to_player_config(strategy, player_id=0, temperature=0.5, top_p=0.9)
            assert result.trust_propensity == pytest.approx(expected_trust), (
                f"trust mismatch for {strategy.name}"
            )
            assert result.vengefulness == pytest.approx(expected_vengefulness), (
                f"vengefulness mismatch for {strategy.name}"
            )


# ---------------------------------------------------------------------------
# get_strategy
# ---------------------------------------------------------------------------


class TestGetStrategy:
    """get_strategy returns the Strategy for a valid name; raises ValueError otherwise."""

    def test_when_get_strategy_called_with_valid_name_then_strategy_is_returned(
        self,
    ) -> None:
        """get_strategy returns a Strategy when given a name present in the catalog."""
        name = STRATEGY_CATALOG[0].name
        result = get_strategy(name)
        assert isinstance(result, Strategy)
        assert result.name == name

    def test_when_get_strategy_called_with_each_valid_name_then_matches_catalog(self) -> None:
        """get_strategy result equals the corresponding entry in STRATEGY_CATALOG."""
        for strategy in STRATEGY_CATALOG:
            assert get_strategy(strategy.name) == strategy

    def test_when_get_strategy_called_with_unknown_name_then_value_error_is_raised(
        self,
    ) -> None:
        """get_strategy raises ValueError for a name not present in the catalog."""
        with pytest.raises(ValueError):
            get_strategy("__nonexistent_strategy__")

    def test_when_get_strategy_called_with_empty_string_then_value_error_is_raised(
        self,
    ) -> None:
        """get_strategy raises ValueError for an empty string."""
        with pytest.raises(ValueError):
            get_strategy("")


# ---------------------------------------------------------------------------
# PlayerConfig fields guard
# ---------------------------------------------------------------------------

# The issue's Context section lists the 7 fields that PlayerConfig must retain
# unchanged: player_id, temperature, top_p, strategy_hint, model,
# trust_propensity, vengefulness.  The strategies module must not add new fields.

_EXPECTED_PLAYER_CONFIG_FIELD_NAMES = frozenset(
    {
        "player_id",
        "temperature",
        "top_p",
        "strategy_hint",
        "model",
        "trust_propensity",
        "vengefulness",
    }
)


class TestPlayerConfigFieldsUnchanged:
    """PlayerConfig must not gain new fields from the strategies module."""

    def test_when_player_config_fields_checked_set_matches_original_contract(self) -> None:
        """dataclasses.fields(PlayerConfig) names equal the original 7-field set."""
        actual = {f.name for f in dataclasses.fields(PlayerConfig)}
        assert actual == _EXPECTED_PLAYER_CONFIG_FIELD_NAMES

    def test_when_player_config_fields_counted_exactly_seven_fields_present(self) -> None:
        """PlayerConfig has exactly 7 fields — no additions from the strategies module."""
        assert len(dataclasses.fields(PlayerConfig)) == 7
