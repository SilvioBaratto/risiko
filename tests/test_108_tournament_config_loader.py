"""Example tests for load_tournament_config and LedgerRecord — Issue #116.

Derived solely from the acceptance criteria; no implementation source was read.
All tests are expected to FAIL (Red) until the implementation is written.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

CONFIG_PATH = pathlib.Path("config/tournament.yaml")

_VALID_STRATEGIES = [
    "australia_lock",
    "south_america_lock",
    "aggressive_blitz",
    "turtle_defensive",
    "card_cycle_hunter",
    "diplomat_coalition",
]

_VALID_MODELS = [
    "glm-5.2:cloud",
    "gemma4:31b-cloud",
    "glm-5.1:cloud",
    "kimi-k2.6:cloud",
    "deepseek-v4-flash:cloud",
    "qwen3.5:cloud",
]


# ---------------------------------------------------------------------------
# load_tournament_config: parses config/tournament.yaml into TournamentConfig
# ---------------------------------------------------------------------------


def test_when_tournament_config_loaded_then_seed_equals_42():
    """Criterion: seed == 42 in the parsed TournamentConfig."""
    from training.tournament import load_tournament_config

    cfg = load_tournament_config(CONFIG_PATH)
    assert cfg.seed == 42


def test_when_tournament_config_loaded_then_n_games_equals_30():
    """Criterion: n_games == 30 in the parsed TournamentConfig."""
    from training.tournament import load_tournament_config

    cfg = load_tournament_config(CONFIG_PATH)
    assert cfg.n_games == 30


def test_when_tournament_config_loaded_then_max_concurrency_equals_6():
    """Criterion: max_concurrency == 6 in the parsed TournamentConfig."""
    from training.tournament import load_tournament_config

    cfg = load_tournament_config(CONFIG_PATH)
    assert cfg.max_concurrency == 6


def test_when_tournament_config_loaded_then_n_rounds_is_at_least_one():
    """n_rounds >= 1 (reduced from 2 to 1 for throughput, paired with cadence)."""
    from training.tournament import load_tournament_config

    cfg = load_tournament_config(CONFIG_PATH)
    assert cfg.n_rounds >= 1


def test_when_tournament_config_loaded_then_negotiation_cadence_is_at_least_one():
    """negotiation_cadence >= 1 (negotiate every Nth player-turn)."""
    from training.tournament import load_tournament_config

    cfg = load_tournament_config(CONFIG_PATH)
    assert cfg.negotiation_cadence >= 1


def test_when_tournament_config_loaded_then_strategies_has_6_entries():
    """Criterion: 6 strategies in the parsed TournamentConfig."""
    from training.tournament import load_tournament_config

    cfg = load_tournament_config(CONFIG_PATH)
    assert len(cfg.strategies) == 6


def test_when_tournament_config_loaded_then_models_has_6_entries():
    """Criterion: 6 models in the parsed TournamentConfig."""
    from training.tournament import load_tournament_config

    cfg = load_tournament_config(CONFIG_PATH)
    assert len(cfg.models) == 6


def test_when_tournament_config_loaded_then_strategies_match_catalog():
    """Criterion: strategy slugs in the config match CATALOG_BY_NAME keys."""
    from training.tournament import load_tournament_config

    cfg = load_tournament_config(CONFIG_PATH)
    assert set(cfg.strategies) == set(_VALID_STRATEGIES)


def test_when_tournament_config_loaded_then_models_match_required_roster():
    """Criterion: models in the config match the required 6-model roster."""
    from training.tournament import load_tournament_config

    cfg = load_tournament_config(CONFIG_PATH)
    assert set(cfg.models) == set(_VALID_MODELS)


def test_when_tournament_config_loaded_then_config_is_frozen():
    """Criterion: TournamentConfig is frozen (mutation raises an error)."""
    from training.tournament import load_tournament_config

    cfg = load_tournament_config(CONFIG_PATH)
    with pytest.raises((AttributeError, TypeError)):
        cfg.seed = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Loader raises a clear error on invalid input
# ---------------------------------------------------------------------------


def _write_yaml(path: pathlib.Path, data: dict) -> pathlib.Path:
    path.write_text(yaml.dump(data))
    return path


def _base_data() -> dict:
    return {
        "name": "test_tournament",
        "seed": 42,
        "n_games": 10,
        "max_concurrency": 2,
        "n_rounds": 2,
        "strategies": _VALID_STRATEGIES,
        "models": _VALID_MODELS,
        "temperature": 0.7,
        "top_p": 0.9,
    }


def test_when_models_count_is_fewer_than_6_then_loader_raises(tmp_path):
    """Criterion: loader raises when models ≠ 6 (too few)."""
    from training.tournament import load_tournament_config

    data = _base_data()
    data["models"] = ["model-a:cloud", "model-b:cloud"]  # only 2
    cfg_file = _write_yaml(tmp_path / "few_models.yaml", data)
    with pytest.raises(ValueError):
        load_tournament_config(cfg_file)


def test_when_models_count_is_more_than_6_then_loader_raises(tmp_path):
    """Criterion: loader raises when models ≠ 6 (too many)."""
    from training.tournament import load_tournament_config

    data = _base_data()
    data["models"] = _VALID_MODELS + ["extra-model:cloud"]  # 7
    cfg_file = _write_yaml(tmp_path / "many_models.yaml", data)
    with pytest.raises(ValueError):
        load_tournament_config(cfg_file)


def test_when_strategies_count_is_fewer_than_6_then_loader_raises(tmp_path):
    """Criterion: loader raises when strategies ≠ 6 (too few)."""
    from training.tournament import load_tournament_config

    data = _base_data()
    data["strategies"] = ["australia_lock"]  # only 1
    cfg_file = _write_yaml(tmp_path / "few_strategies.yaml", data)
    with pytest.raises(ValueError):
        load_tournament_config(cfg_file)


def test_when_strategies_count_is_more_than_6_then_loader_raises(tmp_path):
    """Criterion: loader raises when strategies ≠ 6 (too many)."""
    from training.tournament import load_tournament_config

    data = _base_data()
    data["strategies"] = _VALID_STRATEGIES + ["bonus_strategy"]  # 7
    cfg_file = _write_yaml(tmp_path / "many_strategies.yaml", data)
    with pytest.raises(ValueError):
        load_tournament_config(cfg_file)


def test_when_strategy_slug_not_in_catalog_then_loader_raises(tmp_path):
    """Criterion: loader raises when a strategy slug ∉ CATALOG_BY_NAME."""
    from training.tournament import load_tournament_config

    data = _base_data()
    # Replace one valid slug with an unknown one
    data["strategies"] = _VALID_STRATEGIES[:5] + ["TOTALLY_UNKNOWN_SLUG"]
    cfg_file = _write_yaml(tmp_path / "bad_slug.yaml", data)
    with pytest.raises(ValueError):
        load_tournament_config(cfg_file)


def test_when_all_strategy_slugs_are_invalid_then_loader_raises(tmp_path):
    """Criterion: loader raises when none of the strategy slugs are in CATALOG_BY_NAME."""
    from training.tournament import load_tournament_config

    data = _base_data()
    data["strategies"] = [f"invalid_slug_{i}" for i in range(6)]
    cfg_file = _write_yaml(tmp_path / "all_bad_slugs.yaml", data)
    with pytest.raises(ValueError):
        load_tournament_config(cfg_file)


# ---------------------------------------------------------------------------
# LedgerRecord.to_json_line() ↔ from_json() round-trip
# ---------------------------------------------------------------------------


def _make_minimal_ledger_record():
    """Build a minimal LedgerRecord from plain Python types (no numpy)."""
    from training.tournament import LedgerRecord

    return LedgerRecord(
        game_index=0,
        seed=42,
        winner=2,
        winner_strategy="australia_lock",
        winner_model="glm-5.2:cloud",
        n_turns=120,
        elimination_order=[1, 3, 4, 5, 0, 2],
        seat_strategies={str(i): s for i, s in enumerate(_VALID_STRATEGIES)},
        seat_models={str(i): m for i, m in enumerate(_VALID_MODELS)},
        assignment=dict(zip(_VALID_STRATEGIES, _VALID_MODELS, strict=True)),
        card_trade_turns=[10, 35, 70],
        llm_call_count=480,
    )


def test_when_ledger_record_serialized_then_json_loads_succeeds():
    """Criterion: to_json_line() emits no numpy types; json.loads succeeds."""
    record = _make_minimal_ledger_record()
    line = record.to_json_line()
    # Must not raise — no numpy or non-serialisable types in the output
    parsed = json.loads(line)
    assert isinstance(parsed, dict)


def test_when_ledger_record_round_tripped_then_game_index_is_preserved():
    """Criterion: from_json(json.loads(to_json_line())) round-trips game_index."""
    from training.tournament import LedgerRecord

    record = _make_minimal_ledger_record()
    parsed = json.loads(record.to_json_line())
    restored = LedgerRecord.from_json(parsed)
    assert restored.game_index == record.game_index


def test_when_ledger_record_round_tripped_then_winner_is_preserved():
    """Criterion: from_json(json.loads(to_json_line())) round-trips winner."""
    from training.tournament import LedgerRecord

    record = _make_minimal_ledger_record()
    parsed = json.loads(record.to_json_line())
    restored = LedgerRecord.from_json(parsed)
    assert restored.winner == record.winner


def test_when_ledger_record_round_tripped_then_winner_strategy_is_preserved():
    """Criterion: from_json(json.loads(to_json_line())) round-trips winner_strategy."""
    from training.tournament import LedgerRecord

    record = _make_minimal_ledger_record()
    parsed = json.loads(record.to_json_line())
    restored = LedgerRecord.from_json(parsed)
    assert restored.winner_strategy == record.winner_strategy


def test_when_ledger_record_round_tripped_then_llm_call_count_is_preserved():
    """Criterion: from_json(json.loads(to_json_line())) round-trips llm_call_count."""
    from training.tournament import LedgerRecord

    record = _make_minimal_ledger_record()
    parsed = json.loads(record.to_json_line())
    restored = LedgerRecord.from_json(parsed)
    assert restored.llm_call_count == record.llm_call_count


def test_when_winner_is_none_then_round_trip_preserves_none():
    """Criterion: LedgerRecord with winner=None round-trips without error."""
    from training.tournament import LedgerRecord

    record = LedgerRecord(
        game_index=1,
        seed=43,
        winner=None,
        winner_strategy=None,
        winner_model=None,
        n_turns=1000,
        elimination_order=[],
        seat_strategies={str(i): s for i, s in enumerate(_VALID_STRATEGIES)},
        seat_models={str(i): m for i, m in enumerate(_VALID_MODELS)},
        assignment=dict(zip(_VALID_STRATEGIES, _VALID_MODELS, strict=True)),
        card_trade_turns=[],
        llm_call_count=600,
    )
    line = record.to_json_line()
    parsed = json.loads(line)
    restored = LedgerRecord.from_json(parsed)
    assert restored.winner is None
    assert restored.winner_strategy is None


def test_when_ledger_record_serialized_then_output_is_single_line():
    """Criterion: to_json_line() produces exactly one line (no embedded newlines)."""
    record = _make_minimal_ledger_record()
    line = record.to_json_line()
    # A JSONL record must not contain embedded newlines
    assert "\n" not in line.rstrip("\n")


# ---------------------------------------------------------------------------
# Property: round-trip is an identity over varying game_index and seed
# ---------------------------------------------------------------------------


@given(
    game_index=st.integers(min_value=0, max_value=9999),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_turns=st.integers(min_value=1, max_value=5000),
    llm_call_count=st.integers(min_value=0, max_value=100_000),
)
def test_when_ledger_record_round_tripped_for_any_valid_ints_then_values_are_preserved(
    game_index, seed, n_turns, llm_call_count
):
    """Invariant: round-trip preserves integer fields for any valid input domain."""
    from training.tournament import LedgerRecord

    record = LedgerRecord(
        game_index=game_index,
        seed=seed,
        winner=0,
        winner_strategy="australia_lock",
        winner_model="glm-5.2:cloud",
        n_turns=n_turns,
        elimination_order=[1, 2, 3, 4, 5],
        seat_strategies={str(i): s for i, s in enumerate(_VALID_STRATEGIES)},
        seat_models={str(i): m for i, m in enumerate(_VALID_MODELS)},
        assignment=dict(zip(_VALID_STRATEGIES, _VALID_MODELS, strict=True)),
        card_trade_turns=[],
        llm_call_count=llm_call_count,
    )
    line = record.to_json_line()
    # Must parse without error regardless of the integer values
    parsed = json.loads(line)
    restored = LedgerRecord.from_json(parsed)
    assert restored.game_index == game_index
    assert restored.seed == seed
    assert restored.n_turns == n_turns
    assert restored.llm_call_count == llm_call_count


@given(
    game_index=st.integers(min_value=0, max_value=9999),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_when_ledger_record_serialized_then_json_always_parseable(game_index, seed):
    """Invariant: to_json_line() always produces valid JSON (no numpy or exotic types)."""
    from training.tournament import LedgerRecord

    record = LedgerRecord(
        game_index=game_index,
        seed=seed,
        winner=None,
        winner_strategy=None,
        winner_model=None,
        n_turns=1,
        elimination_order=[],
        seat_strategies={str(i): s for i, s in enumerate(_VALID_STRATEGIES)},
        seat_models={str(i): m for i, m in enumerate(_VALID_MODELS)},
        assignment=dict(zip(_VALID_STRATEGIES, _VALID_MODELS, strict=True)),
        card_trade_turns=[],
        llm_call_count=0,
    )
    # json.loads must not raise for any valid record
    parsed = json.loads(record.to_json_line())
    assert isinstance(parsed, dict)
