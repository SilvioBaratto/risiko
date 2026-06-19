"""
Tests for issue #99: diplomacy persona traits + DiplomacyConfig + gated YAML blocks.

Authored from acceptance criteria only — no implementation source was read.

Criteria tested (scope):
  - PlayerConfig gains trust_propensity (default 0.5) and vengefulness (default 0.5);
    frozen/immutable and picklable preserved
  - DiplomacyConfig exists with enabled=False, n_rounds=1, max_message_tokens=128 defaults
  - A config without a diplomacy: block yields disabled defaults
  - CLI/override path casts diplomacy.enabled=true to a real bool

Criteria tested (global, UNIT-verifiable):
  - get_legal_actions() is never empty in PHASE_TRADE under a forced trade
"""

from __future__ import annotations

import os
import pickle
import tempfile
import textwrap

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.player_config import PlayerConfig
from src.config import DiplomacyConfig, load_config, merge_cli_overrides

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BARE_YAML = textwrap.dedent("""\
    ppo:
      gamma: 0.99
""")

_PLAYER_DEFAULTS = {"player_id": 0, "temperature": 0.5, "top_p": 0.9, "strategy_hint": "test"}


def _make_player(**overrides: object) -> PlayerConfig:
    """Construct a PlayerConfig supplying all required fields."""
    return PlayerConfig(**{**_PLAYER_DEFAULTS, **overrides})


def _tmp_config(content: str) -> str:
    """Write *content* to a temp YAML file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as fh:
        fh.write(content)
    return path


def _load_bare_config():
    path = _tmp_config(_BARE_YAML)
    try:
        return load_config(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# PlayerConfig: new trait defaults
# ---------------------------------------------------------------------------


def test_when_player_config_created_without_args_then_trust_propensity_defaults_to_0_5():
    cfg = _make_player()
    assert cfg.trust_propensity == 0.5


def test_when_player_config_created_without_args_then_vengefulness_defaults_to_0_5():
    cfg = _make_player()
    assert cfg.vengefulness == 0.5


# ---------------------------------------------------------------------------
# PlayerConfig: immutability (frozen)
# ---------------------------------------------------------------------------


def test_when_trust_propensity_is_reassigned_then_frozen_error_is_raised():
    cfg = _make_player()
    with pytest.raises((AttributeError, TypeError)):
        cfg.trust_propensity = 0.9  # type: ignore[misc]


def test_when_vengefulness_is_reassigned_then_frozen_error_is_raised():
    cfg = _make_player()
    with pytest.raises((AttributeError, TypeError)):
        cfg.vengefulness = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PlayerConfig: picklability
# ---------------------------------------------------------------------------


def test_when_player_config_is_pickled_then_trust_propensity_survives_round_trip():
    cfg = _make_player(trust_propensity=0.3, vengefulness=0.7)
    restored = pickle.loads(pickle.dumps(cfg))
    assert restored.trust_propensity == 0.3


def test_when_player_config_is_pickled_then_vengefulness_survives_round_trip():
    cfg = _make_player(trust_propensity=0.3, vengefulness=0.7)
    restored = pickle.loads(pickle.dumps(cfg))
    assert restored.vengefulness == 0.7


@given(
    trust=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    vengefulness=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_when_player_config_with_any_valid_traits_is_pickled_then_it_round_trips(
    trust: float, vengefulness: float
) -> None:
    """Invariant: pickle(PlayerConfig) is lossless for any float traits in [0, 1]."""
    cfg = _make_player(trust_propensity=trust, vengefulness=vengefulness)
    restored = pickle.loads(pickle.dumps(cfg))
    assert restored.trust_propensity == trust
    assert restored.vengefulness == vengefulness


# ---------------------------------------------------------------------------
# DiplomacyConfig: existence and defaults
# ---------------------------------------------------------------------------


def test_when_diplomacy_config_created_without_args_then_enabled_is_false():
    cfg = DiplomacyConfig()
    assert cfg.enabled is False


def test_when_diplomacy_config_created_without_args_then_n_rounds_is_1():
    cfg = DiplomacyConfig()
    assert cfg.n_rounds == 1


def test_when_diplomacy_config_created_without_args_then_max_message_tokens_is_128():
    cfg = DiplomacyConfig()
    assert cfg.max_message_tokens == 128


# ---------------------------------------------------------------------------
# load_config: YAML without diplomacy: block yields disabled defaults
# ---------------------------------------------------------------------------


def test_when_yaml_has_no_diplomacy_block_then_diplomacy_enabled_is_false(tmp_path):
    cfg_file = tmp_path / "no_diplomacy.yaml"
    cfg_file.write_text(_BARE_YAML)
    cfg = load_config(str(cfg_file))
    assert cfg.diplomacy.enabled is False


def test_when_yaml_has_no_diplomacy_block_then_n_rounds_is_default_1(tmp_path):
    cfg_file = tmp_path / "no_diplomacy.yaml"
    cfg_file.write_text(_BARE_YAML)
    cfg = load_config(str(cfg_file))
    assert cfg.diplomacy.n_rounds == 1


def test_when_yaml_has_no_diplomacy_block_then_max_message_tokens_is_default_128(tmp_path):
    cfg_file = tmp_path / "no_diplomacy.yaml"
    cfg_file.write_text(_BARE_YAML)
    cfg = load_config(str(cfg_file))
    assert cfg.diplomacy.max_message_tokens == 128


# ---------------------------------------------------------------------------
# merge_cli_overrides: bool casting for diplomacy.enabled
# ---------------------------------------------------------------------------


def test_when_cli_override_sets_diplomacy_enabled_to_string_true_then_value_is_bool_true():
    cfg = _load_bare_config()
    updated = merge_cli_overrides(cfg, {"diplomacy.enabled": "true"})
    assert updated.diplomacy.enabled is True
    assert isinstance(updated.diplomacy.enabled, bool)


def test_when_cli_override_sets_diplomacy_enabled_to_string_false_then_value_is_bool_false():
    cfg = _load_bare_config()
    updated = merge_cli_overrides(cfg, {"diplomacy.enabled": "false"})
    assert updated.diplomacy.enabled is False
    assert isinstance(updated.diplomacy.enabled, bool)


@given(bool_val=st.booleans())
@settings(max_examples=4)
def test_when_cli_override_is_lowercase_bool_string_then_result_is_python_bool(
    bool_val: bool,
) -> None:
    """Invariant: merge_cli_overrides always casts diplomacy.enabled to a Python bool, never str."""
    cfg = _load_bare_config()
    string_val = str(bool_val).lower()  # "true" or "false"
    updated = merge_cli_overrides(cfg, {"diplomacy.enabled": string_val})
    assert isinstance(updated.diplomacy.enabled, bool)
    assert updated.diplomacy.enabled == bool_val


# ---------------------------------------------------------------------------
# get_legal_actions() in PHASE_TRADE: never empty under forced trade
# ---------------------------------------------------------------------------


def test_when_player_has_valid_card_set_at_turn_start_then_phase_trade_legal_actions_not_empty():
    """
    Criterion: get_legal_actions() is never empty in PHASE_TRADE when a forced trade is required.

    Assumptions (derived from spec, not source):
    - RisikoEnv lives in src/env.py and exposes get_legal_actions().
    - PHASE_TRADE is a module-level constant in src/env.py.
    - env.state.current_player identifies the active player.
    - env.state.hands[player] holds the player's card list.
    - env.state.phase can be overwritten to inject a phase for testing.
    - Three identical symbols (e.g. 0, 0, 0) form a valid 3-of-a-kind set.
    """
    from src.env import PHASE_TRADE, RisikoEnv

    env = RisikoEnv(n_players=3)
    env.reset(seed=42)

    player = env.state.current_player
    env.state.cards[player] = [0, 0, 0]  # valid 3-of-a-kind set (symbol 0 = INFANTRY)
    env.state.phase = PHASE_TRADE

    legal = env.get_legal_actions()
    assert len(legal) > 0, (
        "get_legal_actions() must not return an empty list in PHASE_TRADE "
        "when the active player holds a valid card set (forced trade)."
    )
