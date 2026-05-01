"""Tests for the BAML bridge between env observations and BAML types."""

from __future__ import annotations

import numpy as np
import pytest

from baml_client import types as baml_types
from src.agents.baml_bridge import (
    ActionType,
    _build_cards,
    _build_legal_actions,
    _build_territories,
    baml_action_to_dict,
    build_snapshot,
    is_legal,
)
from src.env import RisikoEnv

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_obs(env: RisikoEnv) -> dict[str, np.ndarray]:
    """Return the current observation from *env*."""
    return env._get_obs()


# ------------------------------------------------------------------
# build_snapshot
# ------------------------------------------------------------------


class TestBuildSnapshot:
    """Conversion from env obs to BAML GameStateSnapshot."""

    @pytest.fixture
    def env(self) -> RisikoEnv:
        return RisikoEnv(n_players=3)

    def test_basic_shape(self, env: RisikoEnv) -> None:
        """Snapshot has 42 territories and correct player count."""
        obs, info = env.reset(seed=42)
        snapshot = build_snapshot(obs, info["legal_actions"])
        assert len(snapshot.territories) == 42
        assert snapshot.n_players == 3
        assert 0 <= snapshot.current_player < 3

    def test_phase_mapping(self, env: RisikoEnv) -> None:
        """Phase integer maps to BAML Phase enum."""
        obs, info = env.reset(seed=42)
        snapshot = build_snapshot(obs, info["legal_actions"])
        assert isinstance(snapshot.phase, baml_types.Phase)

    def test_territory_fields(self, env: RisikoEnv) -> None:
        """Each TerritorySnapshot has id, name, owner, armies, continent."""
        obs, info = env.reset(seed=42)
        snapshot = build_snapshot(obs, info["legal_actions"])
        t0 = snapshot.territories[0]
        assert t0.id == 0
        assert t0.name == "Alaska"
        assert t0.continent == "North America"

    def test_eliminated_is_list(self, env: RisikoEnv) -> None:
        """Eliminated field is a plain list of ints."""
        obs, info = env.reset(seed=42)
        snapshot = build_snapshot(obs, info["legal_actions"])
        assert isinstance(snapshot.eliminated, list)
        assert all(isinstance(x, int) for x in snapshot.eliminated)

    def test_continent_control_is_list(self, env: RisikoEnv) -> None:
        """continent_control field is a plain list of ints."""
        obs, info = env.reset(seed=42)
        snapshot = build_snapshot(obs, info["legal_actions"])
        assert isinstance(snapshot.continent_control, list)
        assert len(snapshot.continent_control) == 6

    def test_legal_actions_present(self, env: RisikoEnv) -> None:
        """legal_actions list is non-empty and matches env info."""
        obs, info = env.reset(seed=42)
        snapshot = build_snapshot(obs, info["legal_actions"])
        assert len(snapshot.legal_actions) > 0
        assert len(snapshot.legal_actions) == len(info["legal_actions"])

    def test_turn_capture_is_bool(self, env: RisikoEnv) -> None:
        """turn_capture is a plain Python bool."""
        obs, info = env.reset(seed=42)
        snapshot = build_snapshot(obs, info["legal_actions"])
        assert isinstance(snapshot.turn_capture, bool)

    def test_when_hint_set_snapshot_has_strategy_hint(self, env: RisikoEnv) -> None:
        obs, info = env.reset(seed=42)
        snapshot = build_snapshot(obs, info["legal_actions"], strategy_hint="Play greedily.")
        assert snapshot.strategy_hint == "Play greedily."

    def test_when_hint_none_snapshot_strategy_hint_is_none(self, env: RisikoEnv) -> None:
        obs, info = env.reset(seed=42)
        snapshot = build_snapshot(obs, info["legal_actions"])
        assert snapshot.strategy_hint is None


# ------------------------------------------------------------------
# baml_action_to_dict
# ------------------------------------------------------------------


class TestBamlActionToDict:
    """Conversion from BAML RisikoAction to env action dict."""

    def test_trade(self) -> None:
        action = baml_types.RisikoAction(
            action_type="trade",
            param_a=0,
            param_b=1,
            param_c=2,
            param_d=0,
        )
        d = baml_action_to_dict(action)
        assert d["action_type"] == ActionType.TRADE

    def test_reinforce(self) -> None:
        action = baml_types.RisikoAction(
            action_type="reinforce",
            param_a=5,
            param_b=2,
            param_c=0,
            param_d=0,
        )
        d = baml_action_to_dict(action)
        assert d["action_type"] == ActionType.REINFORCE
        assert d["param_a"] == 5
        assert d["param_b"] == 2

    def test_attack(self) -> None:
        action = baml_types.RisikoAction(
            action_type="attack",
            param_a=0,
            param_b=1,
            param_c=3,
            param_d=0,
        )
        d = baml_action_to_dict(action)
        assert d["action_type"] == ActionType.ATTACK
        assert d["param_c"] == 3

    def test_skip(self) -> None:
        action = baml_types.RisikoAction(
            action_type="skip",
            param_a=0,
            param_b=0,
            param_c=0,
            param_d=0,
        )
        d = baml_action_to_dict(action)
        assert d["action_type"] == ActionType.SKIP

    def test_unknown_defaults_to_skip(self) -> None:
        action = baml_types.RisikoAction(
            action_type="unknown_action",
            param_a=0,
            param_b=0,
            param_c=0,
            param_d=0,
        )
        d = baml_action_to_dict(action)
        assert d["action_type"] == ActionType.SKIP

    def test_capture_move_with_spaces(self) -> None:
        action = baml_types.RisikoAction(
            action_type="capture move",
            param_a=2,
            param_b=0,
            param_c=0,
            param_d=0,
        )
        d = baml_action_to_dict(action)
        assert d["action_type"] == ActionType.CAPTURE_MOVE
        assert d["param_a"] == 2


# ------------------------------------------------------------------
# is_legal
# ------------------------------------------------------------------


class TestIsLegal:
    """Exact-match validation of parsed actions against legal_actions."""

    def test_exact_match_is_legal(self) -> None:
        legal = [
            {"action_type": 1, "param_a": 5, "param_b": 2, "param_c": 0, "param_d": 0},
            {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0},
        ]
        action = {"action_type": 1, "param_a": 5, "param_b": 2, "param_c": 0, "param_d": 0}
        assert is_legal(action, legal)

    def test_non_match_is_illegal(self) -> None:
        legal = [
            {"action_type": 1, "param_a": 5, "param_b": 2, "param_c": 0, "param_d": 0},
        ]
        action = {"action_type": 1, "param_a": 5, "param_b": 3, "param_c": 0, "param_d": 0}
        assert not is_legal(action, legal)

    def test_single_entry_list(self) -> None:
        legal = [{"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}]
        assert is_legal(legal[0], legal)

    def test_empty_legal_actions(self) -> None:
        assert not is_legal(
            {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0},
            [],
        )


# ------------------------------------------------------------------
# _build_territories
# ------------------------------------------------------------------


class TestBuildTerritories:
    """Territory list construction from observation."""

    def test_count(self) -> None:
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        ts = _build_territories(obs)
        assert len(ts) == 42

    def test_first_territory(self) -> None:
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        ts = _build_territories(obs)
        assert ts[0].name == "Alaska"
        assert ts[0].continent == "North America"

    def test_last_territory(self) -> None:
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        ts = _build_territories(obs)
        assert ts[41].name == "Western Australia"
        assert ts[41].continent == "Australia"


# ------------------------------------------------------------------
# _build_cards
# ------------------------------------------------------------------


class TestBuildCards:
    """Card list construction from observation."""

    def test_empty_at_start(self) -> None:
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        cards = _build_cards(obs)
        # At game start no player holds cards
        assert len(cards) == 0


# ------------------------------------------------------------------
# _build_legal_actions
# ------------------------------------------------------------------


class TestBuildLegalActions:
    """Legal action list construction for the snapshot."""

    def test_skip_mapped(self) -> None:
        legal = [{"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}]
        baml_legal = _build_legal_actions(legal)
        assert len(baml_legal) == 1
        assert baml_legal[0].action_type == "skip"

    def test_non_skip_mapped_to_real_string(self) -> None:
        legal = [{"action_type": 2, "param_a": 0, "param_b": 1, "param_c": 1, "param_d": 0}]
        baml_legal = _build_legal_actions(legal)
        assert baml_legal[0].action_type == "attack"

    def test_description_contains_params(self) -> None:
        legal = [{"action_type": 2, "param_a": 3, "param_b": 7, "param_c": 1, "param_d": 0}]
        baml_legal = _build_legal_actions(legal)
        assert "type=2" in baml_legal[0].description
        assert "params=(3,7,1,0)" in baml_legal[0].description
