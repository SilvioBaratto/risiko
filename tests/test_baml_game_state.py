"""Tests for BAML-generated game-state types and GenerateRisikoAction."""

from __future__ import annotations

import pytest

from baml_client import b as baml_client
from baml_client import types as baml_types
from src.env import RisikoEnv


class TestPhaseEnum:
    """Phase enum values match environment phases."""

    def test_phase_values(self):
        """Phase enum has all 5 phase names."""
        assert baml_types.Phase.TRADE == "TRADE"
        assert baml_types.Phase.REINFORCE == "REINFORCE"
        assert baml_types.Phase.ATTACK == "ATTACK"
        assert baml_types.Phase.CAPTURE_MOVE == "CAPTURE_MOVE"
        assert baml_types.Phase.FORTIFY == "FORTIFY"


class TestTerritorySnapshot:
    """TerritorySnapshot dataclass behaviour."""

    def test_basic_creation(self):
        """Can create a TerritorySnapshot with all fields."""
        t = baml_types.TerritorySnapshot(
            id=0,
            name="Alaska",
            owner=1,
            armies=5,
            continent="North America",
        )
        assert t.id == 0
        assert t.name == "Alaska"
        assert t.owner == 1
        assert t.armies == 5
        assert t.continent == "North America"


class TestCardInfo:
    """CardInfo dataclass behaviour."""

    def test_basic_creation(self):
        """Can create a CardInfo with index and symbol."""
        c = baml_types.CardInfo(index=0, symbol="infantry")
        assert c.index == 0
        assert c.symbol == "infantry"


class TestLegalAction:
    """LegalAction dataclass behaviour."""

    def test_basic_creation(self):
        """Can create a LegalAction with action_type and description."""
        a = baml_types.LegalAction(action_type="skip", description="End attack phase")
        assert a.action_type == "skip"
        assert a.description == "End attack phase"


class TestGameStateSnapshot:
    """GameStateSnapshot construction and field access."""

    def test_minimal_creation(self):
        """Can create a GameStateSnapshot with required nested types."""
        snapshot = baml_types.GameStateSnapshot(
            current_player=0,
            phase=baml_types.Phase.ATTACK,
            territories=[
                baml_types.TerritorySnapshot(
                    id=0, name="Alaska", owner=0, armies=3, continent="North America"
                )
            ],
            cards=[baml_types.CardInfo(index=0, symbol="infantry")],
            reinforcements_remaining=5,
            trade_count=1,
            turn_capture=False,
            n_players=3,
            eliminated=[0, 0, 0],
            continent_control=[1, 0, 0, 0, 0, 0],
            legal_actions=[
                baml_types.LegalAction(action_type="skip", description="End attack phase")
            ],
        )
        assert snapshot.current_player == 0
        assert snapshot.phase == baml_types.Phase.ATTACK
        assert len(snapshot.territories) == 1
        assert len(snapshot.cards) == 1
        assert snapshot.reinforcements_remaining == 5
        assert snapshot.trade_count == 1
        assert snapshot.turn_capture is False
        assert snapshot.n_players == 3
        assert snapshot.eliminated == [0, 0, 0]
        assert snapshot.continent_control == [1, 0, 0, 0, 0, 0]
        assert len(snapshot.legal_actions) == 1


class TestRisikoAction:
    """RisikoAction construction and optional reasoning."""

    def test_basic_creation(self):
        """Can create a RisikoAction with all params and no reasoning."""
        action = baml_types.RisikoAction(
            action_type="attack",
            param_a=0,
            param_b=1,
            param_c=3,
            param_d=0,
        )
        assert action.action_type == "attack"
        assert action.param_a == 0
        assert action.param_b == 1
        assert action.param_c == 3
        assert action.param_d == 0
        assert action.reasoning is None

    def test_with_reasoning(self):
        """Can create a RisikoAction with optional reasoning."""
        action = baml_types.RisikoAction(
            action_type="reinforce",
            param_a=5,
            param_b=2,
            param_c=0,
            param_d=0,
            reasoning="Bolster the front line",
        )
        assert action.reasoning == "Bolster the front line"


class TestEnvToSnapshot:
    """Convert RisikoEnv observation to GameStateSnapshot."""

    @pytest.fixture
    def env(self):
        """Fresh 3-player RisikoEnv."""
        return RisikoEnv(n_players=3)

    def test_snapshot_from_env(self, env):
        """Build a GameStateSnapshot from env observation and info."""
        obs, info = env.reset(seed=42)
        s = env.state
        territories = [
            baml_types.TerritorySnapshot(
                id=i,
                name="",  # names omitted for brevity in test
                owner=int(obs["territory_owner"][i]),
                armies=int(obs["armies"][i]),
                continent="",
            )
            for i in range(42)
        ]
        cards = [
            baml_types.CardInfo(index=i, symbol="infantry")
            for i in range(len(s.cards[s.current_player]))
        ]
        phase_map = {0: "TRADE", 1: "REINFORCE", 2: "ATTACK", 3: "CAPTURE_MOVE", 4: "FORTIFY"}
        phase = baml_types.Phase(phase_map[int(obs["phase"])])

        legal_actions: list[baml_types.LegalAction] = []
        for a in info["legal_actions"]:
            pa, pb, pc, pd = a["param_a"], a["param_b"], a["param_c"], a["param_d"]
            desc = f"type={a['action_type']} params=({pa},{pb},{pc},{pd})"
            legal_actions.append(
                baml_types.LegalAction(
                    action_type="skip" if a["action_type"] == 5 else "action",
                    description=desc,
                )
            )

        snapshot = baml_types.GameStateSnapshot(
            current_player=int(obs["current_player"]),
            phase=phase,
            territories=territories,
            cards=cards,
            reinforcements_remaining=int(obs["reinforcements_remaining"]),
            trade_count=int(obs["trade_count"]),
            turn_capture=bool(obs["turn_capture"]),
            n_players=int(obs["n_players"]),
            eliminated=obs["eliminated"].tolist(),
            continent_control=obs["continent_control"].tolist(),
            legal_actions=legal_actions,
        )
        assert snapshot.current_player == int(obs["current_player"])
        assert snapshot.phase == phase
        assert len(snapshot.territories) == 42
        assert snapshot.n_players == 3


class TestSyncClientFunction:
    """BAML sync client exposes GenerateRisikoAction."""

    def test_method_exists(self):
        """sync_client.b has GenerateRisikoAction method."""
        assert hasattr(baml_client, "GenerateRisikoAction")

    def test_method_is_callable(self):
        """GenerateRisikoAction is a callable bound method."""
        assert callable(baml_client.GenerateRisikoAction)
