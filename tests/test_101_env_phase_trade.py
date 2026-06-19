"""Source-blind example tests for get_legal_actions() in PHASE_TRADE (issue #101).

Derived solely from the acceptance criterion:
  - `get_legal_actions()` is never empty in `PHASE_TRADE` under a forced trade.

These tests fail until the criterion is genuinely met.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.env import PHASE_TRADE, RisikoEnv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_action() -> dict[str, int]:
    return {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}


def _inject_tradeable_set(env: RisikoEnv, player_id: int) -> None:
    """Give *player_id* a valid 3-of-a-kind card set via env.state.

    Assumption: env.state.cards is indexed by player and holds a list/array of
    integer card-symbol values (e.g. 0 = infantry, 1 = cavalry, 2 = artillery,
    3 = wild).  Three identical non-wild symbols form a valid 3-of-a-kind set.
    If the actual card representation differs, this helper will raise an
    AttributeError in the Red phase — fix by aligning with the implementation.
    """
    env.state.cards[player_id] = [0, 0, 0]  # three infantry


def _set_phase(env: RisikoEnv, phase: int) -> None:
    """Force the env into the given phase.

    Assumption: env.state.phase is the mutable integer that drives phase dispatch.
    """
    env.state.phase = phase


# ---------------------------------------------------------------------------
# Example tests — one observable behaviour per test
# ---------------------------------------------------------------------------


class TestPhaseTradeLegalActionsNotEmpty:
    """get_legal_actions() must never be empty in PHASE_TRADE under a forced trade."""

    def test_when_player_holds_tradeable_set_in_phase_trade_then_legal_actions_not_empty(
        self,
    ) -> None:
        """When the current player holds a valid 3-of-a-kind, get_legal_actions() is non-empty.

        A forced trade is required (the player has a valid set); therefore the
        skip action is not legal and the implementation must offer at least one
        valid trade action.
        """
        env = RisikoEnv(n_players=2)
        env.reset(seed=42)
        obs = env._get_obs()
        current = int(obs["current_player"])

        _inject_tradeable_set(env, current)
        _set_phase(env, PHASE_TRADE)

        legal = env.get_legal_actions()
        assert len(legal) > 0, (
            "get_legal_actions() must not return an empty list in PHASE_TRADE "
            "when the current player holds a valid tradeable set."
        )

    def test_when_player_holds_tradeable_set_then_skip_is_not_among_legal_actions(self) -> None:
        """Under a forced trade the skip action (action_type == 5) is illegal.

        Derived from: 'the skip action must not be offered; only valid trade
        actions are legal.'
        """
        env = RisikoEnv(n_players=2)
        env.reset(seed=42)
        obs = env._get_obs()
        current = int(obs["current_player"])

        _inject_tradeable_set(env, current)
        _set_phase(env, PHASE_TRADE)

        legal = env.get_legal_actions()
        skip_present = any(a.get("action_type") == 5 for a in legal)
        assert not skip_present, (
            "Skip action (action_type=5) must not appear in legal actions "
            "when a forced trade is required."
        )

    def test_when_player_holds_tradeable_set_in_6_player_game_then_legal_actions_not_empty(
        self,
    ) -> None:
        """Non-empty legal actions hold for the full 6-player configuration."""
        env = RisikoEnv(n_players=6)
        env.reset(seed=0)
        obs = env._get_obs()
        current = int(obs["current_player"])

        _inject_tradeable_set(env, current)
        _set_phase(env, PHASE_TRADE)

        assert len(env.get_legal_actions()) > 0

    def test_when_player_holds_one_of_each_set_in_phase_trade_then_legal_actions_not_empty(
        self,
    ) -> None:
        """A 1-of-each valid set also forces a non-empty legal action list.

        Assumption: 1-of-each = [0, 1, 2] (one infantry, one cavalry, one
        artillery).  This is one of the three valid card sets per the rules.
        """
        env = RisikoEnv(n_players=2)
        env.reset(seed=42)
        obs = env._get_obs()
        current = int(obs["current_player"])

        env.state.cards[current] = [0, 1, 2]  # one-of-each
        _set_phase(env, PHASE_TRADE)

        assert len(env.get_legal_actions()) > 0

    def test_when_player_has_two_cards_plus_wild_in_phase_trade_then_legal_actions_not_empty(
        self,
    ) -> None:
        """Any-2-plus-wild is also a valid set and must produce non-empty legal actions.

        Assumption: wild is card symbol 3.  Two identical non-wild cards plus a
        wild form a valid set per the rules.
        """
        env = RisikoEnv(n_players=2)
        env.reset(seed=42)
        obs = env._get_obs()
        current = int(obs["current_player"])

        env.state.cards[current] = [1, 1, 3]  # two cavalry + wild
        _set_phase(env, PHASE_TRADE)

        assert len(env.get_legal_actions()) > 0


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@settings(max_examples=40)
@given(n_players=st.integers(min_value=2, max_value=6))
def test_when_any_n_players_and_forced_trade_then_legal_actions_not_empty(
    n_players: int,
) -> None:
    """Invariant: get_legal_actions() is never empty in PHASE_TRADE with a tradeable set,
    regardless of the number of players.

    Derived from: 'get_legal_actions() is never empty in PHASE_TRADE under a
    forced trade'.
    """
    env = RisikoEnv(n_players=n_players)
    env.reset(seed=42)
    obs = env._get_obs()
    current = int(obs["current_player"])

    _inject_tradeable_set(env, current)
    _set_phase(env, PHASE_TRADE)

    assert len(env.get_legal_actions()) > 0
