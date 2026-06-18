"""Source-blind example tests for #94: mandatory card trade enforcement.

Written against the acceptance criteria only; src/ was never read.
Cards 0-13 = infantry (symbol 0), 14-27 = cavalry (symbol 1),
28-41 = artillery (symbol 2), 42-43 = wild (symbol 3).
Valid sets: 3-of-a-kind, 1-of-each, any-2-plus-wild.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from src.env import (
    PHASE_REINFORCE,
    PHASE_TRADE,
    RisikoEnv,
)
from src.utils.constants import get_trade_value

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip() -> dict:
    return {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}


def _trade(a: int, b: int, c: int) -> dict:
    return {"action_type": 0, "param_a": a, "param_b": b, "param_c": c, "param_d": 0}


def _env_with_cards(cards: list[int], *, seed: int = 42) -> tuple["RisikoEnv", int]:
    """Return (env, player) holding the given cards in PHASE_TRADE."""
    env = RisikoEnv(n_players=3)
    env.reset(seed=seed)
    player = env.state.current_player
    env.state.cards[player] = list(cards)
    env.state.phase = PHASE_TRADE
    return env, player


# ---------------------------------------------------------------------------
# Criterion: skip rejected and phase stays PHASE_TRADE while a valid set held
# ---------------------------------------------------------------------------


class TestSkipRejectedWhenSetHeld:
    def test_when_skip_attempted_while_holding_infantry_triple_reward_is_negative(self):
        """Skipping trade while holding a valid set returns the invalid-action penalty."""
        env, _ = _env_with_cards([0, 1, 2])
        _, reward, _, _, _ = env.step(_skip())
        assert reward < 0

    def test_when_skip_attempted_while_holding_valid_set_phase_stays_trade(self):
        """Phase remains PHASE_TRADE after an illegal skip attempt while a set is held."""
        env, _ = _env_with_cards([0, 1, 2])
        obs, _, _, _, _ = env.step(_skip())
        assert obs["phase"] == PHASE_TRADE

    def test_when_skip_attempted_while_holding_mixed_set_reward_is_negative(self):
        """Skipping with a one-of-each (infantry + cavalry + artillery) set is also invalid."""
        env, _ = _env_with_cards([0, 14, 28])
        _, reward, _, _, _ = env.step(_skip())
        assert reward < 0

    def test_when_skip_attempted_multiple_times_phase_never_leaves_trade(self):
        """Repeated skip attempts while a set is held never advance beyond PHASE_TRADE."""
        env, _ = _env_with_cards([0, 1, 2])
        for _ in range(3):
            obs, _, _, _, _ = env.step(_skip())
            assert obs["phase"] == PHASE_TRADE


# ---------------------------------------------------------------------------
# Criterion: valid trade increments trade_count by 1 and adds the correct bonus
# ---------------------------------------------------------------------------


class TestValidTradeIncrementsBonusAndCount:
    def test_when_valid_trade_executed_trade_count_incremented_by_one(self):
        """A valid trade increments the shared trade_count by exactly 1."""
        env, _ = _env_with_cards([0, 1, 2])
        env.state.trade_count = 0
        env.state.reinforcements_remaining = 0
        env.step(_trade(0, 1, 2))
        assert env.state.trade_count == 1

    def test_when_valid_trade_executed_reinforcements_increase_by_trade_value(self):
        """Reinforcements_remaining increases by get_trade_value(prev_trade_count)."""
        env, _ = _env_with_cards([0, 1, 2])
        env.state.trade_count = 2
        env.state.reinforcements_remaining = 0
        expected_bonus = get_trade_value(2)
        env.step(_trade(0, 1, 2))
        assert env.state.reinforcements_remaining == expected_bonus

    def test_when_valid_trade_executed_existing_reinforcements_are_preserved(self):
        """The trade bonus adds on top of any pre-existing reinforcements."""
        env, _ = _env_with_cards([0, 1, 2])
        env.state.trade_count = 0
        env.state.reinforcements_remaining = 5
        bonus = get_trade_value(0)
        env.step(_trade(0, 1, 2))
        assert env.state.reinforcements_remaining == 5 + bonus

    def test_when_successive_trades_trade_count_reflects_both(self):
        """Two sequential trades advance trade_count by 2 in total."""
        env, _ = _env_with_cards([0, 1, 2, 3, 4, 5])
        env.state.trade_count = 0
        env.state.reinforcements_remaining = 0
        env.step(_trade(0, 1, 2))
        env.step(_trade(0, 1, 2))
        assert env.state.trade_count == 2


# ---------------------------------------------------------------------------
# Criterion: valid trade moves exactly 3 cards to discard_pile
# ---------------------------------------------------------------------------


class TestValidTradeDiscardsExactlyThreeCards:
    def test_when_valid_trade_executed_exactly_3_cards_added_to_discard_pile(self):
        """Trading a set removes exactly 3 cards and places them in discard_pile."""
        env, _ = _env_with_cards([0, 1, 2])
        initial_discard = len(env.state.discard_pile)
        env.step(_trade(0, 1, 2))
        assert len(env.state.discard_pile) == initial_discard + 3

    def test_when_valid_trade_executed_hand_shrinks_by_3(self):
        """The player's hand shrinks by exactly 3 after a successful trade."""
        env, player = _env_with_cards([0, 1, 2, 14])
        before = len(env.state.cards[player])
        env.step(_trade(0, 1, 2))
        assert len(env.state.cards[player]) == before - 3

    def test_when_valid_trade_executed_discarded_cards_appear_in_discard_pile(self):
        """The 3 traded cards end up in discard_pile (order not mandated)."""
        env, player = _env_with_cards([0, 1, 2])
        env.step(_trade(0, 1, 2))
        pile = env.state.discard_pile
        assert 0 in pile
        assert 1 in pile
        assert 2 in pile


# ---------------------------------------------------------------------------
# Criterion: chained forcing — phase stays PHASE_TRADE until all sets traded
# ---------------------------------------------------------------------------


class TestChainedForcedTrade:
    def test_when_two_disjoint_sets_held_phase_stays_trade_after_first_trade(self):
        """After trading one set, phase stays PHASE_TRADE while the second set remains."""
        env, _ = _env_with_cards([0, 1, 2, 3, 4, 5])
        env.state.trade_count = 0
        env.state.reinforcements_remaining = 0
        obs, _, _, _, _ = env.step(_trade(0, 1, 2))
        assert obs["phase"] == PHASE_TRADE

    def test_when_two_disjoint_sets_held_second_trade_does_not_return_penalty(self):
        """The second forced trade in a chain is accepted (reward is not negative)."""
        env, _ = _env_with_cards([0, 1, 2, 3, 4, 5])
        env.state.trade_count = 0
        env.state.reinforcements_remaining = 0
        env.step(_trade(0, 1, 2))
        _, reward, _, _, _ = env.step(_trade(0, 1, 2))
        assert reward >= 0

    def test_when_last_set_traded_phase_transitions_to_reinforce(self):
        """After the last valid set is traded and reinforcements exist, phase is REINFORCE."""
        env, _ = _env_with_cards([0, 1, 2])
        env.state.trade_count = 0
        env.state.reinforcements_remaining = 0
        obs, _, _, _, _ = env.step(_trade(0, 1, 2))
        assert obs["phase"] == PHASE_REINFORCE

    def test_when_two_sets_held_after_both_trades_phase_is_reinforce(self):
        """After chained trading of both sets, the final phase is PHASE_REINFORCE."""
        env, _ = _env_with_cards([0, 1, 2, 3, 4, 5])
        env.state.trade_count = 0
        env.state.reinforcements_remaining = 0
        env.step(_trade(0, 1, 2))
        obs, _, _, _, _ = env.step(_trade(0, 1, 2))
        assert obs["phase"] == PHASE_REINFORCE

    def test_when_last_set_traded_skip_in_trade_phase_is_no_longer_needed(self):
        """After the last trade is complete, the resulting phase is outside PHASE_TRADE."""
        env, _ = _env_with_cards([0, 1, 2])
        env.state.trade_count = 0
        env.state.reinforcements_remaining = 0
        obs, _, _, _, _ = env.step(_trade(0, 1, 2))
        assert obs["phase"] != PHASE_TRADE


# ---------------------------------------------------------------------------
# Criterion: legal_actions in PHASE_TRADE under forced trade omit skip
# ---------------------------------------------------------------------------


class TestLegalActionsInForcedTrade:
    def test_when_forced_trade_pending_legal_actions_not_empty(self):
        """`_get_legal_actions()` is non-empty when a forced trade is pending."""
        env, _ = _env_with_cards([0, 1, 2])
        legal = env._get_legal_actions()
        assert len(legal) > 0

    def test_when_forced_trade_pending_legal_actions_contain_no_skip(self):
        """`_get_legal_actions()` has no skip (action_type==5) when a set is held."""
        env, _ = _env_with_cards([0, 1, 2])
        legal = env._get_legal_actions()
        assert all(a["action_type"] != 5 for a in legal)

    def test_when_forced_trade_pending_all_legal_actions_are_trade_actions(self):
        """Every legal action under forced trade has action_type == 0."""
        env, _ = _env_with_cards([0, 1, 2])
        legal = env._get_legal_actions()
        assert all(a["action_type"] == 0 for a in legal)

    def test_when_no_valid_set_in_hand_skip_is_offered_in_legal_actions(self):
        """When no valid set remains, skip (action_type==5) is present in legal actions."""
        # [0, 14]: only 2 cards — cannot form any 3-card set
        env, _ = _env_with_cards([0, 14])
        legal = env._get_legal_actions()
        skip_actions = [a for a in legal if a["action_type"] == 5]
        assert len(skip_actions) >= 1

    def test_when_no_valid_set_skip_successfully_advances_to_reinforce(self):
        """Skip when no valid set is held advances phase to PHASE_REINFORCE."""
        env, _ = _env_with_cards([0, 14])
        env.state.reinforcements_remaining = 3
        obs, reward, _, _, _ = env.step(_skip())
        assert reward >= 0
        assert obs["phase"] == PHASE_REINFORCE

    def test_when_info_from_step_contains_legal_actions_no_skip_under_forced_trade(self):
        """info['legal_actions'] from step() also omits skip when a set is held."""
        env, _ = _env_with_cards([0, 1, 2])
        # Take an invalid step to get info without changing state
        _, _, _, _, info = env.step(_skip())
        legal = info.get("legal_actions", [])
        # After the rejected skip, still in PHASE_TRADE with the set held
        assert all(a["action_type"] != 5 for a in legal)


# ---------------------------------------------------------------------------
# Criterion: determinism — same seed + same actions => identical outcome
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_when_same_seed_and_same_actions_trade_count_and_discard_pile_identical(self):
        """Same seed + same actions produce identical trade_count and discard_pile."""
        cards = [0, 1, 2, 3, 4, 5]

        def run() -> tuple[int, list]:
            env = RisikoEnv(n_players=3)
            env.reset(seed=99)
            player = env.state.current_player
            env.state.cards[player] = list(cards)
            env.state.phase = PHASE_TRADE
            env.state.trade_count = 0
            env.state.reinforcements_remaining = 0
            env.step(_trade(0, 1, 2))
            env.step(_trade(0, 1, 2))
            return env.state.trade_count, list(env.state.discard_pile)

        tc1, dp1 = run()
        tc2, dp2 = run()
        assert tc1 == tc2
        assert dp1 == dp2

    def test_when_different_seeds_trade_count_still_deterministic_per_run(self):
        """Re-running with the same seed always gives the same trade_count."""

        def run(seed: int) -> int:
            env = RisikoEnv(n_players=3)
            env.reset(seed=seed)
            player = env.state.current_player
            env.state.cards[player] = [0, 1, 2]
            env.state.phase = PHASE_TRADE
            env.state.trade_count = 0
            env.state.reinforcements_remaining = 0
            env.step(_trade(0, 1, 2))
            return env.state.trade_count

        assert run(7) == run(7)
        assert run(13) == run(13)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Infantry cards: 0..13 (all symbol 0); any 3 distinct cards form a valid set.
_INFANTRY_CARDS = list(range(14))


@given(
    st.lists(
        st.sampled_from(_INFANTRY_CARDS),
        min_size=3,
        max_size=3,
        unique=True,
    )
)
@settings(max_examples=30)
def test_when_holding_any_infantry_triple_legal_actions_contain_no_skip(cards):
    """For any 3 distinct infantry cards, forced-trade legal actions must omit skip."""
    env, _ = _env_with_cards(cards)
    legal = env._get_legal_actions()
    assert all(a["action_type"] != 5 for a in legal), (
        f"skip found in legal_actions for infantry triple {cards}"
    )


@given(
    st.lists(
        st.sampled_from(_INFANTRY_CARDS),
        min_size=3,
        max_size=3,
        unique=True,
    )
)
@settings(max_examples=30)
def test_when_holding_any_infantry_triple_legal_actions_are_not_empty(cards):
    """For any 3 distinct infantry cards, legal_actions must be non-empty."""
    env, _ = _env_with_cards(cards)
    legal = env._get_legal_actions()
    assert len(legal) > 0, f"empty legal_actions for infantry triple {cards}"


@given(
    st.integers(min_value=0, max_value=10),
)
@settings(max_examples=20)
def test_when_trade_count_is_any_valid_value_valid_trade_increments_it_by_one(prev_count):
    """Invariant: a valid trade always increments trade_count by exactly 1."""
    env, _ = _env_with_cards([0, 1, 2])
    env.state.trade_count = prev_count
    env.state.reinforcements_remaining = 0
    env.step(_trade(0, 1, 2))
    assert env.state.trade_count == prev_count + 1


@given(
    st.integers(min_value=0, max_value=10),
)
@settings(max_examples=20)
def test_when_trade_count_is_any_valid_value_bonus_matches_get_trade_value(prev_count):
    """Invariant: reinforcements bonus always equals get_trade_value(prev_count)."""
    env, _ = _env_with_cards([0, 1, 2])
    env.state.trade_count = prev_count
    env.state.reinforcements_remaining = 0
    expected = get_trade_value(prev_count)
    env.step(_trade(0, 1, 2))
    assert env.state.reinforcements_remaining == expected
