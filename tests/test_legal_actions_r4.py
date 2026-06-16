"""Tests for legal-action enumeration per phase (src/env_core/legal_actions.py).

Issue #51.

State is built via _TestState — a minimal duck-typed dataclass whose fields
match the real GameState interface, augmented with an 'adjacency' override.
get_legal_actions() resolves adjacency via getattr(state, 'adjacency', ADJACENCY),
so real states use the global 42-territory board and test states use a mini-board.

Synthetic 6-territory board:
    0 — 1 — 2
        |
        5       3 — 4

ADJ = {0:[1], 1:[0,2,5], 2:[1], 3:[4], 4:[3], 5:[1]}
Components: {0,1,2,5} and {3,4} are disconnected.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from src.env_core.legal_actions import get_legal_actions
from src.env_core.state import (
    PHASE_ATTACK,
    PHASE_CAPTURE_MOVE,
    PHASE_FORTIFY,
    PHASE_REINFORCE,
    PHASE_TRADE,
)

# ---------------------------------------------------------------------------
# Synthetic mini-board constants
# ---------------------------------------------------------------------------

ADJ: dict[int, list[int]] = {
    0: [1],
    1: [0, 2, 5],
    2: [1],
    3: [4],
    4: [3],
    5: [1],
}
NUM_T = 6

# Integer card symbols (matches src/utils/constants.py CARD_SYMBOLS)
CARD_I = 0  # Infantry
CARD_C = 1  # Cavalry
CARD_A = 2  # Artillery
CARD_W = 3  # Wildcard


# ---------------------------------------------------------------------------
# Minimal test-state fixture
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _TestState:
    """Duck-type compatible with GameState; adds 'adjacency' for mini-board tests."""

    current_player: int
    phase: int
    territory_owner: np.ndarray
    armies: np.ndarray
    cards: list  # cards[player] = list[int]
    reinforcements_remaining: int = 0
    last_attacker: int = -1
    last_capture_dice: int = 0
    last_defender: int = -1
    n_players: int = 2
    turn_capture: int = 0
    adjacency: dict = dataclasses.field(default_factory=lambda: dict(ADJ))


def _make(
    *,
    phase: int = PHASE_REINFORCE,
    player: int = 0,
    owners: list[int] | None = None,
    armies: list[int] | None = None,
    cards: list[int] | None = None,  # current player's hand (int symbols)
    remaining: int = 3,
    last_attacker: int = -1,
    last_captured: int = -1,  # accepted but unused (alias for last_defender)
    last_capture_dice: int = 0,
    n_players: int = 2,
    adjacency: dict | None = None,
) -> _TestState:
    eff_owners = owners if owners is not None else [0] * NUM_T
    n_p = max(max(eff_owners) + 1, n_players)
    all_cards: list = [[] for _ in range(n_p)]
    if cards:
        all_cards[player] = list(cards)
    return _TestState(
        current_player=player,
        phase=phase,
        territory_owner=np.array(eff_owners, dtype=np.int32),
        armies=np.array(armies if armies is not None else [2] * NUM_T, dtype=np.int32),
        cards=all_cards,
        reinforcements_remaining=remaining,
        last_attacker=last_attacker,
        last_capture_dice=last_capture_dice,
        last_defender=last_captured,
        n_players=n_p,
        adjacency=adjacency if adjacency is not None else dict(ADJ),
    )


# ---------------------------------------------------------------------------
# Action introspection helpers
# Actions are dicts: {action_type, param_a, param_b, param_c, param_d}
# ---------------------------------------------------------------------------


def _is_skip(action: Any) -> bool:
    """True iff action is the SKIP action (action_type == 5)."""
    if isinstance(action, dict):
        return action.get("action_type") == 5
    return getattr(action, "action_type", None) == 5


def _attr(action: Any, *names: str) -> Any:
    """Extract a semantic value from an action dict.

    Semantic mapping for dict actions:
      territory / from_territory / attacker / src / source → param_a
      to_territory / defender / target                     → param_b
      dice / dice_count / num_dice                         → param_c
      count / armies / num_armies / move_count:
          CAPTURE_MOVE (type=3) → param_a
          FORTIFY      (type=4) → param_c
          REINFORCE    (type=1) → param_b
    """
    if not isinstance(action, dict):
        for name in names:
            v = getattr(action, name, None)
            if v is not None:
                return v
        return None

    for name in names:
        if name in action:
            return action[name]

    param_a_names = frozenset({"territory", "from_territory", "attacker", "src", "source"})
    param_b_names = frozenset({"to_territory", "defender", "target"})
    param_c_names = frozenset({"dice", "dice_count", "num_dice"})
    count_names = frozenset({"count", "armies", "num_armies", "move_count"})

    atype = action.get("action_type", -1)
    for name in names:
        if name in param_a_names:
            return action.get("param_a")
        if name in param_b_names:
            return action.get("param_b")
        if name in param_c_names:
            return action.get("param_c")
        if name in count_names:
            if atype == 3:
                return action.get("param_a")  # CAPTURE_MOVE: armies = param_a
            if atype == 4:
                return action.get("param_c")  # FORTIFY: count = param_c
            if atype == 1:
                return action.get("param_b")  # REINFORCE: count = param_b
    return None


# ===========================================================================
# CRITERION: get_legal_actions never returns an empty list for any phase
# ===========================================================================


class TestNeverEmpty:
    """AC: get_legal_actions dispatches to per-phase enumerator; never empty."""

    def test_when_reinforce_phase_with_armies_to_place_then_not_empty(self):
        state = _make(phase=PHASE_REINFORCE, remaining=3)
        assert len(get_legal_actions(state)) > 0

    def test_when_reinforce_phase_with_zero_remaining_then_not_empty(self):
        state = _make(phase=PHASE_REINFORCE, remaining=0)
        assert len(get_legal_actions(state)) > 0

    def test_when_trade_phase_with_empty_hand_then_not_empty(self):
        state = _make(phase=PHASE_TRADE, remaining=0, cards=[])
        assert len(get_legal_actions(state)) > 0

    def test_when_attack_phase_then_not_empty(self):
        state = _make(
            phase=PHASE_ATTACK,
            owners=[0, 1, 0, 0, 0, 0],
            armies=[3, 2, 2, 2, 2, 2],
        )
        assert len(get_legal_actions(state)) > 0

    def test_when_capture_move_phase_then_not_empty(self):
        state = _make(
            phase=PHASE_CAPTURE_MOVE,
            owners=[0] * NUM_T,
            armies=[4, 2, 2, 2, 2, 2],
            last_attacker=0,
            last_capture_dice=2,
            remaining=0,
        )
        assert len(get_legal_actions(state)) > 0

    def test_when_fortify_phase_then_not_empty(self):
        state = _make(
            phase=PHASE_FORTIFY,
            owners=[0] * NUM_T,
            armies=[3, 2, 2, 2, 2, 2],
            remaining=0,
        )
        assert len(get_legal_actions(state)) > 0


# ===========================================================================
# CRITERION: TRADE phase
# ===========================================================================


class TestTrade:
    """AC: skip always present; only valid 3-card combos emitted.

    Nothing but skip when hand has < 3 cards.
    """

    def test_when_hand_is_empty_then_only_skip_is_returned(self):
        state = _make(phase=PHASE_TRADE, cards=[], remaining=0)
        actions = get_legal_actions(state)
        assert len(actions) == 1
        assert _is_skip(actions[0])

    def test_when_hand_has_one_card_then_only_skip_is_returned(self):
        state = _make(phase=PHASE_TRADE, cards=[CARD_I], remaining=0)
        actions = get_legal_actions(state)
        assert len(actions) == 1
        assert _is_skip(actions[0])

    def test_when_hand_has_two_cards_then_only_skip_is_returned(self):
        state = _make(phase=PHASE_TRADE, cards=[CARD_I, CARD_C], remaining=0)
        actions = get_legal_actions(state)
        assert len(actions) == 1
        assert _is_skip(actions[0])

    def test_when_hand_has_three_same_type_then_skip_is_present(self):
        state = _make(phase=PHASE_TRADE, cards=[CARD_I, CARD_I, CARD_I], remaining=0)
        assert any(_is_skip(a) for a in get_legal_actions(state))

    def test_when_hand_has_three_same_type_then_trade_action_is_also_emitted(self):
        state = _make(phase=PHASE_TRADE, cards=[CARD_I, CARD_I, CARD_I], remaining=0)
        non_skip = [a for a in get_legal_actions(state) if not _is_skip(a)]
        assert len(non_skip) >= 1

    def test_when_hand_has_one_of_each_type_then_trade_action_is_emitted(self):
        """Infantry + Cavalry + Artillery is a valid Risiko set."""
        state = _make(phase=PHASE_TRADE, cards=[CARD_I, CARD_C, CARD_A], remaining=0)
        non_skip = [a for a in get_legal_actions(state) if not _is_skip(a)]
        assert len(non_skip) >= 1

    def test_when_hand_has_wild_with_two_infantry_then_trade_action_is_emitted(self):
        """Wild substitutes for any type, making the set valid."""
        state = _make(phase=PHASE_TRADE, cards=[CARD_I, CARD_I, CARD_W], remaining=0)
        non_skip = [a for a in get_legal_actions(state) if not _is_skip(a)]
        assert len(non_skip) >= 1

    def test_when_hand_has_two_infantry_one_cavalry_then_only_skip_is_returned(self):
        """2-same + 1-other without wild is invalid in Risiko."""
        state = _make(phase=PHASE_TRADE, cards=[CARD_I, CARD_I, CARD_C], remaining=0)
        non_skip = [a for a in get_legal_actions(state) if not _is_skip(a)]
        assert len(non_skip) == 0, "2 Infantry + 1 Cavalry is not a valid trade combo"

    @given(st.integers(min_value=0, max_value=2))
    def test_when_hand_is_shorter_than_three_then_only_skip_returned(self, n):
        """Invariant: hand size < 3 always yields exactly [skip]."""
        state = _make(phase=PHASE_TRADE, cards=[CARD_I] * n, remaining=0)
        actions = get_legal_actions(state)
        assert len(actions) == 1
        assert _is_skip(actions[0])


# ===========================================================================
# CRITERION: REINFORCE phase
# ===========================================================================


class TestReinforce:
    """AC: only place actions when remaining > 0; skip only when remaining == 0.

    All actions reference owned territories.
    """

    def test_when_remaining_is_zero_then_only_skip_is_returned(self):
        state = _make(
            phase=PHASE_REINFORCE,
            owners=[0, 0, 1, 1, 1, 1],
            armies=[2] * NUM_T,
            remaining=0,
        )
        actions = get_legal_actions(state)
        assert len(actions) == 1
        assert _is_skip(actions[0])

    def test_when_remaining_is_positive_then_place_actions_are_returned(self):
        state = _make(
            phase=PHASE_REINFORCE,
            owners=[0, 0, 1, 1, 1, 1],
            armies=[2] * NUM_T,
            remaining=3,
        )
        non_skip = [a for a in get_legal_actions(state) if not _is_skip(a)]
        assert len(non_skip) > 0

    def test_when_remaining_is_positive_then_skip_is_not_in_actions(self):
        """Skip is emitted ONLY when remaining == 0."""
        state = _make(
            phase=PHASE_REINFORCE,
            owners=[0, 0, 1, 1, 1, 1],
            armies=[2] * NUM_T,
            remaining=2,
        )
        assert not any(_is_skip(a) for a in get_legal_actions(state))

    def test_when_reinforce_then_all_place_actions_target_owned_territory(self):
        current_player = 0
        owners = [0, 0, 1, 1, 1, 1]
        state = _make(
            phase=PHASE_REINFORCE,
            player=current_player,
            owners=owners,
            armies=[2] * NUM_T,
            remaining=3,
        )
        for action in get_legal_actions(state):
            if _is_skip(action):
                continue
            t = _attr(action, "territory", "target", "to_territory")
            if t is not None:
                assert owners[t] == current_player, (
                    f"Place action targets territory {t} owned by player {owners[t]}, "
                    f"not current player {current_player}"
                )

    @given(st.integers(min_value=1, max_value=30))
    def test_when_remaining_positive_then_non_skip_actions_exist(self, remaining):
        """Invariant: any positive remaining yields ≥ 1 place-type action."""
        state = _make(
            phase=PHASE_REINFORCE,
            owners=[0] * NUM_T,
            armies=[2] * NUM_T,
            remaining=remaining,
        )
        assert any(not _is_skip(a) for a in get_legal_actions(state))


# ===========================================================================
# CRITERION: ATTACK phase
# ===========================================================================


class TestAttack:
    """AC: skip always present; no attacks from army==1 territories.

    No attacks to own territories or non-adjacent territories;
    dice ≤ min(3, armies[attacker]-1).
    """

    def test_when_attack_phase_then_skip_is_always_present(self):
        state = _make(
            phase=PHASE_ATTACK,
            owners=[0, 1, 0, 0, 0, 0],
            armies=[3, 2, 2, 2, 2, 2],
            remaining=0,
        )
        assert any(_is_skip(a) for a in get_legal_actions(state))

    def test_when_all_own_border_territories_have_one_army_then_only_skip(self):
        """No attacks possible when every player-0 territory bordering an enemy has 1 army.

        ADJ: 0-[1], 2-[1], 5-[1] — territories 0, 2, 5 are all adjacent to
        enemy territory 1.  All three are given 1 army; 3 and 4 are isolated
        from enemy territory and irrelevant.
        """
        state = _make(
            phase=PHASE_ATTACK,
            player=0,
            owners=[0, 1, 0, 0, 0, 0],
            armies=[1, 2, 1, 2, 2, 1],
            remaining=0,
        )
        assert all(_is_skip(a) for a in get_legal_actions(state))

    def test_when_attack_phase_then_no_attack_originates_from_army_one_territory(self):
        state = _make(
            phase=PHASE_ATTACK,
            player=0,
            owners=[0, 1, 0, 0, 0, 0],
            armies=[1, 2, 3, 2, 2, 2],
            remaining=0,
        )
        for action in get_legal_actions(state):
            if _is_skip(action):
                continue
            src = _attr(action, "from_territory", "attacker", "src")
            if src is not None:
                assert state.armies[src] > 1, f"Attack from territory {src} which has only 1 army"

    def test_when_attack_phase_then_no_attack_targets_own_territory(self):
        player = 0
        owners = [0, 0, 1, 1, 1, 1]
        state = _make(
            phase=PHASE_ATTACK,
            player=player,
            owners=owners,
            armies=[3, 3, 2, 2, 2, 2],
            remaining=0,
        )
        for action in get_legal_actions(state):
            if _is_skip(action):
                continue
            target = _attr(action, "to_territory", "defender", "target")
            if target is not None:
                assert owners[target] != player, f"Attack targets own territory {target}"

    def test_when_attack_phase_then_no_attack_to_non_adjacent_territory(self):
        """ADJ: 0-1, 1-2, 1-5, 3-4. Territory 0 and 2 are NOT adjacent."""
        state = _make(
            phase=PHASE_ATTACK,
            player=0,
            owners=[0, 1, 1, 1, 1, 1],
            armies=[3, 2, 2, 2, 2, 2],
            remaining=0,
        )
        adj = state.adjacency
        for action in get_legal_actions(state):
            if _is_skip(action):
                continue
            src = _attr(action, "from_territory", "attacker", "src")
            dst = _attr(action, "to_territory", "defender", "target")
            if src is not None and dst is not None:
                assert dst in adj.get(src, []), f"Territory {dst} is not adjacent to {src}"

    def test_when_attacker_has_four_armies_then_max_dice_is_three(self):
        """min(3, 4-1) = 3; no action may specify > 3 dice."""
        state = _make(
            phase=PHASE_ATTACK,
            player=0,
            owners=[0, 1, 0, 0, 0, 0],
            armies=[4, 2, 2, 2, 2, 2],
            remaining=0,
        )
        for action in get_legal_actions(state):
            if _is_skip(action):
                continue
            src = _attr(action, "from_territory", "attacker", "src")
            dice = _attr(action, "dice", "dice_count", "num_dice")
            if src is not None and dice is not None:
                assert dice <= min(3, int(state.armies[src]) - 1)

    def test_when_attacker_has_two_armies_then_max_dice_is_one(self):
        """min(3, 2-1) = 1; only 1 die allowed from territory 0."""
        state = _make(
            phase=PHASE_ATTACK,
            player=0,
            owners=[0, 1, 0, 0, 0, 0],
            armies=[2, 2, 2, 2, 2, 2],
            remaining=0,
        )
        for action in get_legal_actions(state):
            if _is_skip(action):
                continue
            src = _attr(action, "from_territory", "attacker", "src")
            dice = _attr(action, "dice", "dice_count", "num_dice")
            if src == 0 and dice is not None:
                assert dice == 1

    @given(st.integers(min_value=2, max_value=10))
    def test_when_attacking_territory_has_n_armies_then_dice_bounded(self, n):
        """Invariant: all emitted dice values ≤ min(3, n-1)."""
        state = _make(
            phase=PHASE_ATTACK,
            player=0,
            owners=[0, 1, 0, 0, 0, 0],
            armies=[n, 2, 2, 2, 2, 2],
            remaining=0,
        )
        max_expected = min(3, n - 1)
        for action in get_legal_actions(state):
            if _is_skip(action):
                continue
            dice = _attr(action, "dice", "dice_count", "num_dice")
            if dice is not None:
                assert dice <= max_expected


# ===========================================================================
# CRITERION: CAPTURE_MOVE phase
# ===========================================================================


class TestCaptureMove:
    """AC: move bounds are [last_capture_dice, armies[last_attacker]-1]."""

    def test_when_capture_move_then_every_count_is_at_least_last_capture_dice(self):
        state = _make(
            phase=PHASE_CAPTURE_MOVE,
            owners=[0] * NUM_T,
            armies=[5, 2, 2, 2, 2, 2],
            last_attacker=0,
            last_capture_dice=3,
            remaining=0,
        )
        for action in get_legal_actions(state):
            count = _attr(action, "count", "armies", "num_armies", "move_count")
            if count is not None:
                assert count >= state.last_capture_dice

    def test_when_capture_move_then_every_count_leaves_at_least_one_army_behind(self):
        state = _make(
            phase=PHASE_CAPTURE_MOVE,
            owners=[0] * NUM_T,
            armies=[5, 2, 2, 2, 2, 2],
            last_attacker=0,
            last_capture_dice=2,
            remaining=0,
        )
        max_move = int(state.armies[state.last_attacker]) - 1
        for action in get_legal_actions(state):
            count = _attr(action, "count", "armies", "num_armies", "move_count")
            if count is not None:
                assert count <= max_move

    def test_when_four_armies_and_two_dice_then_range_is_exactly_two_to_three(self):
        """armies[0]=4, last_capture_dice=2 → valid counts are {2, 3}."""
        state = _make(
            phase=PHASE_CAPTURE_MOVE,
            owners=[0] * NUM_T,
            armies=[4, 2, 2, 2, 2, 2],
            last_attacker=0,
            last_capture_dice=2,
            remaining=0,
        )
        counts = sorted(
            {
                _attr(a, "count", "armies", "num_armies", "move_count")
                for a in get_legal_actions(state)
                if _attr(a, "count", "armies", "num_armies", "move_count") is not None
            }
        )
        assert counts == [2, 3], f"Expected [2, 3], got {counts}"

    @given(
        st.integers(min_value=1, max_value=3),  # last_capture_dice
        st.integers(min_value=2, max_value=10),  # attacker army count
    )
    def test_when_capture_move_then_all_counts_within_bounds(self, dice, army):
        """Invariant: dice ≤ count ≤ army-1 for every emitted action."""
        if dice > army - 1:
            return  # impossible state — skip
        state = _make(
            phase=PHASE_CAPTURE_MOVE,
            owners=[0] * NUM_T,
            armies=[army, 2, 2, 2, 2, 2],
            last_attacker=0,
            last_captured=1,
            last_capture_dice=dice,
            remaining=0,
        )
        for action in get_legal_actions(state):
            count = _attr(action, "count", "armies", "num_armies", "move_count")
            if count is not None:
                assert dice <= count <= army - 1


# ===========================================================================
# CRITERION: FORTIFY phase
# ===========================================================================


class TestFortify:
    """AC: skip always present; no moves from army==1 territories.

    Count leaves ≥ 1 behind; disconnected source/dest pairs excluded.
    """

    def test_when_fortify_phase_then_skip_is_always_present(self):
        state = _make(
            phase=PHASE_FORTIFY,
            owners=[0] * NUM_T,
            armies=[3] * NUM_T,
            remaining=0,
        )
        assert any(_is_skip(a) for a in get_legal_actions(state))

    def test_when_fortify_then_no_move_originates_from_army_one_territory(self):
        # Territory 0 has 1 army; territories 1 and 2 have ≥2. All owned by player 0.
        state = _make(
            phase=PHASE_FORTIFY,
            player=0,
            owners=[0, 0, 0, 1, 1, 1],
            armies=[1, 3, 3, 2, 2, 2],
            remaining=0,
        )
        for action in get_legal_actions(state):
            if _is_skip(action):
                continue
            src = _attr(action, "from_territory", "source", "src")
            if src is not None:
                assert int(state.armies[src]) > 1, (
                    f"Fortify from territory {src} which has only 1 army"
                )

    def test_when_fortify_then_move_count_leaves_at_least_one_army_behind(self):
        state = _make(
            phase=PHASE_FORTIFY,
            player=0,
            owners=[0, 0, 1, 1, 1, 1],
            armies=[4, 2, 2, 2, 2, 2],
            remaining=0,
        )
        for action in get_legal_actions(state):
            if _is_skip(action):
                continue
            src = _attr(action, "from_territory", "source", "src")
            count = _attr(action, "count", "armies", "num_armies", "move_count")
            if src is not None and count is not None:
                assert count <= int(state.armies[src]) - 1, (
                    f"Moving {count} from territory {src} "
                    f"(armies={state.armies[src]}) leaves none behind"
                )

    def test_when_own_territories_are_disconnected_then_no_fortify_action(self):
        """Player 0 owns 0 and 3; components {0,1,2,5} and {3,4} are disconnected.

        No fortify path exists.
        """
        state = _make(
            phase=PHASE_FORTIFY,
            player=0,
            owners=[0, 1, 1, 0, 1, 1],
            armies=[3, 2, 2, 3, 2, 2],
            remaining=0,
        )
        non_skip = [a for a in get_legal_actions(state) if not _is_skip(a)]
        assert len(non_skip) == 0

    def test_when_own_territories_are_adjacent_then_fortify_action_exists(self):
        """Player 0 owns 0 and 1 (adjacent) → at least one fortify action."""
        state = _make(
            phase=PHASE_FORTIFY,
            player=0,
            owners=[0, 0, 1, 1, 1, 1],
            armies=[3, 2, 2, 2, 2, 2],
            remaining=0,
        )
        non_skip = [a for a in get_legal_actions(state) if not _is_skip(a)]
        assert len(non_skip) > 0

    def test_when_connected_through_chain_then_fortify_action_exists(self):
        """Chain 0-1-2 (all player 0) → 0 to 2 is reachable."""
        state = _make(
            phase=PHASE_FORTIFY,
            player=0,
            owners=[0, 0, 0, 1, 1, 1],
            armies=[4, 2, 2, 2, 2, 2],
            remaining=0,
        )
        non_skip = [a for a in get_legal_actions(state) if not _is_skip(a)]
        assert len(non_skip) > 0

    @given(st.integers(min_value=2, max_value=8))
    def test_when_fortify_from_territory_with_n_armies_then_all_counts_safe(self, n):
        """Invariant: every emitted count ≤ n-1 (leaves ≥1 behind)."""
        state = _make(
            phase=PHASE_FORTIFY,
            player=0,
            owners=[0, 0, 1, 1, 1, 1],
            armies=[n, 2, 2, 2, 2, 2],
            remaining=0,
        )
        for action in get_legal_actions(state):
            if _is_skip(action):
                continue
            src = _attr(action, "from_territory", "source", "src")
            count = _attr(action, "count", "armies", "num_armies", "move_count")
            if src is not None and count is not None:
                assert count <= int(state.armies[src]) - 1
