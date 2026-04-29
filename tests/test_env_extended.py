"""Extended tests for the Risiko Gymnasium environment.

Covers card system, dense rewards, fortify pathing, elimination,
capture moves, and edge-case validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.env import (
    MAX_CARDS,
    PHASE_ATTACK,
    PHASE_CAPTURE_MOVE,
    PHASE_FORTIFY,
    PHASE_REINFORCE,
    PHASE_TRADE,
    RisikoEnv,
)
from src.utils.constants import ADJACENCY, CONTINENTS
from src.utils.reward_config import RewardConfig


def _skip() -> dict[str, int]:
    return {
        "action_type": 5,
        "param_a": 0,
        "param_b": 0,
        "param_c": 0,
        "param_d": 0,
    }


def _reinforce(terr: int, rem: int) -> dict[str, int]:
    return {
        "action_type": 1,
        "param_a": terr,
        "param_b": rem,
        "param_c": 0,
        "param_d": 0,
    }


class TestEnvInit:
    """Constructor validation."""

    def test_invalid_n_players_low_raises(self):
        """n_players < 2 must raise ValueError."""
        with pytest.raises(ValueError, match="n_players"):
            RisikoEnv(n_players=1)

    def test_invalid_n_players_high_raises(self):
        """n_players > 6 must raise ValueError."""
        with pytest.raises(ValueError, match="n_players"):
            RisikoEnv(n_players=7)


class TestInvalidActions:
    """Invalid action dict handling."""

    def test_non_dict_action_is_invalid(self):
        """A non-dict action must be rejected with penalty."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        _, reward, _, _, _ = env.step("not_a_dict")  # type: ignore[arg-type]
        assert reward < 0

    def test_missing_key_action_is_invalid(self):
        """A dict missing required keys must be rejected with penalty."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        _, reward, _, _, _ = env.step({"action_type": 0, "param_a": 0})
        assert reward < 0


class TestCardTrading:
    """Card trade mechanics."""

    def test_trade_same_symbols_valid(self):
        """Three cards with the same symbol form a valid set."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        env.state.phase = PHASE_TRADE
        env.state.cards[env.state.current_player] = [0, 1, 2]
        env.state.reinforcements_remaining = 0
        action = {
            "action_type": 0,
            "param_a": 0,
            "param_b": 1,
            "param_c": 2,
            "param_d": 0,
        }
        obs, _, _, _, _ = env.step(action)
        assert obs["phase"] == PHASE_REINFORCE
        assert obs["reinforcements_remaining"] > 0

    def test_trade_different_symbols_valid(self):
        """One of each non-wild symbol is a valid set."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        env.state.phase = PHASE_TRADE
        # Force a hand with one infantry (0), one cavalry (14), one artillery (28)
        env.state.cards[env.state.current_player] = [0, 14, 28]
        env.state.reinforcements_remaining = 0
        action = {
            "action_type": 0,
            "param_a": 0,
            "param_b": 1,
            "param_c": 2,
            "param_d": 0,
        }
        obs, _, _, _, _ = env.step(action)
        assert obs["phase"] == PHASE_REINFORCE

    def test_trade_with_wildcard_always_valid(self):
        """Any two cards plus a wildcard is always valid."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        env.state.phase = PHASE_TRADE
        env.state.cards[env.state.current_player] = [0, 1, 42]  # 42 is a wild
        env.state.reinforcements_remaining = 0
        action = {
            "action_type": 0,
            "param_a": 0,
            "param_b": 1,
            "param_c": 2,
            "param_d": 0,
        }
        obs, _, _, _, _ = env.step(action)
        assert obs["phase"] == PHASE_REINFORCE

    def test_invalid_trade_rejected(self):
        """An invalid trade must not change state and return penalty."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.phase = PHASE_TRADE
        env.state.cards[env.state.current_player] = [0, 0, 0, 14]
        env.state.reinforcements_remaining = 5
        action = {
            "action_type": 0,
            "param_a": 0,
            "param_b": 1,
            "param_c": 3,  # two infantry + cavalry = invalid
            "param_d": 0,
        }
        obs, reward, _, _, _ = env.step(action)
        assert reward < 0
        assert obs["reinforcements_remaining"] == 5

    def test_skip_trade_when_no_tradeable_cards(self):
        """Skipping trade when no valid set exists advances to reinforce."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        env.state.phase = PHASE_TRADE
        env.state.cards[env.state.current_player] = [0, 14]  # only 2 cards
        env.state.reinforcements_remaining = 5
        obs, _, _, _, _ = env.step(_skip())
        assert obs["phase"] == PHASE_REINFORCE

    def test_cards_discarded_after_trade(self):
        """Cards used in a trade are removed from hand."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.phase = PHASE_TRADE
        env.state.cards[env.state.current_player] = [0, 14, 28, 1]
        env.state.reinforcements_remaining = 0
        env.step(
            {
                "action_type": 0,
                "param_a": 0,
                "param_b": 1,
                "param_c": 2,
                "param_d": 0,
            }
        )
        assert len(env.state.cards[env.state.current_player]) == 1

    def test_trade_when_reinforcements_already_present(self):
        """Trading when reinforcements > 0 keeps existing reinforcements."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.phase = PHASE_TRADE
        env.state.cards[env.state.current_player] = [0, 1, 2]
        env.state.reinforcements_remaining = 3
        env.step(
            {
                "action_type": 0,
                "param_a": 0,
                "param_b": 1,
                "param_c": 2,
                "param_d": 0,
            }
        )
        assert env.state.reinforcements_remaining > 3


class TestReinforceEdgeCases:
    """Edge cases in reinforcement placement."""

    def test_reinforce_enemy_territory_invalid(self):
        """Placing on an enemy territory must be invalid."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        while obs["phase"] == PHASE_TRADE:
            env.step(_skip())
            obs = env._get_obs()
        player = int(obs["current_player"])
        enemy = int(np.where(obs["territory_owner"] != player)[0][0])
        action = {
            "action_type": 1,
            "param_a": enemy,
            "param_b": 1,
            "param_c": 0,
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0

    def test_reinforce_zero_advances_phase(self):
        """When reinforcements is 0, skip advances to attack."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        while obs["phase"] == PHASE_TRADE:
            env.step(_skip())
            obs = env._get_obs()
        env.state.reinforcements_remaining = 0
        obs, _, _, _, _ = env.step(_skip())
        assert obs["phase"] == PHASE_ATTACK

    def test_reinforce_over_remaining_invalid(self):
        """Placing more than remaining reinforcements is invalid."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        while obs["phase"] == PHASE_TRADE:
            env.step(_skip())
            obs = env._get_obs()
        player = int(obs["current_player"])
        terr = int(np.where(obs["territory_owner"] == player)[0][0])
        rem = int(obs["reinforcements_remaining"])
        action = {
            "action_type": 1,
            "param_a": terr,
            "param_b": rem + 1,
            "param_c": 0,
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0


class TestAttackEdgeCases:
    """Attack validation beyond adjacency."""

    def test_attack_own_territory_invalid(self):
        """Attacking own territory must be invalid."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        while obs["phase"] != PHASE_ATTACK:
            if obs["phase"] == PHASE_TRADE:
                env.step(_skip())
            elif obs["phase"] == PHASE_REINFORCE:
                player = int(obs["current_player"])
                terr = int(np.where(obs["territory_owner"] == player)[0][0])
                rem = int(obs["reinforcements_remaining"])
                env.step(_reinforce(terr, rem))
            obs = env._get_obs()
        player = int(obs["current_player"])
        owned = np.where((obs["territory_owner"] == player) & (obs["armies"] >= 2))[0]
        assert len(owned) > 0
        attacker = int(owned[0])
        action = {
            "action_type": 2,
            "param_a": attacker,
            "param_b": attacker,
            "param_c": 1,
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0

    def test_attack_from_enemy_territory_invalid(self):
        """Attacking from an enemy territory must be invalid."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        while obs["phase"] != PHASE_ATTACK:
            if obs["phase"] == PHASE_TRADE:
                env.step(_skip())
            elif obs["phase"] == PHASE_REINFORCE:
                player = int(obs["current_player"])
                terr = int(np.where(obs["territory_owner"] == player)[0][0])
                rem = int(obs["reinforcements_remaining"])
                env.step(_reinforce(terr, rem))
            obs = env._get_obs()
        player = int(obs["current_player"])
        enemy = int(np.where(obs["territory_owner"] != player)[0][0])
        target = ADJACENCY[enemy][0]
        action = {
            "action_type": 2,
            "param_a": enemy,
            "param_b": target,
            "param_c": 1,
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0

    def test_attack_with_zero_dice_invalid(self):
        """Attacking with 0 dice is invalid."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        while obs["phase"] != PHASE_ATTACK:
            if obs["phase"] == PHASE_TRADE:
                env.step(_skip())
            elif obs["phase"] == PHASE_REINFORCE:
                player = int(obs["current_player"])
                terr = int(np.where(obs["territory_owner"] == player)[0][0])
                rem = int(obs["reinforcements_remaining"])
                env.step(_reinforce(terr, rem))
            obs = env._get_obs()
        player = int(obs["current_player"])
        attacker = int(np.where((obs["territory_owner"] == player) & (obs["armies"] >= 2))[0][0])
        enemies = [t for t in ADJACENCY[attacker] if obs["territory_owner"][t] != player]
        if enemies:
            target = enemies[0]
            action = {
                "action_type": 2,
                "param_a": attacker,
                "param_b": target,
                "param_c": 0,
                "param_d": 0,
            }
            _, reward, _, _, _ = env.step(action)
            assert reward < 0

    def test_attack_with_too_many_dice_invalid(self):
        """Attacking with more dice than attacker-1 is invalid."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        while obs["phase"] != PHASE_ATTACK:
            if obs["phase"] == PHASE_TRADE:
                env.step(_skip())
            elif obs["phase"] == PHASE_REINFORCE:
                player = int(obs["current_player"])
                terr = int(np.where(obs["territory_owner"] == player)[0][0])
                rem = int(obs["reinforcements_remaining"])
                env.step(_reinforce(terr, rem))
            obs = env._get_obs()
        player = int(obs["current_player"])
        candidates = np.where((obs["territory_owner"] == player) & (obs["armies"] >= 2))[0]
        for attacker in candidates:
            enemies = [t for t in ADJACENCY[attacker] if obs["territory_owner"][t] != player]
            if enemies:
                target = enemies[0]
                max_dice = int(min(3, obs["armies"][attacker] - 1))
                action = {
                    "action_type": 2,
                    "param_a": int(attacker),
                    "param_b": int(target),
                    "param_c": max_dice + 1,
                    "param_d": 0,
                }
                _, reward, _, _, _ = env.step(action)
                assert reward < 0
                break


class TestCaptureMove:
    """Capture-move phase mechanics."""

    def test_capture_move_advances_to_attack_or_fortify(self):
        """After capture move, phase must be attack or fortify."""
        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        # Force capture move state
        env.state.phase = PHASE_CAPTURE_MOVE
        env.state.last_attacker = 0
        env.state.last_defender = 1
        env.state.last_capture_dice = 2
        env.state.armies[0] = 5
        env.state.territory_owner[0] = 0
        env.state.territory_owner[1] = 0
        env.state.current_player = 0
        action = {
            "action_type": 3,
            "param_a": 2,
            "param_b": 0,
            "param_c": 0,
            "param_d": 0,
        }
        obs, _, _, _, _ = env.step(action)
        assert obs["phase"] in (PHASE_ATTACK, PHASE_FORTIFY)

    def test_capture_move_invalid_below_dice(self):
        """Moving fewer armies than dice is invalid."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.phase = PHASE_CAPTURE_MOVE
        env.state.last_attacker = 0
        env.state.last_defender = 1
        env.state.last_capture_dice = 3
        env.state.armies[0] = 5
        env.state.territory_owner[0] = 0
        env.state.territory_owner[1] = 0
        env.state.current_player = 0
        action = {
            "action_type": 3,
            "param_a": 2,  # below dice=3
            "param_b": 0,
            "param_c": 0,
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0

    def test_capture_move_invalid_above_available(self):
        """Moving more armies than available is invalid."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.phase = PHASE_CAPTURE_MOVE
        env.state.last_attacker = 0
        env.state.last_defender = 1
        env.state.last_capture_dice = 2
        env.state.armies[0] = 3
        env.state.territory_owner[0] = 0
        env.state.territory_owner[1] = 0
        env.state.current_player = 0
        action = {
            "action_type": 3,
            "param_a": 4,  # above available armies on attacker
            "param_b": 0,
            "param_c": 0,
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0


class TestFortify:
    """Fortification phase mechanics."""

    def test_fortify_non_adjacent_connected_valid(self):
        """Fortifying over a connected path of own territories is valid."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.phase = PHASE_FORTIFY
        env.state.current_player = 0
        # Create a connected chain: 0-1-8 all owned by player 0
        env.state.territory_owner[0] = 0
        env.state.territory_owner[1] = 0
        env.state.territory_owner[8] = 0
        env.state.armies[0] = 3
        env.state.armies[1] = 1
        env.state.armies[8] = 1
        action = {
            "action_type": 4,
            "param_a": 0,
            "param_b": 8,
            "param_c": 1,
            "param_d": 0,
        }
        obs, _, _, _, _ = env.step(action)
        assert obs["phase"] in (PHASE_TRADE, PHASE_REINFORCE)
        assert env.state.armies[0] == 2
        assert env.state.armies[8] == 2

    def test_fortify_same_territory_invalid(self):
        """Fortifying from a territory to itself is invalid."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.phase = PHASE_FORTIFY
        env.state.current_player = 0
        env.state.territory_owner[0] = 0
        env.state.armies[0] = 3
        action = {
            "action_type": 4,
            "param_a": 0,
            "param_b": 0,
            "param_c": 1,
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0

    def test_fortify_enemy_source_invalid(self):
        """Fortifying from an enemy territory is invalid."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.phase = PHASE_FORTIFY
        env.state.current_player = 0
        env.state.territory_owner[0] = 1
        env.state.territory_owner[1] = 0
        env.state.armies[0] = 3
        env.state.armies[1] = 3
        action = {
            "action_type": 4,
            "param_a": 0,
            "param_b": 1,
            "param_c": 1,
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0

    def test_fortify_enemy_dest_invalid(self):
        """Fortifying to an enemy territory is invalid."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.phase = PHASE_FORTIFY
        env.state.current_player = 0
        env.state.territory_owner[0] = 0
        env.state.territory_owner[1] = 1
        env.state.armies[0] = 3
        env.state.armies[1] = 3
        action = {
            "action_type": 4,
            "param_a": 0,
            "param_b": 1,
            "param_c": 1,
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0

    def test_fortify_leave_at_least_one(self):
        """Fortifying must leave at least 1 army behind."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.phase = PHASE_FORTIFY
        env.state.current_player = 0
        env.state.territory_owner[0] = 0
        env.state.territory_owner[1] = 0
        env.state.armies[0] = 3
        env.state.armies[1] = 1
        action = {
            "action_type": 4,
            "param_a": 0,
            "param_b": 1,
            "param_c": 3,  # would leave 0 behind
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0

    def test_fortify_disconnected_invalid(self):
        """Fortifying between disconnected owned territories is invalid."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.phase = PHASE_FORTIFY
        env.state.current_player = 0
        env.state.territory_owner[0] = 0
        env.state.territory_owner[1] = 1
        env.state.territory_owner[2] = 0
        env.state.armies[0] = 3
        env.state.armies[2] = 1
        action = {
            "action_type": 4,
            "param_a": 0,
            "param_b": 2,
            "param_c": 1,
            "param_d": 0,
        }
        _, reward, _, _, _ = env.step(action)
        assert reward < 0


class TestElimination:
    """Player elimination mechanics."""

    def test_eliminated_player_loses_all_territories(self):
        """When eliminated, a player should have 0 territories."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.territory_owner[:] = 0
        env.state.armies[:] = 10
        env.state.current_player = 0
        env.state.eliminated[1] = 1
        env.state.phase = PHASE_ATTACK
        env.state.armies[0] = 3
        env.state.armies[1] = 1
        env.state.territory_owner[1] = 2
        env._resolve_attack(0, 1, 2)
        assert env.state.eliminated[2] == 1
        assert np.sum(env.state.territory_owner == 2) == 0

    def test_eliminated_player_cards_transferred(self):
        """Eliminated player's cards transfer to attacker."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.territory_owner[:] = 0
        env.state.armies[:] = 10
        env.state.current_player = 0
        env.state.cards[2] = [0, 14, 28]
        env.state.eliminated[1] = 1
        env.state.phase = PHASE_ATTACK
        env.state.armies[0] = 3
        env.state.armies[1] = 1
        env.state.territory_owner[1] = 2
        env._resolve_attack(0, 1, 2)
        assert all(c in env.state.cards[0] for c in [0, 14, 28])

    def test_next_player_skips_eliminated(self):
        """Turn rotation must skip eliminated players."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.current_player = 0
        env.state.eliminated[1] = 1
        env._next_player()
        assert env.state.current_player == 2

    def test_check_win_one_player_remaining(self):
        """When only one player has territories, _check_win is True."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.territory_owner[:] = 0
        env.state.eliminated[1] = 1
        env.state.eliminated[2] = 1
        assert env._check_win()

    def test_check_win_multiple_active(self):
        """When multiple players have territories, _check_win is False."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        assert not env._check_win()

    def test_draw_card_after_capture(self):
        """A successful capture should give a card if under max."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.territory_owner[:] = 0
        env.state.armies[:] = 10
        env.state.current_player = 0
        env.state.eliminated[1] = 1
        env.state.phase = PHASE_ATTACK
        env.state.armies[0] = 3
        env.state.armies[1] = 1
        env.state.territory_owner[1] = 2
        before = len(env.state.cards[0])
        env._resolve_attack(0, 1, 2)
        assert len(env.state.cards[0]) > before

    def test_forced_trade_when_over_max(self):
        """If card transfer exceeds max, a forced trade happens."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.territory_owner[:] = 0
        env.state.armies[:] = 10
        env.state.current_player = 0
        env.state.cards[0] = [0, 1, 2, 3, 4]
        env.state.cards[2] = [10, 11, 12, 13, 14]
        env.state.eliminated[1] = 1
        env.state.phase = PHASE_ATTACK
        env.state.armies[0] = 3
        env.state.armies[1] = 1
        env.state.territory_owner[1] = 2
        env._resolve_attack(0, 1, 2)
        assert len(env.state.cards[0]) <= MAX_CARDS

    def test_deck_reshuffle_when_empty(self):
        """When deck is empty, discard pile is reshuffled into deck."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.deck = []
        env.state.discard_pile = [0, 1, 2]
        env._draw_card(0)
        assert len(env.state.deck) < 3  # one drawn, rest in deck
        assert 0 in env.state.cards[0] or 1 in env.state.cards[0] or 2 in env.state.cards[0]


class TestDenseRewards:
    """Dense reward-shaping signals."""

    def test_territory_delta_reward(self):
        """Capturing a territory gives positive dense territory reward."""
        env = RisikoEnv(
            n_players=3,
            reward_config=RewardConfig(
                sparse_win=0.0,
                sparse_loss=0.0,
                dense_territory_delta=1.0,
                dense_continent_bonus_delta=0.0,
                dense_army_ratio=0.0,
                dense_elimination_bonus=0.0,
                invalid_action_penalty=-1.0,
            ),
        )
        env.reset(seed=42)
        env.state.current_player = 0
        prev = env._snapshot()
        # Simulate capture by changing ownership of one territory
        not_owned = int(np.where(env.state.territory_owner != 0)[0][0])
        env.state.territory_owner[not_owned] = 0
        reward = env._compute_reward(0, prev)
        assert reward == 1.0

    def test_invalid_action_penalty(self):
        """Invalid actions incur the configured penalty."""
        env = RisikoEnv(
            n_players=3,
            reward_config=RewardConfig(invalid_action_penalty=-5.0),
        )
        env.reset(seed=42)
        _, reward, _, _, _ = env.step("bad")  # type: ignore[arg-type]
        assert reward == -5.0

    def test_elimination_bonus(self):
        """Eliminating a player gives dense elimination bonus."""
        env = RisikoEnv(
            n_players=3,
            reward_config=RewardConfig(
                sparse_win=0.0,
                sparse_loss=0.0,
                dense_territory_delta=0.0,
                dense_continent_bonus_delta=0.0,
                dense_army_ratio=0.0,
                dense_elimination_bonus=10.0,
                invalid_action_penalty=-1.0,
            ),
        )
        env.reset(seed=42)
        env.state.current_player = 0
        prev = env._snapshot()
        env.state.eliminated[2] = 1
        reward = env._compute_reward(0, prev)
        # Elimination bonus plus possibly army ratio since total armies > 0
        assert reward >= 10.0

    def test_continent_bonus_delta(self):
        """Gaining a continent gives positive continent bonus reward."""
        env = RisikoEnv(
            n_players=3,
            reward_config=RewardConfig(
                sparse_win=0.0,
                sparse_loss=0.0,
                dense_territory_delta=0.0,
                dense_continent_bonus_delta=2.0,
                dense_army_ratio=0.0,
                dense_elimination_bonus=0.0,
                invalid_action_penalty=-1.0,
            ),
        )
        env.reset(seed=42)
        env.state.current_player = 0
        env.state.phase = PHASE_ATTACK
        # Give player 0 all but one Australia territory
        for t in CONTINENTS["Australia"]:
            env.state.territory_owner[t] = 0
            env.state.armies[t] = 1
        env.state.territory_owner[40] = 2  # New Guinea owned by enemy
        env.state.armies[40] = 1
        env.state.armies[38] = 3  # Eastern Australia has 3 armies
        env.state.eliminated[1] = 1
        _, reward, _, _, _ = env.step(
            {"action_type": 2, "param_a": 38, "param_b": 40, "param_c": 2, "param_d": 0}
        )
        assert reward > 0

    def test_army_ratio_reward(self):
        """Army ratio reward is positive when player has armies."""
        env = RisikoEnv(
            n_players=3,
            reward_config=RewardConfig(
                sparse_win=0.0,
                sparse_loss=0.0,
                dense_territory_delta=0.0,
                dense_continent_bonus_delta=0.0,
                dense_army_ratio=0.5,
                dense_elimination_bonus=0.0,
                invalid_action_penalty=-1.0,
            ),
        )
        env.reset(seed=42)
        env.state.current_player = 0
        env.state.phase = PHASE_ATTACK
        env.state.armies[:] = 1
        env.state.armies[0] = 10
        env.state.territory_owner[0] = 0
        env.state.territory_owner[1] = 1
        env.state.armies[1] = 1
        _, reward, _, _, _ = env.step(
            {"action_type": 2, "param_a": 0, "param_b": 1, "param_c": 2, "param_d": 0}
        )
        assert reward > 0

    def test_sparse_loss_for_eliminated(self):
        """An eliminated player receives sparse_loss."""
        env = RisikoEnv(
            n_players=3,
            reward_config=RewardConfig(
                sparse_win=0.0,
                sparse_loss=-50.0,
            ),
        )
        env.reset(seed=42)
        env.state.current_player = 1
        env.state.eliminated[1] = 1
        prev = env._snapshot()
        reward = env._compute_reward(1, prev)
        assert reward == -50.0

    def test_trade_value_first_trade(self):
        """First trade gives 4 armies."""
        from src.utils.constants import get_trade_value

        assert get_trade_value(0) == 4

    def test_trade_value_sixth_trade(self):
        """Sixth trade gives 15 armies."""
        from src.utils.constants import get_trade_value

        assert get_trade_value(5) == 15

    def test_trade_value_seventh_trade(self):
        """Seventh trade gives 20 armies (15 + 5)."""
        from src.utils.constants import get_trade_value

        assert get_trade_value(6) == 20

    def test_trade_value_escalation(self):
        """Trade values escalate by 5 after the sixth."""
        from src.utils.constants import get_trade_value

        assert get_trade_value(7) == 25

    def test_forms_valid_set_all_wild(self):
        """All wildcards is a valid set."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        assert env._forms_valid_set([3, 3, 3])

    def test_forms_valid_set_two_wild_one_other(self):
        """Two wilds + one other is valid."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        assert env._forms_valid_set([3, 3, 0])

    def test_forms_valid_set_invalid(self):
        """Two infantry + one cavalry is invalid."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        assert not env._forms_valid_set([0, 0, 1])

    def test_has_tradeable_cards_false(self):
        """With fewer than 3 cards, no trade is possible."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.cards[0] = [0, 14]
        assert not env._has_tradeable_cards(0)

    def test_has_tradeable_cards_true(self):
        """With a valid set of 3, trade is possible."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.cards[0] = [0, 1, 2]
        assert env._has_tradeable_cards(0)

    def test_snapshot_returns_copies(self):
        """Snapshot must return independent copies."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        snap = env._snapshot()
        env.state.armies[0] += 1
        assert snap["armies"][0] != env.state.armies[0]

    def test_continent_control_obs(self):
        """Obs continent_control must be 1 for fully owned continents."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        env.state.current_player = 0
        for t in CONTINENTS["Australia"]:
            env.state.territory_owner[t] = 0
        obs = env._get_obs()
        assert obs["continent_control"][5] == 1

    def test_obs_cards_shape(self):
        """Cards observation must have shape (MAX_CARDS, 4)."""
        env = RisikoEnv(n_players=3)
        env.reset(seed=42)
        obs = env._get_obs()
        assert obs["cards"].shape == (MAX_CARDS, 4)
