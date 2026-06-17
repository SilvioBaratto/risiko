"""
Source-blind example tests for compute_per_player_returns (issue #77).

All tests are derived from the acceptance criteria only. Tests written in
Red phase and confirmed failing before any implementation exists.

Design choices (from criterion text):
- compute_per_player_returns(rewards, players, gamma) -> list[float]
- Per-player discounting: discount only counts intervening steps of the SAME
  player (not global step index). A step belonging to player 1 does NOT
  advance player 0's discount chain.
- training/ is a namespace package (no __init__.py); import as
  ``from training.bc_returns import compute_per_player_returns``.
"""

from __future__ import annotations

import inspect

import pytest
from hypothesis import given
from hypothesis import strategies as st

from training.bc_returns import compute_per_player_returns

# ---------------------------------------------------------------------------
# Scope criterion: pure function, no torch/IO imports
# ---------------------------------------------------------------------------


def test_when_module_is_imported_then_torch_is_not_a_dependency():
    """The module must contain no torch import (pure function, no network deps)."""
    import training.bc_returns as _mod

    src = inspect.getsource(_mod)
    assert "import torch" not in src
    assert "from torch" not in src


def test_when_module_is_imported_then_no_io_imports_are_present():
    """No file-IO imports (open, pathlib, os.path writes) in the module source."""
    import training.bc_returns as _mod

    src = inspect.getsource(_mod)
    assert "import io" not in src
    assert "from pathlib" not in src
    # 'open(' in source would indicate file I/O
    assert "open(" not in src


def test_when_called_then_result_is_a_list_of_float():
    """Function must return list[float]."""
    result = compute_per_player_returns([1.0, 2.0], [0, 1], 0.99)
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# Interleaved players criterion
# ---------------------------------------------------------------------------


def test_when_players_interleaved_0101_then_player_0_first_return_skips_player_1_step():
    """
    For players=[0,1,0,1], rewards=[r0,r1,r2,r3], gamma=0.5:
    player 0's return at index 0 = r0 + gamma * r2.
    The player-1 step between them does NOT appear in player 0's chain.
    """
    rewards = [1.0, 10.0, 2.0, 20.0]
    players = [0, 1, 0, 1]
    gamma = 0.5
    result = compute_per_player_returns(rewards, players, gamma)
    expected = 1.0 + 0.5 * 2.0  # 2.0
    assert result[0] == pytest.approx(expected)


def test_when_players_interleaved_0101_then_player_0_last_step_has_no_future():
    """
    Player 0's last step (index 2) has no subsequent player-0 steps.
    Its return must equal r2 exactly.
    """
    rewards = [1.0, 10.0, 2.0, 20.0]
    players = [0, 1, 0, 1]
    gamma = 0.5
    result = compute_per_player_returns(rewards, players, gamma)
    assert result[2] == pytest.approx(2.0)


def test_when_players_interleaved_0101_then_player_1_first_return_skips_player_0_steps():
    """
    For players=[0,1,0,1], player 1's return at index 1 = r1 + gamma * r3.
    Player-0 steps do NOT appear in player 1's discount chain.
    """
    rewards = [1.0, 10.0, 2.0, 20.0]
    players = [0, 1, 0, 1]
    gamma = 0.5
    result = compute_per_player_returns(rewards, players, gamma)
    expected = 10.0 + 0.5 * 20.0  # 20.0
    assert result[1] == pytest.approx(expected)


def test_when_players_interleaved_0101_then_player_1_last_step_has_no_future():
    """Player 1's last step (index 3) has no future: return == r3."""
    rewards = [1.0, 10.0, 2.0, 20.0]
    players = [0, 1, 0, 1]
    gamma = 0.5
    result = compute_per_player_returns(rewards, players, gamma)
    assert result[3] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# gamma extremes criterion
# ---------------------------------------------------------------------------


def test_when_gamma_is_zero_then_returns_equal_immediate_rewards():
    """gamma=0.0 → each return equals its immediate reward with no future contribution."""
    rewards = [1.0, 2.0, 3.0, 4.0]
    players = [0, 1, 0, 1]
    result = compute_per_player_returns(rewards, players, gamma=0.0)
    assert result == pytest.approx(rewards)


def test_when_gamma_is_one_then_player_returns_are_undiscounted_cumulative_sums():
    """
    gamma=1.0 → each player's return is the sum of ALL their future rewards
    (current step included), with no discounting.
    """
    rewards = [1.0, 10.0, 2.0, 20.0]
    players = [0, 1, 0, 1]
    result = compute_per_player_returns(rewards, players, gamma=1.0)
    # Player 0 at index 0: 1.0 + 2.0 = 3.0
    assert result[0] == pytest.approx(3.0)
    # Player 0 at index 2: 2.0 (last player-0 step)
    assert result[2] == pytest.approx(2.0)
    # Player 1 at index 1: 10.0 + 20.0 = 30.0
    assert result[1] == pytest.approx(30.0)
    # Player 1 at index 3: 20.0 (last player-1 step)
    assert result[3] == pytest.approx(20.0)


def test_when_gamma_is_one_single_player_then_return_is_total_sum_of_all_rewards():
    """
    Single player, gamma=1.0: first return = sum of all rewards.
    """
    rewards = [1.0, 2.0, 3.0, 4.0]
    players = [0, 0, 0, 0]
    result = compute_per_player_returns(rewards, players, gamma=1.0)
    assert result[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Edge cases criterion
# ---------------------------------------------------------------------------


def test_when_input_is_empty_then_output_is_empty():
    """Empty rewards and players → empty return list."""
    result = compute_per_player_returns([], [], gamma=0.99)
    assert result == []


def test_when_single_step_then_return_equals_that_reward():
    """A single-step game: the only return equals the immediate reward."""
    result = compute_per_player_returns([7.5], [0], gamma=0.99)
    assert result == pytest.approx([7.5])


def test_when_single_step_any_gamma_then_return_equals_that_reward():
    """With a single step, gamma has no effect; return always equals reward."""
    for gamma in [0.0, 0.5, 0.99, 1.0]:
        result = compute_per_player_returns([-3.5], [2], gamma=gamma)
        assert result == pytest.approx([-3.5]), f"Failed for gamma={gamma}"


def test_when_called_then_len_returns_equals_len_rewards():
    """len(returns) == len(rewards) for typical multi-step input."""
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    players = [0, 1, 2, 0, 1]
    result = compute_per_player_returns(rewards, players, gamma=0.9)
    assert len(result) == len(rewards)


def test_when_called_then_len_returns_equals_len_players():
    """len(returns) == len(players) for typical multi-step input."""
    rewards = [1.0, 2.0, 3.0]
    players = [0, 1, 0]
    result = compute_per_player_returns(rewards, players, gamma=0.5)
    assert len(result) == len(players)


# ---------------------------------------------------------------------------
# Three-player interleaved test (validates N-player generalisation)
# ---------------------------------------------------------------------------


def test_when_three_players_interleaved_then_each_discounts_only_own_steps():
    """
    Three-player rotation [0,1,2,0,1,2]: each player skips the two other
    players' steps between their own decisions.
    """
    rewards = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    players = [0, 1, 2, 0, 1, 2]
    gamma = 0.5
    result = compute_per_player_returns(rewards, players, gamma)
    # Player 0 at index 0: 1.0 + 0.5 * 1.0 = 1.5
    assert result[0] == pytest.approx(1.5)
    # Player 0 at index 3: 1.0 (no future player-0 steps)
    assert result[3] == pytest.approx(1.0)
    # Player 1 at index 1: 1.0 + 0.5 * 1.0 = 1.5
    assert result[1] == pytest.approx(1.5)
    # Player 2 at index 2: 1.0 + 0.5 * 1.0 = 1.5
    assert result[2] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Property-based tests (invariants derived from the criteria)
# ---------------------------------------------------------------------------

_valid_rewards = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)
_valid_gamma = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)


@given(
    rewards=st.lists(_valid_rewards, min_size=0, max_size=60),
    gamma=_valid_gamma,
)
def test_when_any_valid_input_then_output_length_equals_input_length(rewards, gamma):
    """
    Invariant (ordering): len(returns) == len(rewards) for ALL valid inputs.
    Derived from: "len(returns)==len(rewards)==len(players)".
    """
    n = len(rewards)
    players = [i % 2 for i in range(n)]  # alternating two players, always valid
    result = compute_per_player_returns(rewards, players, gamma)
    assert len(result) == n


@given(
    rewards=st.lists(_valid_rewards, min_size=1, max_size=60),
    n_players=st.integers(min_value=1, max_value=6),
)
def test_when_gamma_is_zero_returns_always_equal_immediate_rewards(rewards, n_players):
    """
    Invariant: gamma=0.0 makes every return equal to its immediate reward,
    for ANY input length and player assignment.
    Derived from: "gamma=0.0 → returns equal immediate rewards".
    """
    players = [i % n_players for i in range(len(rewards))]
    result = compute_per_player_returns(rewards, players, gamma=0.0)
    assert result == pytest.approx(rewards)


@given(
    reward=_valid_rewards,
    n=st.integers(min_value=1, max_value=50),
    gamma=_valid_gamma,
)
def test_when_single_player_then_last_return_always_equals_immediate_reward(reward, n, gamma):
    """
    Invariant (ordering): the last step of any player always has no future
    decisions of their own, so its return must always equal the immediate reward.
    Derived from the algorithm: running[p] is seeded to 0.0 for the final step.
    """
    rewards = [reward] * n
    players = [0] * n
    result = compute_per_player_returns(rewards, players, gamma)
    # The last step has no future player-0 steps → return == reward
    assert result[-1] == pytest.approx(reward)


@given(
    rewards=st.lists(_valid_rewards, min_size=2, max_size=60),
)
def test_when_all_same_player_gamma_one_returns_non_increasing_for_nonneg_rewards(
    rewards,
):
    """
    Invariant (ordering/monotonicity): for a single player, gamma=1.0, and
    non-negative rewards, earlier steps have returns >= later steps.
    Proof: returns[t] = sum(rewards[t:]), so returns[t] - returns[t+1] = rewards[t] >= 0.
    This invariant requires EXACTLY gamma=1.0; for gamma < 1 it can fail when
    rewards[t] = 0 and a future reward is large.
    """
    non_negative_rewards = [abs(r) for r in rewards]
    players = [0] * len(non_negative_rewards)
    result = compute_per_player_returns(non_negative_rewards, players, gamma=1.0)
    for i in range(len(result) - 1):
        assert result[i] >= result[i + 1] - 1e-9, (
            f"Earlier return {result[i]} < later return {result[i + 1]} at index {i}"
        )
