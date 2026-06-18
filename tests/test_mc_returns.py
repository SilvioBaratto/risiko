"""MC-return correctness tests for issue #90.

Three layers of verification (from the acceptance criterion):
 1. ``compute_per_player_returns`` on tiny, hand-checkable examples — verifies
    that the interleaved-player discount arithmetic matches the formula.
 2. Shard-level cross-check — monkeypatches the ``compute_per_player_returns``
    call inside ``bc_dataset`` to capture inputs/outputs, then asserts that the
    ``mc_return`` column stored in the shard equals those outputs exactly.
 3. Property-based tests — invariants hold for all valid inputs.

Correct signature (training/bc_returns.py):
    compute_per_player_returns(rewards: list[float],
                               players: list[int],
                               gamma: float) -> list[float]

POST-step attribution note (from bc_dataset.py docstring):
    ``players[t]`` is the POST-step ``current_player`` (from
    ``t.obs["current_player"]``), which on turn-ending steps is the NEXT player
    rather than the agent who made the decision.  This is deliberate; the shard
    ``acting_player`` records the PRE-step decision-maker while ``mc_return``
    uses the POST-step recipient.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from training.bc_returns import compute_per_player_returns

# ---------------------------------------------------------------------------
# Hand-checkable single-player examples
# Players all = 0 ⟹ standard backward accumulation: G[t]=r[t]+γ·G[t+1]
# ---------------------------------------------------------------------------


def test_when_single_player_rewards_zero_zero_eight_gamma_half_then_returns_two_four_eight():
    """rewards=[0,0,8], players=[0,0,0], gamma=0.5 → G=[2,4,8]."""
    result = compute_per_player_returns([0.0, 0.0, 8.0], [0, 0, 0], 0.5)
    np.testing.assert_allclose(result, [2.0, 4.0, 8.0], rtol=1e-5)


def test_when_single_player_first_reward_only_then_returns_correct():
    """rewards=[1,0,0], players=[0,0,0], gamma=0.5 → G=[1,0,0]."""
    result = compute_per_player_returns([1.0, 0.0, 0.0], [0, 0, 0], 0.5)
    np.testing.assert_allclose(result, [1.0, 0.0, 0.0], rtol=1e-5)


def test_when_gamma_zero_then_returns_equal_immediate_rewards():
    """gamma=0 ⟹ no future discounting; G[t]=r[t]."""
    rewards = [1.0, 2.0, 3.0]
    result = compute_per_player_returns(rewards, [0, 1, 0], 0.0)
    assert result == pytest.approx(rewards)


def test_when_single_step_then_return_equals_reward_for_any_gamma():
    """One-step game: G[0]=r[0] regardless of gamma."""
    for gamma in [0.0, 0.5, 0.99, 1.0]:
        result = compute_per_player_returns([5.0], [0], gamma)
        assert result == pytest.approx([5.0]), f"Failed for gamma={gamma}"


# ---------------------------------------------------------------------------
# Two-player interleaved — the key invariant:
# player 0's discount chain skips all player-1 steps
# ---------------------------------------------------------------------------
#
# rewards=[1,10,2,20], players=[0,1,0,1], gamma=0.5
#   Player-0 steps: idx 0,2
#     G[2] = 2.0
#     G[0] = 1.0 + 0.5 × 2.0 = 2.0
#   Player-1 steps: idx 1,3
#     G[3] = 20.0
#     G[1] = 10.0 + 0.5 × 20.0 = 20.0


def test_when_two_players_interleaved_then_player0_first_step_discounts_only_own_steps():
    """G[0] = r[0] + γ·r[2] (player-1 step in between is skipped)."""
    result = compute_per_player_returns([1.0, 10.0, 2.0, 20.0], [0, 1, 0, 1], 0.5)
    assert result[0] == pytest.approx(2.0)


def test_when_two_players_interleaved_then_player0_last_step_has_no_future():
    """G[2] = r[2] — player 0 has no subsequent steps."""
    result = compute_per_player_returns([1.0, 10.0, 2.0, 20.0], [0, 1, 0, 1], 0.5)
    assert result[2] == pytest.approx(2.0)


def test_when_two_players_interleaved_then_player1_first_step_discounts_only_own_steps():
    """G[1] = r[1] + γ·r[3] = 10 + 0.5×20 = 20."""
    result = compute_per_player_returns([1.0, 10.0, 2.0, 20.0], [0, 1, 0, 1], 0.5)
    assert result[1] == pytest.approx(20.0)


def test_when_two_players_interleaved_then_player1_last_step_has_no_future():
    """G[3] = r[3] = 20."""
    result = compute_per_player_returns([1.0, 10.0, 2.0, 20.0], [0, 1, 0, 1], 0.5)
    assert result[3] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Shard-level cross-check
#
# Monkeypatches ``compute_per_player_returns`` inside ``training.bc_dataset``
# to capture every (rewards, players, gamma) call and its return value.
# After ``generate_bc_dataset`` completes, asserts that the ``mc_return``
# column in every shard exactly matches the captured return values.
# ---------------------------------------------------------------------------


def _tiny_cfg_for_mc_test(dataset_dir: str):
    from src.config import BCConfig, TrainingConfig  # noqa: PLC0415

    bc = BCConfig(
        n_games=2,
        n_players=2,
        max_turns=10,
        seed=42,
        dataset_dir=dataset_dir,
        shard_size=500,
    )
    return TrainingConfig(bc=bc)


def test_when_shard_mc_returns_loaded_then_they_match_compute_per_player_returns_output(
    tmp_path, monkeypatch
):
    """
    Cross-check: mc_return values in every shard row equal the outputs that
    compute_per_player_returns produced during dataset generation.

    Design: monkeypatch the ``compute_per_player_returns`` name in the
    bc_dataset module so we can capture all (reward, player, return) triples
    written to the shard, then assert the stored mc_return column matches.
    """
    import pathlib  # noqa: PLC0415

    import training.bc_dataset as _mod  # noqa: PLC0415
    from training.bc_dataset import generate_bc_dataset  # noqa: PLC0415
    from training.bc_returns import compute_per_player_returns as _real  # noqa: PLC0415

    captured_returns: list[float] = []

    def _intercepted(rewards, players, gamma):
        result = _real(rewards, players, gamma)
        captured_returns.extend(result)
        return result

    monkeypatch.setattr(_mod, "compute_per_player_returns", _intercepted)

    cfg = _tiny_cfg_for_mc_test(str(tmp_path))
    generate_bc_dataset(cfg)

    assert captured_returns, "compute_per_player_returns was never called during generation"

    shards = sorted(pathlib.Path(str(tmp_path)).glob("shard_*.npz"))
    assert shards, "No shards were written"

    shard_returns = np.concatenate([np.load(s)["mc_return"] for s in shards])

    assert len(shard_returns) == len(captured_returns), (
        f"Shard has {len(shard_returns)} mc_return values but "
        f"{len(captured_returns)} were captured from compute_per_player_returns"
    )
    np.testing.assert_allclose(
        shard_returns,
        np.array(captured_returns, dtype=np.float32),
        rtol=1e-4,
        err_msg="Shard mc_return values differ from compute_per_player_returns output",
    )


# ---------------------------------------------------------------------------
# Property-based tests — invariants
# ---------------------------------------------------------------------------


@given(
    rewards=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    ),
    gamma=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_when_single_player_then_output_length_equals_input_length(
    rewards: list[float], gamma: float
) -> None:
    """Invariant: len(result) == len(rewards) for any valid single-player input."""
    players = [0] * len(rewards)
    result = compute_per_player_returns(rewards, players, gamma)
    assert len(result) == len(rewards)


@given(
    rewards=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    ),
)
@settings(max_examples=40, deadline=None)
def test_when_gamma_zero_then_returns_always_equal_immediate_rewards(
    rewards: list[float],
) -> None:
    """Invariant: gamma=0 ⟹ G[t]==r[t] for every step and any player sequence."""
    n = len(rewards)
    players = [i % 2 for i in range(n)]
    result = compute_per_player_returns(rewards, players, 0.0)
    assert result == pytest.approx(rewards)


@given(
    rewards=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    ),
    gamma=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_when_called_twice_with_same_inputs_then_results_are_identical(
    rewards: list[float], gamma: float
) -> None:
    """Idempotence: pure function — same inputs produce identical outputs."""
    players = [i % 2 for i in range(len(rewards))]
    r1 = compute_per_player_returns(list(rewards), players, gamma)
    r2 = compute_per_player_returns(list(rewards), players, gamma)
    np.testing.assert_array_equal(r1, r2, err_msg="compute_per_player_returns is not pure")
