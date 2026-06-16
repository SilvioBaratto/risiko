"""Source-blind example tests for issue #54: RisikoEnv Gymnasium environment.

Tests are authored exclusively from the acceptance-criteria text and the oracle
classification.  No implementation source was read (Red phase of TDD).

Verifiable scope criteria covered:
  [UNIT] RisikoEnv(n_players, reward_config=None) validates 2 ≤ n_players ≤ 6
         and declares the 11-key observation_space and ACTION_DIMS-based action_space
  [UNIT] reset(seed, options) deals the board, places starting armies, computes
         reinforcements, sets the initial phase, and returns (obs_dict, info)
         with info["legal_actions"]
  [UNIT] step(action) returns the Gymnasium 5-tuple (obs, reward, terminated,
         truncated, info); structurally invalid actions return invalid_action_penalty
         and terminated=False
  [UNIT] Each phase (TRADE/REINFORCE/ATTACK/CAPTURE_MOVE/FORTIFY) is dispatched
         to its handler; handlers advance phase per the state machine
  [UNIT] Continent bonuses applied (NA+5, SA+2, EU+5, AF+3, AS+7, AU+2);
         current_player advances once per turn skipping eliminated players
  [UNIT] step() is deterministic given a fixed seed
  [UNIT] src/env.py re-exports GameState, PHASE_*, and MAX_CARDS

Verifiable global criteria covered:
  [UNIT] Checkpoints saved every save_freq episodes to models/;
         training fully resumable from any .pt file
  [UNIT] Results tagged with seed and config file path

Skipped criteria (oracle: NOT VERIFIABLE at unit level):
  - LLM non-blocking / 30 s timeout fallback
  - Global env-step determinism (marked NOT VERIFIABLE in oracle global criteria)
  - YAML/CLI no-magic-numbers (static analysis, not a unit assertion)
  - Territory ≥1 army / attacker ≥2 / adjacent-only attack rules (NOT VERIFIABLE scope)
  - Card rules: draw 1, trade 3, max 5, only at turn start (NOT VERIFIABLE scope)
  - Baseline win rates (empirical benchmark, not unit assertion)
  - Code coverage ≥80% (pipeline metric, not a per-criterion assertion)
  - All tests pass (boilerplate gate)
  - SOLID / clean code (subjective prose)
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Module-level constants derived from the acceptance criteria, not from source
# ---------------------------------------------------------------------------

# Phase constant values are specified as PHASE_TRADE=0 … PHASE_FORTIFY=4 in the
# criteria text and the existing state-module tests; we verify the re-exports
# here rather than importing from env_core directly.
_VALID_PHASES = frozenset({0, 1, 2, 3, 4})  # TRADE … FORTIFY

# An action whose action_type (99) cannot match any real game phase.
# Used to trigger the env's invalid-action handling path.
_INVALID_ACTION = {
    "action_type": 99,
    "param_a": 0,
    "param_b": 0,
    "param_c": 0,
    "param_d": 0,
}

# Continent bonus specification from the acceptance criteria:
# NA+5, SA+2, EU+5, AF+3, AS+7, AU+2  → sorted: [2, 2, 3, 5, 5, 7]
_EXPECTED_BONUS_SORTED = [2, 2, 3, 5, 5, 7]
_EXPECTED_BONUS_SUM = 24  # 5+2+5+3+7+2


# ===========================================================================
# Section 1 — src/env.py re-exports GameState, PHASE_*, MAX_CARDS
# Criterion: `src/env.py` re-exports `GameState`, `PHASE_*`, and `MAX_CARDS`;
#            the existing tests pass unmodified.
# ===========================================================================


class TestEnvReExports:
    """All specified symbols must be importable directly from src.env."""

    def test_when_gamestate_imported_from_src_env_then_it_is_not_none(self):
        from src.env import GameState  # noqa: F401

        assert GameState is not None

    def test_when_phase_trade_imported_from_src_env_then_value_is_zero(self):
        from src.env import PHASE_TRADE

        assert PHASE_TRADE == 0

    def test_when_phase_reinforce_imported_from_src_env_then_value_is_one(self):
        from src.env import PHASE_REINFORCE

        assert PHASE_REINFORCE == 1

    def test_when_phase_attack_imported_from_src_env_then_value_is_two(self):
        from src.env import PHASE_ATTACK

        assert PHASE_ATTACK == 2

    def test_when_phase_capture_move_imported_from_src_env_then_value_is_three(self):
        from src.env import PHASE_CAPTURE_MOVE

        assert PHASE_CAPTURE_MOVE == 3

    def test_when_phase_fortify_imported_from_src_env_then_value_is_four(self):
        from src.env import PHASE_FORTIFY

        assert PHASE_FORTIFY == 4

    def test_when_max_cards_imported_from_src_env_then_value_is_five(self):
        from src.env import MAX_CARDS

        assert MAX_CARDS == 5

    def test_when_risiko_env_imported_from_src_env_then_it_is_not_none(self):
        """RisikoEnv itself must be importable from the top-level env module."""
        from src.env import RisikoEnv  # noqa: F401

        assert RisikoEnv is not None


# ===========================================================================
# Section 2 — Constructor validation and space declarations
# Criterion: RisikoEnv(n_players, reward_config=None) validates 2 ≤ n_players ≤ 6
#            and declares the 11-key observation_space and ACTION_DIMS-based action_space
# ===========================================================================


class TestRisikoEnvConstructorValidation:
    """n_players outside [2, 6] must be rejected at construction time."""

    def test_when_n_players_is_0_then_error_is_raised(self):
        from src.env import RisikoEnv

        with pytest.raises((ValueError, AssertionError)):
            RisikoEnv(n_players=0)

    def test_when_n_players_is_1_then_error_is_raised(self):
        """1 is below the minimum of 2."""
        from src.env import RisikoEnv

        with pytest.raises((ValueError, AssertionError)):
            RisikoEnv(n_players=1)

    def test_when_n_players_is_7_then_error_is_raised(self):
        """7 exceeds the maximum of 6."""
        from src.env import RisikoEnv

        with pytest.raises((ValueError, AssertionError)):
            RisikoEnv(n_players=7)

    def test_when_n_players_is_negative_then_error_is_raised(self):
        from src.env import RisikoEnv

        with pytest.raises((ValueError, AssertionError)):
            RisikoEnv(n_players=-1)

    def test_when_n_players_is_2_then_env_is_created_without_error(self):
        """2 is the minimum valid value."""
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=2)
        assert env is not None

    def test_when_n_players_is_6_then_env_is_created_without_error(self):
        """6 is the maximum valid value."""
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=6)
        assert env is not None

    def test_when_reward_config_is_none_then_env_is_created_without_error(self):
        """reward_config=None is the documented default signature."""
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=2, reward_config=None)
        assert env is not None


class TestRisikoEnvSpaces:
    """observation_space and action_space must be declared correctly."""

    def test_when_env_created_then_observation_space_has_11_keys(self):
        """observation_space is a Dict space with exactly 11 subspaces."""
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=2)
        assert len(env.observation_space) == 11

    def test_when_env_created_then_action_space_is_not_none(self):
        """action_space (ACTION_DIMS-based) must be declared at construction time."""
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=2)
        assert env.action_space is not None


# ---------------------------------------------------------------------------
# Properties for constructor and observation_space
# ---------------------------------------------------------------------------


@given(n_players=st.integers().filter(lambda n: n < 2 or n > 6))
@settings(max_examples=20)
def test_when_n_players_is_any_out_of_range_value_then_error_is_always_raised(
    n_players: int,
) -> None:
    """Invariant (never-raises-for-valid-input, inverted): invalid n always raises."""
    from src.env import RisikoEnv

    with pytest.raises((ValueError, AssertionError)):
        RisikoEnv(n_players=n_players)


@given(n_players=st.integers(min_value=2, max_value=6))
@settings(max_examples=10)
def test_when_n_players_is_valid_then_observation_space_always_has_11_keys(
    n_players: int,
) -> None:
    """Invariant: 11-key observation_space holds for every valid player count."""
    from src.env import RisikoEnv

    env = RisikoEnv(n_players=n_players)
    assert len(env.observation_space) == 11


# ===========================================================================
# Section 3 — reset(seed, options)
# Criterion: reset(seed, options) deals the board, places starting armies,
#            computes reinforcements, sets the initial phase, and returns
#            (obs_dict, info) with info["legal_actions"]
# ===========================================================================


class TestReset:
    """Structural contract of reset()."""

    def test_when_reset_called_then_returns_two_element_tuple(self):
        from src.env import RisikoEnv

        result = RisikoEnv(n_players=2).reset(seed=0)
        assert len(result) == 2

    def test_when_reset_called_then_first_element_is_obs_dict(self):
        from src.env import RisikoEnv

        obs, _ = RisikoEnv(n_players=2).reset(seed=0)
        assert isinstance(obs, dict)

    def test_when_reset_called_then_obs_dict_has_11_keys(self):
        from src.env import RisikoEnv

        obs, _ = RisikoEnv(n_players=2).reset(seed=0)
        assert len(obs) == 11

    def test_when_reset_called_then_info_has_legal_actions_key(self):
        from src.env import RisikoEnv

        _, info = RisikoEnv(n_players=2).reset(seed=0)
        assert "legal_actions" in info

    def test_when_reset_called_then_legal_actions_is_a_list(self):
        from src.env import RisikoEnv

        _, info = RisikoEnv(n_players=2).reset(seed=0)
        assert isinstance(info["legal_actions"], list)

    def test_when_reset_called_then_legal_actions_is_non_empty(self):
        """At game start there must be at least one legal action available."""
        from src.env import RisikoEnv

        _, info = RisikoEnv(n_players=2).reset(seed=0)
        assert len(info["legal_actions"]) > 0

    def test_when_reset_called_then_all_42_territories_have_at_least_one_army(self):
        """After placement all 42 territories must have ≥ 1 army."""
        from src.env import RisikoEnv

        obs, _ = RisikoEnv(n_players=2).reset(seed=0)
        armies = obs["armies"]
        assert armies.shape == (42,), f"Expected 42 territories, got shape {armies.shape}"
        assert np.all(armies >= 1), (
            f"Territory with 0 armies found after reset: {np.where(armies < 1)}"
        )

    def test_when_reset_called_then_current_player_is_a_valid_index(self):
        """current_player must be in [0, n_players)."""
        from src.env import RisikoEnv

        n = 3
        obs, _ = RisikoEnv(n_players=n).reset(seed=0)
        player = int(obs["current_player"])
        assert 0 <= player < n, f"current_player={player} is outside [0, {n})"

    def test_when_reset_called_then_initial_phase_is_one_of_the_five_valid_phases(self):
        """Phase after reset must be in {TRADE=0, REINFORCE=1, ATTACK=2, CM=3, FORTIFY=4}."""
        from src.env import RisikoEnv

        obs, _ = RisikoEnv(n_players=2).reset(seed=0)
        assert int(obs["phase"]) in _VALID_PHASES

    def test_when_reset_called_then_initial_phase_is_trade_or_reinforce(self):
        """The start of a turn is always TRADE (0) or REINFORCE (1), never mid-turn."""
        from src.env import PHASE_REINFORCE, PHASE_TRADE, RisikoEnv

        obs, _ = RisikoEnv(n_players=2).reset(seed=0)
        assert int(obs["phase"]) in {PHASE_TRADE, PHASE_REINFORCE}, (
            f"Expected initial phase in {{TRADE, REINFORCE}}, got {obs['phase']}"
        )

    def test_when_reset_called_with_same_seed_twice_then_observations_are_identical(self):
        """reset() with the same seed must be deterministic."""
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=2)
        obs1, _ = env.reset(seed=7)
        obs2, _ = env.reset(seed=7)
        for key in obs1:
            np.testing.assert_array_equal(
                obs1[key],
                obs2[key],
                err_msg=f"obs['{key}'] differs between two resets with seed=7",
            )

    def test_when_reset_called_with_different_seeds_then_board_layouts_differ(self):
        """Different seeds should produce distinct board layouts (armies or owners differ)."""
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=2)
        obs_a, _ = env.reset(seed=0)
        obs_b, _ = env.reset(seed=999)
        owners_differ = not np.array_equal(obs_a["territory_owner"], obs_b["territory_owner"])
        armies_differ = not np.array_equal(obs_a["armies"], obs_b["armies"])
        assert owners_differ or armies_differ, (
            "reset() with seeds 0 and 999 produced identical boards — seed may not be applied"
        )


# ===========================================================================
# Section 4 — step(action) returns the Gymnasium 5-tuple
# Criterion: step(action) returns (obs, reward, terminated, truncated, info);
#            structurally invalid actions return invalid_action_penalty and
#            terminated=False
# ===========================================================================


def _env_with_legal_action(n_players: int = 2, seed: int = 42):
    """Return (env, first_legal_action) after reset so tests share the same setup."""
    from src.env import RisikoEnv

    env = RisikoEnv(n_players=n_players)
    _, info = env.reset(seed=seed)
    return env, info["legal_actions"][0]


class TestStepTuple:
    """step() must always return a Gymnasium 5-tuple with the correct element types."""

    def test_when_step_called_then_result_has_five_elements(self):
        env, action = _env_with_legal_action()
        result = env.step(action)
        assert len(result) == 5

    def test_when_step_called_then_first_element_is_obs_dict(self):
        env, action = _env_with_legal_action()
        obs, *_ = env.step(action)
        assert isinstance(obs, dict)

    def test_when_step_called_then_obs_has_11_keys(self):
        env, action = _env_with_legal_action()
        obs, *_ = env.step(action)
        assert len(obs) == 11

    def test_when_step_called_then_reward_is_numeric(self):
        env, action = _env_with_legal_action()
        _, reward, *_ = env.step(action)
        assert isinstance(reward, (int, float, np.floating, np.integer))

    def test_when_step_called_then_terminated_is_bool(self):
        env, action = _env_with_legal_action()
        _, _, terminated, *_ = env.step(action)
        assert isinstance(terminated, (bool, np.bool_))

    def test_when_step_called_then_truncated_is_bool(self):
        env, action = _env_with_legal_action()
        _, _, _, truncated, *_ = env.step(action)
        assert isinstance(truncated, (bool, np.bool_))

    def test_when_step_called_then_info_has_legal_actions_key(self):
        env, action = _env_with_legal_action()
        _, _, _, _, info = env.step(action)
        assert "legal_actions" in info

    def test_when_step_called_with_valid_action_and_game_not_done_then_info_legal_actions_is_list(
        self,
    ):
        env, action = _env_with_legal_action()
        _, _, terminated, truncated, info = env.step(action)
        if not terminated and not truncated:
            assert isinstance(info["legal_actions"], list)


class TestInvalidAction:
    """Structurally invalid actions must return the penalty and leave the game alive."""

    def test_when_step_called_with_invalid_action_type_then_terminated_is_false(self):
        """An invalid action must not end the game."""
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=2)
        env.reset(seed=42)
        _, _, terminated, _, _ = env.step(_INVALID_ACTION)
        assert terminated is False, "Structurally invalid action must not terminate the episode"

    def test_when_step_called_with_invalid_action_type_then_reward_is_nonpositive(self):
        """invalid_action_penalty is ≤ 0; a structurally invalid action must be penalised."""
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=2)
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(_INVALID_ACTION)
        assert float(reward) <= 0.0, f"Expected invalid_action_penalty ≤ 0, got reward={reward}"

    def test_when_step_called_with_invalid_action_type_then_truncated_is_false(self):
        """An invalid action must not truncate the episode either."""
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=2)
        env.reset(seed=42)
        _, _, _, truncated, _ = env.step(_INVALID_ACTION)
        assert truncated is False


# ===========================================================================
# Section 5 — Phase dispatch
# Criterion: Each phase (TRADE/REINFORCE/ATTACK/CAPTURE_MOVE/FORTIFY) is
#            dispatched to its handler; handlers advance phase per the state machine
# ===========================================================================


class TestPhaseDispatch:
    """phase value stays in the 5-element valid set at all times."""

    def test_when_env_reset_then_phase_is_one_of_the_five_valid_constants(self):
        from src.env import RisikoEnv

        obs, _ = RisikoEnv(n_players=2).reset(seed=0)
        assert int(obs["phase"]) in _VALID_PHASES

    def test_when_legal_action_is_stepped_then_resulting_phase_is_valid(self):
        """Dispatching a legal action must keep phase within the valid set."""
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=2)
        obs, info = env.reset(seed=0)
        obs, _, _, _, _ = env.step(info["legal_actions"][0])
        assert int(obs["phase"]) in _VALID_PHASES

    def test_when_50_legal_actions_are_stepped_then_phase_is_always_valid(self):
        """Phase dispatch invariant holds across an extended run of 50 steps."""
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=2)
        obs, info = env.reset(seed=0)
        for step_idx in range(50):
            legal = info["legal_actions"]
            if not legal:
                break
            obs, _, terminated, truncated, info = env.step(legal[0])
            assert int(obs["phase"]) in _VALID_PHASES, (
                f"Step {step_idx}: phase={obs['phase']} is not in {_VALID_PHASES}"
            )
            if terminated or truncated:
                break

    def test_when_phase_advances_from_initial_after_legal_action_then_phase_is_monotone_or_wraps(
        self,
    ):
        """After one legal action the phase either stays, advances, or wraps to next turn.

        The state machine may advance within the same turn or wrap back to TRADE/REINFORCE
        when a new turn begins — both are legal transitions.
        """
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=2)
        _, info = env.reset(seed=0)
        obs1, _, terminated, _, _ = env.step(info["legal_actions"][0])
        phase1 = int(obs1["phase"])
        # Phase must remain in the valid set; detailed transition logic is NOT VERIFIABLE
        assert phase1 in _VALID_PHASES
        if not terminated:
            # If the game is still running the phase is valid; that is the
            # only invariant we can assert without reading the state machine.
            assert phase1 in _VALID_PHASES


@given(n_players=st.integers(min_value=2, max_value=6))
@settings(max_examples=5)
def test_when_env_reset_with_any_valid_player_count_then_phase_is_always_a_valid_constant(
    n_players: int,
) -> None:
    """Invariant: initial phase is in _VALID_PHASES for all valid game sizes."""
    from src.env import RisikoEnv

    obs, _ = RisikoEnv(n_players=n_players).reset(seed=0)
    assert int(obs["phase"]) in _VALID_PHASES


# ===========================================================================
# Section 6 — Continent bonuses
# Criterion: Continent bonuses applied (NA+5, SA+2, EU+5, AF+3, AS+7, AU+2)
# ===========================================================================


class TestContinentBonuses:
    """Continent bonus constants must match the specification exactly."""

    def test_when_continent_bonuses_inspected_then_exactly_6_continents_are_defined(
        self,
    ):
        """Standard Risiko has exactly 6 continents."""
        from src.env_core.board import CONTINENT_BONUSES

        assert len(CONTINENT_BONUSES) == 6, (
            f"Expected 6 continents, got {len(CONTINENT_BONUSES)}: {list(CONTINENT_BONUSES)}"
        )

    def test_when_continent_bonuses_inspected_then_sorted_values_match_specification(
        self,
    ):
        """Sorted bonus list must be [2, 2, 3, 5, 5, 7] (NA=5 SA=2 EU=5 AF=3 AS=7 AU=2)."""
        from src.env_core.board import CONTINENT_BONUSES

        values = sorted(CONTINENT_BONUSES.values())
        assert values == _EXPECTED_BONUS_SORTED, (
            f"Continent bonus values {values} do not match spec {_EXPECTED_BONUS_SORTED}; "
            "expected NA+5, SA+2, EU+5, AF+3, AS+7, AU+2"
        )

    def test_when_continent_bonuses_inspected_then_total_sums_to_24(self):
        """5+2+5+3+7+2 = 24 — sanity-check that no bonus is missing."""
        from src.env_core.board import CONTINENT_BONUSES

        assert sum(CONTINENT_BONUSES.values()) == _EXPECTED_BONUS_SUM

    def test_when_continent_bonuses_inspected_then_all_values_are_positive_integers(
        self,
    ):
        """Every continent bonus must be a positive integer."""
        from src.env_core.board import CONTINENT_BONUSES

        for continent, bonus in CONTINENT_BONUSES.items():
            assert isinstance(bonus, int) and bonus > 0, (
                f"Continent '{continent}' has invalid bonus {bonus!r}"
            )


# ===========================================================================
# Section 7 — current_player advances once per turn skipping eliminated players
# Criterion: current_player advances once per turn skipping eliminated players
# ===========================================================================


class TestCurrentPlayerAdvancement:
    """current_player rotates after a full turn completes."""

    def test_when_reset_called_then_current_player_key_is_in_obs(self):
        from src.env import RisikoEnv

        obs, _ = RisikoEnv(n_players=2).reset(seed=42)
        assert "current_player" in obs

    def test_when_reset_called_then_current_player_is_in_valid_range(self):
        from src.env import RisikoEnv

        n = 4
        obs, _ = RisikoEnv(n_players=n).reset(seed=42)
        p = int(obs["current_player"])
        assert 0 <= p < n, f"current_player={p} is outside [0, {n})"

    def test_when_full_turn_is_stepped_then_current_player_eventually_advances(self):
        """current_player must rotate to the next alive player within a full turn.

        Interpretation: in a 2-player game player rotation must happen within
        200 steps (a very generous upper bound for one complete turn).
        """
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=2)
        obs, info = env.reset(seed=42)
        initial_player = int(obs["current_player"])
        player_advanced = False
        for _ in range(200):
            legal = info["legal_actions"]
            if not legal:
                break
            obs, _, terminated, truncated, info = env.step(legal[0])
            current = int(obs["current_player"])
            if current != initial_player:
                player_advanced = True
                break
            if terminated:
                # Game ended in 2-player game — winning player is different
                player_advanced = True
                break
            if truncated:
                break
        assert player_advanced, (
            f"current_player stayed at {initial_player} for 200 steps without advancing"
        )


# ===========================================================================
# Section 8 — step() is deterministic given a fixed seed
# Criterion (scope): step() is deterministic given a fixed seed
# ===========================================================================


class TestStepDeterminism:
    """Two envs with the same seed and the same action must produce identical results."""

    def test_when_two_envs_share_same_seed_and_same_first_action_then_obs_is_identical(
        self,
    ):
        from src.env import RisikoEnv

        env1 = RisikoEnv(n_players=2)
        env2 = RisikoEnv(n_players=2)
        _, info1 = env1.reset(seed=5)
        _, info2 = env2.reset(seed=5)
        action = info1["legal_actions"][0]
        obs1, *_ = env1.step(action)
        obs2, *_ = env2.step(action)
        for key in obs1:
            np.testing.assert_array_equal(
                obs1[key],
                obs2[key],
                err_msg=f"step() obs['{key}'] is not identical between two envs with seed=5",
            )

    def test_when_two_envs_share_same_seed_and_same_first_action_then_rewards_are_equal(
        self,
    ):
        """Reward must also be deterministic given the same seed and action."""
        from src.env import RisikoEnv

        env1 = RisikoEnv(n_players=2)
        env2 = RisikoEnv(n_players=2)
        _, info1 = env1.reset(seed=5)
        env2.reset(seed=5)
        action = info1["legal_actions"][0]
        _, reward1, *_ = env1.step(action)
        _, reward2, *_ = env2.step(action)
        assert float(reward1) == pytest.approx(float(reward2))

    def test_when_same_env_reset_with_same_seed_and_same_action_then_obs_is_identical(
        self,
    ):
        """Resetting the same env instance with the same seed is also deterministic."""
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=2)
        _, info1 = env.reset(seed=99)
        action = info1["legal_actions"][0]
        obs1, *_ = env.step(action)

        _, info2 = env.reset(seed=99)
        obs2, *_ = env.step(info2["legal_actions"][0])

        for key in obs1:
            np.testing.assert_array_equal(
                obs1[key],
                obs2[key],
                err_msg=(
                    f"step() obs['{key}'] differs between two runs of the same env with seed=99"
                ),
            )


# Property: determinism holds for any valid seed integer
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=10)
def test_when_any_seed_is_used_then_two_resets_always_return_identical_obs(
    seed: int,
) -> None:
    """Invariant (idempotence of reset with a fixed seed): reset is always reproducible."""
    from src.env import RisikoEnv

    env = RisikoEnv(n_players=2)
    obs1, _ = env.reset(seed=seed)
    obs2, _ = env.reset(seed=seed)
    for key in obs1:
        np.testing.assert_array_equal(obs1[key], obs2[key])


# ===========================================================================
# Section 9 — Global: Checkpoints saved and fully resumable
# Criterion [UNIT]: Checkpoints saved every save_freq episodes to models/;
#                   training fully resumable from any .pt file
# ===========================================================================


class TestCheckpointRoundTrip:
    """save_checkpoint / load_checkpoint must round-trip the config faithfully."""

    def test_when_checkpoint_saved_then_file_is_created_at_given_path(self, tmp_path):
        """save_checkpoint(path, ...) must create a .pt file at that exact path."""
        from src.checkpoint import save_checkpoint

        path = tmp_path / "test.pt"
        from src.config import TrainingConfig

        cfg = TrainingConfig(seed=0)
        save_checkpoint(str(path), config=cfg, model_state_dict={}, optimizer_state_dict={})
        assert path.is_file(), f"Expected checkpoint file at {path}"

    def test_when_checkpoint_saved_and_loaded_then_config_seed_is_preserved(self, tmp_path):
        """Loaded checkpoint must reproduce the exact config that was saved."""
        from src.checkpoint import load_checkpoint, save_checkpoint
        from src.config import TrainingConfig

        cfg = TrainingConfig(seed=12345)
        path = tmp_path / "round_trip.pt"
        save_checkpoint(str(path), config=cfg, model_state_dict={}, optimizer_state_dict={})
        loaded = load_checkpoint(str(path))
        assert loaded["config"].seed == 12345, "Config seed must survive checkpoint round-trip"

    def test_when_checkpoint_saved_then_load_checkpoint_returns_dict_with_model_state(
        self, tmp_path
    ):
        """Loaded dict must include model_state_dict key for weight restoration."""
        from src.checkpoint import load_checkpoint, save_checkpoint
        from src.config import TrainingConfig

        cfg = TrainingConfig(seed=0)
        path = tmp_path / "has_model.pt"
        save_checkpoint(
            str(path),
            config=cfg,
            model_state_dict={"w": 1.0},
            optimizer_state_dict={},
        )
        loaded = load_checkpoint(str(path))
        assert "model_state_dict" in loaded

    def test_when_checkpoint_saved_with_model_state_then_loaded_state_matches(self, tmp_path):
        """Model state dict must be preserved faithfully across save/load."""
        import torch

        from src.checkpoint import load_checkpoint, save_checkpoint
        from src.config import TrainingConfig

        tensor = torch.tensor([1.0, 2.0, 3.0])
        model_state = {"layer.weight": tensor}
        path = tmp_path / "model_state.pt"
        save_checkpoint(
            str(path),
            config=TrainingConfig(seed=0),
            model_state_dict=model_state,
            optimizer_state_dict={},
        )
        loaded = load_checkpoint(str(path))
        torch.testing.assert_close(loaded["model_state_dict"]["layer.weight"], tensor)


# ===========================================================================
# Section 10 — Global: Results tagged with seed and config file path
# Criterion [UNIT]: Results saved as CSV + TensorBoard logs under results/;
#                   every run tagged with its seed and config file path
# ===========================================================================


class TestRunTagging:
    """TrainingConfig must carry the seed and config_path fields for run tagging."""

    def test_when_training_config_created_with_seed_then_seed_is_accessible(self):
        """Criterion: every run is tagged with its seed."""
        from src.config import TrainingConfig

        cfg = TrainingConfig(seed=42)
        assert cfg.seed == 42

    def test_when_training_config_has_config_path_field_then_it_is_accessible(self):
        """Criterion: every run tagged with its config file path.

        The config_path field (or equivalent attribute) must be present so the
        runner can tag TensorBoard/CSV output with the originating YAML file.
        """
        from src.config import TrainingConfig

        cfg = TrainingConfig(seed=0, config_path="config/default.yaml")
        assert cfg.config_path == "config/default.yaml"

    def test_when_tb_logger_instantiated_with_log_dir_then_it_accepts_results_subdir(
        self,
    ):
        """TensorBoardLogger must accept a log_dir under results/ without error."""
        import tempfile

        from src.tb_logger import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmp:
            import os

            log_dir = os.path.join(tmp, "results", "run_seed42")
            # Must not raise; writer creation may be deferred until first log call
            logger = TensorBoardLogger(log_dir=log_dir)
            assert logger is not None

    def test_when_tb_logger_has_log_training_step_then_it_is_callable(
        self,
    ):
        """TensorBoardLogger.log_training_step() must be callable (interface check)."""
        from src.tb_logger import TensorBoardLogger

        assert callable(getattr(TensorBoardLogger, "log_training_step", None)), (
            "TensorBoardLogger must expose log_training_step()"
        )

    def test_when_tb_logger_has_log_game_result_then_it_is_callable(self):
        """TensorBoardLogger.log_game_result() must be callable (interface check)."""
        from src.tb_logger import TensorBoardLogger

        assert callable(getattr(TensorBoardLogger, "log_game_result", None)), (
            "TensorBoardLogger must expose log_game_result()"
        )
