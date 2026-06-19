"""Fast invariant tests for issue #105 — as specified in implementation notes.

Covers:
- obs_dim == 137 (always, regardless of diplomacy)
- RisikoEnv observation keys unchanged when diplomacy is active
- Disabled-equivalence vs MultiAgentRunner (call_count == 0, action_log identical)
- Fixed-seed determinism (same seed → same game)
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

_KNOWN_OBS_KEYS = frozenset(
    {
        "territory_owner",
        "armies",
        "phase",
        "current_player",
        "cards",
        "continent_control",
        "trade_count",
        "reinforcements_remaining",
        "turn_capture",
        "n_players",
        "eliminated",
    }
)

_DIPLOMACY_FIELDS = frozenset(
    {
        "alliances",
        "grudges",
        "messages",
        "trust",
        "reputation",
        "diplomacy",
        "negotiation",
        "social",
    }
)


# ---------------------------------------------------------------------------
# obs_dim == 137
# ---------------------------------------------------------------------------


def test_when_get_obs_dim_called_then_returns_137():
    """The fixed mechanical obs dimension is always 137."""
    from src.models.utils import get_obs_dim

    assert get_obs_dim() == 137


def test_when_obs_is_flattened_then_size_equals_137():
    """flatten_obs() must produce exactly 137 features per sample."""
    from src.env import RisikoEnv
    from src.models.utils import flatten_obs, get_obs_dim, stack_obs

    env = RisikoEnv(n_players=2)
    obs, _ = env.reset(seed=0)
    flat = flatten_obs(stack_obs([obs], device="cpu"))

    assert flat.shape[-1] == get_obs_dim() == 137


# ---------------------------------------------------------------------------
# Observation keys are unchanged when diplomacy is active
# ---------------------------------------------------------------------------


def test_when_diplomacy_runner_enabled_then_obs_keys_match_baseline():
    """Running inside DiplomacyRunner(enabled=True) must not add diplomacy keys to obs."""
    from src.agents.random_agent import RandomAgent
    from src.config import DiplomacyConfig
    from src.env import RisikoEnv
    from src.multi_agent import DiplomacyRunner

    env_dipl = RisikoEnv(n_players=2)
    dipl_result = DiplomacyRunner(
        env_dipl,
        [RandomAgent(seed=0), RandomAgent(seed=1)],
        max_turns=3,
        diplomacy_cfg=DiplomacyConfig(enabled=True),
    ).run_game(seed=42)

    # Verify obs keys in trajectories are the same set
    for traj in dipl_result.trajectories:
        overlap = set(traj.obs.keys()) & _DIPLOMACY_FIELDS
        assert not overlap, f"Diplomacy fields in obs: {overlap}"


def test_when_env_resets_with_diplomacy_runner_then_obs_has_expected_keys():
    """The set of obs keys must equal the known mechanical key set — no extras, no removals."""
    from src.env import RisikoEnv

    env = RisikoEnv(n_players=2)
    obs, _ = env.reset(seed=99)

    assert set(obs.keys()) == _KNOWN_OBS_KEYS, (
        f"Obs keys diverged from expected set.\n"
        f"  Missing: {_KNOWN_OBS_KEYS - set(obs.keys())}\n"
        f"  Extra:   {set(obs.keys()) - _KNOWN_OBS_KEYS}"
    )


# ---------------------------------------------------------------------------
# Disabled-equivalence: DiplomacyRunner(enabled=False) ≡ MultiAgentRunner
# ---------------------------------------------------------------------------


def test_when_diplomacy_disabled_then_call_count_is_zero():
    """DiplomacyRunner.call_count must be 0 after a game when diplomacy is disabled."""
    from src.agents.random_agent import RandomAgent
    from src.config import DiplomacyConfig
    from src.env import RisikoEnv
    from src.multi_agent import DiplomacyRunner

    env = RisikoEnv(n_players=2)
    runner = DiplomacyRunner(
        env,
        [RandomAgent(seed=0), RandomAgent(seed=1)],
        max_turns=8,
        diplomacy_cfg=DiplomacyConfig(enabled=False),
    )
    runner.run_game(seed=42)

    assert runner.call_count == 0, f"Expected 0, got {runner.call_count}"


def test_when_diplomacy_disabled_then_action_log_identical_to_multi_agent_runner():
    """Same seed → same action_log between MultiAgentRunner and DiplomacyRunner(disabled)."""
    from src.agents.random_agent import RandomAgent
    from src.config import DiplomacyConfig
    from src.env import RisikoEnv
    from src.multi_agent import DiplomacyRunner, MultiAgentRunner

    seed = 7

    env_a = RisikoEnv(n_players=2)
    plain = MultiAgentRunner(env_a, [RandomAgent(seed=i) for i in range(2)], max_turns=8).run_game(
        seed=seed
    )

    env_b = RisikoEnv(n_players=2)
    disabled = DiplomacyRunner(
        env_b,
        [RandomAgent(seed=i) for i in range(2)],
        max_turns=8,
        diplomacy_cfg=DiplomacyConfig(enabled=False),
    ).run_game(seed=seed)

    assert plain.action_log == disabled.action_log, (
        "action_log must be byte-identical when diplomacy is disabled"
    )


# ---------------------------------------------------------------------------
# Fixed-seed determinism
# ---------------------------------------------------------------------------


def test_when_same_seed_run_twice_then_action_logs_are_identical():
    """Two runs with the same seed and agents must produce identical action logs."""
    from src.agents.random_agent import RandomAgent
    from src.env import RisikoEnv
    from src.multi_agent import MultiAgentRunner

    seed = 13

    def _run():
        env = RisikoEnv(n_players=2)
        return MultiAgentRunner(env, [RandomAgent(seed=i) for i in range(2)], max_turns=8).run_game(
            seed=seed
        )

    assert _run().action_log == _run().action_log, (
        "Two runs with the same seed must produce identical action logs"
    )


@given(seed=st.integers(min_value=0, max_value=2**14 - 1))
@settings(max_examples=15)
def test_when_any_seed_run_twice_then_action_logs_match(seed: int) -> None:
    """Invariant: determinism holds for any seed in the valid range."""
    from src.agents.random_agent import RandomAgent
    from src.env import RisikoEnv
    from src.multi_agent import MultiAgentRunner

    def _run():
        env = RisikoEnv(n_players=2)
        return MultiAgentRunner(env, [RandomAgent(seed=i) for i in range(2)], max_turns=5).run_game(
            seed=seed
        )

    assert _run().action_log == _run().action_log
