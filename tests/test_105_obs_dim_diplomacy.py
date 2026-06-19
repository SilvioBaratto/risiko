"""
Issue #105 — obs_dim invariant and diplomacy field exclusion.

Criterion: "A fast test asserts get_obs_dim() == 137 and that RisikoEnv observation
keys are unchanged when diplomacy is active (no diplomacy fields ever enter get_obs)"
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

_DIPLOMACY_FIELDS = frozenset(
    {
        "alliances",
        "alliance_state",
        "grudges",
        "grudge_state",
        "messages",
        "message_log",
        "trust",
        "reputation",
        "diplomacy",
        "negotiation",
        "social",
        "attack_priority",
        "propose_alliance_with",
        "accept_alliance_with",
        "declare_war_on",
    }
)


def test_when_get_obs_dim_is_called_then_137_is_returned():
    """get_obs_dim() must always return 137 — the fixed mechanical obs dimension."""
    from src.models.utils import get_obs_dim

    assert get_obs_dim() == 137


def test_when_env_resets_then_observation_keys_exclude_diplomacy_fields():
    """Observation returned by RisikoEnv.reset() must not contain any diplomacy-layer key."""
    from src.env import RisikoEnv

    env = RisikoEnv(n_players=2)
    obs, _ = env.reset(seed=42)

    overlap = set(obs.keys()) & _DIPLOMACY_FIELDS
    assert not overlap, (
        f"Diplomacy fields must never appear in the Gym observation; found: {overlap}"
    )


def test_when_env_steps_then_observation_keys_exclude_diplomacy_fields():
    """Observation returned after env.step() must also never contain diplomacy-layer keys."""
    from src.env import RisikoEnv

    env = RisikoEnv(n_players=2)
    obs, _ = env.reset(seed=7)

    legal = env.get_legal_actions()
    assert legal, "get_legal_actions() must not be empty on a fresh env"

    obs_step, _, _, _, _ = env.step(legal[0])

    overlap = set(obs_step.keys()) & _DIPLOMACY_FIELDS
    assert not overlap, f"Diplomacy fields found in obs after step: {overlap}"


def test_when_obs_is_flattened_then_dimension_equals_get_obs_dim():
    """flatten_obs() must produce a tensor whose last dimension equals get_obs_dim()."""
    from src.env import RisikoEnv
    from src.models.utils import flatten_obs, get_obs_dim, stack_obs

    env = RisikoEnv(n_players=2)
    obs, _ = env.reset(seed=0)
    batched = stack_obs([obs], device="cpu")
    flat = flatten_obs(batched)

    assert flat.shape[-1] == get_obs_dim(), (
        f"flatten_obs produced shape {flat.shape}; expected last dim == {get_obs_dim()}"
    )


@given(seed=st.integers(min_value=0, max_value=2**16 - 1))
@settings(max_examples=20)
def test_when_env_resets_with_any_seed_then_obs_keys_exclude_diplomacy_fields(
    seed: int,
) -> None:
    """Invariant: for any seed, the Gym observation produced at reset never contains
    diplomacy-state keys. Diplomacy is always kept in the LLM/social layer.
    """
    from src.env import RisikoEnv

    env = RisikoEnv(n_players=2)
    obs, _ = env.reset(seed=seed)

    overlap = set(obs.keys()) & _DIPLOMACY_FIELDS
    assert not overlap, f"seed={seed}: diplomacy fields must never appear in obs; found: {overlap}"
