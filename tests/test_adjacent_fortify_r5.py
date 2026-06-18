"""Tests for issue #95: switch fortification to adjacent-only with documented
default-off connected-chain flag.

TDD (Red → Green → Refactor) — all assertions derived from the acceptance criteria.

Action format (fortify):
  action_type == 4, param_a = source, param_b = dest, param_c = count, param_d = 0
Skip action: action_type == 5.
State fields: territory_owner[t], armies[t], phase, current_player.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.env import PHASE_FORTIFY, RisikoEnv
from src.utils.constants import ADJACENCY, FORTIFY_ADJACENT_ONLY_DEFAULT

_SKIP = 5
_FORTIFY = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_chain_triple() -> tuple[int, int, int]:
    """Return (a, b, c): a adj b, b adj c, a NOT adj c."""
    for a, a_adj in ADJACENCY.items():
        for b in a_adj:
            for c in ADJACENCY[b]:
                if c != a and c not in a_adj:
                    return int(a), int(b), int(c)
    raise AssertionError("ADJACENCY contains no chain triple — map data looks wrong")


def _fill_board(env: RisikoEnv, p0: tuple[int, ...], armies: dict[int, int]) -> None:
    """Set board to PHASE_FORTIFY: player 1 owns everything (2 armies each).

    Player 0 owns territories in *p0* with army counts from *armies* (default 3).
    """
    n = len(ADJACENCY)
    for t in range(n):
        env.state.territory_owner[t] = 1
        env.state.armies[t] = 2
    for t in p0:
        env.state.territory_owner[t] = 0
        env.state.armies[t] = armies.get(t, 3)
    env.state.current_player = 0
    env.state.phase = PHASE_FORTIFY


def _drive_to_fortify(env: RisikoEnv, max_steps: int = 800) -> bool:
    """Advance env to PHASE_FORTIFY by preferring skip; fall back to first action."""
    for _ in range(max_steps):
        if env.state.phase == PHASE_FORTIFY:
            return True
        actions = env._get_legal_actions()
        if not actions:
            return False
        skip = next((a for a in actions if a["action_type"] == _SKIP), None)
        _, _, done, truncated, _ = env.step(skip or actions[0])
        if done or truncated:
            return False
    return env.state.phase == PHASE_FORTIFY


def _fortify_actions(env: RisikoEnv) -> list[dict]:
    """Legal actions in PHASE_FORTIFY that are actual fortify moves (not skip)."""
    return [a for a in env._get_legal_actions() if a["action_type"] == _FORTIFY]


# ---------------------------------------------------------------------------
# 1. FORTIFY_ADJACENT_ONLY_DEFAULT lives in constants.py and equals True
# ---------------------------------------------------------------------------


def test_when_fortify_adjacent_only_default_imported_from_constants_then_value_is_true():
    assert FORTIFY_ADJACENT_ONLY_DEFAULT is True


# ---------------------------------------------------------------------------
# 2. Backward-compatibility: all existing RisikoEnv(…) call sites unchanged
# ---------------------------------------------------------------------------


def test_when_env_constructed_without_fortify_flag_then_no_error_is_raised():
    RisikoEnv(n_players=2)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_when_env_constructed_with_n_players_only_then_no_error_is_raised(n):
    RisikoEnv(n_players=n)


def test_when_env_constructed_with_fortify_adjacent_only_true_then_no_error_is_raised():
    RisikoEnv(n_players=2, fortify_adjacent_only=True)


def test_when_env_constructed_with_fortify_adjacent_only_false_then_no_error_is_raised():
    RisikoEnv(n_players=2, fortify_adjacent_only=False)


# ---------------------------------------------------------------------------
# 3. Default mode: only adjacent own territory is offered/accepted
# ---------------------------------------------------------------------------


def test_when_default_mode_nonadjacent_chain_dest_is_absent_from_legal_actions():
    """src→far (chain-only, not adjacent) must not appear in get_legal_actions()."""
    src, mid, far = _find_chain_triple()
    env = RisikoEnv(n_players=2)
    env.reset(seed=0)
    _fill_board(env, (src, mid, far), {src: 5, mid: 3, far: 3})

    illegal = [a for a in _fortify_actions(env) if a["param_a"] == src and a["param_b"] == far]
    assert illegal == [], (
        f"Default mode: src={src}→far={far} (chain-only) must be absent from legal actions"
    )


def test_when_default_mode_adjacent_own_dest_is_present_in_legal_actions():
    """src→mid (adjacent, both owned by player 0) must appear in legal actions."""
    src, mid, _far = _find_chain_triple()
    env = RisikoEnv(n_players=2)
    env.reset(seed=0)
    _fill_board(env, (src, mid, _far), {src: 5, mid: 3, _far: 3})

    legal = [a for a in _fortify_actions(env) if a["param_a"] == src and a["param_b"] == mid]
    assert legal, f"Default mode: src={src}→mid={mid} (adjacent, own) must appear in legal actions"


def test_when_default_mode_nonadjacent_chain_fortify_step_returns_negative_reward():
    """step() with a chain-only fortify in default mode returns the invalid-action penalty."""
    src, mid, far = _find_chain_triple()

    # Obtain a valid src→far action from chain mode
    env_chain = RisikoEnv(n_players=2, fortify_adjacent_only=False)
    env_chain.reset(seed=0)
    _fill_board(env_chain, (src, mid, far), {src: 5, mid: 3, far: 3})
    chain = [a for a in _fortify_actions(env_chain) if a["param_a"] == src and a["param_b"] == far]
    if not chain:
        pytest.skip(f"Chain mode gave no src={src}→far={far} action; board setup may vary")

    env_default = RisikoEnv(n_players=2)
    env_default.reset(seed=0)
    _fill_board(env_default, (src, mid, far), {src: 5, mid: 3, far: 3})
    _, reward, _, _, _ = env_default.step(chain[0])
    assert reward < 0, "Chain-only fortify in default mode must return negative reward"


# ---------------------------------------------------------------------------
# 4. Chain flag disabled: fortify_adjacent_only=False
# ---------------------------------------------------------------------------


def test_when_chain_flag_disabled_connected_nonadjacent_dest_is_present_in_legal_actions():
    """With fortify_adjacent_only=False, src→far (connected via mid) is legal."""
    src, mid, far = _find_chain_triple()
    env = RisikoEnv(n_players=2, fortify_adjacent_only=False)
    env.reset(seed=0)
    _fill_board(env, (src, mid, far), {src: 5, mid: 3, far: 3})

    legal = [a for a in _fortify_actions(env) if a["param_a"] == src and a["param_b"] == far]
    assert legal, f"Chain mode: src={src}→far={far} (connected via {mid}) must be legal"


# ---------------------------------------------------------------------------
# 5. Existing fortify invariants preserved
# ---------------------------------------------------------------------------


def test_when_fortify_source_equals_dest_then_not_in_legal_actions():
    src, mid, far = _find_chain_triple()
    env = RisikoEnv(n_players=2)
    env.reset(seed=0)
    _fill_board(env, (src, mid, far), {src: 5, mid: 3, far: 3})

    self_moves = [a for a in _fortify_actions(env) if a["param_a"] == a["param_b"]]
    assert self_moves == [], "source == dest must never be a legal fortify"


def test_when_fortify_dest_is_enemy_territory_then_not_in_legal_actions():
    src, mid, far = _find_chain_triple()
    env = RisikoEnv(n_players=2)
    env.reset(seed=0)
    _fill_board(env, (src, mid, far), {src: 5, mid: 3, far: 3})

    player = env.state.current_player
    enemy = [a for a in _fortify_actions(env) if env.state.territory_owner[a["param_b"]] != player]
    assert enemy == [], "Fortify to an enemy territory must never be legal"


def test_when_fortify_count_equals_all_armies_then_not_in_legal_actions():
    """Count >= armies[source] must never appear (must leave ≥ 1 army behind)."""
    src, mid, far = _find_chain_triple()
    env = RisikoEnv(n_players=2)
    env.reset(seed=0)
    _fill_board(env, (src, mid, far), {src: 5, mid: 3, far: 3})

    overmoves = [a for a in _fortify_actions(env) if a["param_c"] >= env.state.armies[a["param_a"]]]
    assert overmoves == [], "count >= armies[source] must never be legal"


def test_when_fortify_count_is_zero_then_not_in_legal_actions():
    src, mid, far = _find_chain_triple()
    env = RisikoEnv(n_players=2)
    env.reset(seed=0)
    _fill_board(env, (src, mid, far), {src: 5, mid: 3, far: 3})

    zero = [a for a in _fortify_actions(env) if a["param_c"] <= 0]
    assert zero == [], "count <= 0 must never be a legal fortify"


def test_when_valid_fortify_performed_then_turn_ends():
    """After a valid adjacent fortify the env leaves PHASE_FORTIFY."""
    src, mid, far = _find_chain_triple()
    env = RisikoEnv(n_players=2)
    env.reset(seed=0)
    _fill_board(env, (src, mid, far), {src: 5, mid: 3, far: 3})

    moves = [a for a in _fortify_actions(env) if a["param_a"] == src and a["param_b"] == mid]
    if not moves:
        pytest.skip("No src→mid move available; cannot test turn-advance")

    env.step(moves[0])
    assert env.state.phase != PHASE_FORTIFY, "After a valid fortify the turn must end"


def test_when_default_mode_and_dest_has_one_army_then_fortify_is_valid():
    """Boundary: fortifying INTO a 1-army adjacent own territory is valid.

    The armies≥2 constraint applies only to the source (must keep ≥1 behind),
    not to the destination.
    """
    src, mid, _far = _find_chain_triple()
    env = RisikoEnv(n_players=2)
    env.reset(seed=0)
    # src has 3 armies, mid has exactly 1 army — both owned by player 0
    _fill_board(env, (src, mid, _far), {src: 3, mid: 1, _far: 2})

    # src→mid is adjacent; mid has only 1 army but that's fine as a destination
    legal = [a for a in _fortify_actions(env) if a["param_a"] == src and a["param_b"] == mid]
    assert legal, f"Fortify src={src}→mid={mid} (mid has 1 army) must be legal in default mode"


# ---------------------------------------------------------------------------
# 6. Property: every fortify legal action satisfies dest ∈ ADJACENCY[source]
# ---------------------------------------------------------------------------


@given(st.integers(min_value=0, max_value=99))
@settings(max_examples=25)
def test_given_any_seed_default_mode_all_fortify_actions_are_adjacent(seed: int) -> None:
    """Invariant: in default mode every fortify legal action has dest ∈ ADJACENCY[source]."""
    env = RisikoEnv(n_players=2)
    env.reset(seed=seed)
    if not _drive_to_fortify(env):
        return  # game ended before PHASE_FORTIFY for this seed

    for action in _fortify_actions(env):
        src = action["param_a"]
        dst = action["param_b"]
        assert dst in ADJACENCY[src], (
            f"seed={seed}: dest={dst} not in ADJACENCY[{src}]={set(ADJACENCY[src])}"
        )
