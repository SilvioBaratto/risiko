"""Source-blind example tests for env_core/reward.py (issue #53).

Authored strictly from the acceptance criteria; no implementation file was
read.  Tests start in the Red phase and will pass only when the criteria
are genuinely satisfied.

Skipped (oracle: NOT VERIFIABLE at unit level):
  - LLM non-blocking / timeout fallback
  - env-step determinism via fixed seed
  - YAML/CLI config coverage
  - Baseline win-rate benchmarks
  - Ablation (dense coeff = 0.0 zeroes component) — isolated
      by zeroing coefficients inside other tests, not as a separate criterion
  - SOLID / clean-code prose
  - All-tests-pass gate
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from hypothesis import given
from hypothesis import strategies as st

# ── Modules under test ────────────────────────────────────────────────────────
# ImportError here is the expected Red state until the module is implemented.
from src.env_core.reward import compute_reward, snapshot
from src.utils.reward_config import RewardConfig

# ── Continent constants ───────────────────────────────────────────────────────
# Try to import the canonical board mapping.  Fall back to the well-known
# standard Risk values so the continent-bonus test can express intent even
# before the board module exists (it will still be Red via ImportError above).
try:
    from src.env_core.board import CONTINENT_BONUSES, CONTINENT_TERRITORIES
except ImportError:  # pragma: no cover – Red until board module is implemented
    CONTINENT_TERRITORIES: dict[str, list[int]] = {
        "Australia": [38, 39, 40, 41],
    }
    CONTINENT_BONUSES: dict[str, int] = {
        "Australia": 2,
    }

_AUSTRALIA_IDS: list[int] = CONTINENT_TERRITORIES["Australia"]
_AUSTRALIA_BONUS: int = CONTINENT_BONUSES["Australia"]


# ── Minimal duck-typed state double ──────────────────────────────────────────
# The reward module must accept any object that exposes the three named
# attributes; it must not require a specific concrete class.


@dataclass
class _State:
    """Minimal stand-in for the game state consumed by snapshot() and compute_reward()."""

    territory_owner: dict[int, int] = field(default_factory=dict)
    armies: dict[int, int] = field(default_factory=dict)
    eliminated: set[int] = field(default_factory=set)


def _cfg(**overrides: float) -> RewardConfig:
    """Build a RewardConfig with non-zero canonical defaults; apply overrides."""
    kwargs: dict[str, float] = {
        "sparse_loss": -1.0,
        "dense_territory_delta": 0.01,
        "dense_continent_bonus_delta": 0.05,
        "dense_army_ratio": 0.005,
        "dense_elimination_bonus": 0.1,
    }
    kwargs.update(overrides)
    return RewardConfig(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# snapshot() — independence
#   Criterion: "mutating state afterwards does not alter the snapshot"
# ─────────────────────────────────────────────────────────────────────────────


class TestSnapshotIndependence:
    """snapshot() must return fully independent copies of all three state fields."""

    def test_when_territory_owner_is_mutated_after_snapshot_then_snapshot_is_unchanged(self):
        state = _State(territory_owner={0: 0, 1: 1}, armies={0: 3, 1: 3})
        snap = snapshot(state)
        state.territory_owner[0] = 99  # mutate original
        assert snap.territory_owner[0] == 0

    def test_when_armies_are_mutated_after_snapshot_then_snapshot_is_unchanged(self):
        state = _State(territory_owner={0: 0}, armies={0: 5})
        snap = snapshot(state)
        state.armies[0] = 999  # mutate original
        assert snap.armies[0] == 5

    def test_when_eliminated_is_mutated_after_snapshot_then_snapshot_is_unchanged(self):
        state = _State(territory_owner={0: 0}, armies={0: 3}, eliminated={2})
        snap = snapshot(state)
        state.eliminated.add(3)  # mutate original
        assert 3 not in snap.eliminated

    def test_when_new_territory_is_added_after_snapshot_then_snapshot_does_not_grow(self):
        state = _State(territory_owner={0: 0}, armies={0: 3})
        snap = snapshot(state)
        state.territory_owner[1] = 1  # add a new key
        assert 1 not in snap.territory_owner


# ─────────────────────────────────────────────────────────────────────────────
# compute_reward() — sparse: player eliminated
#   Criterion: "returns cfg.sparse_loss when the player is eliminated"
# ─────────────────────────────────────────────────────────────────────────────


class TestSparseElimination:
    """compute_reward() must return sparse_loss exactly when the player is eliminated."""

    def test_when_player_is_in_eliminated_set_then_reward_is_sparse_loss(self):
        cfg = _cfg()
        prev = snapshot(
            _State(
                territory_owner={0: 0, 1: 1},
                armies={0: 3, 1: 3},
                eliminated=set(),
            )
        )
        # player 0 has been conquered and is now in eliminated
        state = _State(
            territory_owner={0: 1, 1: 1},
            armies={0: 1, 1: 6},
            eliminated={0},
        )
        reward = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg)
        assert reward == pytest.approx(cfg.sparse_loss)

    def test_when_other_player_is_eliminated_then_query_player_reward_is_not_sparse_loss(self):
        """sparse_loss applies only to the queried player, not to opponents."""
        cfg = _cfg(
            dense_territory_delta=0.0,
            dense_continent_bonus_delta=0.0,
            dense_army_ratio=0.0,
            dense_elimination_bonus=0.0,
        )
        prev = snapshot(
            _State(
                territory_owner={0: 0, 1: 1},
                armies={0: 5, 1: 5},
                eliminated=set(),
            )
        )
        # player 1 is eliminated, but we query player 0's reward
        state = _State(
            territory_owner={0: 0, 1: 0},
            armies={0: 5, 1: 0},
            eliminated={1},
        )
        reward = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg)
        assert reward != pytest.approx(cfg.sparse_loss)


# ─────────────────────────────────────────────────────────────────────────────
# compute_reward() — dense: territory delta
#   Criterion: "A net +1 territory contributes cfg.dense_territory_delta"
# ─────────────────────────────────────────────────────────────────────────────


class TestDenseTerritoryDelta:
    """Each net territory gained contributes cfg.dense_territory_delta."""

    def test_when_player_nets_one_territory_then_reward_equals_dense_territory_delta(self):
        # Zero all other dense terms to isolate the territory component.
        cfg = _cfg(
            dense_army_ratio=0.0,
            dense_continent_bonus_delta=0.0,
            dense_elimination_bonus=0.0,
        )
        # prev: player 0 owns t0; player 1 owns t1
        prev = snapshot(
            _State(
                territory_owner={0: 0, 1: 1},
                armies={0: 5, 1: 5},
                eliminated=set(),
            )
        )
        # current: player 0 conquered t1 (net +1 territory)
        state = _State(
            territory_owner={0: 0, 1: 0},
            armies={0: 4, 1: 1},
            eliminated=set(),
        )
        reward = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg)
        assert reward == pytest.approx(cfg.dense_territory_delta)

    def test_when_player_nets_two_territories_then_reward_equals_two_territory_deltas(self):
        cfg = _cfg(
            dense_army_ratio=0.0,
            dense_continent_bonus_delta=0.0,
            dense_elimination_bonus=0.0,
        )
        prev = snapshot(
            _State(
                territory_owner={0: 0, 1: 1, 2: 1},
                armies={0: 8, 1: 4, 2: 4},
                eliminated=set(),
            )
        )
        # player 0 captured t1 and t2 (net +2)
        state = _State(
            territory_owner={0: 0, 1: 0, 2: 0},
            armies={0: 6, 1: 1, 2: 1},
            eliminated=set(),
        )
        reward = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg)
        assert reward == pytest.approx(2 * cfg.dense_territory_delta)


# ─────────────────────────────────────────────────────────────────────────────
# compute_reward() — dense: continent bonus
#   Criterion: "Gaining a continent worth bonus B contributes
#               cfg.dense_continent_bonus_delta * B (e.g. Australia → ×2)"
# ─────────────────────────────────────────────────────────────────────────────


class TestDenseContinentBonus:
    """Gaining a continent worth B contributes cfg.dense_continent_bonus_delta * B."""

    def test_when_player_gains_australia_then_reward_equals_continent_delta_times_australia_bonus(
        self,
    ):
        # Isolate continent component.
        cfg = _cfg(
            dense_territory_delta=0.0,
            dense_army_ratio=0.0,
            dense_elimination_bonus=0.0,
        )
        # prev: player 1 owns all Australia territories (player 0 has none)
        prev = snapshot(
            _State(
                territory_owner=dict.fromkeys(_AUSTRALIA_IDS, 1),
                armies=dict.fromkeys(_AUSTRALIA_IDS, 3),
                eliminated=set(),
            )
        )
        # current: player 0 captured every Australia territory
        state = _State(
            territory_owner=dict.fromkeys(_AUSTRALIA_IDS, 0),
            armies=dict.fromkeys(_AUSTRALIA_IDS, 3),
            eliminated=set(),
        )
        reward = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg)
        expected = cfg.dense_continent_bonus_delta * _AUSTRALIA_BONUS
        assert reward == pytest.approx(expected), (
            f"Gaining Australia (bonus={_AUSTRALIA_BONUS}) should contribute "
            f"{expected:.4f}; got {reward:.4f}"
        )

    def test_when_player_already_held_australia_then_no_continent_bonus_is_added(self):
        """No repeated bonus — gaining a continent already held adds nothing."""
        cfg = _cfg(
            dense_territory_delta=0.0,
            dense_army_ratio=0.0,
            dense_elimination_bonus=0.0,
        )
        # Both prev and current: player 0 fully controls Australia
        full_control = _State(
            territory_owner=dict.fromkeys(_AUSTRALIA_IDS, 0),
            armies=dict.fromkeys(_AUSTRALIA_IDS, 3),
            eliminated=set(),
        )
        prev = snapshot(full_control)
        state = _State(
            territory_owner=dict.fromkeys(_AUSTRALIA_IDS, 0),
            armies=dict.fromkeys(_AUSTRALIA_IDS, 3),
            eliminated=set(),
        )
        reward = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg)
        assert reward == pytest.approx(0.0), (
            "No new continent was gained; continent bonus should be 0.0"
        )


# ─────────────────────────────────────────────────────────────────────────────
# compute_reward() — dense: army ratio
#   Criterion: "Army-ratio term = cfg.dense_army_ratio * (player_armies / total_armies)"
# ─────────────────────────────────────────────────────────────────────────────


class TestDenseArmyRatio:
    """Army-ratio term equals cfg.dense_army_ratio * (player_armies / total_armies)."""

    def test_when_player_holds_one_fifth_of_all_armies_then_army_ratio_term_is_correct(self):
        # Isolate the army-ratio term.
        cfg = _cfg(
            dense_territory_delta=0.0,
            dense_continent_bonus_delta=0.0,
            dense_elimination_bonus=0.0,
        )
        # player 0: 20 armies across t0, t1
        # player 1: 80 armies across t2, t3, t4
        # total: 100 → ratio = 0.20
        state = _State(
            territory_owner={0: 0, 1: 0, 2: 1, 3: 1, 4: 1},
            armies={0: 10, 1: 10, 2: 20, 3: 30, 4: 30},
            eliminated=set(),
        )
        # Identical prev → zero territory-delta and zero continent-delta
        prev = snapshot(state)
        reward = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg)
        expected = cfg.dense_army_ratio * (20 / 100)
        assert reward == pytest.approx(expected)

    def test_when_player_is_sole_survivor_with_all_armies_then_army_ratio_term_is_full_coefficient(
        self,
    ):
        cfg = _cfg(
            dense_territory_delta=0.0,
            dense_continent_bonus_delta=0.0,
            dense_elimination_bonus=0.0,
        )
        # Player 0 controls everything; ratio = 1.0
        state = _State(
            territory_owner={0: 0, 1: 0},
            armies={0: 50, 1: 50},
            eliminated={1},
        )
        prev = snapshot(state)
        reward = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg)
        assert reward == pytest.approx(cfg.dense_army_ratio * 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# compute_reward() — dense: elimination bonus
#   Criterion: "Each opponent newly eliminated this step adds cfg.dense_elimination_bonus"
# ─────────────────────────────────────────────────────────────────────────────


class TestDenseEliminationBonus:
    """Each newly-eliminated opponent adds cfg.dense_elimination_bonus."""

    def test_when_one_opponent_is_newly_eliminated_then_reward_equals_elimination_bonus(self):
        cfg = _cfg(
            dense_territory_delta=0.0,
            dense_continent_bonus_delta=0.0,
            dense_army_ratio=0.0,
        )
        # prev: all players alive
        prev = snapshot(
            _State(
                territory_owner={0: 0, 1: 1},
                armies={0: 5, 1: 5},
                eliminated=set(),
            )
        )
        # current: player 1 newly eliminated
        state = _State(
            territory_owner={0: 0, 1: 0},
            armies={0: 5, 1: 0},
            eliminated={1},
        )
        reward = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg)
        assert reward == pytest.approx(cfg.dense_elimination_bonus)

    def test_when_two_opponents_are_newly_eliminated_then_reward_equals_two_elimination_bonuses(
        self,
    ):
        cfg = _cfg(
            dense_territory_delta=0.0,
            dense_continent_bonus_delta=0.0,
            dense_army_ratio=0.0,
        )
        prev = snapshot(
            _State(
                territory_owner={0: 0, 1: 1, 2: 2},
                armies={0: 10, 1: 5, 2: 5},
                eliminated=set(),
            )
        )
        state = _State(
            territory_owner={0: 0, 1: 0, 2: 0},
            armies={0: 10, 1: 0, 2: 0},
            eliminated={1, 2},
        )
        reward = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg)
        assert reward == pytest.approx(2 * cfg.dense_elimination_bonus)

    def test_when_opponent_was_already_eliminated_before_step_then_no_bonus_is_added(self):
        """Only newly-eliminated opponents count; pre-existing ones must not double-count."""
        cfg = _cfg(
            dense_territory_delta=0.0,
            dense_continent_bonus_delta=0.0,
            dense_army_ratio=0.0,
        )
        # player 2 was already eliminated before this step
        prev = snapshot(
            _State(
                territory_owner={0: 0, 1: 1},
                armies={0: 5, 1: 5},
                eliminated={2},
            )
        )
        # No new eliminations this step
        state = _State(
            territory_owner={0: 0, 1: 1},
            armies={0: 5, 1: 5},
            eliminated={2},
        )
        reward = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg)
        assert reward == pytest.approx(0.0), (
            "A pre-existing elimination must not contribute a bonus again"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Property-based: snapshot independence for any valid board position
#   Invariant (round-trip / copy independence): snapshot(s) is always a
#   full independent copy — no aliasing of territory_owner, armies, or
#   eliminated.
# ─────────────────────────────────────────────────────────────────────────────


@given(
    territory_map=st.dictionaries(
        st.integers(min_value=0, max_value=41),
        st.integers(min_value=0, max_value=5),
        min_size=1,
    ),
    armies_map=st.dictionaries(
        st.integers(min_value=0, max_value=41),
        st.integers(min_value=1, max_value=100),
        min_size=1,
    ),
    elim=st.frozensets(st.integers(min_value=0, max_value=5)),
)
def test_when_any_valid_state_is_snapshotted_then_aggressive_mutation_does_not_alter_snapshot(
    territory_map: dict[int, int],
    armies_map: dict[int, int],
    elim: frozenset[int],
) -> None:
    """Snapshot collections are independent of the source for any valid board position."""
    state = _State(
        territory_owner=dict(territory_map),
        armies=dict(armies_map),
        eliminated=set(elim),
    )
    snap = snapshot(state)

    # Aggressively mutate all three collections in-place
    state.territory_owner.clear()
    state.armies.clear()
    state.eliminated.clear()

    assert snap.territory_owner == territory_map, (
        "territory_owner in snapshot was aliased to the original"
    )
    assert snap.armies == armies_map, "armies in snapshot was aliased to the original"
    assert snap.eliminated == set(elim), "eliminated in snapshot was aliased to the original"
