"""Reward-ablation tests for issue #62.

Verifies that each dense reward coefficient is independently ablatable via
RewardConfig, and that a sparse-only config (all dense coefs = 0) yields
per-step rewards strictly in {-1, 0, +1} across scripted episodes.

Reuses the scripted-game helpers pattern from test_env_acceptance.py (issue #60).
"""

from __future__ import annotations

import pytest

from src.utils.reward_config import RewardConfig

# ---------------------------------------------------------------------------
# Helpers mirroring test_env_acceptance scripted-game pattern
# ---------------------------------------------------------------------------


def _make_env(n_players: int = 2, reward_config: RewardConfig | None = None):
    from src.env import RisikoEnv

    kwargs = {"n_players": n_players}
    if reward_config is not None:
        kwargs["reward_config"] = reward_config
    return RisikoEnv(**kwargs)


def _run_scripted_episode(
    seed: int = 0,
    n_players: int = 2,
    max_steps: int = 300,
    reward_config: RewardConfig | None = None,
) -> list[float]:
    """Run one seeded episode, always picking legal[0], return all rewards."""
    env = _make_env(n_players=n_players, reward_config=reward_config)
    _, info = env.reset(seed=seed)
    rewards: list[float] = []
    for _ in range(max_steps):
        legal = info.get("legal_actions", [])
        if not legal:
            break
        _, reward, done, trunc, info = env.step(legal[0])
        rewards.append(float(reward))
        if done or trunc:
            break
    return rewards


# ---------------------------------------------------------------------------
# Individual coefficient ablation
# ---------------------------------------------------------------------------


class TestDenseCoefficientAblation:
    """Each dense coefficient can be zeroed without affecting the others."""

    def test_dense_territory_delta_default_is_nonzero(self) -> None:
        assert RewardConfig().dense_territory_delta != 0.0

    def test_dense_territory_delta_ablated_to_zero(self) -> None:
        cfg = RewardConfig(dense_territory_delta=0.0)
        assert cfg.dense_territory_delta == 0.0

    def test_dense_continent_bonus_delta_default_is_nonzero(self) -> None:
        assert RewardConfig().dense_continent_bonus_delta != 0.0

    def test_dense_continent_bonus_delta_ablated_to_zero(self) -> None:
        cfg = RewardConfig(dense_continent_bonus_delta=0.0)
        assert cfg.dense_continent_bonus_delta == 0.0

    def test_dense_army_ratio_default_is_nonzero(self) -> None:
        assert RewardConfig().dense_army_ratio != 0.0

    def test_dense_army_ratio_ablated_to_zero(self) -> None:
        cfg = RewardConfig(dense_army_ratio=0.0)
        assert cfg.dense_army_ratio == 0.0

    def test_dense_elimination_bonus_default_is_nonzero(self) -> None:
        assert RewardConfig().dense_elimination_bonus != 0.0

    def test_dense_elimination_bonus_ablated_to_zero(self) -> None:
        cfg = RewardConfig(dense_elimination_bonus=0.0)
        assert cfg.dense_elimination_bonus == 0.0

    def test_zeroing_one_coef_does_not_affect_others(self) -> None:
        """Ablating territory_delta must not change continent_bonus or army_ratio."""
        default = RewardConfig()
        ablated = RewardConfig(dense_territory_delta=0.0)
        assert ablated.dense_continent_bonus_delta == default.dense_continent_bonus_delta
        assert ablated.dense_army_ratio == default.dense_army_ratio
        assert ablated.dense_elimination_bonus == default.dense_elimination_bonus

    def test_sparse_only_config_all_dense_zero(self) -> None:
        cfg = RewardConfig(
            dense_territory_delta=0.0,
            dense_continent_bonus_delta=0.0,
            dense_army_ratio=0.0,
            dense_elimination_bonus=0.0,
        )
        assert cfg.dense_territory_delta == 0.0
        assert cfg.dense_continent_bonus_delta == 0.0
        assert cfg.dense_army_ratio == 0.0
        assert cfg.dense_elimination_bonus == 0.0
        assert cfg.sparse_win == pytest.approx(1.0)
        assert cfg.sparse_loss == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Dense reward contribution tests (via compute_reward)
# ---------------------------------------------------------------------------


class TestComputeRewardDenseComponents:
    """compute_reward sums dense components controlled by their coefficients."""

    def _make_snapshot(self, n_territories: int = 10, player: int = 0):
        from src.env_core.reward import Snapshot

        return Snapshot(
            territory_owner=dict.fromkeys(range(n_territories), player),
            armies=dict.fromkeys(range(42), 1),
            eliminated=set(),
        )

    def _make_state(self, n_territories: int = 10, player: int = 0):
        import numpy as np

        class FakeState:
            territory_owner = np.array(
                [player if i < n_territories else 1 for i in range(42)], dtype=np.int32
            )
            armies = np.ones(42, dtype=np.int32)
            eliminated = np.zeros(6, dtype=np.int32)

        return FakeState()

    def test_when_territory_delta_zero_then_territory_component_absent(self) -> None:
        from src.env_core.reward import compute_reward

        cfg_with = RewardConfig(dense_territory_delta=0.05)
        cfg_without = RewardConfig(dense_territory_delta=0.0)
        prev = self._make_snapshot(10)
        state = self._make_state(12)  # gained 2 territories

        r_with = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg_with)
        r_without = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg_without)
        # cfg_without zeroes the territory component; difference = 2 * 0.05 = 0.1
        assert abs(r_with - r_without) == pytest.approx(0.1, abs=1e-6)

    def test_when_army_ratio_zero_then_army_component_absent(self) -> None:
        from src.env_core.reward import compute_reward

        cfg_with = RewardConfig(dense_army_ratio=0.1)
        cfg_without = RewardConfig(dense_army_ratio=0.0)
        prev = self._make_snapshot(10)
        state = self._make_state(10)  # same territories, same armies

        r_with = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg_with)
        r_without = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg_without)
        # Army ratio must differ when coef is nonzero and player holds armies
        assert r_with != pytest.approx(r_without)

    def test_when_all_dense_zero_then_non_terminal_reward_is_zero(self) -> None:
        from src.env_core.reward import compute_reward

        cfg = RewardConfig(
            dense_territory_delta=0.0,
            dense_continent_bonus_delta=0.0,
            dense_army_ratio=0.0,
            dense_elimination_bonus=0.0,
        )
        prev = self._make_snapshot(10)
        state = self._make_state(10)
        reward = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg)
        assert reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Sparse-only scripted episode
# ---------------------------------------------------------------------------


class TestSparseOnlyEpisode:
    """With all dense coefs = 0, every per-step env reward must be in {-1, 0, +1}."""

    _SPARSE_CFG = RewardConfig(
        dense_territory_delta=0.0,
        dense_continent_bonus_delta=0.0,
        dense_army_ratio=0.0,
        dense_elimination_bonus=0.0,
    )

    def test_sparse_only_2player_episode_rewards_in_valid_set(self) -> None:
        rewards = _run_scripted_episode(seed=42, reward_config=self._SPARSE_CFG)
        assert rewards, "Episode must produce at least one reward"
        invalid = [r for r in rewards if r not in (-1.0, 0.0, 1.0)]
        assert not invalid, f"Sparse-only config yielded non-sparse rewards: {invalid[:5]}"

    def test_sparse_only_episode_with_different_seed_rewards_in_valid_set(self) -> None:
        rewards = _run_scripted_episode(seed=7, reward_config=self._SPARSE_CFG)
        invalid = [r for r in rewards if r not in (-1.0, 0.0, 1.0)]
        assert not invalid, f"Sparse-only config (seed=7) yielded non-sparse rewards: {invalid[:5]}"

    @pytest.mark.parametrize("seed", [0, 1, 13, 99])
    def test_sparse_only_rewards_across_seeds(self, seed: int) -> None:
        rewards = _run_scripted_episode(seed=seed, reward_config=self._SPARSE_CFG)
        invalid = [r for r in rewards if r not in (-1.0, 0.0, 1.0)]
        assert not invalid, f"seed={seed}: non-sparse rewards {invalid[:3]}"

    def test_dense_default_config_produces_non_integer_rewards(self) -> None:
        """Baseline: the default (dense) config must produce at least one non-{-1,0,+1} reward."""
        rewards = _run_scripted_episode(seed=42, reward_config=RewardConfig())
        non_sparse = [r for r in rewards if r not in (-1.0, 0.0, 1.0)]
        assert non_sparse, (
            "Default dense config should produce at least one non-sparse reward; "
            "if all rewards are ±1 or 0, the dense components may not be wired to the env."
        )
