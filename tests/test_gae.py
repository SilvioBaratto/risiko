"""Tests for Generalized Advantage Estimation (GAE)."""

from __future__ import annotations

import pytest
import torch

from src.models.gae import compute_gae


class TestGaeShape:
    """Output shape contracts."""

    def test_returns_two_tensors(self):
        """compute_gae returns a 2-tuple of tensors."""
        rewards = torch.zeros(5)
        values = torch.zeros(6)
        dones = torch.zeros(6)
        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
        assert isinstance(advantages, torch.Tensor)
        assert isinstance(returns, torch.Tensor)

    def test_advantage_shape_matches_steps(self):
        """Advantage tensor has shape (n_steps,)."""
        rewards = torch.zeros(5)
        values = torch.zeros(6)
        dones = torch.zeros(6)
        advantages, _ = compute_gae(rewards, values, dones)
        assert advantages.shape == (5,)

    def test_returns_shape_matches_steps(self):
        """Returns tensor has shape (n_steps,)."""
        rewards = torch.zeros(5)
        values = torch.zeros(6)
        dones = torch.zeros(6)
        _, returns = compute_gae(rewards, values, dones)
        assert returns.shape == (5,)


class TestGaeMath:
    """GAE matches hand-calculated values."""

    def test_three_step_sequence(self):
        """Manual 3-step sequence matches hand calculation."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 1.0, 1.5, 2.0])
        dones = torch.zeros(4)
        gamma = 0.99
        gae_lambda = 0.95

        advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)

        # Hand-calculate
        delta_2 = rewards[2] + gamma * values[3] * (1 - dones[3]) - values[2]
        adv_2 = delta_2
        delta_1 = rewards[1] + gamma * values[2] * (1 - dones[2]) - values[1]
        adv_1 = delta_1 + gamma * gae_lambda * (1 - dones[2]) * adv_2
        delta_0 = rewards[0] + gamma * values[1] * (1 - dones[1]) - values[0]
        adv_0 = delta_0 + gamma * gae_lambda * (1 - dones[1]) * adv_1

        assert advantages[0].item() == pytest.approx(adv_0, abs=1e-5)
        assert advantages[1].item() == pytest.approx(adv_1, abs=1e-5)
        assert advantages[2].item() == pytest.approx(adv_2, abs=1e-5)

    def test_returns_equal_advantages_plus_values(self):
        """Returns == advantages + values[:n_steps]."""
        rewards = torch.randn(10)
        values = torch.randn(11)
        dones = torch.zeros(11)
        advantages, returns = compute_gae(rewards, values, dones)
        expected_returns = advantages + values[:10]
        assert torch.allclose(returns, expected_returns, atol=1e-5)

    def test_done_resets_advantage(self):
        """A done at step t+1 zeros out the bootstrap for step t."""
        rewards = torch.tensor([1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        dones = torch.tensor([0.0, 1.0, 0.0])
        gamma = 0.99
        gae_lambda = 0.95

        advantages, _ = compute_gae(rewards, values, dones, gamma, gae_lambda)

        # done_1 = 1 means the transition from step 0 to step 1 is terminal.

        delta_0 = rewards[0] - values[0]  # done_1 = 1, so no bootstrap
        # But wait, dones[2] is 0, so bootstrap is active... Actually done at index 1
        # means the transition from step 1 to step 2 is terminal.
        # delta_1 = r_1 + gamma * V_2 * (1 - done_2) - V_1
        # The done that matters for step 1's bootstrap is done_2.
        # Let me reconsider: in standard GAE, done_{t+1} affects delta_t.
        # Here dones = [0, 1, 0], so done_2 = 0, meaning delta_1 still bootstraps.
        # But done_1 = 1 means the transition from step 0 to step 1 is terminal,
        # so delta_0 = r_0 - V_0 (no bootstrap because done_1 = 1).

        delta_0 = rewards[0] - values[0]  # done_1 = 1, so no bootstrap
        assert advantages[0].item() == pytest.approx(delta_0, abs=1e-5)

    def test_single_step(self):
        """GAE with one step works."""
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5, 0.5])
        dones = torch.zeros(2)
        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
        delta = rewards[0] + 0.99 * values[1] - values[0]
        assert advantages[0].item() == pytest.approx(delta, abs=1e-5)
        assert returns[0].item() == pytest.approx(delta + values[0], abs=1e-5)

    def test_all_zeros(self):
        """Zero rewards and values give zero advantages and returns."""
        rewards = torch.zeros(5)
        values = torch.zeros(6)
        dones = torch.zeros(6)
        advantages, returns = compute_gae(rewards, values, dones)
        assert torch.allclose(advantages, torch.zeros(5), atol=1e-5)
        assert torch.allclose(returns, torch.zeros(5), atol=1e-5)

    def test_batch_dimensions(self):
        """GAE works with 2D tensors (n_envs, n_steps)."""
        rewards = torch.randn(2, 5)
        values = torch.randn(2, 6)
        dones = torch.zeros(2, 6)
        advantages, returns = compute_gae(rewards, values, dones)
        assert advantages.shape == (2, 5)
        assert returns.shape == (2, 5)

    def test_batch_done_resets(self):
        """Done resets advantage per environment in batched mode."""
        rewards = torch.ones(2, 3)
        values = torch.ones(2, 4)
        dones = torch.zeros(2, 4)
        dones[0, 2] = 1.0  # env 0 has a done after step 1
        gamma = 0.99
        gae_lambda = 0.95

        advantages, _ = compute_gae(rewards, values, dones, gamma, gae_lambda)

        # Env 0, step 1: done_2 = 1, so no bootstrap — advantage is single-step TD
        delta_0_1 = rewards[0, 1] - values[0, 1]
        assert advantages[0, 1].item() == pytest.approx(delta_0_1, abs=1e-5)
        # Env 0, step 2: still single-step (last step, no future)
        delta_0_2 = rewards[0, 2] + gamma * values[0, 3] - values[0, 2]
        assert advantages[0, 2].item() == pytest.approx(delta_0_2, abs=1e-5)
        # Env 1, step 1: no done, so advantage bootstraps from step 2
        advantage_1_2 = rewards[1, 2] + gamma * values[1, 3] - values[1, 2]
        expected_1_1 = (
            rewards[1, 1] + gamma * values[1, 2] - values[1, 1] + gamma * gae_lambda * advantage_1_2
        )
        assert advantages[1, 1].item() == pytest.approx(expected_1_1, abs=1e-5)
