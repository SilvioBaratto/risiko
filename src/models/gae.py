"""Generalized Advantage Estimation (GAE) implementation."""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: Rewards for each step, shape ``(..., n_steps)``.
        values: Value estimates for each step plus bootstrap, shape ``(..., n_steps + 1)``.
        dones: Done flags for each step plus bootstrap, shape ``(..., n_steps + 1)``.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.

    Returns:
        ``(advantages, returns)`` each of shape ``(..., n_steps)``.
    """
    n_steps = rewards.shape[-1]
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(n_steps)):
        next_non_terminal = 1.0 - dones[..., t + 1]
        delta = rewards[..., t] + gamma * values[..., t + 1] * next_non_terminal - values[..., t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[..., t] = last_gae

    returns = advantages + values[..., :n_steps]
    return advantages, returns
