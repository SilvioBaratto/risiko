"""Proximal Policy Optimization (PPO) implementation from scratch."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as nnf

from src.config import PPOConfig
from src.models.actor_critic import ActorCritic
from src.models.replay_buffer import RolloutBuffer
from src.models.utils import flatten_obs
from src.utils.log import get_logger

_log = get_logger("ppo")


class PPOTrainer:
    """PPO clipped-surrogate loss trainer."""

    def __init__(self, net: ActorCritic, config: PPOConfig):
        """Initialize trainer with network and hyperparameters.

        Args:
            net: Actor-Critic network to train.
            config: PPO hyperparameters.
        """
        self.net = net
        self.config = config
        self.optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

    def save_checkpoint(self) -> dict[str, Any]:
        """Return trainer state for checkpointing."""
        return {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }

    def load_checkpoint(self, state: dict[str, Any]) -> None:
        """Restore trainer state from a checkpoint dict."""
        self.net.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run PPO update over the rollout buffer.

        Args:
            buffer: Rollout buffer with computed advantages.

        Returns:
            Dict of training metrics.
        """
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "explained_variance": 0.0,
        }

        for _epoch in range(self.config.n_epochs):
            for batch in buffer.get(
                batch_size=self.config.batch_size,
                action_dims=self.net.action_dims,
            ):
                batch_metrics = self._update_step(batch)
                for key in metrics:
                    metrics[key] += batch_metrics[key]

        n_batches = self.config.n_epochs * self._n_batches(buffer)
        if n_batches > 0:
            for key in metrics:
                metrics[key] /= n_batches

        return metrics

    def _n_batches(self, buffer: RolloutBuffer) -> int:
        """Count total number of mini-batches in one epoch."""
        n = len(buffer)
        batch_size = self.config.batch_size
        return (n + batch_size - 1) // batch_size

    def _update_step(self, batch: dict) -> dict[str, float]:
        """Single gradient step on a mini-batch."""
        obs = flatten_obs(batch["obs"])
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        # Apply the same per-head masks used at sampling time so new_log_prob
        # is comparable to the stored old_log_prob.
        action_masks = batch.get("action_masks")

        _, new_log_probs, entropy, new_values = self.net.get_action_and_value(
            obs,
            action=actions,
            action_masks=action_masks,
        )

        _log.debug(
            "log_probs old(min/max/mean)=%.3f/%.3f/%.3f "
            "new(min/max/mean)=%.3f/%.3f/%.3f adv(min/max)=%.3f/%.3f",
            old_log_probs.min().item(),
            old_log_probs.max().item(),
            old_log_probs.mean().item(),
            new_log_probs.min().item(),
            new_log_probs.max().item(),
            new_log_probs.mean().item(),
            advantages.min().item(),
            advantages.max().item(),
        )

        ratio = torch.exp(new_log_probs - old_log_probs)
        _log.debug(
            "ratio — min=%.3f max=%.3f mean=%.3f (clip_eps=%.2f)",
            ratio.min().item(),
            ratio.max().item(),
            ratio.mean().item(),
            self.config.clip_epsilon,
        )

        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nnf.mse_loss(new_values.squeeze(-1), returns)

        entropy_loss = -entropy.mean()

        total_loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            + self.config.entropy_coef * entropy_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = ((old_log_probs - new_log_probs).abs()).mean().item()
            clip_fraction = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean().item()
            explained_variance = self._explained_variance(returns, new_values.squeeze(-1))

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "explained_variance": explained_variance,
        }

    def _explained_variance(self, returns: torch.Tensor, values: torch.Tensor) -> float:
        """Compute explained variance."""
        return (1.0 - (returns - values).var() / returns.var()).item()
