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

    def __init__(
        self,
        net: ActorCritic,
        config: PPOConfig,
        reference_net: ActorCritic | None = None,
    ):
        """Initialize trainer with network and hyperparameters.

        Args:
            net: Actor-Critic network to train.
            config: PPO hyperparameters.
            reference_net: Optional frozen policy (e.g. the BC clone) to anchor
                the learner toward via ``config.kl_ref_coef`` (RLHF-style). Kept
                in eval mode with grads off; only its log-probs are used.
        """
        self.net = net
        self.config = config
        self.optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
        self.reference_net = reference_net
        if reference_net is not None:
            reference_net.eval()
            for p in reference_net.parameters():
                p.requires_grad = False

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
            "kl_ref": 0.0,
        }

        target_kl = getattr(self.config, "target_kl", 0.0)
        epochs_run = 0
        for _epoch in range(self.config.n_epochs):
            epoch_kl_sum = 0.0
            epoch_batches = 0
            for batch in buffer.get(
                batch_size=self.config.batch_size,
                action_dims=self.net.action_dims,
            ):
                batch_metrics = self._update_step(batch)
                for key in metrics:
                    metrics[key] += batch_metrics[key]
                epoch_kl_sum += batch_metrics["approx_kl"]
                epoch_batches += 1
            epochs_run += 1
            # Target-KL early stop: abandon the remaining epochs once the policy
            # has already moved too far this update. Prevents the KL blow-up seen
            # with a sharp (BC-warm-started) policy run for many epochs over one
            # game's small, correlated rollout.
            if target_kl and epoch_batches and (epoch_kl_sum / epoch_batches) > 1.5 * target_kl:
                _log.info(
                    "target-KL early stop: epoch=%d approx_kl=%.4f > 1.5*target_kl=%.4f",
                    epochs_run,
                    epoch_kl_sum / epoch_batches,
                    1.5 * target_kl,
                )
                break

        n_batches = epochs_run * self._n_batches(buffer)
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
        # Per-mini-batch advantage normalization (standard PPO; SB3 default).
        # Without it the policy-gradient magnitude scales with the raw advantage
        # scale (here: terminal-margin + dense rewards + GAE), making a single
        # epoch move the policy enormously (observed approx_kl ~1). Normalizing
        # to zero-mean/unit-std keeps each update inside the trust region. Skip
        # when the batch advantages are (near-)constant: dividing by ~0 std just
        # amplifies numerical noise into a huge spurious update.
        adv_std = advantages.std()
        if getattr(self.config, "normalize_advantage", True) and adv_std > 1e-6:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
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

        kl_ref = 0.0
        ref_penalty = torch.zeros((), device=new_log_probs.device)
        if self.reference_net is not None and self.config.kl_ref_coef > 0:
            with torch.no_grad():
                _, ref_log_probs, _, _ = self.reference_net.get_action_and_value(
                    obs, action=actions, action_masks=action_masks
                )
            # Anchor toward the reference (BC) policy on the chosen actions:
            # squared log-prob deviation — always ≥0, stable, pulls the policy
            # back when it drifts off the competent BC start.
            ref_penalty = (new_log_probs - ref_log_probs).pow(2).mean()
            kl_ref = ref_penalty.item()

        total_loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            + self.config.entropy_coef * entropy_loss
            + self.config.kl_ref_coef * ref_penalty
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            # Schulman k3 estimator (SB3/standard PPO): low-variance, ≥0, and
            # comparable to published target_kl values. ``ratio`` already equals
            # exp(new - old), so logratio = new - old.
            logratio = new_log_probs - old_log_probs
            approx_kl = ((ratio - 1) - logratio).mean().item()
            clip_fraction = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean().item()
            explained_variance = self._explained_variance(returns, new_values.squeeze(-1))

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "explained_variance": explained_variance,
            "kl_ref": kl_ref,
        }

    def _explained_variance(self, returns: torch.Tensor, values: torch.Tensor) -> float:
        """Compute explained variance."""
        return (1.0 - (returns - values).var() / returns.var()).item()
