"""BC loss functions — full-vocabulary per-head cross-entropy + value MSE + entropy bonus.

No action masking is applied: the Cycle-2 shards store only the chosen per-head
labels, not per-row ``legal_actions``, so the mask cannot be reconstructed at
training time.  Full-vocabulary softmax CE is the correct MLE objective because
the demonstrator's chosen index is guaranteed legal.  The raw entropy computed
here is an upper bound on the masked entropy PPO sees at inference — that is
exactly why the entropy term is used as a *bonus* (not a hard floor), preserving
exploration without assuming the mask shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as nnf
from torch.distributions import Categorical


@dataclass
class LossStats:
    """Scalar metrics for one BC training step — plain floats for logging."""

    policy_ce: float
    value_mse: float
    entropy: float
    top1_acc: float


def policy_cross_entropy(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    label_smoothing: float,
) -> torch.Tensor:
    """Sum F.cross_entropy across all policy heads (no action mask applied)."""
    return sum(
        nnf.cross_entropy(logits[name], targets[name], label_smoothing=label_smoothing)
        for name in logits
    )


def value_mse(value: torch.Tensor, mc_return: torch.Tensor) -> torch.Tensor:
    """MSE between the value head output (B,1) and Monte-Carlo returns (B,)."""
    return nnf.mse_loss(value.squeeze(-1), mc_return)


def policy_entropy(logits: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum mean entropy across all policy heads (matches PPO's head factorisation)."""
    return sum(Categorical(logits=logits[name]).entropy().mean() for name in logits)


def top1_accuracy(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Top-1 accuracy per head plus an aggregate (mean across heads)."""
    per_head = {
        name: (logits[name].argmax(dim=-1) == targets[name]).float().mean().item()
        for name in logits
    }
    per_head["aggregate"] = sum(per_head.values()) / len(per_head)
    return per_head


def combined_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    value: torch.Tensor,
    mc_return: torch.Tensor,
    *,
    value_loss_coef: float,
    entropy_coef: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, LossStats]:
    """Combined BC objective: policy CE + value MSE − entropy bonus.

    Sign convention: entropy is subtracted (bonus), matching the explicit form
    ``loss = policy_ce + value_loss_coef * value_mse - entropy_coef * entropy``.
    """
    ce = policy_cross_entropy(logits, targets, label_smoothing)
    mse = value_mse(value, mc_return)
    ent = policy_entropy(logits)
    loss = ce + value_loss_coef * mse - entropy_coef * ent
    acc = top1_accuracy(logits, targets)
    stats = LossStats(
        policy_ce=ce.item(),
        value_mse=mse.item(),
        entropy=ent.item(),
        top1_acc=acc["aggregate"],
    )
    return loss, stats
