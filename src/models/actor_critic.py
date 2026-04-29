"""Actor-Critic shared-trunk network for PPO training."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyHeads(nn.Module):
    """Collection of independent categorical policy heads."""

    def __init__(self, hidden_size: int, action_dims: dict[str, int]):
        """Create policy heads mapping trunk output to action logits."""
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_size, dim) for name, dim in action_dims.items()}
        )

    def forward(self, trunk_output: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute logits for every action component."""
        return {name: head(trunk_output) for name, head in self.heads.items()}


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic network for PPO."""

    def __init__(
        self,
        obs_dim: int,
        hidden_size: int,
        num_layers: int,
        action_dims: dict[str, int],
    ):
        """Initialize shared trunk, policy heads, and value head.

        Args:
            obs_dim: Flattened observation dimension.
            hidden_size: Hidden layer width.
            num_layers: Number of trunk hidden layers.
            action_dims: Mapping from action component names to cardinalities.
        """
        super().__init__()
        self.action_dims = action_dims

        layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.trunk = nn.Sequential(*layers)

        self.policy = PolicyHeads(hidden_size, action_dims)
        self.value = nn.Linear(hidden_size, 1)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass returning policy logits and state value."""
        trunk_out = self.trunk(obs)
        logits = self.policy(trunk_out)
        value = self.value(trunk_out)
        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: dict[str, torch.Tensor] | None = None,
        action_masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate an action and return auxiliary quantities.

        Args:
            obs: Flattened observation tensor, shape ``(batch, obs_dim)``.
            action: Optional action dict to evaluate instead of sampling.
            action_masks: Optional bool masks, shape matching logits.

        Returns:
            ``(action, log_prob, entropy, value)``.
        """
        logits, value = self.forward(obs)
        distributions = _build_distributions(logits, action_masks)

        if action is None:
            action = {name: dist.sample() for name, dist in distributions.items()}

        log_probs = torch.stack([distributions[name].log_prob(action[name]) for name in logits])
        log_prob = log_probs.sum(dim=0)
        entropies = torch.stack([distributions[name].entropy() for name in logits])
        entropy = entropies.sum(dim=0)
        return action, log_prob, entropy, value


def _build_distributions(
    logits: dict[str, torch.Tensor],
    action_masks: dict[str, torch.Tensor] | None,
) -> dict[str, Categorical]:
    """Create Categorical distributions from logits, applying optional masks."""
    distributions: dict[str, Categorical] = {}
    for name, logit in logits.items():
        mask = action_masks.get(name) if action_masks else None
        masked_logits = _apply_mask(logit, mask)
        distributions[name] = Categorical(logits=masked_logits)
    return distributions


def _apply_mask(
    logits: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Apply a boolean mask by setting masked logits to -inf.

    Raises:
        ValueError: If every logit in a row is masked.
    """
    if mask is None:
        return logits

    masked = logits.masked_fill(~mask, float("-inf"))

    if torch.all(masked == float("-inf"), dim=-1).any():
        raise ValueError("All logits masked for at least one sample.")

    return masked
