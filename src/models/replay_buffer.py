"""Rollout buffer for storing PPO training data."""

from __future__ import annotations

import numpy as np
import torch

from src.models.gae import compute_gae


class RolloutBuffer:
    """Fixed-length rollout buffer storing transitions for PPO updates."""

    def __init__(self, capacity: int, device: str):
        """Initialize the buffer.

        Args:
            capacity: Maximum number of transitions to store.
            device: Target device for tensor outputs.
        """
        self.capacity = capacity
        self.device = device
        self.observations: list[dict[str, np.ndarray]] = []
        self.actions: list[dict[str, torch.Tensor]] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return len(self.observations)

    def add(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, torch.Tensor],
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Add a single transition."""
        if len(self) >= self.capacity:
            raise IndexError("RolloutBuffer is full")
        self.observations.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(float(done))
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_advantages(
        self,
        next_value: torch.Tensor,
        next_done: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE advantages and returns from stored transitions."""
        n = len(self)
        if n == 0:
            raise ValueError("Cannot compute advantages on empty buffer")

        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        # Append bootstrap value and done
        values_ext = torch.cat([values, next_value])
        dones_ext = torch.cat([dones, next_done])

        advantages, returns = compute_gae(
            rewards, values_ext, dones_ext, gamma=gamma, gae_lambda=gae_lambda
        )
        self.advantages = advantages
        self.returns = returns

    def get(self, batch_size: int):
        """Yield shuffled mini-batches.

        Raises:
            ValueError: If compute_advantages has not been called.
        """
        if self.advantages is None or self.returns is None:
            raise ValueError("compute_advantages must be called before get()")

        n = len(self)
        indices = torch.randperm(n)
        advantages = self.advantages
        returns = self.returns
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32)

        # Stack all observations into a batched dict
        obs_keys = self.observations[0].keys()
        obs = {
            k: torch.from_numpy(np.stack([o[k] for o in self.observations])).to(self.device)
            for k in obs_keys
        }

        # Stack all actions into a batched dict
        action_keys = self.actions[0].keys()
        actions = {
            k: torch.stack([a[k] for a in self.actions]).to(self.device) for k in action_keys
        }

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield {
                "obs": {k: v[idx] for k, v in obs.items()},
                "actions": {k: v[idx] for k, v in actions.items()},
                "advantages": advantages[idx].to(self.device),
                "returns": returns[idx].to(self.device),
                "rewards": rewards[idx].to(self.device),
                "values": values[idx].to(self.device),
                "log_probs": log_probs[idx].to(self.device),
            }

    def clear(self) -> None:
        """Reset the buffer."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.advantages = None
        self.returns = None
