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
        self.legal_actions: list[list[dict[str, int]]] = []
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
        legal_actions: list[dict[str, int]] | None = None,
    ) -> None:
        """Add a single transition.

        ``legal_actions`` is the list of legal action dicts at sampling time.
        It is needed at PPO update time to rebuild the per-head action mask so
        ``new_log_prob`` is computed under the same constraint as the stored
        ``old_log_prob``. Without it, the policy distribution at update time
        is uniform-over-all-indices and the PPO ratio collapses.
        """
        if len(self) >= self.capacity:
            raise IndexError("RolloutBuffer is full")
        self.observations.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(float(done))
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.legal_actions.append(list(legal_actions or []))

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

    def get(self, batch_size: int, action_dims: dict[str, int] | None = None):
        """Yield shuffled mini-batches.

        If ``action_dims`` is provided, also yields a batched ``action_masks``
        dict reconstructed from each transition's ``legal_actions`` list. The
        update step uses these masks so ``new_log_prob`` is computed under the
        same per-head constraint that was applied at sampling time.

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

        # Build per-transition action masks once if requested
        masks_per_transition: list[dict[str, torch.Tensor]] | None = None
        if action_dims is not None:
            masks_per_transition = [
                _build_mask_tensor(la, action_dims) for la in self.legal_actions
            ]

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            batch: dict = {
                "obs": {k: v[idx] for k, v in obs.items()},
                "actions": {k: v[idx] for k, v in actions.items()},
                "advantages": advantages[idx].to(self.device),
                "returns": returns[idx].to(self.device),
                "rewards": rewards[idx].to(self.device),
                "values": values[idx].to(self.device),
                "log_probs": log_probs[idx].to(self.device),
            }
            if masks_per_transition is not None:
                batch["action_masks"] = {
                    head: torch.stack([masks_per_transition[i][head] for i in idx.tolist()]).to(
                        self.device
                    )
                    for head in masks_per_transition[0]
                }
            yield batch

    def clear(self) -> None:
        """Reset the buffer."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.legal_actions.clear()
        self.advantages = None
        self.returns = None


def _build_mask_tensor(
    legal_actions: list[dict[str, int]],
    action_dims: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Build per-head bool masks from a list of legal action dicts.

    Mirrors ``src.agents.ppo_agent.build_action_mask`` but kept local to
    avoid a layering dependency from buffer → agent.
    """
    mask = {head: torch.zeros(dim, dtype=torch.bool) for head, dim in action_dims.items()}
    for action in legal_actions:
        for head in action_dims:
            mask[head][action[head]] = True
    # Empty heads get a single dummy True so masking does not collapse.
    for head, dim in action_dims.items():
        del dim
        if not mask[head].any():
            mask[head][0] = True
    return mask
