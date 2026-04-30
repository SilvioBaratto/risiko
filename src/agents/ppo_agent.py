"""PPO agent wrapping ActorCritic with action masking."""

from __future__ import annotations

import numpy as np
import torch

from src.agents.base import Agent
from src.models.actor_critic import ActorCritic
from src.models.utils import flatten_obs, stack_obs


class PPOAgent(Agent):
    """Agent that uses a trained ActorCritic network with action masking."""

    def __init__(self, net: ActorCritic, device: str = "cpu") -> None:
        """Wrap an ActorCritic network for inference."""
        self._net = net.to(device)
        self._net.eval()
        self._device = device

    def act(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> dict[str, int]:
        """Select an action using the policy network with masking."""
        self._validate(legal_actions)
        obs_t = self._prepare_obs(obs)
        masks = self._build_masks(legal_actions)
        action, _log_prob, _value = self._select_action(obs_t, masks, deterministic)
        return action

    def act_with_meta(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
        """Select an action and return log-prob / value tensors as well."""
        self._validate(legal_actions)
        obs_t = self._prepare_obs(obs)
        masks = self._build_masks(legal_actions)
        return self._select_action(obs_t, masks, deterministic)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_obs(self, obs: dict[str, np.ndarray]) -> torch.Tensor:
        stacked = stack_obs([obs], self._device)
        return flatten_obs(stacked)

    def _build_masks(self, legal_actions: list[dict[str, int]]) -> dict[str, torch.Tensor]:
        return build_action_mask(legal_actions, self._net.action_dims)

    def _select_action(
        self,
        obs_t: torch.Tensor,
        masks: dict[str, torch.Tensor],
        deterministic: bool,
    ) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
        if deterministic:
            return self._deterministic_action(obs_t, masks)
        return self._stochastic_action(obs_t, masks)

    def _stochastic_action(
        self,
        obs_t: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            action, log_prob, _, value = self._net.get_action_and_value(obs_t, action_masks=masks)
        action_np = {k: int(v.item()) for k, v in action.items()}
        return action_np, log_prob.squeeze(), value.squeeze()

    def _deterministic_action(
        self,
        obs_t: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits, value = self._net.forward(obs_t)
        action: dict[str, int] = {}
        for name in logits:
            masked = logits[name].masked_fill(~masks[name].unsqueeze(0), float("-inf"))
            action[name] = int(masked.argmax(dim=-1).item())
        log_prob = torch.tensor(0.0)
        return action, log_prob, value.squeeze()

    @staticmethod
    def _validate(legal_actions: list[dict[str, int]]) -> None:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")


def build_action_mask(
    legal_actions: list[dict[str, int]],
    action_dims: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Convert a list of legal action dicts into per-head boolean masks.

    Per-head masking is coarse: it may allow joint combinations that are
    not in *legal_actions*. The environment penalises illegal actions via
    ``invalid_action_penalty`` as a fallback.

    Heads with zero legal values get a single dummy entry at index ``0``
    so ``ActorCritic._apply_mask`` does not raise *ValueError*.
    """
    mask = _initialise_mask(action_dims)
    for action in legal_actions:
        for name in action_dims:
            mask[name][action[name]] = True
    _fill_empty_heads(mask, action_dims)
    return {name: tensor.to(torch.bool) for name, tensor in mask.items()}


def _initialise_mask(action_dims: dict[str, int]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros(dim, dtype=torch.bool) for name, dim in action_dims.items()}


def _fill_empty_heads(
    mask: dict[str, torch.Tensor],
    action_dims: dict[str, int],
) -> None:
    for name, _dim in action_dims.items():
        if not mask[name].any():
            mask[name][0] = True
