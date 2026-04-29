"""Observation flattening and stacking utilities for RisikoEnv."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as nnf

_OBS_DIM = 137

# Normalization denominators for continuous Box fields
_NORM_DENOMINATORS: dict[str, float] = {
    "territory_owner": 5.0,
    "armies": 999.0,
    "trade_count": 999.0,
    "reinforcements_remaining": 999.0,
}

# Discrete fields that need one-hot encoding with their num_classes
_ONE_HOT_CLASSES: dict[str, int] = {
    "phase": 5,
    "current_player": 6,
    "n_players": 7,
}

# Order of concatenation
_CONCAT_ORDER: list[str] = [
    "territory_owner",
    "armies",
    "phase",
    "current_player",
    "cards",
    "continent_control",
    "trade_count",
    "reinforcements_remaining",
    "turn_capture",
    "n_players",
    "eliminated",
]


def get_obs_dim() -> int:
    """Return the flattened observation dimension."""
    return _OBS_DIM


def flatten_obs(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Convert a batched dict observation into a flat float tensor.

    Args:
        obs: Dict of tensors where each value has shape ``(batch, ...)``.

    Returns:
        Float tensor of shape ``(batch, 137)``.
    """
    return torch.cat([_encode(key, obs[key]) for key in _CONCAT_ORDER], dim=-1)


def stack_obs(obs_list: list[dict[str, np.ndarray]], device: str) -> dict[str, torch.Tensor]:
    """Stack a list of individual observations into batched tensors.

    Args:
        obs_list: List of observation dicts from the environment.
        device: Target device for the tensors.

    Returns:
        Dict with the same keys and batched tensor values.
    """
    return {
        key: torch.from_numpy(np.stack([o[key] for o in obs_list])).to(device)
        for key in obs_list[0]
    }


def _encode(key: str, tensor: torch.Tensor) -> torch.Tensor:
    """Encode a single observation component into a flat float tensor."""
    x = tensor.float()

    if key in _ONE_HOT_CLASSES:
        return _one_hot_encode(tensor, _ONE_HOT_CLASSES[key])

    if key in _NORM_DENOMINATORS:
        x = x / _NORM_DENOMINATORS[key]

    if x.ndim == 1:
        x = x.unsqueeze(-1)

    return x.flatten(start_dim=1)


def _one_hot_encode(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot encode a discrete scalar or batched scalar tensor."""
    x = tensor.long()
    if x.ndim == 0:
        x = x.unsqueeze(0)
    encoded = nnf.one_hot(x, num_classes=num_classes).float()
    if tensor.ndim == 0:
        encoded = encoded.squeeze(0)
    return encoded
