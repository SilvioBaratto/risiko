"""Agent loader for resolving checkpoint paths and string tokens."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from src.agents.llm_opponent import LLMOpponent
from src.agents.player_config import load_profiles_from_yaml
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.config import resolve_device
from src.models.actor_critic import ActorCritic
from src.models.utils import get_obs_dim
from src.utils.constants import ACTION_DIMS
from src.utils.safe_torch import safe_torch_load


def load_agent(spec: str, *, seed: int | None = None) -> Any:
    """Load an agent from a checkpoint path or string token.

    Args:
        spec: Either ``'random'``, ``'llm'``, or a path to a ``.pt`` checkpoint.
        seed: Optional seed for ``RandomAgent``.

    Returns:
        An ``Agent`` instance.

    Raises:
        FileNotFoundError: If *spec* is a path that does not exist.
        ValueError: If *spec* is not recognised.
    """
    spec_lower = spec.lower().strip()

    if spec_lower == "random":
        return RandomAgent(seed=seed)

    if spec_lower == "llm":
        return LLMOpponent()

    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {spec}")

    checkpoint = safe_torch_load(path)
    config = checkpoint.get("config", {})
    if dataclasses.is_dataclass(config):
        config = dataclasses.asdict(config)
    network_cfg = config.get("network", {})
    hidden_sizes = network_cfg.get("hidden_sizes", (256, 256))

    net = ActorCritic(
        obs_dim=get_obs_dim(),
        hidden_size=hidden_sizes[0] if hidden_sizes else 256,
        num_layers=len(hidden_sizes) if hidden_sizes else 2,
        action_dims=ACTION_DIMS,
    )
    net.load_state_dict(checkpoint["trainer_state"]["model"])
    device = resolve_device(config.get("device", "cpu"))
    return PPOAgent(net, device=device)


def load_llm_pool(path: Path) -> list[LLMOpponent]:
    """Load one LLMOpponent per PlayerConfig in *path*.

    Args:
        path: YAML file containing a list of player profile dicts.

    Returns:
        One ``LLMOpponent`` per profile, in profile order.
    """
    profiles = load_profiles_from_yaml(path)
    return [LLMOpponent(player_config=pc) for pc in profiles]
