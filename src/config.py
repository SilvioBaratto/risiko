"""Hyperparameter configuration system for Risiko RL training.

All tunables are declared as frozen dataclasses so they can be loaded
from YAML and overridden via CLI without magic numbers in the model source.
"""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

from src.utils.reward_config import RewardConfig

__all__ = [
    "PPOConfig",
    "NetworkConfig",
    "SelfPlayConfig",
    "TrainingConfig",
    "load_config",
    "merge_cli_overrides",
    "resolve_device",
]

T = TypeVar("T")


@dataclass(frozen=True)
class PPOConfig:
    """Proximal Policy Optimization hyperparameters."""

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 2048
    n_epochs: int = 10
    batch_size: int = 64


@dataclass(frozen=True)
class NetworkConfig:
    """Shared-trunk network architecture."""

    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "tanh"


@dataclass(frozen=True)
class SelfPlayConfig:
    """Self-play opponent rotation and promotion settings."""

    opponent_update_freq: int = 50
    best_metric: str = "win_rate"
    promote_threshold: float = 0.55
    eval_games: int = 10
    n_players: int = 2
    llm_profiles_path: str | None = None
    llm_model: str | None = None  # uniform model override for all LLM opponents


@dataclass(frozen=True)
class TrainingConfig:
    """Root configuration bundling PPO, network, training, and self-play settings."""

    total_timesteps: int = 1_000_000
    n_envs: int = 1
    save_freq: int = 100
    eval_freq: int = 50
    seed: int = 42
    device: str = "auto"
    max_turns: int = 2000
    config_path: str | None = None
    ppo: PPOConfig = field(default_factory=PPOConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)


def load_config(path: Path) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text())
    return _from_dict(TrainingConfig, raw)


def _from_dict(cls: Any, raw: dict[str, Any]) -> Any:
    """Recursively construct a frozen dataclass from a plain dict."""
    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        return raw

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        val = raw[f.name]
        resolved = hints.get(f.name, f.type)
        if isinstance(resolved, type) and dataclasses.is_dataclass(resolved):
            kwargs[f.name] = _from_dict(resolved, val)
        elif _is_tuple_int(resolved) and isinstance(val, list):
            kwargs[f.name] = tuple(val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def merge_cli_overrides(cfg: T, overrides: dict[str, str]) -> T:
    """Return a new frozen config with CLI overrides applied.

    Overrides use dot notation for nested fields, e.g. ``{"ppo.lr": "0.001"}``.
    Values are cast to the declared dataclass field type.
    """
    if not overrides:
        return cfg
    raw = dataclasses.asdict(cfg)  # type: ignore[arg-type]
    for key, val in overrides.items():
        parts = key.split(".")
        _validate_path(cfg, parts, key)
        target = raw
        for part in parts[:-1]:
            target = target[part]
        leaf = parts[-1]
        target[leaf] = _cast_value(val, cfg, parts)
    return _from_dict(type(cfg), raw)


def _validate_path(root: Any, parts: list[str], key: str) -> None:
    """Ensure every segment of a dot-path names a real dataclass field."""
    obj_type: type[Any] = type(root)
    for part in parts:
        if not dataclasses.is_dataclass(obj_type) or part not in {f.name for f in fields(obj_type)}:
            raise ValueError(f"Unknown config path: {key}")
        obj_type = get_type_hints(obj_type).get(part, str)


def _cast_value(raw: str, root: Any, path: list[str]) -> Any:
    """Cast a raw string to the type declared on the dataclass field at *path*."""
    field_type = _resolve_field_type(root, path)
    if field_type is bool:
        return raw.lower() in {"true", "1", "yes", "on"}
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    if field_type is str:
        return raw
    if _is_tuple_int(field_type):
        parsed = ast.literal_eval(raw)
        return tuple(parsed) if isinstance(parsed, list) else parsed
    return raw


def _resolve_field_type(root: Any, path: list[str]) -> type[Any]:
    """Walk the dataclass hierarchy to find the declared type of the leaf field."""
    obj_type: type[Any] = type(root)
    for part in path:
        hints = get_type_hints(obj_type)
        for f in fields(obj_type):
            if f.name == part:
                obj_type = hints.get(f.name, f.type)
                break
        else:
            return str
    return obj_type


def _is_tuple_int(annotation: type[Any]) -> bool:
    """Check whether an annotation is ``tuple[int, ...]``."""
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())
    return bool(origin is tuple and args and args[0] is int)


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` or ``"cpu"``; passthrough otherwise."""
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
