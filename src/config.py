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
    "BCConfig",
    "DiplomacyConfig",
    "HeuristicConfig",
    "PPOConfig",
    "NetworkConfig",
    "SelfPlayConfig",
    "EarlyStopConfig",
    "TrainingConfig",
    "load_config",
    "merge_cli_overrides",
    "resolve_device",
]

T = TypeVar("T")


@dataclass(frozen=True)
class HeuristicConfig:
    """Thresholds and weights for the HeuristicAgent scripted policy."""

    reinforce_border_weight: float = 1.0
    reinforce_continent_weight: float = 0.5
    reinforce_threat_weight: float = 0.8
    attack_min_odds: float = 1.5
    attack_odds_stop: float = 1.0
    attack_continent_weight: float = 1.5
    attack_weakest_enemy_weight: float = 0.5
    attack_dice_policy: str = "max"
    fortify_threat_weight: float = 1.0
    fortify_min_interior_armies: int = 2
    capture_move_fraction: float = 0.5
    card_target_weakest_weight: float = 1.0


@dataclass(frozen=True)
class BCConfig:
    """Behaviour-cloning pipeline configuration."""

    n_games: int = 10_000
    n_players: int = 6
    max_turns: int = 500
    seed: int = 42
    dataset_dir: str = "data/bc"
    shard_size: int = 10_000
    demonstrator: str = "heuristic"
    epochs: int = 10
    batch_size: int = 512
    lr: float = 3e-4
    value_loss_coef: float = 0.5
    val_split: float = 0.1
    early_stop_patience: int = 5
    output_path: str = "models/pretrained.pt"
    explore_eps: float = 0.1
    label_smoothing: float = 0.05
    entropy_coef: float = 0.01
    heuristic: HeuristicConfig = field(default_factory=HeuristicConfig)


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
    # Stop the epoch loop early once the policy has drifted past this approx-KL
    # (guards against divergence from many epochs over a small, correlated
    # buffer — e.g. a BC-warm-started policy). 0.0 disables the guard.
    target_kl: float = 0.05
    # Normalize advantages to zero-mean/unit-std per mini-batch (SB3 default).
    # Essential for stability: raw advantages (terminal-margin + dense + GAE)
    # otherwise produce unbounded policy-gradient steps (approx_kl ~1).
    normalize_advantage: bool = True
    # Coefficient for an anchor penalty pulling the policy toward a frozen
    # reference (the BC clone), RLHF-style. > 0 enables fine-tuning a sharp
    # BC-warm-started policy without the divergence that plain PPO causes
    # (the reference is loaded from ``BCConfig.output_path``). 0.0 disables.
    kl_ref_coef: float = 0.0


@dataclass(frozen=True)
class NetworkConfig:
    """Shared-trunk network architecture."""

    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "tanh"
    # "mlp" (flat trunk) or "graphsage" (GraphSAGE over the territory adjacency
    # graph — board-structured trunk, same action heads). len(hidden_sizes) sets
    # the number of SAGE layers in graphsage mode.
    arch: str = "mlp"


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
    # Reverse curriculum: seed near-won boards so a from-scratch learner reaches
    # wins, then anneal difficulty up. Disabled by default.
    curriculum_enabled: bool = False
    # "territory": learner owns all but k territories per opponent, k grows.
    # "army": balanced territories but the learner starts army-rich (multiplier),
    #         annealed toward 1.0 — teaches converting a material edge into a win,
    #         which transfers to full games (territory mode tops out near-won).
    curriculum_mode: str = "territory"
    curriculum_start: int = 2  # (territory) opponent territories at the easiest stage
    curriculum_step: int = 2  # (territory) territories added per opponent on promotion
    curriculum_army_start: float = 4.0  # (army) learner army multiplier at the easiest stage
    curriculum_army_step: float = 0.5  # (army) multiplier reduction per promotion (toward 1.0)
    curriculum_promote_threshold: float = 0.7  # learner win-rate to advance a stage
    curriculum_window: int = 50  # games over which the stage win-rate is measured
    # Fraction of episodes that use a balanced full-game start instead of the
    # curriculum near-won start. The curriculum teaches closeouts but only ever
    # shows near-won states, so the policy never learns full-game play and
    # fails to generalize (0% vs random in full games). Mixing in balanced
    # episodes trains on the eval distribution. These episodes do not count
    # toward curriculum-stage promotion.
    curriculum_balanced_fraction: float = 0.0


@dataclass(frozen=True)
class DiplomacyConfig:
    """Eval-only diplomacy layer configuration; disabled by default."""

    enabled: bool = False
    n_rounds: int = 1
    max_message_tokens: int = 128


@dataclass(frozen=True)
class EarlyStopConfig:
    """Early stopping on a fixed-baseline metric (win-rate vs ``RandomAgent``).

    Research-backed for self-play / sparse-reward training: evaluate against a
    FIXED reference rather than the (changing, circular) training opponent,
    smooth the noisy win-rate over a window, and stop after ``patience`` checks
    with no ``> min_delta`` improvement — keeping the best checkpoint, since
    over-training self-play agents tends to degrade past the peak.

    Disabled by default; opt in via config. ``eval_games`` use ``RandomAgent``
    seats only, so a check costs no LLM calls.
    """

    enabled: bool = False
    eval_every: int = 50  # episodes between baseline evaluations
    eval_games: int = 20  # games per baseline evaluation
    window: int = 5  # moving-average window over evaluations (smooths noise)
    patience: int = 12  # evaluations without improvement before stopping
    min_delta: float = 0.01  # smallest smoothed gain that counts as improvement
    restore_best: bool = True  # reload the best checkpoint when stopping


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
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    diplomacy: DiplomacyConfig = field(default_factory=DiplomacyConfig)


def load_config(path: Path | str) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file."""
    path = Path(path)
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
