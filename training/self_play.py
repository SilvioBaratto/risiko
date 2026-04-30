"""Self-play training loop with TensorBoard logging and checkpointing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.agents.base import Agent
from src.agents.ppo_agent import PPOAgent
from src.checkpoint import CheckpointManager
from src.config import TrainingConfig
from src.env import RisikoEnv
from src.models.actor_critic import ActorCritic
from src.models.ppo import PPOTrainer
from src.models.replay_buffer import RolloutBuffer
from src.models.utils import get_obs_dim, stack_obs
from src.tb_logger import TensorBoardLogger
from src.utils.seed import set_global_seeds


def train_self_play(
    cfg: TrainingConfig,
    checkpoint_dir: Path = Path("models"),
    log_dir: Path | None = None,
) -> None:
    """Run a self-play training loop with logging and periodic checkpoints.

    Args:
        cfg: Training hyperparameters.
        checkpoint_dir: Where to write ``.pt`` checkpoints.
        log_dir: TensorBoard log directory. Defaults to
            ``results/runs/self_play_{timestamp}/``.
    """
    device = _resolve_device(cfg.device)
    rng = set_global_seeds(cfg.seed)
    env = RisikoEnv(n_players=2)
    trainer, agent = _build_trainer(cfg, device)
    logger = _build_logger(log_dir)
    buffer = RolloutBuffer(capacity=cfg.ppo.n_steps, device=device)
    episode = 0

    while episode < cfg.total_timesteps:
        result = _run_episode(env, agent, agent, episode, cfg.seed, buffer)
        _update_and_log(trainer, buffer, result, agent, episode, logger)
        _maybe_checkpoint(checkpoint_dir, trainer, env, rng, episode, cfg, cfg.save_freq)
        episode += 1

    logger.close()


def _resolve_device(device_str: str) -> str:
    """Resolve ``'auto'`` to ``'cuda'`` or ``'cpu'``."""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _build_trainer(cfg: TrainingConfig, device: str) -> tuple[PPOTrainer, PPOAgent]:
    """Create a ``PPOTrainer`` and its ``PPOAgent``."""
    action_dims = {
        "action_type": 6,
        "param_a": 42,
        "param_b": 42,
        "param_c": 43,
        "param_d": 43,
    }
    net = ActorCritic(
        obs_dim=get_obs_dim(),
        hidden_size=cfg.network.hidden_sizes[0],
        num_layers=len(cfg.network.hidden_sizes),
        action_dims=action_dims,
    ).to(device)
    trainer = PPOTrainer(net, cfg.ppo)
    agent = PPOAgent(net, device=device)
    return trainer, agent


def _build_logger(log_dir: Path | None) -> TensorBoardLogger:
    """Return a ``TensorBoardLogger`` with a sensible default path."""
    if log_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("results") / "runs" / f"self_play_{stamp}"
    return TensorBoardLogger(log_dir)


def _run_episode(
    env: RisikoEnv,
    agent: Agent,
    opponent: Agent,
    episode: int,
    base_seed: int,
    buffer: RolloutBuffer,
) -> Any:
    """Play one episode, append transitions to *buffer*, and return results."""
    obs, info = env.reset(seed=base_seed + episode)
    terminated = False
    truncated = False
    trajectories: list[tuple] = []

    while not (terminated or truncated):
        player = int(obs["current_player"])
        legal = info["legal_actions"]
        current_agent = agent if player == 0 else opponent
        action, log_prob, value = current_agent.act_with_meta(obs, legal)
        next_obs, reward, terminated, truncated, info = env.step(action)
        if player == 0:
            trajectories.append((obs.copy(), action, reward, terminated, value, log_prob))
        obs = next_obs

    for obs_i, action_i, reward_i, done_i, value_i, log_prob_i in trajectories:
        buffer.add(
            obs_i,
            action_i,
            reward_i,
            done_i,
            value_i.item(),
            log_prob_i.item(),
        )

    from src.multi_agent import GameResult

    return GameResult(
        winner=0 if not terminated else None,
        n_turns=len(trajectories),
        territory_history=[],
        elimination_order=[],
        card_trade_turns=[],
        action_log=[{"action_type": 5, "player": 0}] * len(trajectories),
        trajectories=[
            (obs_i, action_i, float(reward_i))
            for obs_i, action_i, reward_i, _d, _v, _lp in trajectories
        ],
    )


def _update_and_log(
    trainer: PPOTrainer,
    buffer: RolloutBuffer,
    result: Any,
    agent: PPOAgent,
    episode: int,
    logger: TensorBoardLogger,
) -> None:
    """Run a PPO update, log metrics, and log episode stats."""
    if len(buffer) == 0:
        logger.log_game_result(result, player_id=0, episode=episode)
        return

    with torch.no_grad():
        last_obs = stack_obs([result.trajectories[-1][0]], agent._device)
        flat_last = torch.cat(
            [last_obs[k].float().flatten(1) for k in last_obs],
            dim=-1,
        )
        _, last_value = agent._net(flat_last)

    metrics = trainer.update(buffer)
    logger.log_training_step(metrics, episode=episode)
    logger.log_game_result(result, player_id=0, episode=episode)
    buffer.clear()


def _maybe_checkpoint(
    checkpoint_dir: Path,
    trainer: PPOTrainer,
    env: RisikoEnv,
    rng: np.random.Generator,
    episode: int,
    config: TrainingConfig,
    save_freq: int,
) -> None:
    """Save a checkpoint every *save_freq* episodes."""
    if save_freq <= 0 or episode % save_freq != 0:
        return
    import random

    rng_state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
        "env": env.state.rng.bit_generator.state,
    }
    path = checkpoint_dir / f"checkpoint_{episode}.pt"
    CheckpointManager.save(path, trainer, env, rng_state, episode, config)
