"""Regression tests for risiko_rl/agent_loader.py — Issue #87."""

from __future__ import annotations

import torch


def _write_minimal_bc_checkpoint(path):
    """Write a minimal PPO-compatible BC checkpoint."""
    from src.config import TrainingConfig  # noqa: PLC0415
    from src.models.actor_critic import ActorCritic  # noqa: PLC0415
    from src.models.utils import get_obs_dim  # noqa: PLC0415
    from src.utils.constants import ACTION_DIMS  # noqa: PLC0415
    from training.bc_checkpoint import write_bc_checkpoint  # noqa: PLC0415

    cfg = TrainingConfig()
    net = ActorCritic(
        obs_dim=get_obs_dim(),
        hidden_size=cfg.network.hidden_sizes[0],
        num_layers=len(cfg.network.hidden_sizes),
        action_dims=ACTION_DIMS,
    )
    opt = torch.optim.Adam(net.parameters(), lr=cfg.bc.lr)
    write_bc_checkpoint(path, net, opt, cfg)


def test_when_real_bc_checkpoint_loaded_then_load_agent_returns_ppo_agent(tmp_path):
    """Regression: a real .pt checkpoint must produce a PPOAgent, not raise."""
    from risiko_rl.agent_loader import load_agent  # noqa: PLC0415
    from src.agents.ppo_agent import PPOAgent  # noqa: PLC0415

    ckpt = tmp_path / "pretrained.pt"
    _write_minimal_bc_checkpoint(ckpt)

    agent = load_agent(str(ckpt))

    assert isinstance(agent, PPOAgent)


def test_when_real_checkpoint_loaded_then_device_is_resolved(tmp_path):
    """load_agent routes device through resolve_device; 'auto' must not be passed raw."""
    from risiko_rl.agent_loader import load_agent  # noqa: PLC0415
    from src.agents.ppo_agent import PPOAgent  # noqa: PLC0415

    ckpt = tmp_path / "pretrained.pt"
    _write_minimal_bc_checkpoint(ckpt)

    agent = load_agent(str(ckpt))

    assert isinstance(agent, PPOAgent)
    # device must be a concrete device string, not 'auto'
    assert agent._device in ("cpu", "cuda", "mps")
