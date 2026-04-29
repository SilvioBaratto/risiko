"""Typer CLI entry point for risiko-rl."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from src.config import load_config, merge_cli_overrides, resolve_device
from src.models.actor_critic import ActorCritic
from src.models.ppo import PPOTrainer
from src.models.utils import get_obs_dim

_ACTION_DIMS = {
    "action_type": 6,
    "param_a": 42,
    "param_b": 42,
    "param_c": 43,
    "param_d": 43,
}

app = typer.Typer(help="Risiko RL — train and evaluate PPO agents")


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML config file",
        ),
    ] = Path("config/default.yaml"),
    lr: Annotated[
        float | None,
        typer.Option("--lr", help="Learning rate (overrides config)"),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed (overrides config)"),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Device: auto, cpu, or cuda (overrides config)"),
    ] = None,
    timesteps: Annotated[
        int | None,
        typer.Option("--timesteps", help="Total training timesteps (overrides config)"),
    ] = None,
    override: Annotated[
        list[str] | None,
        typer.Option(
            "--override",
            "-o",
            help="Generic override as key=value (e.g. ppo.gamma=0.97)",
        ),
    ] = None,
):
    """Train a PPO agent on the Risiko environment."""
    cfg = load_config(config)
    overrides: dict[str, str] = {}
    if lr is not None:
        overrides["ppo.lr"] = str(lr)
    if seed is not None:
        overrides["seed"] = str(seed)
    if device is not None:
        overrides["device"] = device
    if timesteps is not None:
        overrides["total_timesteps"] = str(timesteps)
    if override:
        for o in override:
            if "=" not in o:
                raise typer.BadParameter(f"Override must be key=value, got: {o}")
            key, value = o.split("=", 1)
            overrides[key] = value

    cfg = merge_cli_overrides(cfg, overrides)
    device_str = resolve_device(cfg.device)
    typer.echo(f"Training with config: {cfg}")
    typer.echo(f"Resolved device: {device_str}")

    net = ActorCritic(
        obs_dim=get_obs_dim(),
        hidden_size=cfg.network.hidden_sizes[0],
        num_layers=len(cfg.network.hidden_sizes),
        action_dims=_ACTION_DIMS,
    ).to(device_str)
    trainer = PPOTrainer(net, cfg.ppo)
    typer.echo(f"Model has {sum(p.numel() for p in net.parameters())} parameters")
    typer.echo(f"PPOTrainer ready: {trainer}")


def main():
    """Run the risiko-rl CLI."""
    app()


if __name__ == "__main__":
    main()
