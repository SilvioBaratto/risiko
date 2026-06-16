"""Typer CLI entry point for risiko-rl."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

import typer

# training/ lives at the project root but is not an installed package.
# Insert the root so it is importable regardless of cwd.
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import TrainingConfig, load_config, merge_cli_overrides  # noqa: E402
from src.utils.env import ensure_env_loaded  # noqa: E402
from src.utils.log import setup_logging  # noqa: E402

from .agent_loader import load_agent  # noqa: E402, I001

ensure_env_loaded()

app = typer.Typer(help="Risiko RL — train and evaluate PPO agents")


def _preflight_validate_model(cfg: TrainingConfig) -> None:
    """Fail fast if an LLM opponent model is not available on the Ollama server.

    Only runs in LLM mode (``self_play.llm_profiles_path`` set). Resolves the
    effective model set — the ``--model`` override if given, else the distinct
    models declared in the profile YAML — and checks each against the live
    model list. If the server can't be queried, logs a warning and proceeds
    (cloud ``/v1/models`` may differ); only a definitively-missing model aborts.
    """
    from src.agents.ollama_client import list_ollama_models
    from src.agents.player_config import load_profiles_from_yaml

    profiles_path = cfg.self_play.llm_profiles_path
    if not profiles_path:
        return

    if cfg.self_play.llm_model:
        wanted = {cfg.self_play.llm_model}
    else:
        wanted = {p.model for p in load_profiles_from_yaml(profiles_path)}

    available = list_ollama_models()
    if available is None:
        typer.echo("Warning: could not query Ollama model list; skipping model pre-flight check.")
        return

    missing = sorted(wanted - available)
    if missing:
        raise typer.BadParameter(
            f"Model(s) not available on Ollama server: {', '.join(missing)}. "
            f"Available: {', '.join(sorted(available))}. "
            "Pull the model (`ollama pull <name>`) or pass a valid --model."
        )


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
    checkpoint_dir: Annotated[
        Path,
        typer.Option("--checkpoint-dir", help="Checkpoint save directory"),
    ] = Path("models"),
    log_dir: Annotated[
        Path | None,
        typer.Option("--log-dir", help="TensorBoard log directory"),
    ] = None,
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
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Ollama model for all LLM opponents (overrides YAML profiles)",
        ),
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
    setup_logging()
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
    if model is not None:
        overrides["self_play.llm_model"] = model
    if override:
        for o in override:
            if "=" not in o:
                raise typer.BadParameter(f"Override must be key=value, got: {o}")
            key, value = o.split("=", 1)
            overrides[key] = value

    cfg = merge_cli_overrides(cfg, overrides)
    _preflight_validate_model(cfg)
    typer.echo(f"Training with config: {cfg}")

    from training.self_play import SelfPlayTrainer

    trainer = SelfPlayTrainer(cfg, checkpoint_dir=checkpoint_dir, log_dir=log_dir)
    trainer.train()
    typer.echo(f"Training complete. Episodes: {trainer._episode}")


@app.command()
def watch(
    agent1: str = typer.Option("random", "--agent1", help="Agent spec for player 1"),
    agent2: str = typer.Option("random", "--agent2", help="Agent spec for player 2"),
    seed: int = typer.Option(42, "--seed", help="Game RNG seed"),
    max_turns: int = typer.Option(10_000, "--max-turns", help="Turn cap per game"),
    mode: str = typer.Option("ascii", "--mode", help="Rendering mode: ascii or matplotlib"),
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Export replay to file"),
    ] = None,
):
    """Watch a single game between two agents, rendered frame-by-frame."""
    setup_logging()
    from src.env import RisikoEnv
    from src.multi_agent import MultiAgentRunner
    from visualization.render_game import ReplayExporter, render_ascii, render_matplotlib

    if mode not in {"ascii", "matplotlib"}:
        raise typer.BadParameter(f"Invalid mode: {mode}. Choose 'ascii' or 'matplotlib'.")

    env = RisikoEnv(n_players=2)
    agents = [load_agent(agent1), load_agent(agent2)]
    runner = MultiAgentRunner(env, agents, max_turns=max_turns)
    result = runner.run_game(seed=seed)

    exporter = ReplayExporter() if output else None

    for t in result.trajectories:
        if mode == "ascii":
            typer.echo(render_ascii(t.obs))
            typer.echo("-" * 40)
        elif mode == "matplotlib":
            fig = render_matplotlib(t.obs)
            # In CLI watch mode, matplotlib figures are not shown interactively
            import matplotlib.pyplot as plt

            plt.close(fig)
        if exporter is not None:
            exporter.add_frame(t.obs)

    winner_text = f"Player {result.winner}" if result.winner is not None else "Draw"
    typer.echo(f"Game over. Winner: {winner_text} (turns: {result.n_turns})")

    if exporter is not None and output is not None:
        if output.suffix == ".gif":
            exporter.export_gif(output)
        else:
            exporter.export_gif(output.with_suffix(".gif"))
        typer.echo(f"Replay saved to {output}")


@app.command()
def evaluate(
    agent_a: str = typer.Option("random", "--agent-a", help="Agent spec for player A"),
    agent_b: str = typer.Option("random", "--agent-b", help="Agent spec for player B"),
    n_games: int = typer.Option(100, "--n-games", help="Number of games to play"),
    n_players: int = typer.Option(2, "--n-players", help="Total players (2+ fills with random)"),
    seed: int = typer.Option(42, "--seed", help="Evaluation RNG seed"),
    max_turns: int = typer.Option(10_000, "--max-turns", help="Turn cap per game"),
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Export CSV summary"),
    ] = None,
):
    """Run a head-to-head evaluation and print win-rate statistics."""
    setup_logging()
    from training.evaluate import evaluate_agents

    agent_a_obj = load_agent(agent_a)
    agent_b_obj = load_agent(agent_b)

    typer.echo(f"Evaluating {agent_a} vs {agent_b} over {n_games} games ...")
    result = evaluate_agents(
        agent_a_obj,
        agent_b_obj,
        n_games=n_games,
        n_players=n_players,
        seed=seed,
        max_turns=max_turns,
    )

    typer.echo("")
    typer.echo(f"{'Metric':<20} {'Agent A':<15} {'Agent B':<15}")
    typer.echo("-" * 50)
    typer.echo(f"{'Win rate':<20} {result.win_rate_a:<15.3f} {result.win_rate_b:<15.3f}")
    typer.echo(f"{'Draw rate':<20} {result.draw_rate:<15.3f}")
    typer.echo(f"{'Avg length':<20} {result.avg_length:<15.1f}")
    typer.echo(f"{'CI (A)':<20} [{result.ci_a[0]:.3f}, {result.ci_a[1]:.3f}]")
    typer.echo(f"{'CI (B)':<20} [{result.ci_b[0]:.3f}, {result.ci_b[1]:.3f}]")

    if output is not None:
        result.to_csv(output)
        typer.echo(f"\nCSV saved to {output}")


@app.command()
def benchmark(
    games: int = typer.Option(50, "--games", help="Games per matchup"),
    n_players: int = typer.Option(2, "--n-players", help="Total players per game"),
    seed: int = typer.Option(42, "--seed", help="RNG seed"),
    max_turns: int = typer.Option(10_000, "--max-turns", help="Turn cap per game"),
    rl_checkpoint: Annotated[
        Path | None,
        typer.Option("--rl-checkpoint", help="Optional RL checkpoint path"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Export CSV summary"),
    ] = None,
):
    """Run Monte Carlo baselines: random vs random, random vs LLM, and optionally RL vs LLM."""
    from training.monte_carlo import run_baseline_tournament

    matchups: list[tuple[str, Any, Any]] = [
        ("random_vs_random", load_agent("random"), load_agent("random")),
        ("random_vs_llm", load_agent("random"), load_agent("llm")),
    ]
    if rl_checkpoint is not None:
        matchups.append(("rl_vs_llm", load_agent(str(rl_checkpoint)), load_agent("llm")))

    typer.echo(f"Running {len(matchups)} matchups, {games} games each ...")
    results = run_baseline_tournament(
        matchups,
        n_games=games,
        n_players=n_players,
        seed=seed,
        max_turns=max_turns,
    )

    typer.echo("")
    typer.echo(
        f"{'Matchup':<20} {'Win L':<8} {'Win R':<8} {'Draw':<8} "
        f"{'Avg len':<10} {'CI L':<20} {'CI R':<20}"
    )
    typer.echo("-" * 100)
    for r in results.values():
        ci_l = f"[{r.ci_left[0]:.3f}, {r.ci_left[1]:.3f}]"
        ci_r = f"[{r.ci_right[0]:.3f}, {r.ci_right[1]:.3f}]"
        typer.echo(
            f"{r.matchup:<20} {r.win_rate_left:<8.3f} {r.win_rate_right:<8.3f} "
            f"{r.draw_rate:<8.3f} {r.avg_length:<10.1f} {ci_l:<20} {ci_r:<20}"
        )

    if output is not None:
        from training.monte_carlo import BaselineResult

        path = output if output.suffix == ".csv" else output.with_suffix(".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        BaselineResult.to_csv_all(results, path)
        typer.echo(f"\nCSV saved to {path}")


def main():
    """Run the risiko-rl CLI."""
    app()


if __name__ == "__main__":
    main()
