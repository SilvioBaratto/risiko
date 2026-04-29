"""Typer CLI entry point for risiko-rl."""

import typer

app = typer.Typer(help="Risiko RL — train and evaluate PPO agents")


def main():
    """Run the risiko-rl CLI."""
    app()


if __name__ == "__main__":
    main()
