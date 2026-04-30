"""Tests for the risiko_rl watch command and agent loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from risiko_rl.agent_loader import load_agent
from risiko_rl.cli import app

runner = CliRunner()


class TestLoadAgent:
    """Agent loader resolves checkpoint paths and string tokens."""

    def test_random_token_returns_random_agent(self) -> None:
        agent = load_agent("random")
        from src.agents.random_agent import RandomAgent

        assert isinstance(agent, RandomAgent)

    def test_random_token_with_seed(self) -> None:
        agent = load_agent("random", seed=42)
        from src.agents.random_agent import RandomAgent

        assert isinstance(agent, RandomAgent)

    def test_llm_token_returns_agent(self) -> None:
        agent = load_agent("llm")
        assert agent is not None

    def test_nonexistent_checkpoint_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_agent("/nonexistent/checkpoint.pt")


class TestWatchCommand:
    """Watch command renders a single game frame-by-frame."""

    def test_watch_random_vs_random_runs(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "watch",
                "--agent1",
                "random",
                "--agent2",
                "random",
                "--seed",
                "42",
                "--max-turns",
                "5",
                "--mode",
                "ascii",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_watch_with_output_flag(self, tmp_path: Path) -> None:
        out_path = tmp_path / "replay.gif"
        result = runner.invoke(
            app,
            [
                "watch",
                "--agent1",
                "random",
                "--agent2",
                "random",
                "--seed",
                "42",
                "--max-turns",
                "5",
                "--mode",
                "matplotlib",
                "--output",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()

    def test_watch_matplotlib_mode(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "watch",
                "--agent1",
                "random",
                "--agent2",
                "random",
                "--seed",
                "42",
                "--max-turns",
                "3",
                "--mode",
                "matplotlib",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_watch_invalid_mode_raises(self) -> None:
        result = runner.invoke(
            app,
            [
                "watch",
                "--agent1",
                "random",
                "--agent2",
                "random",
                "--mode",
                "invalid",
            ],
        )
        assert result.exit_code != 0
