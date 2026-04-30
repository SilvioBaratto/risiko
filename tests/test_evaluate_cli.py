"""Tests for the risiko_rl evaluate CLI command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from risiko_rl.cli import app
from src.agents.llm_opponent import LLMOpponent

runner = CliRunner()


def _fake_call(self, snapshot):
    from baml_client import types as baml_types

    return baml_types.RisikoAction(
        action_type="skip",
        param_a=0,
        param_b=0,
        param_c=0,
        param_d=0,
    )


class TestEvaluateCommand:
    """Evaluate command runs head-to-head games and prints stats."""

    def test_evaluate_random_vs_random(self) -> None:
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--agent-a",
                "random",
                "--agent-b",
                "random",
                "--n-games",
                "3",
                "--n-players",
                "2",
                "--seed",
                "42",
                "--max-turns",
                "20",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Win rate" in result.output

    def test_evaluate_llm_vs_random(self, monkeypatch) -> None:
        monkeypatch.setattr(
            LLMOpponent,
            "_call_with_timeout",
            _fake_call,
        )

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--agent-a",
                "llm",
                "--agent-b",
                "random",
                "--n-games",
                "2",
                "--n-players",
                "2",
                "--seed",
                "42",
                "--max-turns",
                "10",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Win rate" in result.output

    def test_evaluate_with_output_csv(self, tmp_path: Path) -> None:
        out_path = tmp_path / "results.csv"
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--agent-a",
                "random",
                "--agent-b",
                "random",
                "--n-games",
                "2",
                "--n-players",
                "2",
                "--seed",
                "42",
                "--max-turns",
                "10",
                "--output",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        content = out_path.read_text()
        assert "win_rate_a" in content

    def test_evaluate_three_players(self) -> None:
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--agent-a",
                "random",
                "--agent-b",
                "random",
                "--n-games",
                "2",
                "--n-players",
                "3",
                "--seed",
                "42",
                "--max-turns",
                "10",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Win rate" in result.output

    def test_evaluate_invalid_checkpoint_raises(self) -> None:
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--agent-a",
                "/nonexistent/checkpoint.pt",
                "--agent-b",
                "random",
                "--n-games",
                "1",
            ],
        )
        assert result.exit_code != 0
        assert "Checkpoint not found" in str(result.exception)

    def test_evaluate_summary_table(self) -> None:
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--agent-a",
                "random",
                "--agent-b",
                "random",
                "--n-games",
                "3",
                "--seed",
                "42",
                "--max-turns",
                "10",
            ],
        )
        assert result.exit_code == 0
        assert "Win rate" in result.output
        assert "Draw rate" in result.output
        assert "Avg length" in result.output
        assert "CI" in result.output
