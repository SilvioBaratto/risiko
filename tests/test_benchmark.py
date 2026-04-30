"""Tests for the risiko_rl benchmark CLI command."""

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


class TestBenchmarkCommand:
    """Benchmark command runs Monte Carlo baseline matchups."""

    def test_benchmark_random_vs_random_runs(self, monkeypatch) -> None:
        monkeypatch.setattr(
            LLMOpponent,
            "_call_with_timeout",
            _fake_call,
        )

        result = runner.invoke(
            app,
            [
                "benchmark",
                "--games",
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
        assert "random_vs_random" in result.output

    def test_benchmark_includes_llm_matchup(self, monkeypatch) -> None:
        monkeypatch.setattr(
            LLMOpponent,
            "_call_with_timeout",
            _fake_call,
        )

        result = runner.invoke(
            app,
            [
                "benchmark",
                "--games",
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
        assert "random_vs_llm" in result.output

    def test_benchmark_summary_table_headers(self, monkeypatch) -> None:
        monkeypatch.setattr(
            LLMOpponent,
            "_call_with_timeout",
            _fake_call,
        )

        result = runner.invoke(
            app,
            [
                "benchmark",
                "--games",
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
        assert "Matchup" in result.output
        assert "Win L" in result.output

    def test_benchmark_with_output_csv(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            LLMOpponent,
            "_call_with_timeout",
            _fake_call,
        )

        out_path = tmp_path / "benchmark.csv"
        result = runner.invoke(
            app,
            [
                "benchmark",
                "--games",
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
        assert "matchup" in content

    def test_benchmark_three_players(self, monkeypatch) -> None:
        monkeypatch.setattr(
            LLMOpponent,
            "_call_with_timeout",
            _fake_call,
        )

        result = runner.invoke(
            app,
            [
                "benchmark",
                "--games",
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
        assert "random_vs_random" in result.output

    def test_benchmark_invalid_checkpoint_raises(self) -> None:
        result = runner.invoke(
            app,
            [
                "benchmark",
                "--games",
                "1",
                "--rl-checkpoint",
                "/nonexistent/checkpoint.pt",
            ],
        )
        assert result.exit_code != 0
        assert "Checkpoint not found" in str(result.exception)
