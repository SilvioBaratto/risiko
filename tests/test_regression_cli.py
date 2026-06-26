"""Source-blind regression tests for issue #132.

Acceptance criteria pinned here:
- `train --help`, `pretrain --help`, `evaluate --help` each exit 0 and list
  their original options (regression via CliRunner).
- `risiko_rl.cli.app` exposes all pre-existing commands alongside `tournament`.
- Importing `training.tournament_cli` and `risiko_rl.cli` does not break
  `src.models.actor_critic`, `src.models.ppo`, `src.config.TrainingConfig`,
  or `src.checkpoint`.
- `ruff check . tests` exits 0 (clean).
- Games execute concurrently: `max_concurrency` is configurable on
  `TournamentConfig` (property test: any positive integer is accepted).

No implementation source was read while authoring these tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Constants derived from acceptance criteria (CLAUDE.md / requirements.md)
# ---------------------------------------------------------------------------

#: All commands that must exist BEFORE the tournament extension landed.
#: Keys are the Python function names; values are the CLI names (Typer converts
#: underscores to hyphens, so social_eval → social-eval in the CLI).
_PRE_EXISTING_COMMANDS: frozenset[str] = frozenset(
    ["train", "pretrain", "evaluate", "watch", "social_eval", "analyze", "benchmark"]
)

#: CLI names (as shown in --help) for pre-existing commands.
#: Typer converts function-name underscores to hyphens.
_PRE_EXISTING_CLI_NAMES: frozenset[str] = frozenset(
    ["train", "pretrain", "evaluate", "watch", "social-eval", "analyze", "benchmark"]
)

#: The new command introduced in this cycle.
_NEW_COMMAND = "tournament"

#: Known options that `train --help` MUST still list (derived from CLAUDE.md).
_TRAIN_REQUIRED_OPTIONS = ["--config", "--lr", "--seed", "--override"]

#: Known options that `pretrain --help` MUST still list (derived from CLAUDE.md).
_PRETRAIN_REQUIRED_OPTIONS = ["--config", "--override"]

#: Known options that `evaluate --help` MUST still list.
#: evaluate uses --agent-a / --agent-b (not --checkpoint) per the current CLI.
_EVALUATE_REQUIRED_OPTIONS = ["--agent-a", "--agent-b"]

_runner = CliRunner()


def _app():
    from risiko_rl.cli import app  # noqa: PLC0415

    return app


def _registered_function_names() -> set[str]:
    """Return the Python function names of all registered Typer commands.

    Typer's CommandInfo.name is None when the command is registered without an
    explicit name override — the actual name derives from callback.__name__.
    """
    return {cmd.callback.__name__ for cmd in _app().registered_commands if cmd.callback is not None}


# ===========================================================================
# Scope criterion: train / pretrain / evaluate --help exit 0 + options intact
# ===========================================================================


class TestHelpRegressionExitCode:
    """Each pre-existing command's --help exits 0 (regression gate)."""

    def test_when_train_help_invoked_then_exit_code_is_0(self) -> None:
        result = _runner.invoke(_app(), ["train", "--help"])
        assert result.exit_code == 0, (
            f"`train --help` exited {result.exit_code}:\n{result.output[:400]}"
        )

    def test_when_pretrain_help_invoked_then_exit_code_is_0(self) -> None:
        result = _runner.invoke(_app(), ["pretrain", "--help"])
        assert result.exit_code == 0, (
            f"`pretrain --help` exited {result.exit_code}:\n{result.output[:400]}"
        )

    def test_when_evaluate_help_invoked_then_exit_code_is_0(self) -> None:
        result = _runner.invoke(_app(), ["evaluate", "--help"])
        assert result.exit_code == 0, (
            f"`evaluate --help` exited {result.exit_code}:\n{result.output[:400]}"
        )


class TestHelpRegressionOptions:
    """Each pre-existing command's --help still lists all original options."""

    def test_when_train_help_invoked_then_config_option_is_listed(self) -> None:
        result = _runner.invoke(_app(), ["train", "--help"])
        assert "--config" in result.output

    def test_when_train_help_invoked_then_lr_option_is_listed(self) -> None:
        result = _runner.invoke(_app(), ["train", "--help"])
        assert "--lr" in result.output

    def test_when_train_help_invoked_then_seed_option_is_listed(self) -> None:
        result = _runner.invoke(_app(), ["train", "--help"])
        assert "--seed" in result.output

    def test_when_train_help_invoked_then_override_option_is_listed(self) -> None:
        result = _runner.invoke(_app(), ["train", "--help"])
        assert "--override" in result.output

    def test_when_pretrain_help_invoked_then_config_option_is_listed(self) -> None:
        result = _runner.invoke(_app(), ["pretrain", "--help"])
        assert "--config" in result.output

    def test_when_pretrain_help_invoked_then_override_option_is_listed(self) -> None:
        result = _runner.invoke(_app(), ["pretrain", "--help"])
        assert "--override" in result.output

    def test_when_evaluate_help_invoked_then_agent_a_option_is_listed(self) -> None:
        result = _runner.invoke(_app(), ["evaluate", "--help"])
        assert "--agent-a" in result.output

    def test_when_evaluate_help_invoked_then_agent_b_option_is_listed(self) -> None:
        result = _runner.invoke(_app(), ["evaluate", "--help"])
        assert "--agent-b" in result.output


# Property: for every pre-existing CLI command name, --help exits 0
@given(
    command=st.sampled_from(sorted(_PRE_EXISTING_CLI_NAMES)),
)
def test_when_any_preexisting_command_help_invoked_then_exit_code_is_0(
    command: str,
) -> None:
    """Ordering invariant: no pre-existing command may have a broken --help."""
    result = _runner.invoke(_app(), [command, "--help"])
    assert result.exit_code == 0, (
        f"`{command} --help` exited {result.exit_code}:\n{result.output[:400]}"
    )


# ===========================================================================
# Scope criterion: app exposes all pre-existing commands + tournament
# ===========================================================================


class TestCommandRegistry:
    """risiko_rl.cli.app must expose all 8 pre-existing commands + tournament."""

    def test_when_app_inspected_then_train_is_registered(self) -> None:
        assert "train" in _registered_function_names()

    def test_when_app_inspected_then_pretrain_is_registered(self) -> None:
        assert "pretrain" in _registered_function_names()

    def test_when_app_inspected_then_evaluate_is_registered(self) -> None:
        assert "evaluate" in _registered_function_names()

    def test_when_app_inspected_then_watch_is_registered(self) -> None:
        assert "watch" in _registered_function_names()

    def test_when_app_inspected_then_social_eval_is_registered(self) -> None:
        assert "social_eval" in _registered_function_names()

    def test_when_app_inspected_then_analyze_is_registered(self) -> None:
        assert "analyze" in _registered_function_names()

    def test_when_app_inspected_then_benchmark_is_registered(self) -> None:
        assert "benchmark" in _registered_function_names()

    def test_when_app_inspected_then_tournament_is_registered(self) -> None:
        assert "tournament" in _registered_function_names()

    def test_when_app_inspected_then_all_preexisting_commands_are_present(self) -> None:
        """Single aggregate check: all pre-existing commands survive the extension."""
        registered = _registered_function_names()
        missing = _PRE_EXISTING_COMMANDS - registered
        assert not missing, f"Commands removed from app: {missing}"


# ===========================================================================
# Scope criterion: importing tournament_cli / cli does not break existing modules
# ===========================================================================


class TestImportGuards:
    """Importing tournament_cli or risiko_rl.cli must not break existing modules."""

    def test_when_tournament_cli_imported_then_actor_critic_is_importable(self) -> None:
        import src.models.actor_critic  # noqa: F401, PLC0415
        import training.tournament_cli  # noqa: F401, PLC0415

    def test_when_tournament_cli_imported_then_ppo_is_importable(self) -> None:
        import src.models.ppo  # noqa: F401, PLC0415
        import training.tournament_cli  # noqa: F401, PLC0415

    def test_when_tournament_cli_imported_then_training_config_is_importable(self) -> None:
        import training.tournament_cli  # noqa: F401, PLC0415
        from src.config import TrainingConfig  # noqa: F401, PLC0415

    def test_when_tournament_cli_imported_then_checkpoint_is_importable(self) -> None:
        import src.checkpoint  # noqa: F401, PLC0415
        import training.tournament_cli  # noqa: F401, PLC0415

    def test_when_risiko_rl_cli_imported_then_actor_critic_is_importable(self) -> None:
        import risiko_rl.cli  # noqa: F401, PLC0415
        import src.models.actor_critic  # noqa: F401, PLC0415

    def test_when_risiko_rl_cli_imported_then_ppo_is_importable(self) -> None:
        import risiko_rl.cli  # noqa: F401, PLC0415
        import src.models.ppo  # noqa: F401, PLC0415

    def test_when_risiko_rl_cli_imported_then_training_config_is_importable(self) -> None:
        import risiko_rl.cli  # noqa: F401, PLC0415
        from src.config import TrainingConfig  # noqa: F401, PLC0415

    def test_when_risiko_rl_cli_imported_then_checkpoint_is_importable(self) -> None:
        import risiko_rl.cli  # noqa: F401, PLC0415
        import src.checkpoint  # noqa: F401, PLC0415


# ===========================================================================
# Scope criterion: ruff check . tests is clean
# ===========================================================================


class TestRuffGate:
    """ruff check . tests must exit 0 (no lint errors in source or test tree)."""

    def test_when_ruff_check_run_then_exit_code_is_0(self) -> None:
        """Criterion: `ruff check . tests` is clean."""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", ".", "tests"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0, (
            f"ruff check exited {result.returncode}.\n"
            f"stdout:\n{result.stdout[:1000]}\n"
            f"stderr:\n{result.stderr[:1000]}"
        )


# ===========================================================================
# Global criterion: max_concurrency is configurable (concurrency acceptance)
# ===========================================================================


@given(max_concurrency=st.integers(min_value=1, max_value=32))
def test_when_any_positive_max_concurrency_given_then_tournament_config_accepts_it(
    max_concurrency: int,
) -> None:
    """Never-raises invariant: TournamentConfig must accept any max_concurrency ≥ 1.

    Derived from criterion: 'Games execute concurrently (configurable
    max_concurrency)'. The invariant is that the configuration layer never
    rejects a valid concurrency value — raising here would prevent the user
    from controlling concurrency at all.
    """
    from training.tournament import TournamentConfig  # noqa: PLC0415

    strategies = (
        "australia_lock",
        "south_america_lock",
        "aggressive_blitz",
        "turtle_defensive",
        "card_cycle_hunter",
        "diplomat_coalition",
    )
    models = ("m1", "m2", "m3", "m4", "m5", "m6")

    cfg = TournamentConfig(
        name="concurrency_test",
        seed=0,
        n_games=1,
        max_concurrency=max_concurrency,
        n_rounds=1,
        strategies=strategies,
        models=models,
        temperature=0.7,
        top_p=0.9,
    )
    assert cfg.max_concurrency == max_concurrency
