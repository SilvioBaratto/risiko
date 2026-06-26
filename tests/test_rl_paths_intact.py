"""Verification tests for issue #132.

Criterion: the `tournament` command + `training/tournament_cli.py` did not
disturb the existing RL paths: `train`, `pretrain`, and `evaluate` remain
importable, their CLI surfaces unchanged, and `training.tournament_cli` and
`risiko_rl.cli` leave core RL modules intact.

Pattern mirrors `test_when_tournament_imported_then_*` in
`tests/test_122_tournament_orchestration.py`.
"""

from __future__ import annotations

from typer.testing import CliRunner

_runner = CliRunner()


def _app():
    from risiko_rl.cli import app  # noqa: PLC0415

    return app


# ---------------------------------------------------------------------------
# CliRunner --help regression: train / pretrain / evaluate
# ---------------------------------------------------------------------------


def test_when_train_help_invoked_then_exit_code_is_0() -> None:
    """`train --help` must exit 0 (unchanged after tournament addition)."""
    result = _runner.invoke(_app(), ["train", "--help"])
    assert result.exit_code == 0


def test_when_pretrain_help_invoked_then_exit_code_is_0() -> None:
    """`pretrain --help` must exit 0 (unchanged after tournament addition)."""
    result = _runner.invoke(_app(), ["pretrain", "--help"])
    assert result.exit_code == 0


def test_when_evaluate_help_invoked_then_exit_code_is_0() -> None:
    """`evaluate --help` must exit 0 (unchanged after tournament addition)."""
    result = _runner.invoke(_app(), ["evaluate", "--help"])
    assert result.exit_code == 0


def test_when_train_help_invoked_then_config_option_listed() -> None:
    result = _runner.invoke(_app(), ["train", "--help"])
    assert "--config" in result.output


def test_when_train_help_invoked_then_lr_option_listed() -> None:
    result = _runner.invoke(_app(), ["train", "--help"])
    assert "--lr" in result.output


def test_when_train_help_invoked_then_seed_option_listed() -> None:
    result = _runner.invoke(_app(), ["train", "--help"])
    assert "--seed" in result.output


def test_when_train_help_invoked_then_override_option_listed() -> None:
    result = _runner.invoke(_app(), ["train", "--help"])
    assert "--override" in result.output


def test_when_pretrain_help_invoked_then_config_option_listed() -> None:
    result = _runner.invoke(_app(), ["pretrain", "--help"])
    assert "--config" in result.output


def test_when_pretrain_help_invoked_then_override_option_listed() -> None:
    result = _runner.invoke(_app(), ["pretrain", "--help"])
    assert "--override" in result.output


def test_when_evaluate_help_invoked_then_agent_a_option_listed() -> None:
    result = _runner.invoke(_app(), ["evaluate", "--help"])
    assert "--agent-a" in result.output


def test_when_evaluate_help_invoked_then_agent_b_option_listed() -> None:
    result = _runner.invoke(_app(), ["evaluate", "--help"])
    assert "--agent-b" in result.output


# ---------------------------------------------------------------------------
# App command surface: pre-existing commands + tournament all present
# ---------------------------------------------------------------------------

_EXPECTED_COMMANDS = frozenset(
    ["train", "pretrain", "evaluate", "watch", "social_eval", "analyze", "benchmark", "tournament"]
)


def _registered_function_names() -> set[str]:
    """Typer stores commands with name=None; derive name from callback.__name__."""
    return {cmd.callback.__name__ for cmd in _app().registered_commands if cmd.callback is not None}


def test_when_app_inspected_then_all_expected_commands_are_present() -> None:
    """All pre-existing commands and the new tournament command must be registered."""
    registered = _registered_function_names()
    missing = _EXPECTED_COMMANDS - registered
    assert not missing, f"Commands missing from risiko_rl.cli.app: {missing}"


def test_when_app_inspected_then_tournament_command_is_present() -> None:
    assert "tournament" in _registered_function_names()


# ---------------------------------------------------------------------------
# Import guards: tournament_cli + risiko_rl.cli must not break RL modules
# Mirrors the pattern in test_122_tournament_orchestration.py
# ---------------------------------------------------------------------------


def test_when_tournament_cli_imported_then_actor_critic_is_still_importable() -> None:
    """training.tournament_cli must not break src.models.actor_critic.ActorCritic."""
    import src.models.actor_critic  # noqa: F401
    import training.tournament_cli  # noqa: F401

    assert hasattr(src.models.actor_critic, "ActorCritic")


def test_when_tournament_cli_imported_then_ppo_trainer_is_still_importable() -> None:
    """training.tournament_cli must not break src.models.ppo.PPOTrainer."""
    import src.models.ppo  # noqa: F401
    import training.tournament_cli  # noqa: F401

    assert hasattr(src.models.ppo, "PPOTrainer")


def test_when_tournament_cli_imported_then_training_config_is_still_loadable() -> None:
    """training.tournament_cli must not break src.config.TrainingConfig."""
    import training.tournament_cli  # noqa: F401
    from src.config import TrainingConfig  # noqa: F401

    assert TrainingConfig is not None


def test_when_tournament_cli_imported_then_checkpoint_module_is_intact() -> None:
    """training.tournament_cli must not break src.checkpoint."""
    import src.checkpoint  # noqa: F401
    import training.tournament_cli  # noqa: F401

    assert hasattr(src.checkpoint, "save_checkpoint")
    assert hasattr(src.checkpoint, "load_checkpoint")


def test_when_risiko_rl_cli_imported_then_actor_critic_is_still_importable() -> None:
    """risiko_rl.cli must not break src.models.actor_critic.ActorCritic."""
    import risiko_rl.cli  # noqa: F401
    import src.models.actor_critic  # noqa: F401

    assert hasattr(src.models.actor_critic, "ActorCritic")


def test_when_risiko_rl_cli_imported_then_ppo_trainer_is_still_importable() -> None:
    """risiko_rl.cli must not break src.models.ppo.PPOTrainer."""
    import risiko_rl.cli  # noqa: F401
    import src.models.ppo  # noqa: F401

    assert hasattr(src.models.ppo, "PPOTrainer")


def test_when_risiko_rl_cli_imported_then_training_config_is_still_loadable() -> None:
    """risiko_rl.cli must not break src.config.TrainingConfig."""
    import risiko_rl.cli  # noqa: F401
    from src.config import TrainingConfig  # noqa: F401

    assert TrainingConfig is not None


def test_when_risiko_rl_cli_imported_then_checkpoint_module_is_intact() -> None:
    """risiko_rl.cli must not break src.checkpoint."""
    import risiko_rl.cli  # noqa: F401
    import src.checkpoint  # noqa: F401

    assert hasattr(src.checkpoint, "save_checkpoint")
    assert hasattr(src.checkpoint, "load_checkpoint")
