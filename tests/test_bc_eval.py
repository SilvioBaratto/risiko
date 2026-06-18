"""Tests for training/bc_eval.py and risiko_rl/agent_loader.py — Issue #87."""

from __future__ import annotations

import dataclasses
import inspect
from unittest.mock import MagicMock, patch

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Helpers — fixtures built from spec and actual APIs
# ---------------------------------------------------------------------------


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


def _run_stubbed(ci=(0.10, 0.35), gate=0.16, wins=10, n_games=50):
    """Call evaluate_pretrained with all side-effects patched out."""
    from training.bc_eval import evaluate_pretrained  # noqa: PLC0415

    mock_result = MagicMock()
    mock_result.win_rate_a = wins / n_games
    mock_result.ci_a = ci

    with (
        patch("training.bc_eval.load_agent", return_value=MagicMock()),
        patch("training.bc_eval.evaluate_agents", return_value=mock_result),
    ):
        return evaluate_pretrained("fake.pt", n_games=n_games, seed=0, max_turns=100, gate=gate)


# ---------------------------------------------------------------------------
# Criterion: load_agent() loads a real BC/PPO checkpoint into a PPOAgent
# ---------------------------------------------------------------------------


def test_when_bc_checkpoint_written_via_write_bc_checkpoint_then_load_agent_returns_ppo_agent(
    tmp_path,
):
    """Regression: write_bc_checkpoint → load_agent(path) → PPOAgent instance."""
    from risiko_rl.agent_loader import load_agent  # noqa: PLC0415
    from src.agents.ppo_agent import PPOAgent  # noqa: PLC0415

    ckpt = tmp_path / "pretrained.pt"
    _write_minimal_bc_checkpoint(ckpt)

    agent = load_agent(str(ckpt))

    assert isinstance(agent, PPOAgent)


# ---------------------------------------------------------------------------
# Criterion: evaluate_pretrained entry-point signature
# ---------------------------------------------------------------------------


def test_when_evaluate_pretrained_signature_inspected_then_n_players_defaults_to_6():
    from training.bc_eval import evaluate_pretrained  # noqa: PLC0415

    sig = inspect.signature(evaluate_pretrained)
    assert sig.parameters["n_players"].default == 6


def test_when_evaluate_pretrained_signature_inspected_then_gate_defaults_to_0_16():
    from training.bc_eval import evaluate_pretrained  # noqa: PLC0415

    sig = inspect.signature(evaluate_pretrained)
    assert sig.parameters["gate"].default == pytest.approx(0.16)


def test_when_evaluate_pretrained_signature_inspected_then_checkpoint_path_is_first_positional():
    from training.bc_eval import evaluate_pretrained  # noqa: PLC0415

    sig = inspect.signature(evaluate_pretrained)
    assert list(sig.parameters.keys())[0] == "checkpoint_path"


# ---------------------------------------------------------------------------
# Criterion: result carries win_rate, ci, n_games, baseline, passed
# ---------------------------------------------------------------------------


def test_when_evaluate_pretrained_called_then_result_has_win_rate():
    assert hasattr(_run_stubbed(), "win_rate")


def test_when_evaluate_pretrained_called_then_result_has_ci():
    assert hasattr(_run_stubbed(), "ci")


def test_when_evaluate_pretrained_called_then_result_has_n_games():
    assert hasattr(_run_stubbed(), "n_games")


def test_when_evaluate_pretrained_called_then_result_has_baseline():
    assert hasattr(_run_stubbed(), "baseline")


def test_when_evaluate_pretrained_called_then_result_has_passed():
    assert hasattr(_run_stubbed(), "passed")


def test_when_evaluate_pretrained_called_then_result_is_a_frozen_dataclass():
    result = _run_stubbed()
    assert dataclasses.is_dataclass(result)
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        result.passed = not result.passed  # type: ignore[misc]


def test_when_evaluate_pretrained_called_then_it_does_not_raise():
    assert _run_stubbed() is not None


# ---------------------------------------------------------------------------
# Criterion: passed = ci_lower_bound > gate (Wilson lower, not point estimate)
# Stubs evaluate_agents to assert gate logic without playing games.
# ---------------------------------------------------------------------------


def test_when_ci_is_0_30_to_0_45_then_passed_is_true():
    """CI (0.30, 0.45) → lower 0.30 > gate 0.16 → passed = True."""
    assert _run_stubbed(ci=(0.30, 0.45), gate=0.16).passed is True


def test_when_ci_is_0_05_to_0_20_then_passed_is_false():
    """CI (0.05, 0.20) → lower 0.05 < gate 0.16 → passed = False."""
    assert _run_stubbed(ci=(0.05, 0.20), gate=0.16).passed is False


def test_when_ci_lower_equals_gate_exactly_then_passed_is_false():
    """Strict inequality: ci_lower == gate is not sufficient."""
    assert _run_stubbed(ci=(0.16, 0.30), gate=0.16).passed is False


def test_when_gate_is_zero_and_ci_lower_is_positive_then_passed_is_true():
    assert _run_stubbed(ci=(0.01, 0.10), gate=0.0).passed is True


def test_when_point_estimate_above_gate_but_ci_lower_below_then_passed_is_false():
    """Gate uses Wilson LOWER bound, not the point win_rate estimate."""
    assert _run_stubbed(ci=(0.10, 0.40), gate=0.16).passed is False


# ---------------------------------------------------------------------------
# Property: passed == (ci_lower > gate) for all valid inputs
# ---------------------------------------------------------------------------


@given(
    ci_lower=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ci_width=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    gate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_when_any_valid_ci_and_gate_then_passed_equals_ci_lower_strictly_gt_gate(
    ci_lower,
    ci_width,
    gate,
):
    """Invariant: passed == (ci_lower > gate) for every probability triple in [0, 1]."""
    from training.bc_eval import evaluate_pretrained  # noqa: PLC0415

    mock_result = MagicMock()
    mock_result.win_rate_a = 0.2
    mock_result.ci_a = (ci_lower, min(1.0, ci_lower + ci_width))

    with (
        patch("training.bc_eval.load_agent", return_value=MagicMock()),
        patch("training.bc_eval.evaluate_agents", return_value=mock_result),
    ):
        result = evaluate_pretrained("fake.pt", n_games=100, seed=0, max_turns=100, gate=gate)

    assert result.passed == (ci_lower > gate)


# ---------------------------------------------------------------------------
# Criterion: BCConfig is a frozen dataclass (convention compliance)
# ---------------------------------------------------------------------------


def test_when_bcconfig_inspected_then_it_is_frozen():
    from src.config import BCConfig  # noqa: PLC0415

    assert dataclasses.is_dataclass(BCConfig)
    assert BCConfig.__dataclass_params__.frozen


# ---------------------------------------------------------------------------
# Criterion: round-trip — write BC checkpoint → load into SelfPlayTrainer
# ---------------------------------------------------------------------------


def test_when_bc_checkpoint_loaded_into_selfplay_trainer_then_weights_are_preserved(
    tmp_path,
):
    """Pretrain → load into SelfPlayTrainer; architecture must be compatible."""
    from src.config import TrainingConfig  # noqa: PLC0415
    from src.models.actor_critic import ActorCritic  # noqa: PLC0415
    from src.models.utils import get_obs_dim  # noqa: PLC0415
    from src.utils.constants import ACTION_DIMS  # noqa: PLC0415
    from training.bc_checkpoint import write_bc_checkpoint  # noqa: PLC0415
    from training.self_play import SelfPlayTrainer  # noqa: PLC0415

    cfg = TrainingConfig()
    net = ActorCritic(
        obs_dim=get_obs_dim(),
        hidden_size=cfg.network.hidden_sizes[0],
        num_layers=len(cfg.network.hidden_sizes),
        action_dims=ACTION_DIMS,
    )
    opt = torch.optim.Adam(net.parameters(), lr=cfg.bc.lr)
    ckpt = tmp_path / "pretrained.pt"
    write_bc_checkpoint(ckpt, net, opt, cfg)

    # Save a specific parameter value for comparison after load
    param_name = next(iter(net.state_dict()))
    saved_param = net.state_dict()[param_name].clone()

    trainer = SelfPlayTrainer(cfg)
    trainer.load_checkpoint(str(ckpt))

    # Verify the loaded agent's weights match the saved checkpoint (no arch changes)
    loaded_param = trainer._agent._net.state_dict()[param_name]
    assert torch.allclose(saved_param, loaded_param.cpu())


# ---------------------------------------------------------------------------
# Integration test: generate → pretrain → evaluate_pretrained end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_when_tiny_generate_pretrain_pipeline_runs_then_result_returned_without_raising(
    tmp_path,
):
    """Tiny generate → pretrain → evaluate_pretrained (gate=0.0) must return, not raise."""
    import dataclasses as dc  # noqa: PLC0415

    from src.config import BCConfig, TrainingConfig  # noqa: PLC0415
    from training.bc_dataset import generate_bc_dataset  # noqa: PLC0415
    from training.bc_eval import evaluate_pretrained  # noqa: PLC0415
    from training.bc_trainer import pretrain  # noqa: PLC0415

    dataset_dir = str(tmp_path / "data" / "bc")
    output_path = str(tmp_path / "models" / "pretrained.pt")
    bc_cfg = BCConfig(
        n_games=5,
        n_players=2,
        max_turns=20,
        seed=0,
        dataset_dir=dataset_dir,
        shard_size=100,
        output_path=output_path,
    )
    cfg = dc.replace(TrainingConfig(), bc=bc_cfg)

    generate_bc_dataset(cfg)
    pretrain(cfg, output_path=output_path)

    result = evaluate_pretrained(
        checkpoint_path=output_path,
        n_games=3,
        n_players=bc_cfg.n_players,
        seed=bc_cfg.seed,
        max_turns=bc_cfg.max_turns,
        gate=0.0,
    )

    assert result is not None
    assert hasattr(result, "win_rate")
    assert hasattr(result, "passed")
