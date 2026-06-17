"""
Source-blind example tests for training.bc_checkpoint.write_bc_checkpoint (issue #84).

Tests derived exclusively from the acceptance criteria for issue #84:
  "feat: PPO-compatible BC checkpoint writer — SelfPlayTrainer.load_checkpoint round-trip"
No implementation source has been read.  Tests are in the Red phase (TDD) and are expected
to fail until write_bc_checkpoint satisfies each criterion.

Skipped (per oracle — NOT VERIFIABLE):
  - pytest -m "not integration" passes (suite-level gate, no per-criterion assertion)
  - All tests pass (boilerplate suite gate)
  - SOLID, clean code (subjective code-quality prose)
"""

from __future__ import annotations

import inspect
import pathlib
import tempfile

import pytest
import torch
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from src.config import PPOConfig, TrainingConfig
from src.models.actor_critic import ActorCritic
from src.models.ppo import PPOTrainer

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_net(obs_dim: int = 16, hidden_size: int = 8, num_layers: int = 1) -> ActorCritic:
    return ActorCritic(
        obs_dim=obs_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        action_dims={
            "action_type": 6,
            "param_a": 42,
            "param_b": 42,
            "param_c": 42,
            "param_d": 42,
        },
    )


def _make_cfg() -> TrainingConfig:
    return TrainingConfig(total_timesteps=1000, ppo=PPOConfig(lr=1e-3))


def _write_and_load(path: pathlib.Path, net: ActorCritic, cfg: TrainingConfig) -> dict:
    """Write a BC checkpoint then load the raw payload back."""
    from training.bc_checkpoint import write_bc_checkpoint

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    write_bc_checkpoint(path, net, optimizer, cfg)
    return torch.load(path, weights_only=False)


# ── Module accessibility ──────────────────────────────────────────────────────


def test_when_module_imported_then_write_bc_checkpoint_is_accessible():
    """Criterion: training/bc_checkpoint.py exposes write_bc_checkpoint."""
    from training.bc_checkpoint import write_bc_checkpoint  # noqa: F401


def test_when_write_bc_checkpoint_imported_then_signature_has_four_parameters():
    """
    Criterion: write_bc_checkpoint(path, net, optimizer, cfg) — spec names exactly
    these four positional parameters.
    """
    from training.bc_checkpoint import write_bc_checkpoint

    params = list(inspect.signature(write_bc_checkpoint).parameters.keys())
    assert params == ["path", "net", "optimizer", "cfg"], (
        f"Expected signature (path, net, optimizer, cfg), got {params}"
    )


# ── Criterion: file is created at the given path ──────────────────────────────


def test_when_write_bc_checkpoint_called_then_file_is_created(tmp_path):
    """Criterion: write_bc_checkpoint(path, net, optimizer, cfg) writes a file at path."""
    from training.bc_checkpoint import write_bc_checkpoint

    out = tmp_path / "pretrained.pt"
    write_bc_checkpoint(out, _make_net(), torch.optim.Adam(_make_net().parameters()), _make_cfg())
    assert out.exists(), "write_bc_checkpoint must create a file at the given path"


# ── Criterion: parent directory is created (mkdir parents=True, exist_ok=True) ─


def test_when_parent_directory_does_not_exist_then_write_bc_checkpoint_creates_it(tmp_path):
    """
    Criterion: the parent directory is created (mkdir(parents=True, exist_ok=True)).
    Passing a deeply-nested path whose parents do not exist must not raise.
    """
    from training.bc_checkpoint import write_bc_checkpoint

    nested = tmp_path / "deep" / "nested" / "dir" / "pretrained.pt"
    net = _make_net()
    write_bc_checkpoint(nested, net, torch.optim.Adam(net.parameters()), _make_cfg())
    assert nested.exists(), "write_bc_checkpoint must create the full parent directory tree"


# ── Criterion: payload top-level structure ────────────────────────────────────


def test_when_checkpoint_loaded_then_payload_contains_trainer_state_key(tmp_path):
    """Criterion: payload has 'trainer_state' key (mirrors SelfPlayTrainer checkpoint layout)."""
    payload = _write_and_load(tmp_path / "ck.pt", _make_net(), _make_cfg())
    assert "trainer_state" in payload, "payload must contain 'trainer_state'"


def test_when_checkpoint_loaded_then_payload_contains_top_level_config_key(tmp_path):
    """Criterion: payload has top-level 'config' key — the full TrainingConfig."""
    payload = _write_and_load(tmp_path / "ck.pt", _make_net(), _make_cfg())
    assert "config" in payload, "payload must contain top-level 'config'"


def test_when_checkpoint_loaded_then_payload_contains_episode_key(tmp_path):
    """Criterion: payload has 'episode' key."""
    payload = _write_and_load(tmp_path / "ck.pt", _make_net(), _make_cfg())
    assert "episode" in payload, "payload must contain 'episode'"


def test_when_checkpoint_loaded_then_episode_is_zero(tmp_path):
    """
    Criterion: payload['episode'] == 0 — BC checkpoint seeds the net only,
    PPO self-play starts fresh from episode 0.
    """
    payload = _write_and_load(tmp_path / "ck.pt", _make_net(), _make_cfg())
    assert payload["episode"] == 0, f"Expected episode=0, got {payload['episode']!r}"


# ── Criterion: trainer_state sub-structure ────────────────────────────────────


def test_when_checkpoint_loaded_then_trainer_state_contains_model_key(tmp_path):
    """Criterion: trainer_state['model'] == net.state_dict()."""
    payload = _write_and_load(tmp_path / "ck.pt", _make_net(), _make_cfg())
    assert "model" in payload["trainer_state"], "trainer_state must contain 'model'"


def test_when_checkpoint_loaded_then_trainer_state_contains_optimizer_key(tmp_path):
    """Criterion: trainer_state['optimizer'] == optimizer.state_dict()."""
    payload = _write_and_load(tmp_path / "ck.pt", _make_net(), _make_cfg())
    assert "optimizer" in payload["trainer_state"], "trainer_state must contain 'optimizer'"


def test_when_checkpoint_loaded_then_trainer_state_contains_config_key(tmp_path):
    """Criterion: trainer_state['config'] is a PPOConfig (what PPOTrainer expects)."""
    payload = _write_and_load(tmp_path / "ck.pt", _make_net(), _make_cfg())
    assert "config" in payload["trainer_state"], "trainer_state must contain 'config'"


# ── Criterion: config types are correct ───────────────────────────────────────


def test_when_checkpoint_loaded_then_trainer_state_config_is_ppo_config(tmp_path):
    """
    Criterion: trainer_state['config'] is a PPOConfig — the exact type PPOTrainer expects,
    not the full TrainingConfig (which would cause PPOTrainer.load_checkpoint to mismatch).
    """
    payload = _write_and_load(tmp_path / "ck.pt", _make_net(), _make_cfg())
    ts_cfg = payload["trainer_state"]["config"]
    assert isinstance(ts_cfg, PPOConfig), (
        f"trainer_state['config'] must be PPOConfig, got {type(ts_cfg).__name__}"
    )


def test_when_checkpoint_loaded_then_top_level_config_is_training_config(tmp_path):
    """
    Criterion: top-level payload['config'] is the full TrainingConfig
    (mirrors SelfPlayTrainer.save_checkpoint).
    """
    payload = _write_and_load(tmp_path / "ck.pt", _make_net(), _make_cfg())
    assert isinstance(payload["config"], TrainingConfig), (
        f"top-level 'config' must be TrainingConfig, got {type(payload['config']).__name__}"
    )


def test_when_checkpoint_loaded_then_trainer_state_ppo_config_matches_cfg_ppo(tmp_path):
    """
    Criterion: trainer_state['config'] is cfg.ppo — the PPOConfig sub-object of the
    TrainingConfig that was passed to write_bc_checkpoint.
    """
    cfg = TrainingConfig(total_timesteps=5000, ppo=PPOConfig(lr=7e-4, gamma=0.98))
    payload = _write_and_load(tmp_path / "ck.pt", _make_net(), cfg)
    ts_cfg = payload["trainer_state"]["config"]
    assert isinstance(ts_cfg, PPOConfig)
    assert ts_cfg.lr == pytest.approx(7e-4), (
        f"trainer_state['config'].lr expected 7e-4, got {ts_cfg.lr}"
    )
    assert ts_cfg.gamma == pytest.approx(0.98), (
        f"trainer_state['config'].gamma expected 0.98, got {ts_cfg.gamma}"
    )


# ── Criterion: model weights are exactly preserved ────────────────────────────


def test_when_checkpoint_loaded_then_model_state_dict_matches_original_net(tmp_path):
    """
    Criterion: trainer_state['model'] == net.state_dict() — all weight tensors
    are preserved exactly under torch.equal.
    """
    net = _make_net()
    original_sd = {k: v.clone() for k, v in net.state_dict().items()}
    payload = _write_and_load(tmp_path / "ck.pt", net, _make_cfg())
    loaded_sd = payload["trainer_state"]["model"]
    assert set(loaded_sd.keys()) == set(original_sd.keys()), (
        "state_dict key sets differ between original net and checkpoint"
    )
    for k in original_sd:
        assert torch.equal(original_sd[k], loaded_sd[k]), (
            f"state_dict['{k}'] differs between original net and loaded checkpoint"
        )


# ── Criterion: PPOTrainer round-trip ─────────────────────────────────────────


def test_when_ppo_trainer_loads_trainer_state_then_no_error_is_raised(tmp_path):
    """
    Criterion: a real PPOTrainer built on a matching ActorCritic loads
    payload['trainer_state'] without error.
    """
    net = _make_net()
    cfg = _make_cfg()
    payload = _write_and_load(tmp_path / "ck.pt", net, cfg)

    fresh_net = _make_net()
    trainer = PPOTrainer(fresh_net, cfg.ppo)
    trainer.load_checkpoint(payload["trainer_state"])  # must not raise


def test_when_ppo_trainer_loads_trainer_state_then_net_state_dict_round_trips_equal(tmp_path):
    """
    Criterion: net.state_dict() round-trips equal (torch.equal) after
    PPOTrainer.load_checkpoint is called on the written payload.
    This is the key round-trip guarantee: BC checkpoint → PPOTrainer → same weights.
    """
    from training.bc_checkpoint import write_bc_checkpoint

    net = _make_net()
    cfg = _make_cfg()
    original_sd = {k: v.clone() for k, v in net.state_dict().items()}

    out = tmp_path / "pretrained.pt"
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    write_bc_checkpoint(out, net, optimizer, cfg)
    payload = torch.load(out, weights_only=False)

    fresh_net = _make_net()
    trainer = PPOTrainer(fresh_net, cfg.ppo)
    trainer.load_checkpoint(payload["trainer_state"])

    restored_sd = trainer.net.state_dict()
    assert set(restored_sd.keys()) == set(original_sd.keys())
    for k in original_sd:
        assert torch.equal(original_sd[k], restored_sd[k]), (
            f"Round-trip failed: state_dict['{k}'] differs between original and restored"
        )


# ── Criterion: omitted keys are absent (intentionally) ───────────────────────


def test_when_checkpoint_loaded_then_opponent_state_key_is_absent(tmp_path):
    """
    Criterion: opponent_state is omitted — BC seeds the net only; PPO self-play
    starts fresh (SelfPlayTrainer guards this key with an existence check).
    """
    payload = _write_and_load(tmp_path / "ck.pt", _make_net(), _make_cfg())
    assert "opponent_state" not in payload, (
        "opponent_state must be absent from BC checkpoint (intentionally omitted per spec)"
    )


def test_when_checkpoint_loaded_then_buffer_state_key_is_absent(tmp_path):
    """
    Criterion: buffer_state is omitted — BC produces a net warm-start only;
    no replay-buffer state exists at the BC stage.
    """
    payload = _write_and_load(tmp_path / "ck.pt", _make_net(), _make_cfg())
    assert "buffer_state" not in payload, (
        "buffer_state must be absent from BC checkpoint (intentionally omitted per spec)"
    )


def test_when_checkpoint_loaded_then_rng_state_key_is_absent(tmp_path):
    """
    Criterion: rng_state is omitted — PPO self-play seeds its own RNG from scratch
    after loading a BC warm-start.
    """
    payload = _write_and_load(tmp_path / "ck.pt", _make_net(), _make_cfg())
    assert "rng_state" not in payload, (
        "rng_state must be absent from BC checkpoint (intentionally omitted per spec)"
    )


def test_when_checkpoint_loaded_then_best_metric_value_key_is_absent(tmp_path):
    """
    Criterion: best_metric_value is omitted — no prior PPO win-rate metric exists
    in a brand-new BC checkpoint.
    """
    payload = _write_and_load(tmp_path / "ck.pt", _make_net(), _make_cfg())
    assert "best_metric_value" not in payload, (
        "best_metric_value must be absent from BC checkpoint (intentionally omitted per spec)"
    )


# ── Property-based test: round-trip is an invariant for all valid networks ────


@given(
    obs_dim=st.integers(min_value=4, max_value=32),
    hidden_size=st.integers(min_value=4, max_value=32),
    num_layers=st.integers(min_value=1, max_value=2),
)
@hyp_settings(max_examples=10, deadline=None)
def test_pbt_when_any_valid_network_written_then_state_dict_round_trips_equal(
    obs_dim,
    hidden_size,
    num_layers,
):
    """
    Invariant (round-trip): for any valid ActorCritic architecture, writing then loading
    via PPOTrainer.load_checkpoint yields torch.equal weights for every state_dict key.
    Derived from: 'net.state_dict() round-trips equal (torch.equal)'.
    """
    from training.bc_checkpoint import write_bc_checkpoint

    net = ActorCritic(
        obs_dim=obs_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        action_dims={
            "action_type": 6,
            "param_a": 42,
            "param_b": 42,
            "param_c": 42,
            "param_d": 42,
        },
    )
    cfg = _make_cfg()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    original_sd = {k: v.clone() for k, v in net.state_dict().items()}

    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "ck.pt"
        write_bc_checkpoint(out, net, optimizer, cfg)
        payload = torch.load(out, weights_only=False)

        fresh_net = ActorCritic(
            obs_dim=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            action_dims={
                "action_type": 6,
                "param_a": 42,
                "param_b": 42,
                "param_c": 42,
                "param_d": 42,
            },
        )
        trainer = PPOTrainer(fresh_net, cfg.ppo)
        trainer.load_checkpoint(payload["trainer_state"])
        restored_sd = trainer.net.state_dict()

        for k in original_sd:
            assert torch.equal(original_sd[k], restored_sd[k]), (
                f"Round-trip failed for key '{k}' "
                f"(obs_dim={obs_dim}, hidden_size={hidden_size}, num_layers={num_layers})"
            )
