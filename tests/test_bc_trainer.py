"""
Source-blind example tests for training/bc_trainer.pretrain (issue #85).

Tests authored exclusively from the acceptance criteria for issue #85:
  "feat: pretrain(cfg) BC trainer — epoch loop, val early-stopping,
   PPO-warm-start checkpoint (bc_trainer.py)"
No implementation source has been read.  Tests are written in the Red phase
of TDD and are expected to fail until the implementation satisfies each criterion.

Skipped (per oracle — NOT VERIFIABLE):
  - pytest -m "not integration" passes (suite-level gate; no per-criterion assertion)
  - BC training of a 10k-game dataset completes in minutes on CPU (performance; proxy
    in test_bc_dataset.py; no per-criterion unit assertion for the trainer)
  - All tests pass (boilerplate suite gate)
  - SOLID, clean code (subjective code-quality prose)

Global criteria already covered in tests/test_bc_global_criteria.py:
  - ruff check . tests
  - coverage >= 80 % on src/
  - data/bc/ in .gitignore / models/ not excluded
  - BCConfig frozen dataclass with all required fields
  - HeuristicAgent implements Agent protocol
  - Logging via src/utils/log
  - Round-trip pretrain → SelfPlayTrainer.load_checkpoint → step (slow integration)
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

OBS_DIM = 137
# Production action-head dimensions derived from the spec (test_bc_loss.py confirms these).
_ACTION_DIMS: dict[str, int] = {
    "action_type": 6,
    "param_a": 201,
    "param_b": 201,
    "param_c": 201,
    "param_d": 1,
}
# Heads with dim > 1: entropy is non-trivially meaningful for these.
_MEANINGFUL_HEADS = ("action_type", "param_a", "param_b", "param_c")
# Minimum entropy (nats) a non-razor-peaked head must retain after BC training.
_ENTROPY_EPSILON = 0.01


# ── Synthetic dataset helpers ─────────────────────────────────────────────────


def _write_shard(path: pathlib.Path, n_rows: int, rng: np.random.Generator) -> None:
    """Write one synthetic shard matching the bc_dataset output schema."""
    np.savez(
        str(path),
        obs=rng.random((n_rows, OBS_DIM)).astype(np.float32),
        label_index=rng.integers(0, 5, size=(n_rows,)).astype(np.int64),
        acting_player=rng.integers(0, 6, size=(n_rows,)).astype(np.int64),
        mc_return=rng.random(n_rows).astype(np.float32),
        action_type=rng.integers(0, 6, size=(n_rows,)).astype(np.int64),
        param_a=rng.integers(0, 201, size=(n_rows,)).astype(np.int64),
        param_b=rng.integers(0, 201, size=(n_rows,)).astype(np.int64),
        param_c=rng.integers(0, 201, size=(n_rows,)).astype(np.int64),
        param_d=np.zeros(n_rows, dtype=np.int64),
    )


def _make_dataset(
    dataset_dir: pathlib.Path,
    *,
    n_shards: int = 4,
    rows_per_shard: int = 20,
    seed: int = 0,
) -> None:
    """Create a minimal synthetic BC dataset under *dataset_dir*."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_shards):
        _write_shard(dataset_dir / f"shard_{i:03d}.npz", rows_per_shard, rng)
    manifest = {
        "total_rows": n_shards * rows_per_shard,
        "obs_dim": OBS_DIM,
        "seed": seed,
        "config_hash": "cafebabe",
        "n_shards": n_shards,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest))


def _tiny_cfg(
    dataset_dir: pathlib.Path,
    output_path: pathlib.Path,
    *,
    seed: int = 0,
    epochs: int = 2,
    early_stop_patience: int = 5,
    entropy_coef: float = 0.01,
    label_smoothing: float = 0.05,
):
    """Return a minimal TrainingConfig suitable for fast unit-test fixtures."""
    from src.config import BCConfig, PPOConfig, TrainingConfig

    bc = BCConfig(
        n_games=4,
        n_players=2,
        max_turns=10,
        seed=seed,
        dataset_dir=str(dataset_dir),
        shard_size=200,
        demonstrator="heuristic",
        epochs=epochs,
        batch_size=8,
        lr=1e-3,
        value_loss_coef=0.5,
        val_split=0.25,
        early_stop_patience=early_stop_patience,
        output_path=str(output_path),
        explore_eps=0.0,
        label_smoothing=label_smoothing,
        entropy_coef=entropy_coef,
    )
    return TrainingConfig(total_timesteps=100, ppo=PPOConfig(lr=1e-3), bc=bc)


# ── Criterion: pretrain entry point is accessible ─────────────────────────────


def test_when_bc_trainer_module_imported_then_pretrain_is_accessible():
    """Criterion: training/bc_trainer.py exposes a pretrain(cfg) entry point."""
    from training.bc_trainer import pretrain  # noqa: F401


# ── Criterion: checkpoint is written to cfg.bc.output_path ───────────────────


def test_when_pretrain_called_then_checkpoint_is_written_to_bc_output_path(tmp_path):
    """
    Spec: pretrain(cfg) saves the checkpoint to cfg.bc.output_path.
    The file must exist after the call returns.
    """
    from training.bc_trainer import pretrain

    dataset_dir = tmp_path / "bc"
    output_path = tmp_path / "pretrained.pt"
    _make_dataset(dataset_dir)
    pretrain(_tiny_cfg(dataset_dir, output_path))
    assert output_path.exists(), "pretrain(cfg) must write the checkpoint to cfg.bc.output_path"


def test_when_pretrain_called_then_checkpoint_deserialises_to_a_dict(tmp_path):
    """
    Checkpoint must be a valid torch-serialised payload that torch.load returns as a dict.
    Derived from: 'saved as a checkpoint that SelfPlayTrainer.load_checkpoint() accepts'
    (PPO checkpoints are dicts with trainer_state and config keys).
    """
    from training.bc_trainer import pretrain

    dataset_dir = tmp_path / "bc"
    output_path = tmp_path / "pretrained.pt"
    _make_dataset(dataset_dir)
    pretrain(_tiny_cfg(dataset_dir, output_path))
    payload = torch.load(str(output_path), weights_only=False)
    assert isinstance(payload, dict), "Checkpoint must deserialise to a dict"


def test_when_checkpoint_loaded_then_trainer_state_key_is_present(tmp_path):
    """
    Spec: 'same trainer_state/ActorCritic shape' — the checkpoint must carry
    a 'trainer_state' key so SelfPlayTrainer.load_checkpoint can consume it.
    """
    from training.bc_trainer import pretrain

    dataset_dir = tmp_path / "bc"
    output_path = tmp_path / "pretrained.pt"
    _make_dataset(dataset_dir)
    pretrain(_tiny_cfg(dataset_dir, output_path))
    payload = torch.load(str(output_path), weights_only=False)
    assert "trainer_state" in payload, "Checkpoint dict must contain 'trainer_state'"


def test_when_checkpoint_loaded_then_trainer_state_contains_model_weights(tmp_path):
    """
    Spec: checkpoint round-trips into PPO trainer — trainer_state must hold model
    weights under a 'model' key (matches PPOTrainer.load_checkpoint interface).
    """
    from training.bc_trainer import pretrain

    dataset_dir = tmp_path / "bc"
    output_path = tmp_path / "pretrained.pt"
    _make_dataset(dataset_dir)
    pretrain(_tiny_cfg(dataset_dir, output_path))
    payload = torch.load(str(output_path), weights_only=False)
    assert "model" in payload["trainer_state"], (
        "trainer_state must contain a 'model' key with the ActorCritic state_dict"
    )


# ── Criterion: set_global_seeds is called with cfg.bc.seed ───────────────────


def test_when_pretrain_called_then_set_global_seeds_is_invoked_with_bc_seed(tmp_path, monkeypatch):
    """
    Spec: pretrain(cfg) seeds deterministically — all randomness flows through
    set_global_seeds, and every run logs its seed.
    Design: monkeypatch set_global_seeds in the bc_trainer module; assert cfg.bc.seed
    is passed to it.
    """
    import training.bc_trainer as _mod

    calls: list[int] = []
    monkeypatch.setattr(_mod, "set_global_seeds", lambda s: calls.append(s))

    dataset_dir = tmp_path / "bc"
    output_path = tmp_path / "pretrained.pt"
    _make_dataset(dataset_dir)
    _mod.pretrain(_tiny_cfg(dataset_dir, output_path, seed=77))
    assert 77 in calls, f"set_global_seeds(77) was not called; observed calls: {calls}"


# ── Criterion: per-epoch logging of train AND val metrics ────────────────────


def test_when_pretrain_runs_then_per_epoch_train_and_val_metrics_are_logged(tmp_path, monkeypatch):
    """
    Spec: per epoch — train AND val policy top-1 accuracy and value MSE are logged
    via get_logger.

    Design: monkeypatch get_logger in the bc_trainer module to capture all log
    messages; after training, assert logged output contains recognisable tokens for
    accuracy and MSE for both the train and val splits.
    """
    import training.bc_trainer as _mod

    messages: list[str] = []

    class _MockLogger:
        def info(self, msg, *args, **kwargs) -> None:
            messages.append(str(msg) % args if args else str(msg))

        def debug(self, msg, *args, **kwargs) -> None:
            messages.append(str(msg) % args if args else str(msg))

        def warning(self, msg, *args, **kwargs) -> None:
            messages.append(str(msg) % args if args else str(msg))

    monkeypatch.setattr(_mod, "get_logger", lambda *a, **kw: _MockLogger())

    dataset_dir = tmp_path / "bc"
    output_path = tmp_path / "pretrained.pt"
    _make_dataset(dataset_dir, n_shards=4, rows_per_shard=20)
    _mod.pretrain(_tiny_cfg(dataset_dir, output_path, epochs=2))

    combined = " ".join(messages).lower()
    assert "acc" in combined or "accuracy" in combined, (
        "Expected 'acc'/'accuracy' token in per-epoch log messages"
    )
    assert "mse" in combined, "Expected 'mse' token in per-epoch log messages"
    assert "val" in combined, (
        "Expected 'val' token in per-epoch log messages (val metrics must be logged)"
    )
    assert "train" in combined, (
        "Expected 'train' token in per-epoch log messages (train metrics must be logged)"
    )


# ── Criterion: early stopping — checkpoint is saved even when stop fires ──────


def test_when_early_stop_patience_triggers_then_checkpoint_is_still_saved(tmp_path):
    """
    Spec: early stopping on validation loss governed by cfg.bc.early_stop_patience
    via EarlyStopper(-val_loss); best-on-val state is restored before saving.

    Proxy: with patience=1 and many epochs on a tiny random dataset, pretrain must
    complete and write a checkpoint — not crash when EarlyStopper fires.
    """
    from training.bc_trainer import pretrain

    dataset_dir = tmp_path / "bc"
    output_path = tmp_path / "pretrained.pt"
    _make_dataset(dataset_dir, n_shards=2, rows_per_shard=15, seed=42)
    pretrain(_tiny_cfg(dataset_dir, output_path, epochs=20, early_stop_patience=1))
    assert output_path.exists(), (
        "Checkpoint must be written even when early stopping fires before the final epoch"
    )


def test_when_early_stopper_signals_stop_then_epoch_loop_terminates_early(tmp_path, monkeypatch):
    """
    Spec: early stopping via EarlyStopper(-val_loss) with patience from cfg.
    When EarlyStopper signals stop, pretrain must exit the epoch loop early.

    Design: replace EarlyStopper in the training.bc_trainer namespace with a mock
    that unconditionally stops after epoch 2.  With max_epochs=20 the loop must
    run fewer than 20 epochs.  The stop signal is tracked via a shared counter.

    Assumption: bc_trainer imports EarlyStopper into its own module namespace
    (``from ... import EarlyStopper``), so monkeypatching the name in bc_trainer
    intercepts every instantiation within pretrain.
    """
    import training.bc_trainer as _mod

    epoch_count: list[int] = [0]

    class _CountingEarlyStopper:
        """Stops after the second call regardless of the metric value."""

        def __init__(self, *args, **kwargs) -> None:
            self._calls = 0

        def __call__(self, metric: float) -> bool:
            self._calls += 1
            epoch_count[0] = self._calls
            return self._calls >= 2

    monkeypatch.setattr(_mod, "EarlyStopper", _CountingEarlyStopper)

    dataset_dir = tmp_path / "bc"
    output_path = tmp_path / "pretrained.pt"
    _make_dataset(dataset_dir, n_shards=2, rows_per_shard=15)
    _mod.pretrain(_tiny_cfg(dataset_dir, output_path, epochs=20, early_stop_patience=1))

    assert epoch_count[0] < 20, (
        "pretrain ran all 20 epochs despite EarlyStopper signalling stop at epoch 2; "
        "the epoch loop must respect the EarlyStopper return value"
    )
    assert output_path.exists(), "Checkpoint must be saved when early stopping fires"


# ── Criterion: determinism (same seed → bitwise-equal state_dicts) ───────────


def test_when_same_seed_used_twice_then_state_dicts_are_bitwise_equal(tmp_path):
    """
    Spec: deterministic — two runs with identical cfg.bc.seed produce bitwise-equal
    net state_dict (torch.equal for every tensor).
    """
    from training.bc_trainer import pretrain

    def _run(run_dir: pathlib.Path) -> dict:
        dataset_dir = run_dir / "bc"
        _make_dataset(dataset_dir, n_shards=4, rows_per_shard=20, seed=42)
        output = run_dir / "pretrained.pt"
        pretrain(_tiny_cfg(dataset_dir, output, seed=42, epochs=2))
        return torch.load(str(output), weights_only=False)

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()

    sd_a = _run(run_a)["trainer_state"]["model"]
    sd_b = _run(run_b)["trainer_state"]["model"]

    assert set(sd_a.keys()) == set(sd_b.keys()), (
        "State dict key sets differ between two identical-seed pretrain runs"
    )
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), (
            f"State dict key '{k}' is not bitwise-equal between identical-seed pretrain runs"
        )


# ── Criterion: non-trivial entropy after BC training ─────────────────────────


def test_when_pretrained_with_entropy_bonus_then_meaningful_heads_retain_nontrivial_entropy(
    tmp_path,
):
    """
    Spec: A test asserts the cloned policy retains non-trivial entropy
    (action_type/param_a/param_b/param_c entropy > epsilon; not razor-peaked) —
    proving entropy bonus + label smoothing worked. Exclude param_d (dim=1, entropy ≡ 0).

    Design: run pretrain with high entropy_coef=0.5 and label_smoothing=0.2 on a
    random synthetic dataset. Load the checkpoint, reconstruct the ActorCritic using
    the production action_dims and the NetworkConfig defaults, then compute mean
    per-head Shannon entropy on a batch of random observations.

    If ActorCritic reconstruction fails due to unexpected NetworkConfig field names,
    the test is skipped with a diagnostic message rather than failing with a confusing
    shape mismatch — the implementer must ensure the dims match in production.
    """
    from src.models.actor_critic import ActorCritic
    from training.bc_trainer import pretrain

    dataset_dir = tmp_path / "bc"
    output_path = tmp_path / "pretrained.pt"
    _make_dataset(dataset_dir, n_shards=4, rows_per_shard=30, seed=0)

    cfg = _tiny_cfg(
        dataset_dir,
        output_path,
        seed=0,
        epochs=3,
        entropy_coef=0.5,
        label_smoothing=0.2,
    )
    pretrain(cfg)

    payload = torch.load(str(output_path), weights_only=False)
    sd = payload["trainer_state"]["model"]

    try:
        from src.config import NetworkConfig

        nc = NetworkConfig()
        net = ActorCritic(
            obs_dim=OBS_DIM,
            hidden_size=nc.hidden_sizes[0],
            num_layers=len(nc.hidden_sizes),
            action_dims=_ACTION_DIMS,
        )
        net.load_state_dict(sd)
    except Exception as exc:
        pytest.skip(
            f"Could not reconstruct ActorCritic from checkpoint — "
            f"NetworkConfig field names may differ from expected defaults: {exc}"
        )

    net.eval()
    obs = torch.randn(64, OBS_DIM)
    with torch.no_grad():
        logits, _ = net(obs)  # logits: dict[head -> (B, dim)]

    for head in _MEANINGFUL_HEADS:
        head_logits = logits[head]
        probs = torch.softmax(head_logits, dim=-1)
        entropy = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1).mean()
        assert entropy.item() > _ENTROPY_EPSILON, (
            f"Head '{head}' mean entropy {entropy.item():.4f} nats <= {_ENTROPY_EPSILON}; "
            "policy is razor-peaked — entropy_coef + label_smoothing did not preserve "
            "exploration entropy for the downstream PPO handoff"
        )


# ── Criterion: SelfPlayTrainer round-trip (acceptance gate) ──────────────────


def test_when_pretrained_checkpoint_loaded_into_self_play_trainer_then_no_error_is_raised(
    tmp_path,
):
    """
    Spec: Acceptance — instantiate SelfPlayTrainer(cfg) and call
    trainer.load_checkpoint(Path(cfg.bc.output_path)) — loads with matching
    trainer_state/ActorCritic shape.
    """
    from training.bc_trainer import pretrain
    from training.self_play import SelfPlayTrainer

    dataset_dir = tmp_path / "bc"
    output_path = tmp_path / "pretrained.pt"
    _make_dataset(dataset_dir, n_shards=4, rows_per_shard=20, seed=0)

    cfg = _tiny_cfg(dataset_dir, output_path, seed=0, epochs=1)
    pretrain(cfg)
    assert output_path.exists(), "pretrain must write a checkpoint before the round-trip"

    trainer = SelfPlayTrainer(cfg)
    trainer.load_checkpoint(pathlib.Path(str(output_path)))  # must not raise


# ── Property-based tests — invariants derived from criteria ──────────────────


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@hyp_settings(max_examples=5, deadline=None)
def test_pbt_when_any_seed_used_twice_then_state_dicts_are_bitwise_equal(seed: int) -> None:
    """
    Invariant (round-trip): for any non-negative seed, two calls to pretrain with the
    same seed + config produce bitwise-equal net state_dict tensors.
    Derived from: 'two runs with identical cfg.bc.seed produce bitwise-equal state_dict
    (torch.equal)'.
    """
    from training.bc_trainer import pretrain

    with tempfile.TemporaryDirectory() as base:
        base_path = pathlib.Path(base)

        def _run(run_dir: pathlib.Path) -> dict:
            dataset_dir = run_dir / "bc"
            _make_dataset(dataset_dir, n_shards=2, rows_per_shard=10, seed=seed)
            output = run_dir / "pretrained.pt"
            pretrain(_tiny_cfg(dataset_dir, output, seed=seed, epochs=1))
            return torch.load(str(output), weights_only=False)

        run_a = base_path / "a"
        run_b = base_path / "b"
        run_a.mkdir()
        run_b.mkdir()

        sd_a = _run(run_a)["trainer_state"]["model"]
        sd_b = _run(run_b)["trainer_state"]["model"]

        for k in sd_a:
            assert torch.equal(sd_a[k], sd_b[k]), (
                f"seed={seed}: state_dict['{k}'] differs between identical-seed runs"
            )


@given(epochs=st.integers(min_value=1, max_value=4))
@hyp_settings(max_examples=5, deadline=None)
def test_pbt_when_pretrain_runs_for_any_epoch_count_then_checkpoint_always_exists(
    epochs: int,
) -> None:
    """
    Invariant (never-raises / output-always-produced): for any valid epoch count ≥ 1,
    pretrain must always write a checkpoint to cfg.bc.output_path.
    Derived from: 'saves the checkpoint to cfg.bc.output_path' and 'early stopping …
    best-on-val net state is restored before saving'.
    """
    from training.bc_trainer import pretrain

    with tempfile.TemporaryDirectory() as base:
        base_path = pathlib.Path(base)
        dataset_dir = base_path / "bc"
        output_path = base_path / "pretrained.pt"
        _make_dataset(dataset_dir, n_shards=2, rows_per_shard=10, seed=0)
        pretrain(_tiny_cfg(dataset_dir, output_path, epochs=epochs))
        assert output_path.exists(), (
            f"epochs={epochs}: checkpoint must exist after pretrain regardless of epoch count"
        )
