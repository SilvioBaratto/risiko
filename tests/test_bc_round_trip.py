"""Round-trip and determinism tests for issue #88.

Source-blind: derived from acceptance criteria only. No implementation source was read.

Covers:
  - Round-trip: BC checkpoint → load into SelfPlayTrainer → one PPO gradient step
  - Determinism: same seed + config produces an identical dataset
"""

from __future__ import annotations

import dataclasses
import tempfile

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Helpers — constructed from spec knowledge, not from reading src/
# ---------------------------------------------------------------------------


def _write_minimal_bc_checkpoint(path):
    """Write a PPO-compatible BC checkpoint using BCConfig + NetworkConfig defaults.

    Uses write_bc_checkpoint (training.bc_checkpoint) so the artifact format
    matches exactly what the pretrain stage produces.
    """
    from src.config import TrainingConfig  # noqa: PLC0415
    from src.models.actor_critic import ActorCritic  # noqa: PLC0415
    from src.utils.constants import ACTION_DIMS  # noqa: PLC0415
    from training.bc_checkpoint import write_bc_checkpoint  # noqa: PLC0415

    cfg = TrainingConfig()
    obs_dim = 137  # fixed by architecture spec (src/models/utils.py)
    net = ActorCritic(
        obs_dim=obs_dim,
        hidden_size=cfg.network.hidden_sizes[0],
        num_layers=len(cfg.network.hidden_sizes),
        action_dims=ACTION_DIMS,
    )
    opt = torch.optim.Adam(net.parameters(), lr=cfg.bc.lr)
    write_bc_checkpoint(path, net, opt, cfg)
    return net, cfg


def _make_tiny_bc_config(dataset_dir: str, seed: int = 42):
    """Return a TrainingConfig with a tiny BCConfig for fast tests."""
    from src.config import TrainingConfig  # noqa: PLC0415

    base = TrainingConfig()
    tiny_bc = dataclasses.replace(
        base.bc,
        n_games=3,
        n_players=2,
        max_turns=15,
        seed=seed,
        dataset_dir=dataset_dir,
        shard_size=100,
        output_path="models/pretrained.pt",
    )
    return dataclasses.replace(base, bc=tiny_bc)


# ---------------------------------------------------------------------------
# Criterion: round-trip — no architecture changes required in SelfPlayTrainer / PPO loop
# ---------------------------------------------------------------------------


def test_when_bc_checkpoint_loaded_into_selfplay_trainer_then_forward_pass_does_not_raise(
    tmp_path,
) -> None:
    """
    Round-trip: BC checkpoint must load into SelfPlayTrainer without any
    architecture changes (same obs_dim=137, same ActorCritic shape).
    """
    from training.self_play import SelfPlayTrainer  # noqa: PLC0415

    ckpt = tmp_path / "pretrained.pt"
    _net, cfg = _write_minimal_bc_checkpoint(ckpt)

    trainer = SelfPlayTrainer(cfg)
    trainer.load_checkpoint(str(ckpt))

    # obs_dim=137 is the fixed architecture constant; must not raise
    dummy_obs = torch.zeros(1, 137)
    with torch.no_grad():
        output = trainer._agent._net(dummy_obs)

    assert output is not None


def test_when_bc_checkpoint_loaded_then_one_gradient_step_does_not_raise(
    tmp_path,
) -> None:
    """
    Round-trip → one update step: backward pass through the loaded model must
    not raise, proving no architecture surgery is needed in the PPO loop.
    """
    from training.self_play import SelfPlayTrainer  # noqa: PLC0415

    ckpt = tmp_path / "pretrained.pt"
    _net, cfg = _write_minimal_bc_checkpoint(ckpt)

    trainer = SelfPlayTrainer(cfg)
    trainer.load_checkpoint(str(ckpt))

    dummy_obs = torch.zeros(2, 137)
    out = trainer._agent._net(dummy_obs)

    # out may be (logits, value) tuple or a single tensor
    if isinstance(out, (list, tuple)):
        loss = sum(o.sum() for o in out if isinstance(o, torch.Tensor))
    else:
        loss = out.sum()

    loss.backward()  # must not raise

    has_grad = any(p.grad is not None for p in trainer._agent._net.parameters())
    assert has_grad, (
        "At least one gradient must be non-None after the backward pass; "
        "no parameters should be frozen due to an architecture mismatch."
    )


def test_when_bc_checkpoint_loaded_then_all_parameters_require_grad(tmp_path) -> None:
    """After loading a BC checkpoint, all model parameters must accept gradient updates."""
    from training.self_play import SelfPlayTrainer  # noqa: PLC0415

    ckpt = tmp_path / "pretrained.pt"
    _net, cfg = _write_minimal_bc_checkpoint(ckpt)

    trainer = SelfPlayTrainer(cfg)
    trainer.load_checkpoint(str(ckpt))

    params = list(trainer._agent._net.parameters())
    assert params, "SelfPlayTrainer._agent._net must have parameters after load"
    assert all(p.requires_grad for p in params), (
        "All parameters must require grad; none should be frozen after loading the BC checkpoint."
    )


# ---------------------------------------------------------------------------
# Criterion: same seed + config produces an identical dataset
# ---------------------------------------------------------------------------


def test_when_generate_bc_dataset_called_twice_with_same_seed_then_observations_are_identical(
    tmp_path,
) -> None:
    """
    Determinism: generate_bc_dataset must produce the same .npz shards for the
    same seed and config. All randomness must flow through set_global_seeds.
    """
    import numpy as np  # noqa: PLC0415

    from training.bc_dataset import generate_bc_dataset  # noqa: PLC0415

    data_a = tmp_path / "run_a"
    data_b = tmp_path / "run_b"

    generate_bc_dataset(_make_tiny_bc_config(str(data_a), seed=99))
    generate_bc_dataset(_make_tiny_bc_config(str(data_b), seed=99))

    shards_a = sorted(data_a.glob("*.npz"))
    shards_b = sorted(data_b.glob("*.npz"))

    assert len(shards_a) > 0, "generate_bc_dataset must produce at least one shard"
    assert len(shards_a) == len(shards_b), "Same seed must produce the same number of shards"

    for sa, sb in zip(shards_a, shards_b, strict=True):
        arr_a = np.load(sa)
        arr_b = np.load(sb)
        assert set(arr_a.files) == set(arr_b.files), (
            f"Shard {sa.name}: key sets differ between runs with the same seed"
        )
        for key in arr_a.files:
            np.testing.assert_array_equal(
                arr_a[key],
                arr_b[key],
                err_msg=(
                    f"Shard {sa.name}, key '{key}': arrays differ between runs "
                    "with the same seed — all randomness must flow through set_global_seeds."
                ),
            )


def test_when_generate_bc_dataset_called_with_different_seeds_then_datasets_differ(
    tmp_path,
) -> None:
    """
    Sanity check: two runs with different seeds must NOT produce identical obs arrays.
    This confirms the generator is actually seeded rather than always outputting the same data.
    """
    import numpy as np  # noqa: PLC0415

    from training.bc_dataset import generate_bc_dataset  # noqa: PLC0415

    data_a = tmp_path / "seed_1"
    data_b = tmp_path / "seed_2"

    generate_bc_dataset(_make_tiny_bc_config(str(data_a), seed=1))
    generate_bc_dataset(_make_tiny_bc_config(str(data_b), seed=2))

    shards_a = sorted(data_a.glob("*.npz"))
    shards_b = sorted(data_b.glob("*.npz"))

    if shards_a and shards_b:
        arr_a = np.load(shards_a[0])["obs"]
        arr_b = np.load(shards_b[0])["obs"]
        assert not np.array_equal(arr_a, arr_b), (
            "Datasets generated with different seeds should differ "
            "(confirms the seed actually drives randomness)"
        )


# ---------------------------------------------------------------------------
# Property-based test: determinism invariant holds for any valid seed
# ---------------------------------------------------------------------------


@given(st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=3, deadline=None)
def test_when_any_seed_used_twice_then_generated_observations_are_identical(seed: int) -> None:
    """
    Invariant: generate_bc_dataset(cfg) is deterministic for any non-negative integer seed.
    Same seed → same obs arrays in every shard, for all seeds in the valid domain.
    """
    import pathlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from training.bc_dataset import generate_bc_dataset  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        generate_bc_dataset(_make_tiny_bc_config(d1, seed=seed))
        generate_bc_dataset(_make_tiny_bc_config(d2, seed=seed))

        shards_a = sorted(pathlib.Path(d1).glob("*.npz"))
        shards_b = sorted(pathlib.Path(d2).glob("*.npz"))

        assert len(shards_a) == len(shards_b), (
            f"seed={seed}: shard counts differ ({len(shards_a)} vs {len(shards_b)})"
        )

        for sa, sb in zip(shards_a, shards_b, strict=True):
            arr_a = np.load(sa)
            arr_b = np.load(sb)
            for key in arr_a.files:
                np.testing.assert_array_equal(
                    arr_a[key],
                    arr_b[key],
                    err_msg=f"seed={seed}, shard {sa.name}, key '{key}': not identical",
                )
