"""Source-blind tests for training.bc_shard_writer.ShardWriter (issue #79).

Derived exclusively from the acceptance criteria for issue #79:
"feat: sharded .npz writer + manifest for BC dataset (bc_shard_writer.py)".
No implementation source has been read; tests specify the contract.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from training.bc_shard_writer import ShardWriter

OBS_DIM = 137


# ---------------------------------------------------------------------------
# Minimal BCConfig stub (fields inferred from acceptance criteria only)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _BCConfig:
    shard_size: int
    dataset_dir: str
    seed: int = 0
    n_games: int = 2
    explore_eps: float = 0.1
    n_players: int = 6
    max_turns: int = 200
    batch_size: int = 32
    lr: float = 0.001


@dataclasses.dataclass(frozen=True)
class _NetworkConfig:
    hidden_sizes: tuple[int, ...] = (256, 256)


@dataclasses.dataclass(frozen=True)
class _CfgStub:
    bc: _BCConfig
    network: _NetworkConfig = dataclasses.field(default_factory=_NetworkConfig)


def _make_obs() -> np.ndarray:
    return np.random.rand(OBS_DIM).astype(np.float32)


def _make_labels() -> dict:
    return {
        "action_type": 0,
        "param_a": 1,
        "param_b": 2,
        "param_c": 3,
        "param_d": 4,
        "label_index": 5,
        "acting_player": 0,
        "mc_return": 1.5,
    }


def _write_n(writer: ShardWriter, n: int) -> list[np.ndarray]:
    obs_list = [_make_obs() for _ in range(n)]
    for obs in obs_list:
        writer.write(obs, _make_labels())
    return obs_list


# ===========================================================================
# Criterion: shard count
# ===========================================================================


def test_two_full_shards_and_one_partial_when_remainder_rows_written(tmp_path):
    shard_size, r = 10, 3
    cfg = _BCConfig(shard_size=shard_size, dataset_dir=str(tmp_path))
    with ShardWriter(cfg) as w:
        _write_n(w, 2 * shard_size + r)
    shards = sorted(tmp_path.glob("shard_*.npz"))
    assert len(shards) == 3
    row_counts = [np.load(s)["obs"].shape[0] for s in shards]
    assert row_counts == [shard_size, shard_size, r]


def test_one_full_shard_no_empty_trailing_when_exactly_shard_size_rows(tmp_path):
    shard_size = 8
    cfg = _BCConfig(shard_size=shard_size, dataset_dir=str(tmp_path))
    with ShardWriter(cfg) as w:
        _write_n(w, shard_size)
    shards = sorted(tmp_path.glob("shard_*.npz"))
    assert len(shards) == 1
    assert np.load(shards[0])["obs"].shape[0] == shard_size


# ===========================================================================
# Criterion: .npz structure
# ===========================================================================


def test_obs_matrix_has_shape_rows_by_137_and_float32_dtype_when_shard_loaded(tmp_path):
    shard_size = n_rows = 5
    cfg = _BCConfig(shard_size=shard_size, dataset_dir=str(tmp_path))
    with ShardWriter(cfg) as w:
        _write_n(w, n_rows)
    data = np.load(next(tmp_path.glob("shard_*.npz")))
    assert data["obs"].shape == (n_rows, OBS_DIM)
    assert data["obs"].dtype == np.float32


def test_label_arrays_are_aligned_with_correct_dtypes_when_shard_loaded(tmp_path):
    shard_size = n_rows = 4
    cfg = _BCConfig(shard_size=shard_size, dataset_dir=str(tmp_path))
    with ShardWriter(cfg) as w:
        _write_n(w, n_rows)
    data = np.load(next(tmp_path.glob("shard_*.npz")))
    int_keys = (
        "action_type",
        "param_a",
        "param_b",
        "param_c",
        "param_d",
        "label_index",
        "acting_player",
    )
    for name in int_keys:
        assert data[name].shape == (n_rows,), name
        got = data[name].dtype
        assert np.issubdtype(got, np.integer), f"{name}: expected int dtype, got {got}"
    assert data["mc_return"].shape == (n_rows,)
    assert np.issubdtype(data["mc_return"].dtype, np.floating)


# ===========================================================================
# Criterion: close() writes trailing partial shard + manifest.json
# ===========================================================================


def test_partial_shard_written_when_close_called_with_fewer_than_shard_size_rows(tmp_path):
    cfg = _BCConfig(shard_size=10, dataset_dir=str(tmp_path))
    with ShardWriter(cfg) as w:
        _write_n(w, 7)
    shards = sorted(tmp_path.glob("shard_*.npz"))
    assert len(shards) == 1
    assert np.load(shards[0])["obs"].shape[0] == 7


def test_manifest_json_contains_required_fields_when_close_called(tmp_path):
    cfg = _BCConfig(shard_size=10, dataset_dir=str(tmp_path), seed=42)
    with ShardWriter(cfg) as w:
        _write_n(w, 7)
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    for field in ("total_rows", "obs_dim", "seed", "config_hash"):
        assert field in manifest, f"manifest missing required field '{field}'"


def test_manifest_total_rows_equals_sum_of_shard_row_counts_when_close_called(tmp_path):
    shard_size, n_rows = 6, 15
    cfg = _BCConfig(shard_size=shard_size, dataset_dir=str(tmp_path))
    with ShardWriter(cfg) as w:
        _write_n(w, n_rows)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    shard_total = sum(np.load(f)["obs"].shape[0] for f in tmp_path.glob("shard_*.npz"))
    assert manifest["total_rows"] == n_rows
    assert manifest["total_rows"] == shard_total


def test_manifest_obs_dim_is_137_when_close_called(tmp_path):
    cfg = _BCConfig(shard_size=5, dataset_dir=str(tmp_path))
    with ShardWriter(cfg) as w:
        _write_n(w, 3)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["obs_dim"] == OBS_DIM


def test_manifest_seed_matches_config_when_close_called(tmp_path):
    cfg = _BCConfig(shard_size=5, dataset_dir=str(tmp_path), seed=77)
    with ShardWriter(cfg) as w:
        _write_n(w, 3)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["seed"] == 77


# ===========================================================================
# Criterion: round-trip — np.load reproduces written rows in order
# ===========================================================================


def test_rows_reproduced_in_order_when_shards_reloaded(tmp_path):
    shard_size, n_rows = 4, 9
    cfg = _BCConfig(shard_size=shard_size, dataset_dir=str(tmp_path))
    np.random.seed(0)
    written = [np.random.rand(OBS_DIM).astype(np.float32) for _ in range(n_rows)]
    with ShardWriter(cfg) as w:
        for obs in written:
            w.write(obs, _make_labels())
    shards = sorted(tmp_path.glob("shard_*.npz"))
    loaded = np.concatenate([np.load(f)["obs"] for f in shards], axis=0)
    np.testing.assert_array_equal(loaded, np.stack(written))


# ===========================================================================
# Criterion: obs-encoding helper converts obs dict → 137-vector
# ===========================================================================


def test_137_vector_returned_when_obs_dict_encoded():
    from src.env import RisikoEnv
    from training.bc_shard_writer import encode_obs

    env = RisikoEnv()
    obs, _ = env.reset()
    result = encode_obs(obs)
    assert isinstance(result, np.ndarray)
    assert result.shape == (OBS_DIM,)
    assert result.dtype == np.float32


# ===========================================================================
# Criterion: same seed + config → identical dataset (reproducibility)
# ===========================================================================


def test_identical_dataset_produced_when_same_seed_and_config_used(tmp_path):
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    dir_a.mkdir()
    dir_b.mkdir()

    np.random.seed(42)
    rows = [(np.random.rand(OBS_DIM).astype(np.float32), _make_labels()) for _ in range(10)]

    for d in (dir_a, dir_b):
        cfg = _BCConfig(shard_size=4, dataset_dir=str(d), seed=42)
        with ShardWriter(cfg) as w:
            for obs, labels in rows:
                w.write(obs, labels)

    shards_a = sorted(dir_a.glob("shard_*.npz"))
    shards_b = sorted(dir_b.glob("shard_*.npz"))
    assert len(shards_a) == len(shards_b)
    for fa, fb in zip(shards_a, shards_b, strict=True):
        np.testing.assert_array_equal(np.load(fa)["obs"], np.load(fb)["obs"])
    manifest_a = json.loads((dir_a / "manifest.json").read_text())
    manifest_b = json.loads((dir_b / "manifest.json").read_text())
    assert manifest_a["config_hash"] == manifest_b["config_hash"]
    assert manifest_a["total_rows"] == manifest_b["total_rows"]


# ===========================================================================
# Criterion: data/bc/ in .gitignore
# ===========================================================================


def test_data_bc_directory_is_excluded_when_gitignore_read():
    gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
    assert gitignore.exists(), ".gitignore not found at project root"
    content = gitignore.read_text()
    assert "data/bc" in content, "Expected 'data/bc' pattern in .gitignore"


# ===========================================================================
# Criterion: pretrain → load pretrained.pt into PPO → one update step
# ===========================================================================


@pytest.mark.integration
def test_one_update_step_succeeds_when_pretrained_pt_loaded_into_ppo_trainer(tmp_path):
    from training.bc_trainer import pretrain
    from training.self_play import SelfPlayTrainer

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_path = tmp_path / "pretrained.pt"

    np.random.seed(0)
    tiny_bc = _BCConfig(shard_size=5, dataset_dir=str(data_dir), seed=0)
    with ShardWriter(tiny_bc) as w:
        for _ in range(10):
            w.write(np.random.rand(OBS_DIM).astype(np.float32), _make_labels())

    cfg = _CfgStub(bc=tiny_bc)
    pretrain(cfg, output_path=str(output_path))
    assert output_path.exists()

    trainer = SelfPlayTrainer.from_checkpoint(str(output_path))
    trainer.update_step()


# ===========================================================================
# Property-based tests (Hypothesis)
# ===========================================================================


@given(st.integers(min_value=1, max_value=200), st.integers(min_value=1, max_value=20))
@settings(max_examples=30)
def test_pbt_reloaded_rows_match_original_for_any_n_rows(n_rows, shard_size):
    with tempfile.TemporaryDirectory() as d:
        cfg = _BCConfig(shard_size=shard_size, dataset_dir=d)
        np.random.seed(0)
        written = [np.random.rand(OBS_DIM).astype(np.float32) for _ in range(n_rows)]
        with ShardWriter(cfg) as w:
            for obs in written:
                w.write(obs, _make_labels())
        shards = sorted(Path(d).glob("shard_*.npz"))
        loaded = np.concatenate([np.load(f)["obs"] for f in shards], axis=0)
        np.testing.assert_array_equal(loaded, np.stack(written))


@given(st.integers(min_value=1, max_value=300), st.integers(min_value=1, max_value=20))
@settings(max_examples=30)
def test_pbt_every_shard_has_at_most_shard_size_rows_for_any_n(n_rows, shard_size):
    with tempfile.TemporaryDirectory() as d:
        cfg = _BCConfig(shard_size=shard_size, dataset_dir=d)
        np.random.seed(0)
        with ShardWriter(cfg) as w:
            for _ in range(n_rows):
                w.write(np.random.rand(OBS_DIM).astype(np.float32), _make_labels())
        for f in Path(d).glob("shard_*.npz"):
            assert np.load(f)["obs"].shape[0] <= shard_size


@given(st.integers(min_value=0, max_value=500), st.integers(min_value=1, max_value=20))
@settings(max_examples=30)
def test_pbt_manifest_total_rows_equals_shard_sum_for_any_n(n_rows, shard_size):
    with tempfile.TemporaryDirectory() as d:
        cfg = _BCConfig(shard_size=shard_size, dataset_dir=d)
        np.random.seed(0)
        with ShardWriter(cfg) as w:
            for _ in range(n_rows):
                w.write(np.random.rand(OBS_DIM).astype(np.float32), _make_labels())
        manifest = json.loads((Path(d) / "manifest.json").read_text())
        shard_files = list(Path(d).glob("shard_*.npz"))
        shard_total = sum(np.load(f)["obs"].shape[0] for f in shard_files) if shard_files else 0
        assert manifest["total_rows"] == n_rows
        assert manifest["total_rows"] == shard_total


@given(st.integers(min_value=1, max_value=5))
@settings(max_examples=10)
def test_pbt_encode_obs_never_raises_for_valid_env_obs(_n):
    from src.env import RisikoEnv
    from training.bc_shard_writer import encode_obs

    env = RisikoEnv()
    obs, _ = env.reset()
    result = encode_obs(obs)
    assert result.shape == (OBS_DIM,)
    assert result.dtype == np.float32
