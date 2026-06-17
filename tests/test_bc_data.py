"""
Source-blind example tests for training/bc_data.py (issue #83).

Tests authored exclusively from the acceptance criteria for issue #83:
"feat: memory-bounded sharded BC dataset loader + seeded train/val split (bc_data.py)".
No implementation source has been read.  Tests are written in the Red phase of TDD
and are expected to fail until the implementation satisfies each criterion.

Skipped (per oracle — NOT VERIFIABLE):
  - pytest -m "not integration" (suite-level gate; no per-criterion assertion possible)
  - All tests pass (suite gate)
  - SOLID / code-quality prose

Global criteria already covered in tests/test_bc_global_criteria.py:
  - ruff check
  - Coverage >= 80%
  - Identical datasets from same generation seed
  - BCConfig fields present and frozen
  - data/bc/ in .gitignore / models/ not excluded
  - HeuristicAgent protocol + logging convention
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

OBS_DIM = 137
_HEAD_NAMES = ("action_type", "param_a", "param_b", "param_c", "param_d")


# ── Test fixtures ─────────────────────────────────────────────────────────────


def _write_shard(path: Path, n_rows: int, rng: np.random.Generator) -> None:
    """Write a synthetic shard file matching the bc_dataset output schema."""
    np.savez(
        str(path),
        obs=rng.random((n_rows, OBS_DIM)).astype(np.float32),
        label_index=rng.integers(0, 5, size=(n_rows,)).astype(np.int64),
        acting_player=rng.integers(0, 6, size=(n_rows,)).astype(np.int64),
        mc_return=rng.random(n_rows).astype(np.float32),
        action_type=rng.integers(0, 6, size=(n_rows,)).astype(np.int64),
        param_a=rng.integers(0, 42, size=(n_rows,)).astype(np.int64),
        param_b=rng.integers(0, 42, size=(n_rows,)).astype(np.int64),
        param_c=rng.integers(0, 42, size=(n_rows,)).astype(np.int64),
        param_d=np.zeros(n_rows, dtype=np.int64),
    )


def _make_dataset(
    dataset_dir: Path,
    n_shards: int = 4,
    rows_per_shard: int = 20,
    seed: int = 42,
    obs_dim: int = OBS_DIM,
) -> dict:
    """Create a minimal synthetic BC dataset in *dataset_dir*. Returns the manifest dict."""
    rng = np.random.default_rng(seed)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    total_rows = n_shards * rows_per_shard
    for i in range(n_shards):
        _write_shard(dataset_dir / f"shard_{i:03d}.npz", rows_per_shard, rng)
    manifest = {
        "total_rows": total_rows,
        "obs_dim": obs_dim,
        "seed": seed,
        "config_hash": "deadbeef",
        "n_shards": n_shards,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest


# ── Module accessibility ──────────────────────────────────────────────────────


def test_when_module_imported_then_read_manifest_is_accessible():
    """Criterion: training/bc_data.py must expose read_manifest."""
    from training.bc_data import read_manifest  # noqa: F401


def test_when_module_imported_then_list_shards_is_accessible():
    """Criterion: training/bc_data.py must expose list_shards."""
    from training.bc_data import list_shards  # noqa: F401


def test_when_module_imported_then_shard_stream_class_is_accessible():
    """Criterion: training/bc_data.py must expose ShardStream."""
    from training.bc_data import ShardStream  # noqa: F401


# ── read_manifest ─────────────────────────────────────────────────────────────


def test_when_manifest_obs_dim_matches_get_obs_dim_then_read_manifest_returns_dict(tmp_path):
    """
    Criterion: read_manifest(dataset_dir) loads manifest.json and asserts
    obs_dim == get_obs_dim() (which returns 137). When the stored obs_dim is
    correct, read_manifest must return the manifest contents as a dict.
    """
    _make_dataset(tmp_path, obs_dim=OBS_DIM)
    from training.bc_data import read_manifest

    result = read_manifest(tmp_path)
    assert isinstance(result, dict), "read_manifest must return a dict"
    assert result["obs_dim"] == OBS_DIM


def test_when_manifest_obs_dim_is_wrong_then_read_manifest_raises(tmp_path):
    """
    Criterion: read_manifest asserts obs_dim == get_obs_dim().
    A manifest whose obs_dim != 137 must cause an AssertionError, ValueError,
    or RuntimeError — never silently succeed.
    """
    _make_dataset(tmp_path, obs_dim=999)
    from training.bc_data import read_manifest

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        read_manifest(tmp_path)


def test_when_manifest_loaded_then_total_rows_field_is_present_and_correct(tmp_path):
    """read_manifest must return a dict containing the correct total_rows value."""
    expected = _make_dataset(tmp_path, n_shards=3, rows_per_shard=10)
    from training.bc_data import read_manifest

    result = read_manifest(tmp_path)
    assert "total_rows" in result, "Manifest dict must contain 'total_rows'"
    assert result["total_rows"] == expected["total_rows"]


# ── list_shards ───────────────────────────────────────────────────────────────


def test_when_dataset_has_shards_then_list_shards_returns_all_npz_files(tmp_path):
    """
    Criterion: list_shards(dataset_dir) returns sorted(glob("shard_*.npz")).
    All shard files present in the directory must appear in the result.
    """
    _make_dataset(tmp_path, n_shards=4)
    from training.bc_data import list_shards

    result = list_shards(tmp_path)
    assert len(result) == 4, f"Expected 4 shards, got {len(result)}"


def test_when_shards_written_in_arbitrary_order_then_list_shards_returns_sorted(tmp_path):
    """
    Criterion: deterministic order matching the zero-padded writer naming.
    Shard files inserted in non-sequential order must still come back sorted.
    """
    rng = np.random.default_rng(0)
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in [3, 1, 0, 2]:  # deliberately out of order
        _write_shard(tmp_path / f"shard_{i:03d}.npz", 5, rng)

    from training.bc_data import list_shards

    result = list_shards(tmp_path)
    names = [Path(str(r)).name for r in result]
    assert names == sorted(names), f"Shards not sorted: {names}"


def test_when_directory_has_no_shards_then_list_shards_returns_empty(tmp_path):
    """list_shards on a directory with no shard_*.npz files must return an empty sequence."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    from training.bc_data import list_shards

    result = list_shards(tmp_path)
    assert len(result) == 0, f"Expected empty result; got {len(result)} item(s)"


# ── ShardStream: Batch contents ───────────────────────────────────────────────


def test_when_shard_stream_iterated_then_batch_obs_is_float32_tensor_with_137_columns(tmp_path):
    """
    Criterion: ShardStream yields Batch with obs:(B,137) float32 tensor.
    """
    _make_dataset(tmp_path, n_shards=2, rows_per_shard=20)
    from training.bc_data import ShardStream

    stream = ShardStream(dataset_dir=tmp_path, split="train", batch_size=8, seed=0, val_split=0.5)
    batch = next(iter(stream))

    assert hasattr(batch, "obs"), "Batch must have 'obs' attribute"
    assert isinstance(batch.obs, torch.Tensor), "obs must be a torch.Tensor"
    assert batch.obs.dtype == torch.float32, f"obs.dtype must be float32, got {batch.obs.dtype}"
    assert batch.obs.ndim == 2, f"obs must be 2-D, got ndim={batch.obs.ndim}"
    assert batch.obs.shape[1] == OBS_DIM, (
        f"obs.shape[1] must be {OBS_DIM}, got {batch.obs.shape[1]}"
    )


def test_when_shard_stream_iterated_then_batch_targets_is_dict_of_long_tensors(tmp_path):
    """
    Criterion: Batch.targets is dict[head -> long(B)].
    Each value must be a 1-D torch.long tensor with the same leading dimension as obs.
    """
    _make_dataset(tmp_path, n_shards=2, rows_per_shard=20)
    from training.bc_data import ShardStream

    stream = ShardStream(dataset_dir=tmp_path, split="train", batch_size=8, seed=0, val_split=0.5)
    batch = next(iter(stream))

    assert hasattr(batch, "targets"), "Batch must have 'targets' attribute"
    assert isinstance(batch.targets, dict), "targets must be a dict"
    n_rows = batch.obs.shape[0]
    for head, t in batch.targets.items():
        assert isinstance(t, torch.Tensor), f"targets['{head}'] must be a Tensor"
        assert t.dtype == torch.long, f"targets['{head}'] dtype must be long, got {t.dtype}"
        assert t.shape == (n_rows,), f"targets['{head}'] shape must be ({n_rows},), got {t.shape}"


def test_when_shard_stream_iterated_then_targets_contains_all_five_action_head_names(tmp_path):
    """
    Criterion: targets dict contains one entry per action head.
    All five action-component heads required by the policy network must be present.
    """
    _make_dataset(tmp_path, n_shards=2, rows_per_shard=20)
    from training.bc_data import ShardStream

    stream = ShardStream(dataset_dir=tmp_path, split="train", batch_size=8, seed=0, val_split=0.5)
    batch = next(iter(stream))

    expected = set(_HEAD_NAMES)
    actual = set(batch.targets.keys())
    assert expected.issubset(actual), f"targets missing head names: {expected - actual}"


def test_when_shard_stream_iterated_then_mc_return_is_1d_tensor_matching_batch_size(tmp_path):
    """
    Criterion: Batch.mc_return is a (B,) tensor sharing the batch dimension with obs.
    """
    _make_dataset(tmp_path, n_shards=2, rows_per_shard=20)
    from training.bc_data import ShardStream

    stream = ShardStream(dataset_dir=tmp_path, split="train", batch_size=8, seed=0, val_split=0.5)
    batch = next(iter(stream))

    assert hasattr(batch, "mc_return"), "Batch must have 'mc_return' attribute"
    n_rows = batch.obs.shape[0]
    assert isinstance(batch.mc_return, torch.Tensor), "mc_return must be a torch.Tensor"
    assert batch.mc_return.shape == (n_rows,), (
        f"mc_return shape must be ({n_rows},), got {batch.mc_return.shape}"
    )


def test_when_shard_stream_batch_yielded_then_obs_targets_mc_return_share_batch_dim(tmp_path):
    """All three Batch fields must have the same leading batch dimension."""
    _make_dataset(tmp_path, n_shards=2, rows_per_shard=20)
    from training.bc_data import ShardStream

    stream = ShardStream(dataset_dir=tmp_path, split="train", batch_size=8, seed=0, val_split=0.5)
    batch = next(iter(stream))

    n_rows = batch.obs.shape[0]
    assert batch.mc_return.shape[0] == n_rows, (
        f"mc_return batch dim {batch.mc_return.shape[0]} != obs batch dim {n_rows}"
    )
    for head, t in batch.targets.items():
        assert t.shape[0] == n_rows, (
            f"targets['{head}'] batch dim {t.shape[0]} != obs batch dim {n_rows}"
        )


# ── Train/val split ───────────────────────────────────────────────────────────


def test_when_train_and_val_streams_fully_iterated_then_combined_row_count_equals_total(tmp_path):
    """
    Criterion: train and val shard sets are disjoint — their combined row count
    must equal manifest total_rows (partition with no overlap and no loss).
    """
    _make_dataset(tmp_path, n_shards=6, rows_per_shard=10)
    from training.bc_data import ShardStream, read_manifest

    manifest = read_manifest(tmp_path)
    total = manifest["total_rows"]

    def _stream(split):
        return ShardStream(dataset_dir=tmp_path, split=split, batch_size=5, seed=3, val_split=0.3)

    train_rows = sum(b.obs.shape[0] for b in _stream("train"))
    val_rows = sum(b.obs.shape[0] for b in _stream("val"))

    assert train_rows + val_rows == total, (
        f"train({train_rows}) + val({val_rows}) = {train_rows + val_rows} != total({total}); "
        "splits must be a disjoint partition of all shards"
    )


def test_when_val_split_nonzero_then_val_stream_yields_at_least_one_batch(tmp_path):
    """
    Criterion: val split is non-empty when val_split > 0 and there are enough shards.
    With 6 shards and val_split=0.3, at least 1 shard must go to val.
    """
    _make_dataset(tmp_path, n_shards=6, rows_per_shard=10)
    from training.bc_data import ShardStream

    stream = ShardStream(dataset_dir=tmp_path, split="val", batch_size=5, seed=0, val_split=0.3)
    val_rows = sum(b.obs.shape[0] for b in stream)
    assert val_rows > 0, "val split must be non-empty when val_split=0.3 and n_shards=6"


def test_when_train_stream_constructed_twice_with_same_seed_then_first_batch_is_identical(tmp_path):
    """
    Criterion: reproducible across two constructions with the same seed.
    The same seed must produce the same shard assignment and hence identical data.
    """
    _make_dataset(tmp_path, n_shards=6, rows_per_shard=10)
    from training.bc_data import ShardStream

    def _mk():
        return ShardStream(
            dataset_dir=tmp_path, split="train", batch_size=8, seed=42, val_split=0.3
        )

    stream_a = _mk()
    stream_b = _mk()

    batch_a = next(iter(stream_a))
    batch_b = next(iter(stream_b))

    assert torch.equal(batch_a.obs, batch_b.obs), (
        "First batch obs differs between two ShardStream constructions with the same seed"
    )
    assert torch.equal(batch_a.mc_return, batch_b.mc_return), (
        "First batch mc_return differs between two ShardStream constructions with the same seed"
    )


def test_when_val_stream_constructed_twice_with_same_seed_then_row_counts_match(tmp_path):
    """
    Criterion: reproducible across two constructions with the same seed.
    Val split must yield the same total rows on both invocations.
    """
    _make_dataset(tmp_path, n_shards=6, rows_per_shard=10)
    from training.bc_data import ShardStream

    def _val_stream():
        return ShardStream(dataset_dir=tmp_path, split="val", batch_size=5, seed=7, val_split=0.3)

    rows_a = sum(b.obs.shape[0] for b in _val_stream())
    rows_b = sum(b.obs.shape[0] for b in _val_stream())

    assert rows_a == rows_b, (
        f"Val row count differs between identical-seed constructions: {rows_a} vs {rows_b}"
    )


# ── Memory-bounded / lazy loading ─────────────────────────────────────────────


def test_when_shard_stream_constructed_then_no_shards_are_loaded_eagerly(tmp_path, monkeypatch):
    """
    Criterion: memory-bounded — a test asserts shards load lazily (not all upfront).
    Proxy: np.load must not be called during ShardStream.__init__; only during
    iteration when each shard is actually needed.
    """
    _make_dataset(tmp_path, n_shards=4, rows_per_shard=10)
    from training.bc_data import ShardStream

    call_count = [0]
    real_load = np.load

    def _counting_load(path, *args, **kwargs):
        call_count[0] += 1
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", _counting_load)

    _ = ShardStream(dataset_dir=tmp_path, split="train", batch_size=5, seed=0, val_split=0.5)

    assert call_count[0] == 0, (
        f"np.load was called {call_count[0]} time(s) during ShardStream construction; "
        "expected 0 — shards must load lazily only when iterated"
    )


def test_when_shard_stream_iterated_then_at_most_one_shard_context_open_at_once(
    tmp_path, monkeypatch
):
    """
    Criterion: at most one shard's arrays resident at a time
    ('with np.load(...) as z:' copy-out pattern, released before the next shard).
    Tracked by wrapping each NpzFile in a context-manager counter.
    """
    _make_dataset(tmp_path, n_shards=4, rows_per_shard=10)
    from training.bc_data import ShardStream

    concurrent = [0]
    max_concurrent = [0]
    real_load = np.load

    class _TrackingFile:
        """Wraps NpzFile to count how many context managers are open simultaneously."""

        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            concurrent[0] += 1
            max_concurrent[0] = max(max_concurrent[0], concurrent[0])
            return self._inner.__enter__()

        def __exit__(self, *exc):
            concurrent[0] -= 1
            return self._inner.__exit__(*exc)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def _tracking_load(path, *args, **kwargs):
        return _TrackingFile(real_load(path, *args, **kwargs))

    monkeypatch.setattr(np, "load", _tracking_load)

    stream = ShardStream(dataset_dir=tmp_path, split="train", batch_size=5, seed=0, val_split=0.5)
    for _ in stream:
        pass

    assert max_concurrent[0] <= 1, (
        f"Maximum concurrent np.load context managers open: {max_concurrent[0]}; expected ≤1"
    )


# ── Total rows / batch partition ──────────────────────────────────────────────


def test_when_entire_train_split_streamed_then_row_count_equals_train_contribution(tmp_path):
    """
    Criterion: total streamed rows == manifest total_rows (for the chosen split).
    With val_split=0.0 all shards go to train; streamed rows must equal total_rows.
    """
    _make_dataset(tmp_path, n_shards=4, rows_per_shard=10)
    from training.bc_data import ShardStream, read_manifest

    manifest = read_manifest(tmp_path)
    total = manifest["total_rows"]

    train_stream = ShardStream(
        dataset_dir=tmp_path, split="train", batch_size=7, seed=0, val_split=0.0
    )
    streamed = sum(b.obs.shape[0] for b in train_stream)

    assert streamed == total, (
        f"Streamed {streamed} rows but manifest total_rows={total}; "
        "val_split=0.0 assigns all shards to train"
    )


def test_when_batches_contain_unique_row_ids_then_no_row_appears_twice(tmp_path):
    """
    Criterion: batches partition rows with no overlap or loss within a split.
    mc_return stores global unique float IDs (row indices); no ID may appear twice.
    """
    n_shards, rows_per_shard = 3, 10
    total = n_shards * rows_per_shard
    rng = np.random.default_rng(1)
    tmp_path.mkdir(parents=True, exist_ok=True)

    row_id = 0
    for i in range(n_shards):
        ids = np.arange(row_id, row_id + rows_per_shard, dtype=np.float32)
        row_id += rows_per_shard
        np.savez(
            str(tmp_path / f"shard_{i:03d}.npz"),
            obs=rng.random((rows_per_shard, OBS_DIM)).astype(np.float32),
            label_index=np.zeros(rows_per_shard, dtype=np.int64),
            acting_player=np.zeros(rows_per_shard, dtype=np.int64),
            mc_return=ids,
            action_type=np.zeros(rows_per_shard, dtype=np.int64),
            param_a=np.zeros(rows_per_shard, dtype=np.int64),
            param_b=np.zeros(rows_per_shard, dtype=np.int64),
            param_c=np.zeros(rows_per_shard, dtype=np.int64),
            param_d=np.zeros(rows_per_shard, dtype=np.int64),
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "total_rows": total,
                "obs_dim": OBS_DIM,
                "seed": 1,
                "config_hash": "test",
            }
        )
    )

    from training.bc_data import ShardStream

    all_ids: list[float] = []
    stream = ShardStream(dataset_dir=tmp_path, split="train", batch_size=4, seed=0, val_split=0.0)
    for batch in stream:
        all_ids.extend(batch.mc_return.tolist())

    unique_ids = {int(x) for x in all_ids}
    assert len(all_ids) == len(unique_ids), (
        f"Duplicate row IDs found: {len(all_ids)} rows streamed but only "
        f"{len(unique_ids)} unique IDs — batches must not overlap"
    )
    assert len(all_ids) == total, (
        f"Expected {total} rows, got {len(all_ids)} — batches must cover all rows (no loss)"
    )


# ── Conventions ───────────────────────────────────────────────────────────────


def test_when_bc_data_module_exists_then_it_logs_via_src_utils_log():
    """
    Criterion: new conventions follow the existing codebase — logging via src/utils/log.
    """
    module_path = Path("training/bc_data.py")
    if not module_path.exists():
        pytest.skip("training/bc_data.py does not exist yet — skipping convention check")
    source = module_path.read_text()
    uses_src_log = "src.utils.log" in source or "from src.utils" in source
    assert uses_src_log, (
        "training/bc_data.py must import logging from src/utils/log, "
        "not use bare logging.basicConfig or plain print statements"
    )


# ── Boundary: 3 shards + val_split=0.1 rounding guard ────────────────────────


def test_when_3_shards_and_val_split_01_then_both_splits_are_nonempty(tmp_path):
    """
    Issue-comment boundary test: with n_shards=3 and val_split=0.1, a naive
    int(3*0.1)=0 would leave val empty.  The rounding guard must ensure both
    splits are non-empty: n_val = max(1, round(3*0.1)) = 1.
    """
    _make_dataset(tmp_path, n_shards=3, rows_per_shard=10)
    from training.bc_data import ShardStream

    def _mk(split):
        return ShardStream(dataset_dir=tmp_path, split=split, batch_size=5, seed=0, val_split=0.1)

    train_rows = sum(b.obs.shape[0] for b in _mk("train"))
    val_rows = sum(b.obs.shape[0] for b in _mk("val"))

    assert val_rows > 0, (
        "val split must be non-empty: rounding guard must use max(1, round(3*0.1))=1"
    )
    assert train_rows > 0, "train split must be non-empty"
    assert train_rows + val_rows == 30, (
        f"train({train_rows}) + val({val_rows}) must equal total 30 rows"
    )


def test_when_3_shards_and_val_split_01_split_is_reproducible(tmp_path):
    """
    Issue-comment boundary test: two ShardStream constructions with the same seed
    must produce identical row counts (disjoint, reproducible split).
    """
    _make_dataset(tmp_path, n_shards=3, rows_per_shard=10)
    from training.bc_data import ShardStream

    def _mk(split):
        return ShardStream(dataset_dir=tmp_path, split=split, batch_size=5, seed=99, val_split=0.1)

    rows_train_a = sum(b.obs.shape[0] for b in _mk("train"))
    rows_train_b = sum(b.obs.shape[0] for b in _mk("train"))
    rows_val_a = sum(b.obs.shape[0] for b in _mk("val"))

    assert rows_train_a == rows_train_b, (
        f"train row count not reproducible: {rows_train_a} vs {rows_train_b}"
    )
    assert rows_train_a + rows_val_a == 30, "disjoint: train+val must cover all 30 rows"


# ── Property-based tests ──────────────────────────────────────────────────────


@given(st.integers(min_value=2, max_value=8))
@hyp_settings(max_examples=5, deadline=None)
def test_pbt_when_any_number_of_shards_then_list_shards_is_always_sorted(n_shards):
    """
    Invariant: list_shards always returns shards in sorted (ascending) order,
    for any number of shards in [2, 8].
    Derived from: 'sorted(glob("shard_*.npz")) — deterministic order'.
    """
    from training.bc_data import list_shards

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        for i in range(n_shards):
            _write_shard(p / f"shard_{i:03d}.npz", 5, rng)
        result = list_shards(p)
        names = [Path(str(r)).name for r in result]
        assert names == sorted(names), f"list_shards not sorted for n_shards={n_shards}: {names}"


@given(
    st.integers(min_value=4, max_value=8),
    st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False),
    st.integers(min_value=0, max_value=999),
)
@hyp_settings(max_examples=5, deadline=None)
def test_pbt_when_any_valid_split_params_then_train_plus_val_equals_total_rows(
    n_shards, val_split, seed
):
    """
    Invariant: train_rows + val_rows == manifest total_rows for all valid
    (n_shards ∈ [4,8], val_split ∈ [0.1, 0.5], seed) combinations.
    Derived from: 'train and val shard sets are disjoint' (partition property).
    """
    from training.bc_data import ShardStream, read_manifest

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        _make_dataset(p, n_shards=n_shards, rows_per_shard=5)
        manifest = read_manifest(p)
        total = manifest["total_rows"]

        def _mk(split):
            return ShardStream(
                dataset_dir=p, split=split, batch_size=5, seed=seed, val_split=val_split
            )

        train_rows = sum(b.obs.shape[0] for b in _mk("train"))
        val_rows = sum(b.obs.shape[0] for b in _mk("val"))

        assert train_rows + val_rows == total, (
            f"n_shards={n_shards} val_split={val_split:.2f} seed={seed}: "
            f"train({train_rows}) + val({val_rows}) = {train_rows + val_rows} != total({total})"
        )


@given(st.integers(min_value=0, max_value=9999))
@hyp_settings(max_examples=5, deadline=None)
def test_pbt_when_same_seed_used_twice_then_train_row_count_is_always_identical(seed):
    """
    Invariant: the same seed always produces the same shard assignment → same
    training row count, for any seed in [0, 9999].
    Derived from: 'reproducible across two constructions with the same seed'.
    """
    from training.bc_data import ShardStream

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        _make_dataset(p, n_shards=6, rows_per_shard=5, seed=0)

        def _mk():
            return ShardStream(dataset_dir=p, split="train", batch_size=5, seed=seed, val_split=0.3)

        rows_a = sum(b.obs.shape[0] for b in _mk())
        rows_b = sum(b.obs.shape[0] for b in _mk())

        assert rows_a == rows_b, (
            f"seed={seed}: train row counts differ between identical-seed constructions: "
            f"{rows_a} vs {rows_b}"
        )
