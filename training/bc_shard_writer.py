"""Sharded .npz writer and obs-encoding helper for BC dataset generation.

Determinism contract (array-level, not file-byte-level):
  Two ``ShardWriter`` runs from identical (obs, labels) sequences produce
  shards whose arrays compare equal under ``np.array_equal``.  File-byte
  equality is NOT guaranteed — ``np.savez`` embeds per-entry timestamps in the
  ZIP container (``ZipInfo.date_time`` defaults to the current local time), so
  two shards written from byte-identical data differ as raw bytes across runs.
  All round-trip tests should use ``np.load`` + ``np.array_equal`` / assertion
  helpers, never ``filecmp.cmp``.  See: numpy/savez + zipfile.ZipInfo.date_time.

  ``manifest.json`` IS byte-stable: keys are sorted and no wall-clock value
  enters it; only ``total_rows``, ``obs_dim``, ``seed``, and ``config_hash``
  (a SHA-256 of a canonical sorted-key JSON dump of the config).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.models.utils import flatten_obs, get_obs_dim, stack_obs

__all__ = ["ShardWriter", "encode_obs", "compute_config_hash"]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def encode_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Convert a single env obs dict to a 137-dim float32 vector."""
    batched = stack_obs([obs], "cpu")
    flat = flatten_obs(batched)
    return flat[0].detach().numpy().astype(np.float32)


_PATH_FIELDS = frozenset({"dataset_dir", "output_path"})


def compute_config_hash(cfg: Any) -> str:
    """Return a hex-digest SHA-256 of the config's canonical JSON dump.

    Deterministic: same hyperparameter fields → same hash. Output-path fields
    (``dataset_dir``, ``output_path``) are excluded so two runs writing to
    different directories but using the same hyperparameters share a hash.
    Wall-clock / nondeterministic values must NOT enter ``cfg``.
    """
    raw = dataclasses.asdict(cast(Any, cfg)) if dataclasses.is_dataclass(cfg) else dict(cfg)
    for key in _PATH_FIELDS:
        raw.pop(key, None)
    canonical = json.dumps(raw, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Internal buffer
# ---------------------------------------------------------------------------


class _RowBuffer:
    """Accumulates (obs, labels) rows; drains them as stacked numpy arrays."""

    def __init__(self) -> None:
        self._obs: list[np.ndarray] = []
        self._labels: list[dict[str, Any]] = []

    def append(self, obs: np.ndarray, labels: dict[str, Any]) -> None:
        self._obs.append(obs)
        self._labels.append(labels)

    def size(self) -> int:
        return len(self._obs)

    def drain(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return stacked arrays then clear the buffer."""
        obs_arr = np.stack(self._obs).astype(np.float32)
        label_arrs = {k: np.array([lab[k] for lab in self._labels]) for k in self._labels[0]}
        self._obs.clear()
        self._labels.clear()
        return obs_arr, label_arrs


# ---------------------------------------------------------------------------
# Shard flusher
# ---------------------------------------------------------------------------


class _ShardFlusher:
    """Serialises one drained buffer to a zero-padded ``.npz`` file."""

    def __init__(self, dataset_dir: Path) -> None:
        self._dir = dataset_dir

    def write(self, index: int, obs: np.ndarray, labels: dict[str, np.ndarray]) -> int:
        """Write shard and return the number of rows written."""
        path = self._dir / f"shard_{index:04d}.npz"
        np.savez(str(path), obs=obs, **labels)
        return int(obs.shape[0])


# ---------------------------------------------------------------------------
# ShardWriter
# ---------------------------------------------------------------------------


class ShardWriter:
    """Buffers BC rows and writes fixed-width ``.npz`` shards to disk.

    Usage::

        with ShardWriter(cfg) as writer:
            writer.write(obs_vector, labels_dict)
        # close() is called automatically; manifest.json is written.

    ``cfg`` must expose ``dataset_dir`` (str/Path), ``shard_size`` (int), and
    ``seed`` (int).  Any frozen dataclass with those fields is accepted.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialise writer from a config object with dataset_dir/shard_size/seed."""
        self._dir = Path(cfg.dataset_dir)
        self._shard_size: int = cfg.shard_size
        self._seed: int = cfg.seed
        self._cfg_hash: str = compute_config_hash(cfg)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._buf = _RowBuffer()
        self._flusher = _ShardFlusher(self._dir)
        self._shard_idx: int = 0
        self._total_rows: int = 0

    def write(self, obs: np.ndarray, labels: dict[str, Any]) -> None:
        """Append one row; flush a shard when the buffer reaches ``shard_size``."""
        self._buf.append(obs, labels)
        if self._buf.size() >= self._shard_size:
            self._flush()

    def close(self) -> None:
        """Flush any remaining rows then write ``manifest.json``."""
        if self._buf.size() > 0:
            self._flush()
        self._write_manifest()

    def _flush(self) -> None:
        obs, label_arrs = self._buf.drain()
        rows = self._flusher.write(self._shard_idx, obs, label_arrs)
        self._shard_idx += 1
        self._total_rows += rows

    def _write_manifest(self) -> None:
        manifest = {
            "config_hash": self._cfg_hash,
            "obs_dim": get_obs_dim(),
            "seed": self._seed,
            "total_rows": self._total_rows,
        }
        (self._dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))

    def __enter__(self) -> ShardWriter:
        """Return self for use as a context manager."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Call close() on context exit."""
        self.close()
