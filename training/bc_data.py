"""Memory-bounded sharded BC dataset loader with seeded train/val split.

Reads Cycle-2 shards written by ``bc_shard_writer.ShardWriter``.

Shuffling trade-off
-------------------
Full global shuffle requires loading all shards at once, violating the memory
bound.  Rows within a shard are already drawn across many games, so the
per-shard distribution is approximately i.i.d.  Shard visitation order is
determined by the seeded split permutation (fixed at construction).

Degenerate case: n_shards == 1 with val_split > 0 is not splittable.  Both
splits return the single shard with a warning.  Prefer datasets with >= 2 shards.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from src.models.utils import get_obs_dim
from src.utils.log import get_logger

_log = get_logger("bc_data")
_HEAD_NAMES = ("action_type", "param_a", "param_b", "param_c", "param_d")

__all__ = ["Batch", "read_manifest", "list_shards", "ShardStream"]


@dataclasses.dataclass(frozen=True)
class Batch:
    """One mini-batch from the BC dataset."""

    obs: torch.Tensor  # (B, 137) float32
    targets: dict[str, torch.Tensor]  # head → (B,) long
    mc_return: torch.Tensor  # (B,) float32


def read_manifest(dataset_dir: Path | str) -> dict:
    """Load manifest.json and assert obs_dim matches get_obs_dim().

    Raises:
        ValueError: if obs_dim in the manifest != get_obs_dim() (137).
    """
    path = Path(dataset_dir) / "manifest.json"
    manifest = json.loads(path.read_text())
    expected = get_obs_dim()
    if manifest["obs_dim"] != expected:
        raise ValueError(f"obs_dim mismatch: manifest={manifest['obs_dim']}, expected={expected}")
    return manifest


def list_shards(dataset_dir: Path | str) -> list[Path]:
    """Return sorted shard_*.npz paths — deterministic, zero-padded order."""
    return sorted(Path(dataset_dir).glob("shard_*.npz"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _n_val_shards(n: int, val_split: float) -> int:
    """Val shard count with a rounding guard.

    Returns 0 when val_split == 0.  Otherwise ``max(1, min(round(n*val_split),
    n-1))`` — guarantees both splits are non-empty when n >= 2.
    """
    if val_split <= 0.0:
        return 0
    return max(1, min(round(n * val_split), n - 1))


def _assign_shards(
    all_shards: list[Path],
    split: str,
    val_split: float,
    rng: np.random.Generator,
) -> list[Path]:
    """Partition shards into train/val via a seeded permutation."""
    n = len(all_shards)
    if n == 0:
        return []
    if n == 1 and val_split > 0.0:
        _log.warning("Only 1 shard; val and train will share it.")
        return all_shards
    perm: list[int] = rng.permutation(n).tolist()
    n_val = _n_val_shards(n, val_split)
    val_idx = set(perm[:n_val])
    if split == "val":
        return [all_shards[i] for i in sorted(val_idx)]
    return [all_shards[i] for i in sorted(set(range(n)) - val_idx)]


def _load_batches(path: Path, batch_size: int) -> Iterator[Batch]:
    """Yield Batch objects from one shard; file handle is released before first yield."""
    with np.load(path) as z:
        obs = z["obs"].copy()
        tgts = {h: z[h].copy() for h in _HEAD_NAMES}
        mc_ret = z["mc_return"].copy()
    n = obs.shape[0]
    for s in range(0, n, batch_size):
        e = s + batch_size
        yield Batch(
            obs=torch.from_numpy(obs[s:e]),
            targets={h: torch.from_numpy(tgts[h][s:e]).long() for h in _HEAD_NAMES},
            mc_return=torch.from_numpy(mc_ret[s:e]),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ShardStream:
    """Stream mini-batches from a sharded BC dataset for one split.

    Construction is O(1) and loads no shard data.  At most one shard's arrays
    are resident in memory at a time (copy-out pattern inside ``_load_batches``).

    Args:
        dataset_dir: Directory with shard_*.npz files and manifest.json.
        split: ``"train"`` or ``"val"``.
        batch_size: Rows per Batch.
        seed: Seed for the shard-level permutation that determines the split.
        val_split: Fraction of shards assigned to validation (0 <= val_split < 1).
            When 0, all shards go to train.  When > 0 and n_shards >= 2, at
            least 1 shard is guaranteed in each split (rounding guard).
    """

    def __init__(
        self,
        dataset_dir: Path | str,
        split: str,
        batch_size: int,
        seed: int,
        val_split: float,
    ) -> None:
        """Assign shards to this split; no shard data is loaded here."""
        rng = np.random.default_rng(seed)
        all_shards = list_shards(dataset_dir)
        self._shards = _assign_shards(all_shards, split, val_split, rng)
        self._batch_size = batch_size
        _log.debug("ShardStream(%s): %d/%d shards", split, len(self._shards), len(all_shards))

    def __iter__(self) -> Iterator[Batch]:
        """Yield all Batch objects for this split, one shard at a time."""
        for path in self._shards:
            yield from _load_batches(path, self._batch_size)
