"""Hardened ``torch.load`` that refuses arbitrary code execution.

``torch.load(..., weights_only=False)`` unpickles arbitrary Python objects, so a
malicious ``.pt`` file can run code on load (pickle ``__reduce__`` -> RCE). We
instead load with ``weights_only=True`` — which executes no code — and explicitly
allowlist only the benign types our own checkpoints legitimately contain (the
frozen config dataclasses and the numpy RNG state). An attacker therefore cannot
smuggle a dangerous global past the allowlist, while genuine checkpoints (and
resume) keep working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

_REGISTERED = False


def _config_safe_globals() -> list[Any]:
    """Project config dataclasses stored inside checkpoints."""
    from src.config import (
        BCConfig,
        EarlyStopConfig,
        HeuristicConfig,
        NetworkConfig,
        PPOConfig,
        SelfPlayConfig,
        TrainingConfig,
    )
    from src.utils.reward_config import RewardConfig

    return [
        TrainingConfig,
        PPOConfig,
        NetworkConfig,
        SelfPlayConfig,
        EarlyStopConfig,
        RewardConfig,
        BCConfig,
        HeuristicConfig,
    ]


def _numpy_safe_globals() -> list[Any]:
    """Globals needed to unpickle ``np.random.get_state()`` (an ndarray of uint32)."""
    import numpy as np

    allow: list[Any] = [np.ndarray, np.dtype]
    core = getattr(np, "_core", None) or getattr(np, "core", None)
    reconstruct = getattr(getattr(core, "multiarray", None), "_reconstruct", None)
    if reconstruct is not None:
        allow.append(reconstruct)
    dtypes_mod = getattr(np, "dtypes", None)
    if dtypes_mod is not None:
        for name in ("UInt32DType", "Int64DType", "Float64DType"):
            klass = getattr(dtypes_mod, name, None)
            if klass is not None:
                allow.append(klass)
    return allow


def _ensure_registered() -> None:
    """Register the allowlist with torch exactly once (process-wide, idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    torch.serialization.add_safe_globals(_config_safe_globals() + _numpy_safe_globals())
    _REGISTERED = True


def safe_torch_load(path: str | Path) -> Any:
    """Load a checkpoint safely: ``weights_only=True`` plus a benign-type allowlist.

    Raises:
        pickle.UnpicklingError: If the file references any global outside the
            allowlist (e.g. a malicious payload attempting code execution).
    """
    _ensure_registered()
    return torch.load(Path(path), weights_only=True)


__all__ = ["safe_torch_load"]
