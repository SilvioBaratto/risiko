"""Tests for the hardened checkpoint loader (src.utils.safe_torch).

Covers both halves of the contract:
- A malicious checkpoint (pickle reduce -> arbitrary callable) is rejected, not
  executed — the whole point of weights_only=True.
- A genuine checkpoint containing numpy RNG state and a config dataclass still
  loads, so resume keeps working.
"""

from __future__ import annotations

import os
import pickle
import random

import numpy as np
import pytest
import torch

from src.config import PPOConfig, TrainingConfig
from src.utils.safe_torch import safe_torch_load


class _ReduceToCallable:
    """Pickles into a call of an un-allowlisted global (simulates an RCE payload)."""

    def __reduce__(self):
        # os.getcwd is harmless if it ever ran; weights_only must still REFUSE it.
        return (os.getcwd, ())


def test_when_checkpoint_references_unlisted_global_then_load_is_refused(tmp_path):
    evil = tmp_path / "evil.pt"
    torch.save({"payload": _ReduceToCallable()}, evil)

    with pytest.raises(pickle.UnpicklingError):
        safe_torch_load(evil)


def test_when_checkpoint_has_numpy_rng_and_config_then_load_succeeds(tmp_path):
    np.random.seed(7)
    random.seed(7)
    torch.manual_seed(7)
    payload = {
        "trainer_state": {
            "model": {"w": torch.randn(2, 2)},
            "optimizer": {"step": 1},
            "config": PPOConfig(lr=0.001),  # dataclass embedded by PPOTrainer.save_checkpoint
        },
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),  # contains a uint32 ndarray
            "random": random.getstate(),
            "env": {"bit_generator": "PCG64", "state": {"state": 1, "inc": 3}},
        },
        "episode": 42,
        "config": {"seed": 42, "ppo": {"lr": 0.001}},
    }
    path = tmp_path / "good.pt"
    torch.save(payload, path)

    loaded = safe_torch_load(path)

    assert loaded["episode"] == 42
    assert isinstance(loaded["trainer_state"]["config"], PPOConfig)
    assert loaded["rng_state"]["numpy"][0] == "MT19937"
    assert isinstance(loaded["rng_state"]["numpy"][1], np.ndarray)


def test_when_config_dataclass_round_trips_then_values_match(tmp_path):
    cfg = TrainingConfig(total_timesteps=1234, ppo=PPOConfig(lr=0.002))
    path = tmp_path / "cfg.pt"
    torch.save({"config": cfg}, path)

    loaded = safe_torch_load(path)

    assert loaded["config"].total_timesteps == 1234
    assert loaded["config"].ppo.lr == pytest.approx(0.002)
