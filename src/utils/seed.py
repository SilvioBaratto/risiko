"""Global seeding utility for reproducible training.

Fixes torch, numpy, and the built-in random module together.
Returns a local numpy Generator so callers never rely on the global RNG.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seeds(seed: int) -> np.random.Generator:
    """Fix all global random seeds and return a local numpy Generator.

    Args:
        seed: Integer seed for all backends.

    Returns:
        A new numpy Generator seeded with the same value.
        Callers should use this Generator for sampling instead of
        np.random or random to avoid state pollution.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return np.random.default_rng(seed)
