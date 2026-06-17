"""Write a PPO-compatible checkpoint from a BC-trained network.

The payload mirrors the shape that ``SelfPlayTrainer.load_checkpoint`` and
``PPOTrainer.load_checkpoint`` expect, so ``risiko-rl train --resume`` can
warm-start from a BC artifact with no architecture changes.

Omitted keys (``opponent_state``, ``buffer_state``, ``rng_state``,
``best_metric_value``) are intentional: BC seeds the policy net only; PPO
self-play initialises all of them fresh on startup.  ``SelfPlayTrainer``
guards every optional key with ``.get``/``in``, so their absence is safe.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim

from src.config import TrainingConfig
from src.utils.log import get_logger

_log = get_logger("bc_checkpoint")


def write_bc_checkpoint(
    path: Path,
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
) -> None:
    """Save a BC-trained network as a PPO-loadable checkpoint.

    Args:
        path: Destination file path (parent directories are created).
        net: Trained ``ActorCritic`` whose weights to persist.
        optimizer: Optimizer whose state to persist (carries BC ``lr``).
        cfg: Full ``TrainingConfig``; ``cfg.ppo`` is written into
             ``trainer_state["config"]`` so ``PPOTrainer`` reads it correctly.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trainer_state": {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.ppo,
        },
        "config": cfg,
        "episode": 0,
    }
    torch.save(payload, path)
    _log.info("BC checkpoint written to %s", path)
