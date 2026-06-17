"""BC pretraining loop — supervised imitation learning into ActorCritic."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from src.config import TrainingConfig
from src.models.actor_critic import ActorCritic
from src.models.utils import get_obs_dim
from src.utils.constants import ACTION_DIMS
from src.utils.log import get_logger
from src.utils.seed import set_global_seeds
from training.bc_checkpoint import write_bc_checkpoint
from training.bc_data import ShardStream
from training.bc_loss import LossStats, combined_loss
from training.early_stopping import EarlyStopper


@dataclasses.dataclass(frozen=True)
class BCResult:
    """Summary returned by pretrain()."""

    epochs_run: int
    best_val_loss: float
    final_train_acc: float
    final_val_acc: float


def pretrain(cfg: TrainingConfig, *, output_path: str | None = None) -> BCResult:
    """Supervised BC pretraining loop; saves a PPO-warm-start checkpoint.

    Args:
        cfg: Training configuration with BC settings.
        output_path: Override cfg.bc.output_path if provided.
    """
    set_global_seeds(cfg.bc.seed)
    net, opt = _build_net(cfg)
    train_s, val_s = _build_streams(cfg)
    trainer = _Trainer(net, opt, train_s, val_s, cfg, output_path=output_path)
    return trainer.run()


def _build_net(cfg: TrainingConfig) -> tuple[ActorCritic, torch.optim.Optimizer]:
    net = ActorCritic(
        obs_dim=get_obs_dim(),
        hidden_size=cfg.network.hidden_sizes[0],
        num_layers=len(cfg.network.hidden_sizes),
        action_dims=ACTION_DIMS,
    )
    return net, torch.optim.Adam(net.parameters(), lr=cfg.bc.lr)


def _build_streams(cfg: TrainingConfig) -> tuple[ShardStream, ShardStream]:
    kw = {
        "dataset_dir": cfg.bc.dataset_dir,
        "batch_size": cfg.bc.batch_size,
        "seed": cfg.bc.seed,
        "val_split": cfg.bc.val_split,
    }
    train_s = ShardStream(split="train", **kw)
    val_s = ShardStream(split="val", **kw)
    assert train_s._shards, "BC train split has zero shards — increase dataset or reduce val_split"
    assert val_s._shards, "BC val split has zero shards — reduce val_split or add more shards"
    return train_s, val_s


def _snapshot(net: ActorCritic) -> dict:
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _log_epoch(epoch: int, train: LossStats, val: LossStats) -> None:
    get_logger("bc_trainer").info(
        "epoch=%d | train acc=%.4f mse=%.4f | val acc=%.4f mse=%.4f",
        epoch,
        train.top1_acc,
        train.value_mse,
        val.top1_acc,
        val.value_mse,
    )


def _compute_loss(net: ActorCritic, batch, cfg: TrainingConfig) -> tuple:
    logits, value = net(batch.obs)
    return combined_loss(
        logits,
        batch.targets,
        value,
        batch.mc_return,
        value_loss_coef=cfg.bc.value_loss_coef,
        entropy_coef=cfg.bc.entropy_coef,
        label_smoothing=cfg.bc.label_smoothing,
    )


def _train_step(
    net: ActorCritic, optimizer: torch.optim.Optimizer, batch, cfg: TrainingConfig
) -> LossStats:
    loss, stats = _compute_loss(net, batch, cfg)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    return stats


def _train_epoch(
    net: ActorCritic, optimizer: torch.optim.Optimizer, stream: ShardStream, cfg: TrainingConfig
) -> LossStats:
    net.train()
    return _avg_stats([_train_step(net, optimizer, b, cfg) for b in stream])


def _eval_step(net: ActorCritic, batch, cfg: TrainingConfig) -> LossStats:
    _, stats = _compute_loss(net, batch, cfg)
    return stats


def _eval_epoch(net: ActorCritic, stream: ShardStream, cfg: TrainingConfig) -> LossStats:
    net.eval()
    stats: list[LossStats] = []
    with torch.no_grad():
        for b in stream:
            stats.append(_eval_step(net, b, cfg))
    return _avg_stats(stats)


def _avg_stats(stats_list: list[LossStats]) -> LossStats:
    if not stats_list:
        return LossStats(policy_ce=0.0, value_mse=0.0, entropy=0.0, top1_acc=0.0)
    n = len(stats_list)
    return LossStats(
        policy_ce=sum(s.policy_ce for s in stats_list) / n,
        value_mse=sum(s.value_mse for s in stats_list) / n,
        entropy=sum(s.entropy for s in stats_list) / n,
        top1_acc=sum(s.top1_acc for s in stats_list) / n,
    )


class _BestTracker:
    """Tracks the best-on-val network state with a deep-copied state_dict."""

    def __init__(self, net: ActorCritic) -> None:
        self.val_loss = float("inf")
        self.state = _snapshot(net)
        self.final_train_acc = 0.0
        self.final_val_acc = 0.0

    def update(self, val_loss: float, net: ActorCritic, train_acc: float, val_acc: float) -> None:
        self.final_train_acc, self.final_val_acc = train_acc, val_acc
        if val_loss < self.val_loss:
            self.val_loss, self.state = val_loss, _snapshot(net)


class _Trainer:
    """Epoch-loop coordinator — delegates to module-level step helpers."""

    def __init__(
        self, net, opt, train_s, val_s, cfg: TrainingConfig, *, output_path: str | None = None
    ) -> None:
        self._net, self._opt, self._cfg = net, opt, cfg
        self._train, self._val = train_s, val_s
        self._stopper = EarlyStopper(patience=cfg.bc.early_stop_patience)
        self._best = _BestTracker(net)
        self._epoch = 0
        self._output_path = output_path or cfg.bc.output_path

    def run(self) -> BCResult:
        for epoch in range(self._cfg.bc.epochs):
            self._epoch = epoch
            if self._step():
                break
        self._save()
        return BCResult(
            self._epoch + 1,
            self._best.val_loss,
            self._best.final_train_acc,
            self._best.final_val_acc,
        )

    def _step(self) -> bool:
        t = _train_epoch(self._net, self._opt, self._train, self._cfg)
        v = _eval_epoch(self._net, self._val, self._cfg)
        val_loss = v.policy_ce + self._cfg.bc.value_loss_coef * v.value_mse
        _log_epoch(self._epoch, t, v)
        self._best.update(val_loss, self._net, t.top1_acc, v.top1_acc)
        return self._stopper(-val_loss)

    def _save(self) -> None:
        self._net.load_state_dict(self._best.state)
        write_bc_checkpoint(Path(self._output_path), self._net, self._opt, self._cfg)
