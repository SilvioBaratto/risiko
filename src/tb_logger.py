"""TensorBoard logging wrapper for Risiko RL training metrics."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.multi_agent import GameResult
from src.utils.constants import CONTINENTS


class TensorBoardLogger:
    """Write training metrics and episode statistics to TensorBoard."""

    def __init__(self, log_dir: Path | str) -> None:
        """Create a logger backed by a SummaryWriter.

        Args:
            log_dir: Directory for tfevents files (Path or str).
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._win_history: deque[int] = deque(maxlen=100)

    def log_training_step(self, metrics: dict[str, float], episode: int) -> None:
        """Log scalar metrics from a PPO update step.

        Args:
            metrics: Dict returned by ``PPOTrainer.update()``.
            episode: Current episode number for the x-axis.
        """
        for key, tag in _METRIC_TAGS.items():
            if key in metrics:
                self._writer.add_scalar(tag, metrics[key], episode)
        self._writer.flush()

    def log_game_result(
        self,
        result: GameResult,
        player_id: int,
        episode: int,
    ) -> None:
        """Log episode-level statistics.

        Args:
            result: Outcome of a single game.
            player_id: Index of the player whose perspective to log.
            episode: Current episode number for the x-axis.
        """
        reward = self._compute_reward(result, player_id)
        self._writer.add_scalar("episode/reward", reward, episode)

        win = self._compute_win(result, player_id)
        self._win_history.append(win)
        self._writer.add_scalar("episode/win", win, episode)
        self._writer.add_scalar("episode/win_rate_100", self._rolling_win_rate(), episode)

        self._writer.add_scalar("episode/length", result.n_turns, episode)

        card_freq = len(result.card_trade_turns) / result.n_turns if result.n_turns > 0 else 0.0
        self._writer.add_scalar("episode/card_trade_frequency", card_freq, episode)

        if result.territory_history:
            mean_terr = float(np.mean([t[player_id] for t in result.territory_history]))
        else:
            mean_terr = 0.0
        self._writer.add_scalar("episode/mean_territory", mean_terr, episode)

        ratios = self._continent_ratios(result, player_id)
        for name, ratio in ratios.items():
            self._writer.add_scalar(f"continent/{name}", ratio, episode)

        self._writer.flush()

    def log_scalar(self, tag: str, value: float, episode: int) -> None:
        """Log a single named scalar at *episode* (e.g. early-stop metrics).

        Args:
            tag: TensorBoard scalar tag, e.g. ``"early_stop/win_rate_vs_random"``.
            value: Scalar value to record.
            episode: Current episode number for the x-axis.
        """
        self._writer.add_scalar(tag, value, episode)
        self._writer.flush()

    def close(self) -> None:
        """Flush and close the underlying writer."""
        self._writer.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_reward(self, result: GameResult, player_id: int) -> float:
        """Sum rewards for *player_id* across trajectories."""
        total = 0.0
        for i, t in enumerate(result.trajectories):
            if i < len(result.action_log) and result.action_log[i]["player"] == player_id:
                total += t.reward
        return total

    def _compute_win(self, result: GameResult, player_id: int) -> int:
        """Return 1 if *player_id* won the game, else 0."""
        return 1 if result.winner == player_id else 0

    def _rolling_win_rate(self) -> float:
        """Return mean of the last 100 win outcomes."""
        if not self._win_history:
            return 0.0
        return sum(self._win_history) / len(self._win_history)

    def _continent_ratios(self, result: GameResult, player_id: int) -> dict[str, float]:
        """Compute per-continent control ratios for *player_id*."""
        if not result.trajectories:
            return dict.fromkeys(CONTINENTS, 0.0)
        last_obs = result.trajectories[-1].obs
        owner = last_obs.get("territory_owner", np.zeros(42, dtype=np.int32))
        return {name: self._ratio(owner, player_id, tids) for name, tids in CONTINENTS.items()}

    @staticmethod
    def _ratio(owner: np.ndarray, player_id: int, territory_ids: list[int]) -> float:
        """Return fraction of *territory_ids* owned by *player_id*."""
        if not territory_ids:
            return 0.0
        owned = sum(1 for tid in territory_ids if owner[tid] == player_id)
        return owned / len(territory_ids)


_METRIC_TAGS: dict[str, str] = {
    "entropy_loss": "train/entropy_loss",
    "value_loss": "train/value_loss",
    "explained_variance": "train/explained_variance",
    "policy_loss": "train/policy_loss",
    "approx_kl": "train/approx_kl",
    "clip_fraction": "train/clip_fraction",
}
