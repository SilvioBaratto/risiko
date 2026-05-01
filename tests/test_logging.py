"""Tests for TensorBoard logging wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from src.multi_agent import GameResult
from src.tb_logger import TensorBoardLogger


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    """Provide a fresh temporary log directory."""
    d = tmp_path / "tb_logs"
    d.mkdir()
    return d


def _read_scalars(log_dir: Path, tag: str) -> list[tuple[int, float]]:
    """Read scalar values from the first tfevents file in *log_dir*."""
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert event_files, f"No event files found in {log_dir}"
    ea = EventAccumulator(str(event_files[0]))
    ea.Reload()
    events = ea.Scalars(tag)
    return [(int(e.step), e.value) for e in events]


def _close_and_wait(logger: TensorBoardLogger) -> None:
    """Flush and close the writer so data appears on disk."""
    logger._writer.close()


class TestTensorBoardLoggerInit:
    """Construction sets up the log directory and writer."""

    def test_creates_log_dir(self, log_dir: Path) -> None:
        target = log_dir / "run_1"
        logger = TensorBoardLogger(target)
        assert target.exists()
        _close_and_wait(logger)

    def test_creates_subdir_for_run(self, log_dir: Path) -> None:
        target = log_dir / "my_run"
        logger = TensorBoardLogger(target)
        assert target.exists()
        _close_and_wait(logger)


class TestLogTrainingStep:
    """Training metrics are written as scalars."""

    def test_logs_entropy(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        metrics = {
            "policy_loss": 0.5,
            "value_loss": 1.2,
            "entropy_loss": 0.3,
            "approx_kl": 0.01,
            "clip_fraction": 0.1,
            "explained_variance": 0.8,
        }
        logger.log_training_step(metrics, episode=5)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "train/entropy_loss")
        assert len(values) == 1
        assert values[0] == (5, pytest.approx(0.3))

    def test_logs_value_loss(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        metrics = {"value_loss": 2.0, "entropy_loss": 0.1, "explained_variance": 0.5}
        logger.log_training_step(metrics, episode=3)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "train/value_loss")
        assert values[0] == (3, pytest.approx(2.0))

    def test_logs_explained_variance(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        metrics = {"entropy_loss": 0.1, "explained_variance": 0.95}
        logger.log_training_step(metrics, episode=1)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "train/explained_variance")
        assert values[0] == (1, pytest.approx(0.95))

    def test_ignores_missing_keys(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        logger.log_training_step({"value_loss": 1.0}, episode=1)
        _close_and_wait(logger)
        # Should not raise
        values = _read_scalars(log_dir, "train/value_loss")
        assert len(values) == 1


class TestLogGameResult:
    """Episode-level metrics are written from GameResult."""

    def test_logs_episode_reward(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        result = GameResult(
            winner=0,
            n_turns=3,
            territory_history=[np.array([21, 21]), np.array([20, 22])],
            elimination_order=[],
            card_trade_turns=[],
            action_log=[
                {
                    "action_type": 0,
                    "param_a": 0,
                    "param_b": 0,
                    "param_c": 0,
                    "param_d": 0,
                    "player": 0,
                },
                {
                    "action_type": 1,
                    "param_a": 0,
                    "param_b": 0,
                    "param_c": 0,
                    "param_d": 0,
                    "player": 0,
                },
                {
                    "action_type": 5,
                    "param_a": 0,
                    "param_b": 0,
                    "param_c": 0,
                    "param_d": 0,
                    "player": 0,
                },
            ],
            trajectories=[
                ({}, {}, 1.0),
                ({}, {}, 2.0),
                ({}, {}, 3.0),
            ],
        )
        logger.log_game_result(result, player_id=0, episode=7)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "episode/reward")
        assert len(values) == 1
        assert values[0] == (7, pytest.approx(6.0))

    def test_logs_win(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        result = GameResult(
            winner=0,
            n_turns=1,
            territory_history=[np.array([21, 21])],
            elimination_order=[],
            card_trade_turns=[],
            action_log=[],
            trajectories=[({}, {}, 0.0)],
        )
        logger.log_game_result(result, player_id=0, episode=1)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "episode/win")
        assert values[0] == (1, pytest.approx(1.0))

    def test_logs_loss(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        result = GameResult(
            winner=1,
            n_turns=1,
            territory_history=[np.array([21, 21])],
            elimination_order=[],
            card_trade_turns=[],
            action_log=[],
            trajectories=[({}, {}, 0.0)],
        )
        logger.log_game_result(result, player_id=0, episode=2)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "episode/win")
        assert values[0] == (2, pytest.approx(0.0))

    def test_logs_draw(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        result = GameResult(
            winner=None,
            n_turns=1,
            territory_history=[np.array([21, 21])],
            elimination_order=[],
            card_trade_turns=[],
            action_log=[],
            trajectories=[({}, {}, 0.0)],
        )
        logger.log_game_result(result, player_id=0, episode=3)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "episode/win")
        assert values[0] == (3, pytest.approx(0.0))


class TestRollingWinRate:
    """Rolling win rate is computed over the last 100 episodes."""

    def test_logs_rolling_win_rate(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        for i in range(10):
            winner = 0 if i < 5 else 1
            result = GameResult(
                winner=winner,
                n_turns=1,
                territory_history=[np.array([21, 21])],
                elimination_order=[],
                card_trade_turns=[],
                action_log=[],
                trajectories=[({}, {}, 0.0)],
            )
            logger.log_game_result(result, player_id=0, episode=i + 1)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "episode/win_rate_100")
        assert len(values) == 10
        # Last value should be 5/10 = 0.5
        assert values[-1] == (10, pytest.approx(0.5))

    def test_uses_maxlen_100(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        for i in range(110):
            winner = 0 if i < 50 else 1
            result = GameResult(
                winner=winner,
                n_turns=1,
                territory_history=[np.array([21, 21])],
                elimination_order=[],
                card_trade_turns=[],
                action_log=[],
                trajectories=[({}, {}, 0.0)],
            )
            logger.log_game_result(result, player_id=0, episode=i + 1)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "episode/win_rate_100")
        # Last 100: episodes 10-109, winners 0 for 10-49 (40 wins),
        # 1 for 50-109 (60 losses). So 40/100 = 0.4
        assert values[-1] == (110, pytest.approx(0.4))


class TestContinentControl:
    """Per-continent control ratios are logged from territory ownership."""

    def test_logs_continent_control_from_obs(self, log_dir: Path) -> None:
        """Continent control computed from the last observation in trajectories."""
        logger = TensorBoardLogger(log_dir)
        # Player 0 owns all North America (0-8), player 1 owns rest
        territory_owner = np.zeros(42, dtype=np.int32)
        territory_owner[9:42] = 1
        result = GameResult(
            winner=0,
            n_turns=1,
            territory_history=[np.array([9, 33])],
            elimination_order=[],
            card_trade_turns=[],
            action_log=[],
            trajectories=[
                (
                    {"territory_owner": territory_owner},
                    {},
                    0.0,
                ),
            ],
        )
        logger.log_game_result(result, player_id=0, episode=1)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "continent/North America")
        assert len(values) == 1
        assert values[0] == (1, pytest.approx(1.0))

    def test_logs_all_continents(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        territory_owner = np.zeros(42, dtype=np.int32)
        territory_owner[9:42] = 1
        result = GameResult(
            winner=0,
            n_turns=1,
            territory_history=[np.array([9, 33])],
            elimination_order=[],
            card_trade_turns=[],
            action_log=[],
            trajectories=[
                ({"territory_owner": territory_owner}, {}, 0.0),
            ],
        )
        logger.log_game_result(result, player_id=0, episode=1)
        _close_and_wait(logger)
        for name in ["North America", "South America", "Europe", "Africa", "Asia", "Australia"]:
            values = _read_scalars(log_dir, f"continent/{name}")
            assert len(values) == 1


class TestEpisodeMetrics:
    """New per-episode scalars: length, card-trade frequency, mean territory."""

    def _make_result(
        self,
        n_turns: int = 4,
        card_trade_turns: list[int] | None = None,
        territory_history: list[np.ndarray] | None = None,
    ) -> GameResult:
        if card_trade_turns is None:
            card_trade_turns = []
        if territory_history is None:
            territory_history = [np.array([20, 22]) for _ in range(n_turns)]
        return GameResult(
            winner=0,
            n_turns=n_turns,
            territory_history=territory_history,
            elimination_order=[],
            card_trade_turns=card_trade_turns,
            action_log=[],
            trajectories=[],
        )

    def test_when_n_turns_5_logs_length_5(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        result = self._make_result(n_turns=5)
        logger.log_game_result(result, player_id=0, episode=1)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "episode/length")
        assert values[0] == (1, pytest.approx(5.0))

    def test_when_two_trades_in_4_turns_logs_frequency_0_5(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        result = self._make_result(n_turns=4, card_trade_turns=[1, 3])
        logger.log_game_result(result, player_id=0, episode=2)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "episode/card_trade_frequency")
        assert values[0] == (2, pytest.approx(0.5))

    def test_when_n_turns_is_0_card_trade_frequency_is_0(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        result = self._make_result(n_turns=0, card_trade_turns=[], territory_history=[])
        logger.log_game_result(result, player_id=0, episode=3)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "episode/card_trade_frequency")
        assert values[0] == (3, pytest.approx(0.0))

    def test_when_territory_history_logs_mean_territory(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        history = [np.array([10, 32]), np.array([12, 30]), np.array([14, 28])]
        result = self._make_result(n_turns=3, territory_history=history)
        logger.log_game_result(result, player_id=0, episode=4)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "episode/mean_territory")
        assert values[0] == (4, pytest.approx(12.0))

    def test_when_territory_history_empty_mean_territory_is_0(self, log_dir: Path) -> None:
        logger = TensorBoardLogger(log_dir)
        result = self._make_result(n_turns=0, territory_history=[])
        logger.log_game_result(result, player_id=0, episode=5)
        _close_and_wait(logger)
        values = _read_scalars(log_dir, "episode/mean_territory")
        assert values[0] == (5, pytest.approx(0.0))
