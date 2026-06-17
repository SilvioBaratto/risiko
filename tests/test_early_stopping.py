"""Tests for plateau-detection early stopping and its config wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import EarlyStopConfig, TrainingConfig, load_config, merge_cli_overrides
from training.early_stopping import EarlyStopper

# ---------------------------------------------------------------------------
# EarlyStopper — pure plateau-detection logic
# ---------------------------------------------------------------------------


class TestEarlyStopper:
    def test_when_first_update_then_is_best_and_does_not_stop(self) -> None:
        stopper = EarlyStopper(patience=3)
        assert stopper.update(0.1) is False
        assert stopper.is_best
        assert stopper.best_value == pytest.approx(0.1)
        assert stopper.best_step == 0

    def test_when_monotonic_improvement_then_never_stops(self) -> None:
        stopper = EarlyStopper(patience=2)
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            assert stopper.update(v) is False
            assert stopper.is_best
        assert stopper.num_bad_evals == 0

    def test_when_plateau_then_stops_after_exactly_patience(self) -> None:
        stopper = EarlyStopper(patience=3, min_delta=0.0, window=1)
        assert stopper.update(0.5) is False  # best
        # three non-improving evals → stop on the third
        assert stopper.update(0.5) is False  # bad 1
        assert stopper.update(0.5) is False  # bad 2
        assert stopper.update(0.5) is True  # bad 3 → stop
        assert stopper.should_stop
        assert stopper.num_bad_evals == 3

    def test_when_improvement_resets_patience(self) -> None:
        stopper = EarlyStopper(patience=2, min_delta=0.0, window=1)
        stopper.update(0.5)  # best
        stopper.update(0.5)  # bad 1
        assert stopper.num_bad_evals == 1
        assert stopper.update(0.6) is False  # improves → reset
        assert stopper.num_bad_evals == 0
        assert stopper.is_best

    def test_when_gain_below_min_delta_then_counts_as_bad(self) -> None:
        stopper = EarlyStopper(patience=2, min_delta=0.05, window=1)
        stopper.update(0.50)  # best
        # +0.02 < min_delta 0.05 → not an improvement
        assert stopper.update(0.52) is False
        assert not stopper.is_best
        assert stopper.num_bad_evals == 1
        # +0.10 > min_delta → improvement
        assert stopper.update(0.62) is False
        assert stopper.is_best

    def test_when_window_set_then_decision_uses_moving_average(self) -> None:
        stopper = EarlyStopper(patience=5, min_delta=0.0, window=2)
        stopper.update(0.0)  # smoothed 0.0  → best
        stopper.update(1.0)  # smoothed 0.5  → best
        # raw drops but the 2-window average (1.0+0.9)/2=0.95 still beats 0.5
        assert stopper.update(0.9) is False
        assert stopper.is_best
        assert stopper.best_value == pytest.approx(0.95)

    @pytest.mark.parametrize("bad_kwargs", [{"patience": 0}, {"window": 0}, {"min_delta": -1.0}])
    def test_when_invalid_args_then_raises(self, bad_kwargs: dict) -> None:
        kwargs = {"patience": 3, **bad_kwargs}
        with pytest.raises(ValueError):
            EarlyStopper(**kwargs)


# ---------------------------------------------------------------------------
# EarlyStopConfig — defaults, YAML loading, CLI overrides
# ---------------------------------------------------------------------------


class TestEarlyStopConfig:
    def test_when_default_then_disabled(self) -> None:
        cfg = TrainingConfig()
        assert isinstance(cfg.early_stop, EarlyStopConfig)
        assert cfg.early_stop.enabled is False
        assert cfg.early_stop.restore_best is True

    def test_when_yaml_has_early_stop_then_loaded(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text(
            yaml.dump(
                {
                    "seed": 1,
                    "early_stop": {
                        "enabled": True,
                        "eval_every": 25,
                        "patience": 7,
                        "min_delta": 0.02,
                        "window": 3,
                    },
                }
            )
        )
        cfg = load_config(path)
        assert cfg.early_stop.enabled is True
        assert cfg.early_stop.eval_every == 25
        assert cfg.early_stop.patience == 7
        assert cfg.early_stop.min_delta == pytest.approx(0.02)
        assert cfg.early_stop.window == 3

    def test_when_cli_override_then_applies_to_nested_field(self) -> None:
        cfg = TrainingConfig()
        merged = merge_cli_overrides(
            cfg, {"early_stop.enabled": "true", "early_stop.patience": "9"}
        )
        assert merged.early_stop.enabled is True
        assert merged.early_stop.patience == 9


# ---------------------------------------------------------------------------
# Trainer wiring — _maybe_early_stop drives the stop decision (games mocked)
# ---------------------------------------------------------------------------


class TestTrainerEarlyStopWiring:
    def _make_trainer(self, tmp_path: Path, **es_kwargs):
        from src.config import SelfPlayConfig
        from training.self_play import SelfPlayTrainer

        cfg = TrainingConfig(
            seed=0,
            self_play=SelfPlayConfig(n_players=2),
            early_stop=EarlyStopConfig(enabled=True, **es_kwargs),
        )
        return SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, log_dir=tmp_path / "tb")

    def test_when_disabled_then_no_stopper(self, tmp_path: Path) -> None:
        from training.self_play import SelfPlayTrainer

        cfg = TrainingConfig()  # early_stop disabled by default
        trainer = SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, log_dir=tmp_path / "tb")
        assert trainer._early_stopper is None
        assert trainer._maybe_early_stop() is False

    def test_when_plateau_then_stops_and_saves_best(self, tmp_path: Path, monkeypatch) -> None:
        trainer = self._make_trainer(
            tmp_path, eval_every=1, patience=3, min_delta=0.0, window=1, restore_best=False
        )
        scripted = iter([0.10, 0.20, 0.20, 0.20, 0.20])
        monkeypatch.setattr(trainer, "_evaluate_against_random", lambda: next(scripted))

        stops = []
        for ep in range(1, 6):
            trainer._episode = ep
            stops.append(trainer._maybe_early_stop())

        # best at the 0.20 eval (ep 2); then 3 flat evals exhaust patience
        assert stops == [False, False, False, False, True]
        assert (tmp_path / "best.pt").exists()
        assert trainer._early_stopper.best_value == pytest.approx(0.20)

    def test_when_off_schedule_then_skips_eval(self, tmp_path: Path, monkeypatch) -> None:
        trainer = self._make_trainer(tmp_path, eval_every=50, patience=3)
        called = {"n": 0}

        def _spy() -> float:
            called["n"] += 1
            return 0.5

        monkeypatch.setattr(trainer, "_evaluate_against_random", _spy)
        trainer._episode = 37  # not a multiple of eval_every
        assert trainer._maybe_early_stop() is False
        assert called["n"] == 0
