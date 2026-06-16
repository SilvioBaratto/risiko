"""Self-play smoke tests for issue #62.

Covers:
- Opponent promotion: weights copied after _promote_current_to_opponent()
- GAE on compute_advantages(): buffer stores transitions, advantages are computed
- NaN log_prob transitions: skipped by SelfPlayTrainer._buffer_learner_transitions()
- Checkpoint resumability: episode counter, RNG state, opponent round-trip
- LLM pool wiring: N-1 LLMOpponents built from profiles YAML (Ollama mocked)

All tests use tiny sizes and seed=0 for speed.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

from src.config import SelfPlayConfig, TrainingConfig
from src.models.replay_buffer import RolloutBuffer
from src.utils.reward_config import RewardConfig

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _minimal_cfg(**overrides) -> TrainingConfig:
    """Minimal TrainingConfig for smoke tests (tiny n_steps so buffer fills fast)."""
    from src.config import PPOConfig

    base = {
        "seed": 0,
        "self_play": SelfPlayConfig(n_players=2),
        "ppo": PPOConfig(n_steps=8),
        "reward": RewardConfig(),
    }
    base.update(overrides)
    return TrainingConfig(**base)


def _make_trainer(tmp_path: pathlib.Path, **cfg_overrides):
    from training.self_play import SelfPlayTrainer

    cfg = _minimal_cfg(**cfg_overrides)
    return SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=30)


def _minimal_obs() -> dict:
    return {
        "current_player": np.array(0, dtype=np.int32),
        "phase": np.array(1, dtype=np.int32),
        "reinforcements_remaining": np.array(3, dtype=np.int32),
        "trade_count": np.array(0, dtype=np.int32),
        "cards": np.zeros((5, 4), dtype=np.int32),
        "eliminated": np.zeros(6, dtype=np.int32),
        "territory_owner": np.zeros(42, dtype=np.int32),
        "armies": np.ones(42, dtype=np.int32),
        "continent_control": np.zeros(6, dtype=np.int32),
        "turn_capture": np.array(0, dtype=np.int32),
        "n_players": np.array(2, dtype=np.int32),
    }


def _minimal_action() -> dict:
    return {k: torch.tensor(0) for k in ("action_type", "param_a", "param_b", "param_c", "param_d")}


def _fake_ollama_response(index: int = 0) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "message": {"content": json.dumps({"action_index": index})}
    }
    resp.raise_for_status = MagicMock(return_value=None)
    return resp


def _write_player_profiles(path: pathlib.Path, n: int) -> None:
    profiles = [
        {
            "player_id": i,
            "temperature": 0.1,
            "top_p": 0.9,
            "strategy_hint": "play optimally",
            "model": "gpt-oss:120b",
        }
        for i in range(n)
    ]
    path.write_text(yaml.dump(profiles))


# ---------------------------------------------------------------------------
# Opponent promotion
# ---------------------------------------------------------------------------


class TestOpponentPromotion:
    """_promote_current_to_opponent() copies learner weights to the frozen net."""

    def test_after_promote_opponent_weights_equal_learner(self, tmp_path) -> None:
        trainer = _make_trainer(tmp_path)

        with torch.no_grad():
            for p in trainer._agent._net.parameters():
                torch.nn.init.constant_(p, 7.0)

        trainer._promote_current_to_opponent()

        for (k, lv), (_, ov) in zip(
            trainer._agent._net.state_dict().items(),
            trainer._opponent_net.state_dict().items(),
            strict=True,
        ):
            assert torch.allclose(lv, ov), f"Param '{k}' mismatch after promotion"

    def test_promote_threshold_default_is_0_55(self, tmp_path) -> None:
        trainer = _make_trainer(tmp_path)
        assert trainer._cfg.self_play.promote_threshold == pytest.approx(0.55)

    def test_no_promotion_when_win_rate_equals_threshold(self, tmp_path) -> None:
        """Promotion is strictly greater-than; equal must NOT promote."""
        trainer = _make_trainer(tmp_path)
        before = {k: v.clone() for k, v in trainer._opponent_net.state_dict().items()}

        with torch.no_grad():
            for p in trainer._agent._net.parameters():
                torch.nn.init.constant_(p, 99.0)

        freq = trainer._cfg.self_play.opponent_update_freq
        trainer._episode = freq
        with patch.object(trainer, "_evaluate_against_opponent", return_value=0.55):
            trainer._maybe_evaluate_and_promote()

        for k, v in before.items():
            assert torch.allclose(trainer._opponent_net.state_dict()[k], v), (
                f"Param '{k}' changed at win_rate == threshold; promotion must be strictly >"
            )

    def test_promotion_occurs_when_win_rate_above_threshold(self, tmp_path) -> None:
        """Promotion must fire when win_rate > threshold."""
        trainer = _make_trainer(tmp_path)

        with torch.no_grad():
            for p in trainer._agent._net.parameters():
                torch.nn.init.constant_(p, 55.0)

        freq = trainer._cfg.self_play.opponent_update_freq
        trainer._episode = freq
        with patch.object(trainer, "_evaluate_against_opponent", return_value=0.80):
            trainer._maybe_evaluate_and_promote()

        for (k, lv), (_, ov) in zip(
            trainer._agent._net.state_dict().items(),
            trainer._opponent_net.state_dict().items(),
            strict=True,
        ):
            assert torch.allclose(lv, ov), f"Param '{k}' not updated after promotion"


# ---------------------------------------------------------------------------
# GAE on compute_advantages
# ---------------------------------------------------------------------------


class TestGaeOnComputeAdvantages:
    """RolloutBuffer.compute_advantages() must compute GAE advantages in-place."""

    def _fill_buffer(self, buf: RolloutBuffer, n: int) -> None:
        for i in range(n):
            buf.add(
                state=_minimal_obs(),
                action=_minimal_action(),
                reward=float(i % 3),
                done=(i == n - 1),
                value=0.5,
                log_prob=-1.0,
            )

    def test_advantages_none_before_compute(self) -> None:
        buf = RolloutBuffer(capacity=8, device="cpu")
        self._fill_buffer(buf, 4)
        assert buf.advantages is None

    def test_advantages_populated_after_compute(self) -> None:
        buf = RolloutBuffer(capacity=8, device="cpu")
        n = 6
        self._fill_buffer(buf, n)
        buf.compute_advantages(
            next_value=torch.tensor([0.0]),
            next_done=torch.tensor([1.0]),
        )
        assert buf.advantages is not None
        assert len(buf.advantages) == n

    def test_advantages_are_finite(self) -> None:
        buf = RolloutBuffer(capacity=8, device="cpu")
        self._fill_buffer(buf, 5)
        buf.compute_advantages(
            next_value=torch.tensor([0.0]),
            next_done=torch.tensor([1.0]),
            gamma=0.99,
            gae_lambda=0.95,
        )
        assert torch.isfinite(buf.advantages).all(), "All computed advantages must be finite"

    def test_returns_populated_after_compute(self) -> None:
        buf = RolloutBuffer(capacity=8, device="cpu")
        self._fill_buffer(buf, 4)
        buf.compute_advantages(
            next_value=torch.tensor([0.0]),
            next_done=torch.tensor([1.0]),
        )
        assert buf.returns is not None
        assert len(buf.returns) == 4

    def test_clear_resets_advantages(self) -> None:
        buf = RolloutBuffer(capacity=8, device="cpu")
        self._fill_buffer(buf, 3)
        buf.compute_advantages(
            next_value=torch.tensor([0.0]),
            next_done=torch.tensor([0.0]),
        )
        buf.clear()
        assert buf.advantages is None
        assert buf.returns is None
        assert len(buf) == 0


# ---------------------------------------------------------------------------
# NaN log_prob filtering (in trainer's buffer helper)
# ---------------------------------------------------------------------------


class TestNanLogprobFiltering:
    """SelfPlayTrainer._buffer_learner_transitions skips NaN log_prob/value transitions."""

    def _make_transition(self, log_prob: float, player_id: int = 0):
        """Return a minimal Transition with the given log_prob and player."""
        from src.multi_agent import Transition

        obs = _minimal_obs()
        obs["current_player"] = np.array(player_id, dtype=np.int32)
        import math as _math

        return Transition(
            obs=obs,
            action=dict.fromkeys(("action_type", "param_a", "param_b", "param_c", "param_d"), 0),
            reward=0.0,
            log_prob=log_prob,
            value=float("nan") if _math.isnan(log_prob) else 0.5,
            legal_actions=[],
        )

    def test_nan_logprob_transitions_are_not_added_to_buffer(self, tmp_path) -> None:
        """NaN log_prob transitions must be silently skipped."""
        from src.multi_agent import GameResult
        from training.self_play import SelfPlayTrainer

        cfg = _minimal_cfg()
        trainer = SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=30)
        buf = RolloutBuffer(capacity=10, device="cpu")

        traj_valid = self._make_transition(-1.0, player_id=0)
        traj_nan = self._make_transition(float("nan"), player_id=0)

        result = GameResult(
            winner=None,
            n_turns=2,
            territory_history=[],
            elimination_order=[],
            card_trade_turns=[],
            action_log=[],
            trajectories=[traj_valid, traj_nan, traj_valid],
        )
        trainer._buffer_learner_transitions(result, buf)
        # 2 valid + 1 NaN → only 2 should be in buffer
        assert len(buf) == 2, f"Expected 2 valid transitions, got {len(buf)}"

    def test_non_learner_transitions_not_added(self, tmp_path) -> None:
        """Transitions from player_id != 0 (opponent) must not enter the buffer."""
        from src.multi_agent import GameResult
        from training.self_play import SelfPlayTrainer

        cfg = _minimal_cfg()
        trainer = SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=30)
        buf = RolloutBuffer(capacity=10, device="cpu")

        learner = self._make_transition(-1.0, player_id=0)
        opponent = self._make_transition(-1.0, player_id=1)

        result = GameResult(
            winner=None,
            n_turns=2,
            territory_history=[],
            elimination_order=[],
            card_trade_turns=[],
            action_log=[],
            trajectories=[learner, opponent, learner],
        )
        trainer._buffer_learner_transitions(result, buf)
        assert len(buf) == 2, f"Only learner transitions expected, got {len(buf)}"


# ---------------------------------------------------------------------------
# Checkpoint resumability
# ---------------------------------------------------------------------------


class TestCheckpointResumability:
    """save_checkpoint / load_checkpoint must round-trip episode, RNG, and opponent."""

    def test_episode_counter_survives_round_trip(self, tmp_path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer._episode = 77
        ckpt = tmp_path / "ep77.pt"
        trainer.save_checkpoint(ckpt)

        trainer2 = _make_trainer(tmp_path)
        trainer2.load_checkpoint(ckpt)
        assert trainer2._episode == 77

    def test_opponent_weights_survive_round_trip(self, tmp_path) -> None:
        trainer = _make_trainer(tmp_path)
        with torch.no_grad():
            for p in trainer._opponent_net.parameters():
                torch.nn.init.constant_(p, 3.14)

        ckpt = tmp_path / "opp.pt"
        trainer.save_checkpoint(ckpt)

        trainer2 = _make_trainer(tmp_path)
        trainer2.load_checkpoint(ckpt)

        for (k, v1), (_, v2) in zip(
            trainer._opponent_net.state_dict().items(),
            trainer2._opponent_net.state_dict().items(),
            strict=True,
        ):
            assert torch.allclose(v1, v2), f"Opponent param '{k}' mismatch after load"

    def test_rng_state_key_present_in_checkpoint(self, tmp_path) -> None:
        trainer = _make_trainer(tmp_path)
        ckpt = tmp_path / "rng.pt"
        trainer.save_checkpoint(ckpt)
        payload = torch.load(ckpt, weights_only=False)
        assert "rng_state" in payload, "Checkpoint must include 'rng_state' key"
        assert payload["rng_state"], "rng_state must be non-empty"

    def test_env_rng_survives_round_trip(self, tmp_path) -> None:
        """Env RNG state must be captured and restored so seeded runs can resume."""
        trainer = _make_trainer(tmp_path)
        env_rng_before = dict(trainer._env.state.rng.bit_generator.state)

        ckpt = tmp_path / "env_rng.pt"
        trainer.save_checkpoint(ckpt)

        trainer2 = _make_trainer(tmp_path)
        trainer2.load_checkpoint(ckpt)
        env_rng_after = dict(trainer2._env.state.rng.bit_generator.state)

        assert env_rng_before["state"]["state"] == env_rng_after["state"]["state"], (
            "Env RNG state must survive checkpoint round-trip"
        )

    def test_best_metric_value_survives_round_trip(self, tmp_path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer._best_metric_value = 0.73
        ckpt = tmp_path / "metric.pt"
        trainer.save_checkpoint(ckpt)

        trainer2 = _make_trainer(tmp_path)
        trainer2.load_checkpoint(ckpt)
        assert trainer2._best_metric_value == pytest.approx(0.73)


# ---------------------------------------------------------------------------
# LLM pool wiring
# ---------------------------------------------------------------------------


class TestLlmPoolWiring:
    """With llm_profiles_path set, trainer builds N-1 LLMOpponents."""

    def test_3_players_yields_2_llm_opponents(self, tmp_path, monkeypatch) -> None:
        from src.agents.llm_opponent import LLMOpponent
        from training.self_play import SelfPlayTrainer

        profiles = tmp_path / "profiles.yaml"
        _write_player_profiles(profiles, 2)

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama")

        cfg = TrainingConfig(
            seed=0,
            self_play=SelfPlayConfig(n_players=3, llm_profiles_path=str(profiles)),
        )
        with patch("httpx.post", return_value=_fake_ollama_response(0)):
            trainer = SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=10)

        assert trainer._llm_opponents is not None
        assert len(trainer._llm_opponents) == 2
        assert all(isinstance(o, LLMOpponent) for o in trainer._llm_opponents)

    def test_4_players_yields_3_llm_opponents(self, tmp_path, monkeypatch) -> None:
        from src.agents.llm_opponent import LLMOpponent
        from training.self_play import SelfPlayTrainer

        profiles = tmp_path / "profiles.yaml"
        _write_player_profiles(profiles, 3)

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama")

        cfg = TrainingConfig(
            seed=0,
            self_play=SelfPlayConfig(n_players=4, llm_profiles_path=str(profiles)),
        )
        with patch("httpx.post", return_value=_fake_ollama_response(0)):
            trainer = SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=10)

        assert trainer._llm_opponents is not None
        assert len(trainer._llm_opponents) == 3
        assert all(isinstance(o, LLMOpponent) for o in trainer._llm_opponents)

    def test_no_http_call_on_trainer_construction(self, tmp_path, monkeypatch) -> None:
        """LLMOpponent construction must not make any Ollama HTTP requests."""
        from training.self_play import SelfPlayTrainer

        profiles = tmp_path / "profiles.yaml"
        _write_player_profiles(profiles, 1)

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama")

        cfg = TrainingConfig(
            seed=0,
            self_play=SelfPlayConfig(n_players=2, llm_profiles_path=str(profiles)),
        )

        calls: list[Any] = []

        def _counting(*a: Any, **kw: Any) -> MagicMock:
            calls.append(1)
            return _fake_ollama_response(0)

        with patch("httpx.post", side_effect=_counting):
            SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=10)

        assert len(calls) == 0, f"Expected 0 HTTP calls on construction, got {len(calls)}"

    def test_no_llm_profiles_yields_none(self, tmp_path) -> None:
        """Without llm_profiles_path, _llm_opponents must be None."""
        from training.self_play import SelfPlayTrainer

        cfg = TrainingConfig(seed=0, self_play=SelfPlayConfig(n_players=2))
        trainer = SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=10)
        assert trainer._llm_opponents is None
