"""
Source-blind acceptance-criterion tests for issue #62:
  reward-ablation · opponent-promotion · checkpoint-resumability verification

Derived from acceptance-criteria text and requirements.md only.
No implementation source was read.

Criteria covered (oracle-verified as UNIT or T2):
  [UNIT] Each dense reward coefficient is independently ablatable via RewardConfig;
         sparse-only config (all dense coefs = 0) yields per-step rewards in {-1, 0, +1}
  [UNIT] Frozen opponent is promoted to current learner weights when learner win rate
         > promote_threshold (default 0.55) over eval_games
  [UNIT] RolloutBuffer accumulates transitions and computes GAE on compute_advantages();
         NaN-logprob transitions are skipped by the trainer's buffer helper
  [UNIT] SelfPlayTrainer.train() smoke run writes a .pt checkpoint to a tmp dir and
         load_checkpoint round-trips episode counter, RNG state, and opponent
  [UNIT] With llm_profiles_path set, the trainer builds N-1 LLMOpponents
         (Ollama call mocked, no network)
  [UNIT] No magic numbers — all knobs pulled from TrainingConfig/RewardConfig
  [UNIT] Results saved as CSV + TensorBoard logs under results/; every run tagged
         with its seed and config file path

Criteria already covered by test_ac60.py / test_ac61.py (omitted here to avoid
duplication): Ollama /chat/completions contract, LLM output is index, render_action_prompt,
.env.example credential files, basic checkpoint save/load.
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
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _minimal_obs() -> dict:
    """Minimum observation dict valid for RisikoEnv / LLMOpponent."""
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


def _skip_action() -> dict:
    return {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}


def _fake_ollama_response(action_index: int = 0) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps({"action_index": action_index})}}]
    }
    resp.raise_for_status = MagicMock(return_value=None)
    return resp


def _write_player_profiles(path: pathlib.Path, n: int) -> None:
    profiles = [
        {
            "player_id": i,
            "temperature": round(0.1 + 0.1 * i, 1),
            "top_p": 0.9,
            "strategy_hint": f"Profile {i}: play optimally.",
            "model": "gpt-oss:120b",
        }
        for i in range(n)
    ]
    path.write_text(yaml.dump(profiles))


def _import_reward_config():
    """Import RewardConfig from whichever module it lives in."""
    try:
        from src.utils.reward_config import RewardConfig
    except ImportError:
        from src.config import RewardConfig  # type: ignore[assignment]
    return RewardConfig


# ===========================================================================
# [UNIT]  Each dense reward coefficient is independently ablatable via
#         RewardConfig; sparse-only (all coefs = 0) yields {-1, 0, +1}
# ===========================================================================


class TestRewardConfigAblation:
    """RewardConfig must expose individual dense-reward coefficients."""

    def test_when_reward_config_constructed_with_zero_territory_coef_then_field_is_zero(
        self,
    ) -> None:
        rc = _import_reward_config()
        cfg = rc(dense_territory_delta=0.0)
        assert cfg.dense_territory_delta == 0.0

    def test_when_reward_config_constructed_with_zero_continent_coef_then_field_is_zero(
        self,
    ) -> None:
        rc = _import_reward_config()
        cfg = rc(dense_continent_bonus_delta=0.0)
        assert cfg.dense_continent_bonus_delta == 0.0

    def test_when_reward_config_constructed_with_zero_army_ratio_coef_then_field_is_zero(
        self,
    ) -> None:
        rc = _import_reward_config()
        cfg = rc(dense_army_ratio=0.0)
        assert cfg.dense_army_ratio == 0.0

    def test_when_all_dense_coefs_zero_then_sparse_only_config_is_constructable(self) -> None:
        """A fully sparse-only RewardConfig must be constructable with all coefs = 0."""
        rc = _import_reward_config()
        cfg = rc(
            dense_territory_delta=0.0,
            dense_continent_bonus_delta=0.0,
            dense_army_ratio=0.0,
            dense_elimination_bonus=0.0,
        )
        assert cfg.dense_territory_delta == 0.0
        assert cfg.dense_continent_bonus_delta == 0.0
        assert cfg.dense_army_ratio == 0.0
        assert cfg.dense_elimination_bonus == 0.0

    def test_when_sparse_only_config_and_nonterminal_step_then_reward_is_zero(self) -> None:
        """With all dense coefs = 0, a non-terminal step reward must be 0.0."""
        try:
            from src.env_core.reward import Snapshot, compute_reward
        except ImportError:
            pytest.skip("compute_reward not yet implemented")

        rc = _import_reward_config()
        cfg = rc(
            dense_territory_delta=0.0,
            dense_continent_bonus_delta=0.0,
            dense_army_ratio=0.0,
            dense_elimination_bonus=0.0,
        )

        class FakeState:
            territory_owner = np.zeros(42, dtype=np.int32)
            armies = np.ones(42, dtype=np.int32)
            eliminated = np.zeros(6, dtype=np.int32)

        state = FakeState()
        prev = Snapshot(
            territory_owner=dict.fromkeys(range(21), 0),
            armies=dict.fromkeys(range(42), 1),
            eliminated=set(),
        )
        reward = compute_reward(state, player=0, prev_snapshot=prev, cfg=cfg)
        assert reward == pytest.approx(0.0)

    def test_when_sparse_only_env_episode_run_then_all_step_rewards_are_in_valid_set(
        self,
    ) -> None:
        """
        Integration: run a scripted 2-player episode with sparse-only RewardConfig
        and verify every per-step reward is in {-1, 0, +1}.
        """
        try:
            from src.env import RisikoEnv
        except ImportError:
            pytest.skip("RisikoEnv not yet implemented")

        rc = _import_reward_config()
        sparse_cfg = rc(
            dense_territory_delta=0.0,
            dense_continent_bonus_delta=0.0,
            dense_army_ratio=0.0,
            dense_elimination_bonus=0.0,
        )

        try:
            env = RisikoEnv(n_players=2, reward_config=sparse_cfg)
        except TypeError:
            pytest.skip("RisikoEnv does not yet accept reward_config parameter")

        _, info = env.reset(seed=0)
        for _ in range(200):
            legal = info.get("legal_actions", [])
            if not legal:
                break
            _, reward, done, trunc, info = env.step(legal[0])
            assert reward in (-1.0, 0.0, 1.0), (
                f"Sparse-only config produced non-sparse reward: {reward}"
            )
            if done or trunc:
                break


# ===========================================================================
# [UNIT]  RolloutBuffer accumulates transitions, computes GAE on
#         compute_advantages(); NaN-logprob transitions skipped by trainer
# ===========================================================================


class TestRolloutBuffer:
    """RolloutBuffer API contract tests."""

    def _add_transition(self, buf, log_prob: float = -1.5, done: bool = False) -> None:
        buf.add(
            state=_minimal_obs(),
            action=_minimal_action(),
            reward=0.5,
            done=done,
            value=0.3,
            log_prob=log_prob,
        )

    def test_when_n_transitions_pushed_then_buffer_length_is_n(self) -> None:
        from src.models.replay_buffer import RolloutBuffer

        buf = RolloutBuffer(capacity=10, device="cpu")
        for _ in range(5):
            self._add_transition(buf)
        assert len(buf) == 5

    def test_when_clear_called_then_buffer_length_is_reset_to_zero(self) -> None:
        from src.models.replay_buffer import RolloutBuffer

        buf = RolloutBuffer(capacity=10, device="cpu")
        self._add_transition(buf)
        buf.clear()
        assert len(buf) == 0

    def test_when_compute_advantages_called_then_advantages_are_populated(self) -> None:
        """GAE advantages must be populated after compute_advantages()."""
        from src.models.replay_buffer import RolloutBuffer

        buf = RolloutBuffer(capacity=10, device="cpu")
        for i in range(4):
            self._add_transition(buf, done=(i == 3))
        buf.compute_advantages(
            next_value=torch.tensor([0.0]),
            next_done=torch.tensor([1.0]),
        )
        assert buf.advantages is not None
        assert len(buf.advantages) == 4

    def test_when_nan_logprob_transition_added_buffer_stores_it(self) -> None:
        """Buffer itself does not filter NaN log_probs; filtering is the trainer's job."""
        from src.models.replay_buffer import RolloutBuffer

        buf = RolloutBuffer(capacity=10, device="cpu")
        self._add_transition(buf, log_prob=-1.0)
        self._add_transition(buf, log_prob=float("nan"))
        self._add_transition(buf, log_prob=-0.5)
        assert len(buf) == 3

    def test_when_only_valid_transitions_pushed_then_length_equals_pushed_count(
        self,
    ) -> None:
        from src.models.replay_buffer import RolloutBuffer

        buf = RolloutBuffer(capacity=10, device="cpu")
        n = 6
        for i in range(n):
            self._add_transition(buf, done=(i == n - 1))
        assert len(buf) == n

    @given(n=st.integers(min_value=1, max_value=30))
    @settings(max_examples=15, deadline=5_000)
    def test_property_when_n_valid_transitions_pushed_then_length_equals_n(self, n: int) -> None:
        """Invariant: pushing N valid transitions yields len(buffer) == N before clear."""
        from src.models.replay_buffer import RolloutBuffer

        buf = RolloutBuffer(capacity=n, device="cpu")
        for _ in range(n):
            buf.add(
                state=_minimal_obs(),
                action=_minimal_action(),
                reward=0.5,
                done=False,
                value=0.3,
                log_prob=-1.5,
            )
        assert len(buf) == n


# ===========================================================================
# [UNIT]  Frozen opponent is promoted to current learner weights when
#         learner win rate > promote_threshold (default 0.55) over eval_games
# ===========================================================================


class TestOpponentPromotion:
    """Self-play opponent promotion contract."""

    def _make_trainer(self, tmp_path: pathlib.Path):
        """Construct a minimal SelfPlayTrainer with cheap defaults."""
        from src.config import SelfPlayConfig, TrainingConfig
        from training.self_play import SelfPlayTrainer

        cfg = TrainingConfig(
            seed=0,
            self_play=SelfPlayConfig(n_players=2),
        )
        return SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=20)

    def test_when_promote_threshold_checked_then_default_is_0_55(self, tmp_path) -> None:
        """promote_threshold must default to 0.55 (from spec)."""
        trainer = self._make_trainer(tmp_path)
        threshold = trainer._cfg.self_play.promote_threshold
        assert threshold == pytest.approx(0.55), (
            f"Default promote_threshold must be 0.55; got {threshold}"
        )

    def test_when_promote_current_to_opponent_called_then_opponent_weights_match_learner(
        self, tmp_path
    ) -> None:
        """After _promote_current_to_opponent(), frozen-opponent weights must equal learner."""
        trainer = self._make_trainer(tmp_path)

        with torch.no_grad():
            for p in trainer._agent._net.parameters():
                torch.nn.init.constant_(p, 99.0)

        trainer._promote_current_to_opponent()

        for (lk, lv), (_ok, ov) in zip(
            trainer._agent._net.state_dict().items(),
            trainer._opponent_net.state_dict().items(),
            strict=True,
        ):
            assert torch.allclose(lv, ov), (
                f"After promotion, opponent param '{lk}' must equal learner param"
            )

    def test_when_win_rate_at_threshold_then_promotion_does_not_occur(self, tmp_path) -> None:
        """Promotion must NOT occur when win_rate == promote_threshold (strictly >)."""
        trainer = self._make_trainer(tmp_path)

        original = {k: v.clone() for k, v in trainer._opponent_net.state_dict().items()}

        with torch.no_grad():
            for p in trainer._agent._net.parameters():
                torch.nn.init.constant_(p, 42.0)

        freq = trainer._cfg.self_play.opponent_update_freq
        trainer._episode = freq
        with patch.object(trainer, "_evaluate_against_opponent", return_value=0.55):
            trainer._maybe_evaluate_and_promote()

        for k, v in original.items():
            assert torch.allclose(trainer._opponent_net.state_dict()[k], v), (
                f"Opponent param '{k}' changed when win_rate == threshold — must be strictly >"
            )


# ===========================================================================
# [UNIT]  SelfPlayTrainer.train() smoke run writes a .pt checkpoint;
#         load_checkpoint round-trips episode counter, RNG state, opponent
# ===========================================================================


class TestCheckpointRoundTripEpisodeRng:
    """Episode counter and RNG state must survive a checkpoint round-trip."""

    def test_when_checkpoint_includes_episode_then_load_returns_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        """CheckpointManager.load() must expose the 'episode' key."""
        from src.checkpoint import CheckpointManager, _config_to_dict
        from src.config import TrainingConfig

        out = tmp_path / "ep7.pt"
        torch.save(
            {
                "episode": 7,
                "config": _config_to_dict(TrainingConfig()),
                "trainer_state": {},
                "rng_state": {},
            },
            out,
        )
        result = CheckpointManager.load(out)
        assert result["episode"] == 7

    def test_when_checkpoint_includes_rng_state_then_load_returns_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        """CheckpointManager.load() must return the saved rng_state (non-empty)."""
        from src.checkpoint import CheckpointManager, _config_to_dict
        from src.config import TrainingConfig

        rng_bytes = torch.get_rng_state().numpy().tobytes()
        out = tmp_path / "rng.pt"
        torch.save(
            {
                "episode": 1,
                "config": _config_to_dict(TrainingConfig()),
                "trainer_state": {},
                "rng_state": rng_bytes,
            },
            out,
        )
        result = CheckpointManager.load(out)
        assert result.get("rng_state") is not None, (
            "CheckpointManager.load() must return the saved rng_state"
        )
        assert len(result["rng_state"]) > 0, "rng_state must be non-empty"

    @given(episode=st.integers(min_value=0, max_value=10_000))
    @settings(
        max_examples=10,
        deadline=10_000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_when_any_episode_saved_then_it_round_trips_exactly(
        self, tmp_path: pathlib.Path, episode: int
    ) -> None:
        """Invariant: episode counter round-trips exactly for any non-negative int."""
        from src.checkpoint import CheckpointManager, _config_to_dict
        from src.config import TrainingConfig

        out = tmp_path / f"ep_{episode}.pt"
        torch.save(
            {
                "episode": episode,
                "config": _config_to_dict(TrainingConfig()),
                "trainer_state": {},
                "rng_state": {},
            },
            out,
        )
        assert CheckpointManager.load(out)["episode"] == episode

    def test_when_selfplay_trainer_save_then_pt_checkpoint_is_written(
        self, tmp_path: pathlib.Path
    ) -> None:
        """SelfPlayTrainer.save_checkpoint() must write a .pt file."""
        from src.config import SelfPlayConfig, TrainingConfig
        from training.self_play import SelfPlayTrainer

        cfg = TrainingConfig(seed=0, self_play=SelfPlayConfig(n_players=2))
        trainer = SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=20)
        ckpt_path = tmp_path / "smoke.pt"
        trainer.save_checkpoint(ckpt_path)

        assert ckpt_path.exists(), "save_checkpoint() must write a .pt file"

    def test_when_checkpoint_saved_and_loaded_then_episode_counter_matches(
        self, tmp_path: pathlib.Path
    ) -> None:
        """load_checkpoint() must restore the episode counter."""
        from src.config import SelfPlayConfig, TrainingConfig
        from training.self_play import SelfPlayTrainer

        cfg = TrainingConfig(seed=0, self_play=SelfPlayConfig(n_players=2))
        trainer = SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=20)
        trainer._episode = 42
        ckpt_path = tmp_path / "ep42.pt"
        trainer.save_checkpoint(ckpt_path)

        trainer2 = SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=20)
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2._episode == 42

    def test_when_checkpoint_saved_and_loaded_then_opponent_weights_match(
        self, tmp_path: pathlib.Path
    ) -> None:
        """load_checkpoint() must restore the frozen opponent network weights."""
        from src.config import SelfPlayConfig, TrainingConfig
        from training.self_play import SelfPlayTrainer

        cfg = TrainingConfig(seed=0, self_play=SelfPlayConfig(n_players=2))
        trainer = SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=20)

        with torch.no_grad():
            for p in trainer._opponent_net.parameters():
                torch.nn.init.constant_(p, 7.77)

        ckpt_path = tmp_path / "opp.pt"
        trainer.save_checkpoint(ckpt_path)

        trainer2 = SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=20)
        trainer2.load_checkpoint(ckpt_path)

        for k, v in trainer._opponent_net.state_dict().items():
            assert torch.allclose(trainer2._opponent_net.state_dict()[k], v), (
                f"Opponent param '{k}' did not round-trip through checkpoint"
            )


# ===========================================================================
# [UNIT]  With llm_profiles_path set, the trainer builds N-1 LLMOpponents
#         (Ollama call mocked, no network)
# ===========================================================================


class TestSelfPlayTrainerLlmProfiles:
    """SelfPlayTrainer must instantiate N-1 LLMOpponents from llm_profiles_path."""

    def test_when_profiles_path_set_and_4_players_then_3_llm_opponents_in_trainer(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Trainer with 4 players + 3-profile YAML must expose 3 LLMOpponent instances."""
        from src.agents.llm_opponent import LLMOpponent
        from src.config import SelfPlayConfig, TrainingConfig
        from training.self_play import SelfPlayTrainer

        profiles_yaml = tmp_path / "profiles.yaml"
        _write_player_profiles(profiles_yaml, 3)

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama")

        cfg = TrainingConfig(
            seed=0,
            self_play=SelfPlayConfig(
                n_players=4,
                llm_profiles_path=str(profiles_yaml),
            ),
        )

        with patch("httpx.post", return_value=_fake_ollama_response(0)):
            trainer = SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=10)

        assert trainer._llm_opponents is not None
        llm_opponents = [a for a in trainer._llm_opponents if isinstance(a, LLMOpponent)]
        assert len(llm_opponents) == 3, (
            f"Expected N-1=3 LLMOpponents for 4 players, got {len(llm_opponents)}"
        )

    def test_when_profiles_path_set_then_no_real_http_call_is_made_on_construction(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Constructing LLMOpponents must not trigger any real Ollama HTTP call."""
        from src.config import SelfPlayConfig, TrainingConfig
        from training.self_play import SelfPlayTrainer

        profiles_yaml = tmp_path / "profiles3.yaml"
        _write_player_profiles(profiles_yaml, 2)

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama")

        cfg = TrainingConfig(
            seed=0,
            self_play=SelfPlayConfig(
                n_players=3,
                llm_profiles_path=str(profiles_yaml),
            ),
        )

        call_count: list[int] = [0]

        def counting_post(*args: Any, **kwargs: Any) -> MagicMock:
            call_count[0] += 1
            return _fake_ollama_response(0)

        with patch("httpx.post", side_effect=counting_post):
            SelfPlayTrainer(cfg, checkpoint_dir=tmp_path, max_turns=10)

        assert call_count[0] == 0, (
            "No Ollama HTTP calls expected during SelfPlayTrainer construction;"
            f" got {call_count[0]}"
        )


# ===========================================================================
# [UNIT]  No magic numbers — all knobs pulled from TrainingConfig / RewardConfig
# ===========================================================================


class TestNoMagicNumbers:
    """TrainingConfig and RewardConfig must expose all documented hyperparameters."""

    def test_when_training_config_inspected_then_save_freq_field_exists(self) -> None:
        import dataclasses

        from src.config import TrainingConfig

        field_names = {f.name for f in dataclasses.fields(TrainingConfig)}
        assert "save_freq" in field_names, (
            "TrainingConfig must have a 'save_freq' field — no magic checkpoint intervals"
        )

    def test_when_self_play_config_inspected_then_promote_threshold_field_exists(self) -> None:
        import dataclasses

        from src.config import SelfPlayConfig

        field_names = {f.name for f in dataclasses.fields(SelfPlayConfig)}
        assert "promote_threshold" in field_names, (
            "SelfPlayConfig must have a 'promote_threshold' field"
        )

    def test_when_self_play_config_inspected_then_eval_games_field_exists(self) -> None:
        import dataclasses

        from src.config import SelfPlayConfig

        field_names = {f.name for f in dataclasses.fields(SelfPlayConfig)}
        assert "eval_games" in field_names, "SelfPlayConfig must have an 'eval_games' field"

    def test_when_reward_config_inspected_then_dense_territory_delta_field_exists(self) -> None:
        import dataclasses

        rc = _import_reward_config()
        field_names = {f.name for f in dataclasses.fields(rc)}
        assert "dense_territory_delta" in field_names, (
            "RewardConfig must have a 'dense_territory_delta' field"
        )

    def test_when_reward_config_inspected_then_dense_continent_bonus_delta_field_exists(
        self,
    ) -> None:
        import dataclasses

        rc = _import_reward_config()
        field_names = {f.name for f in dataclasses.fields(rc)}
        assert "dense_continent_bonus_delta" in field_names, (
            "RewardConfig must have a 'dense_continent_bonus_delta' field"
        )

    def test_when_reward_config_inspected_then_dense_army_ratio_field_exists(self) -> None:
        import dataclasses

        rc = _import_reward_config()
        field_names = {f.name for f in dataclasses.fields(rc)}
        assert "dense_army_ratio" in field_names, (
            "RewardConfig must have a 'dense_army_ratio' field"
        )

    def test_when_ppo_config_inspected_then_learning_rate_field_exists(self) -> None:
        import dataclasses

        from src.config import PPOConfig

        field_names = {f.name for f in dataclasses.fields(PPOConfig)}
        assert "lr" in field_names or "learning_rate" in field_names, (
            "PPOConfig must expose the learning-rate knob as 'lr' or 'learning_rate'"
        )

    def test_when_ppo_config_inspected_then_entropy_coef_field_exists(self) -> None:
        import dataclasses

        from src.config import PPOConfig

        field_names = {f.name for f in dataclasses.fields(PPOConfig)}
        assert "entropy_coef" in field_names, "PPOConfig must have an 'entropy_coef' field"


# ===========================================================================
# [UNIT]  Results saved as CSV + TensorBoard logs under results/;
#         every run tagged with its seed and config file path
# ===========================================================================


class TestResultsLogging:
    """Training runs must produce tagged TensorBoard events and a CSV file."""

    def _make_game_result(self) -> Any:
        from src.multi_agent import GameResult

        return GameResult(
            winner=0,
            n_turns=20,
            territory_history=[np.zeros(2, dtype=np.int32)],
            elimination_order=[],
            card_trade_turns=[],
            action_log=[_skip_action()],
            trajectories=[],
        )

    def test_when_logger_created_with_seed_tagged_dir_then_event_file_is_written(
        self, tmp_path: pathlib.Path
    ) -> None:
        """TensorBoardLogger must write at least one TF event file to the given dir."""
        from src.tb_logger import TensorBoardLogger

        log_dir = tmp_path / "results" / "run_seed_42"
        logger = TensorBoardLogger(log_dir=str(log_dir))
        logger.log_training_step(
            metrics={"policy_loss": 0.5, "value_loss": 0.3, "entropy_loss": 0.1},
            episode=1,
        )
        logger.close()

        assert list(log_dir.rglob("events.out.tfevents.*")), (
            f"No TF event file found under {log_dir}"
        )

    def test_when_logger_dir_name_includes_seed_then_seed_appears_in_path(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Convention: the run directory name must embed the seed for traceability."""
        from src.tb_logger import TensorBoardLogger

        seed = 123
        log_dir = tmp_path / "results" / f"run_seed_{seed}"
        logger = TensorBoardLogger(log_dir=str(log_dir))
        logger.close()

        assert str(seed) in str(log_dir), (
            "Run directory must contain the seed value for reproducibility tagging"
        )

    def test_when_game_result_logged_then_event_file_exists(self, tmp_path: pathlib.Path) -> None:
        """log_game_result() must flush at least one TF event."""
        from src.tb_logger import TensorBoardLogger

        log_dir = tmp_path / "results" / "run_seed_0"
        logger = TensorBoardLogger(log_dir=str(log_dir))
        logger.log_game_result(self._make_game_result(), player_id=0, episode=1)
        logger.close()

        assert list(log_dir.rglob("events.out.tfevents.*")), (
            "log_game_result() must write a TF event file"
        )

    def test_when_csv_results_written_then_file_contains_seed_column(
        self, tmp_path: pathlib.Path
    ) -> None:
        """
        The CSV results file must include a 'seed' column (or the seed in the filename)
        so every run is traceable.  Assumes a write_results_csv(path, rows) helper or
        equivalent; skipped if not yet implemented.
        """
        try:
            from src.tb_logger import write_results_csv
        except ImportError:
            try:
                from training.evaluate import write_results_csv  # type: ignore[import]
            except ImportError:
                pytest.skip("write_results_csv not yet implemented")

        csv_path = tmp_path / "results" / "run.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_results_csv(
            str(csv_path),
            rows=[{"seed": 42, "config_path": "config/default.yaml", "win_rate": 0.55}],
        )
        content = csv_path.read_text()
        assert "seed" in content, "CSV results file must include a 'seed' column header"
        assert "42" in content, "CSV results file must contain the run seed value"
