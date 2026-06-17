"""
Acceptance tests for issue #61:
feat: wire Ollama LLM opponents into CLI, agent loader, and self-play training

Criteria the oracle marks NOT VERIFIABLE (non-blocking timeout, determinism,
hyperparameter-YAML wiring, baseline win-rates, all-tests-pass, clean-code)
are omitted.  Code-coverage >= 80% is a CI gate, not a unit assertion.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Project-root anchor
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fake_httpx_response(action_index: int = 0) -> MagicMock:
    """Return a mock that looks like an httpx.Response from Ollama."""
    content = json.dumps({"action_index": action_index})
    body = {"message": {"content": content}}
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    resp.raise_for_status = MagicMock(return_value=None)
    return resp


def _minimal_obs() -> dict:
    """Return the minimum observation dict that render_action_prompt accepts."""
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
        "n_players": np.array(6, dtype=np.int32),
    }


def _minimal_legal() -> list[dict]:
    """Return one legal 'skip' action."""
    return [{"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}]


def _write_player_profiles(path: pathlib.Path, n: int) -> None:
    """Write N player profiles as a flat YAML list.

    player_id runs 0..n-1 so all ids stay in the valid 0-5 range for n <= 6.
    """
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


# ---------------------------------------------------------------------------
# [T2] LLM opponent calls Ollama via the native /api/chat endpoint
#      with an enforced JSON-schema `format` field
# ---------------------------------------------------------------------------


class TestOllamaApiContract:
    """The Ollama client must POST to the native /api/chat with an enforced `format`."""

    def _call_with_capture(self, monkeypatch) -> dict:
        """Call call_ollama_for_action_index; return the captured httpx.post kwargs."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama")

        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return _fake_httpx_response(0)

        from src.agents.ollama_client import call_ollama_for_action_index

        with (
            patch("httpx.post", side_effect=fake_post),
            patch("src.agents.ollama_client.render_action_prompt", return_value="test prompt"),
        ):
            call_ollama_for_action_index(
                _minimal_obs(),
                _minimal_legal(),
                model="gpt-oss:120b",
            )
        return captured

    def test_when_action_requested_then_url_targets_api_chat(self, monkeypatch) -> None:
        """POST URL must target the native /api/chat endpoint."""
        captured = self._call_with_capture(monkeypatch)
        assert captured["url"].endswith("/api/chat"), (
            f"Expected URL to end with /api/chat; got {captured['url']!r}"
        )
        assert "/v1/" not in captured["url"], (
            f"Native endpoint must not contain /v1/; got {captured['url']!r}"
        )

    def test_when_action_requested_then_format_type_is_object(self, monkeypatch) -> None:
        """Request body must have format.type == 'object' (raw schema)."""
        captured = self._call_with_capture(monkeypatch)
        body = captured["kwargs"].get("json", {})
        fmt = body.get("format", {})
        assert fmt.get("type") == "object", f"format.type must be 'object'; got {fmt!r}"

    def test_when_action_requested_then_thinking_is_disabled_by_default(self, monkeypatch) -> None:
        """Thinking is OFF by default: the answer is a single legal-action index
        needing no chain-of-thought, and reasoning models otherwise spend
        10-100x the latency emitting a discarded thinking trace. ``think`` must
        be sent explicitly (absence defaults to True server-side).
        """
        captured = self._call_with_capture(monkeypatch)
        body = captured["kwargs"].get("json", {})
        assert body.get("think") is False, f"think must be False; got {body.get('think')!r}"

    def test_when_think_true_passed_then_forwarded(self, monkeypatch) -> None:
        """``think=True`` is forwarded for models that benefit from reasoning."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama")
        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["kwargs"] = kwargs
            return _fake_httpx_response(0)

        from src.agents.ollama_client import call_ollama_for_action_index

        with (
            patch("httpx.post", side_effect=fake_post),
            patch("src.agents.ollama_client.render_action_prompt", return_value="test prompt"),
        ):
            call_ollama_for_action_index(
                _minimal_obs(), _minimal_legal(), model="gpt-oss:120b", think=True
            )
        assert captured["kwargs"]["json"].get("think") is True

    def test_when_action_requested_then_schema_constrains_action_index_property(
        self, monkeypatch
    ) -> None:
        """The `format` schema must define an 'action_index' property."""
        captured = self._call_with_capture(monkeypatch)
        body = captured["kwargs"].get("json", {})
        schema = body.get("format", {})
        assert "action_index" in schema.get("properties", {}), (
            f"Schema must declare 'action_index'; got {schema!r}"
        )

    def test_when_action_requested_then_authorization_bearer_header_is_set(
        self, monkeypatch
    ) -> None:
        """Request must use 'Authorization: Bearer <key>' header (Ollama convention)."""
        captured = self._call_with_capture(monkeypatch)
        headers = captured["kwargs"].get("headers", {})
        lower_keys = {k.lower(): v for k, v in headers.items()}
        assert "authorization" in lower_keys, (
            f"'Authorization' header must be present; got {list(headers)!r}"
        )
        assert lower_keys["authorization"].startswith("Bearer "), (
            f"Authorization must use Bearer scheme; got {lower_keys['authorization']!r}"
        )


# ---------------------------------------------------------------------------
# [UNIT] Ollama credentials load from .env; .env.example is committed
# ---------------------------------------------------------------------------


class TestOllamaCredentials:
    def test_when_env_example_checked_then_file_exists_at_project_root(self) -> None:
        """.env.example must exist at the project root (committed to git)."""
        assert (_PROJECT_ROOT / ".env.example").exists(), (
            ".env.example must be present and committed"
        )

    def test_when_env_example_read_then_ollama_base_url_placeholder_is_present(
        self,
    ) -> None:
        """.env.example must contain OLLAMA_BASE_URL."""
        content = (_PROJECT_ROOT / ".env.example").read_text()
        assert "OLLAMA_BASE_URL" in content

    def test_when_env_example_read_then_ollama_api_key_placeholder_is_present(
        self,
    ) -> None:
        """.env.example must contain OLLAMA_API_KEY."""
        content = (_PROJECT_ROOT / ".env.example").read_text()
        assert "OLLAMA_API_KEY" in content

    def test_when_gitignore_checked_then_dotenv_file_is_ignored(self) -> None:
        """.env must appear in .gitignore so credentials are never committed."""
        gitignore = _PROJECT_ROOT / ".gitignore"
        if not gitignore.exists():
            pytest.skip(".gitignore not found at project root")
        assert ".env" in gitignore.read_text(), ".env must be listed in .gitignore"

    def test_when_ensure_env_loaded_called_twice_then_no_error_is_raised(self, monkeypatch) -> None:
        """ensure_env_loaded() must be idempotent — calling it twice must not raise."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama")
        from src.utils.env import ensure_env_loaded

        ensure_env_loaded()
        ensure_env_loaded()  # must not raise

    def test_when_no_creds_in_env_then_client_defaults_to_localhost(self) -> None:
        """Without env vars the client must fall back to localhost, not raise."""
        import os

        from src.agents.ollama_client import DEFAULT_API_KEY, DEFAULT_BASE_URL

        assert DEFAULT_BASE_URL.startswith("http://localhost"), (
            f"DEFAULT_BASE_URL must point at localhost; got {DEFAULT_BASE_URL!r}"
        )
        assert DEFAULT_API_KEY == "ollama", (
            f"DEFAULT_API_KEY must be 'ollama'; got {DEFAULT_API_KEY!r}"
        )

        # Verify the client builds the correct URL/key when env is absent

        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"message": {"content": json.dumps({"action_index": 0})}}
            resp.raise_for_status = MagicMock(return_value=None)
            return resp

        excluded = {"OLLAMA_BASE_URL", "OLLAMA_API_KEY"}
        clean_env = {k: v for k, v in os.environ.items() if k not in excluded}
        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch("httpx.post", side_effect=fake_post),
        ):
            from src.agents.ollama_client import call_ollama_for_action_index

            call_ollama_for_action_index(_minimal_obs(), _minimal_legal(), model="gpt-oss:120b")

        assert "localhost" in captured.get("url", ""), (
            f"URL must contain localhost when no env var set; got {captured.get('url')!r}"
        )
        auth = captured.get("headers", {}).get("Authorization", "")
        assert "Bearer" in auth, f"Authorization must use Bearer scheme; got {auth!r}"


# ---------------------------------------------------------------------------
# [UNIT] LLM output is an index into legal_actions — always legal
# ---------------------------------------------------------------------------


class TestLlmOutputLegality:
    """Test at the LLMOpponent level: the selected action is always in legal_actions."""

    def test_when_llm_returns_index_1_then_action_equals_legal_actions_at_1(
        self,
    ) -> None:
        """LLMOpponent picks legal_actions[1] when the LLM returns index 1."""
        from src.agents.llm_opponent import LLMOpponent

        legal = [
            {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0},
            {"action_type": 1, "param_a": 2, "param_b": 3, "param_c": 0, "param_d": 0},
            {"action_type": 2, "param_a": 0, "param_b": 1, "param_c": 1, "param_d": 0},
        ]
        agent = LLMOpponent()
        with patch.object(agent, "_call_with_timeout", return_value=1):
            result = agent.act(_minimal_obs(), legal)
        assert result == legal[1]

    def test_when_llm_returns_index_0_then_action_equals_legal_actions_at_0(
        self,
    ) -> None:
        """LLMOpponent picks legal_actions[0] when the LLM returns index 0."""
        from src.agents.llm_opponent import LLMOpponent

        legal = [
            {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0},
            {"action_type": 1, "param_a": 2, "param_b": 3, "param_c": 0, "param_d": 0},
        ]
        agent = LLMOpponent()
        with patch.object(agent, "_call_with_timeout", return_value=0):
            result = agent.act(_minimal_obs(), legal)
        assert result == legal[0]

    def test_when_llm_returns_none_then_result_is_still_a_legal_action(self) -> None:
        """When the LLM returns None (e.g. parse error), fallback yields a legal action."""
        from src.agents.llm_opponent import LLMOpponent

        legal = [
            {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0},
            {"action_type": 1, "param_a": 2, "param_b": 3, "param_c": 0, "param_d": 0},
        ]
        agent = LLMOpponent()
        with patch.object(agent, "_call_with_timeout", return_value=None):
            result = agent.act(_minimal_obs(), legal)
        assert result in legal

    @given(st.integers(min_value=0, max_value=4))
    @settings(max_examples=10)
    def test_when_llm_returns_valid_index_then_result_is_always_in_legal_actions(
        self, idx: int
    ) -> None:
        """For any in-range index, act() returns legal_actions[idx]."""
        from src.agents.llm_opponent import LLMOpponent

        legal = [
            {
                "action_type": t,
                "param_a": 0,
                "param_b": 0,
                "param_c": 0,
                "param_d": 0,
            }
            for t in range(5)
        ]
        agent = LLMOpponent()
        with patch.object(agent, "_call_with_timeout", return_value=idx):
            result = agent.act(_minimal_obs(), legal)
        assert result in legal


# ---------------------------------------------------------------------------
# [UNIT] render_action_prompt() is the single source of truth;
#        the API key is never hardcoded in source
# ---------------------------------------------------------------------------


class TestRenderActionPrompt:
    def test_when_render_action_prompt_called_then_non_empty_string_is_returned(
        self,
    ) -> None:
        """render_action_prompt() must return a non-empty string."""
        from src.agents.action_prompt import render_action_prompt

        result = render_action_prompt(_minimal_obs(), _minimal_legal())
        assert isinstance(result, str) and len(result) > 0

    def test_when_strategy_hint_provided_then_hint_appears_in_prompt(self) -> None:
        """strategy_hint must be embedded in the rendered prompt."""
        from src.agents.action_prompt import render_action_prompt

        hint = "Focus on securing full continents before expanding."
        result = render_action_prompt(_minimal_obs(), _minimal_legal(), strategy_hint=hint)
        assert hint in result

    def test_when_api_key_is_set_then_it_never_appears_in_rendered_prompt(
        self, monkeypatch
    ) -> None:
        """The rendered prompt must never contain the raw API key value."""
        secret = "super-secret-ollama-key-xyz-98765"
        monkeypatch.setenv("OLLAMA_API_KEY", secret)
        from src.agents.action_prompt import render_action_prompt

        result = render_action_prompt(_minimal_obs(), _minimal_legal())
        assert secret not in result

    def test_when_source_files_scanned_then_no_hardcoded_api_key_pattern_is_found(
        self,
    ) -> None:
        """No source in src/ or risiko_rl/ may contain a hardcoded API key literal."""
        result = subprocess.run(
            [
                "grep",
                "-rEn",
                r'(api[_-]?key|API[_-]?KEY)\s*=\s*["\'][A-Za-z0-9+/=\-]{20,}["\']',
                str(_PROJECT_ROOT / "src"),
                str(_PROJECT_ROOT / "risiko_rl"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.stdout == "", f"Hardcoded API key found in source:\n{result.stdout}"

    @given(st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=6))
    @settings(max_examples=10)
    def test_when_any_legal_actions_list_given_then_prompt_is_returned_without_error(
        self, action_types: list[int]
    ) -> None:
        """render_action_prompt must not raise for any non-empty list of skip actions."""
        from src.agents.action_prompt import render_action_prompt

        legal = [
            {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}
            for _ in action_types
        ]
        result = render_action_prompt(_minimal_obs(), legal)
        assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# [UNIT] Checkpoints saved every save_freq episodes to models/;
#        training fully resumable from any .pt file
# ---------------------------------------------------------------------------


class TestCheckpoints:
    def test_when_checkpoint_saved_then_pt_file_exists_at_given_path(self, tmp_path) -> None:
        """save_checkpoint must write a .pt file at the requested path."""
        import torch

        from src.checkpoint import save_checkpoint
        from src.config import TrainingConfig

        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters())
        out = tmp_path / "chk.pt"

        save_checkpoint(
            str(out),
            config=TrainingConfig(),
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
        )
        assert out.exists(), f"Expected .pt file at {out}"

    def test_when_checkpoint_loaded_then_model_weights_match_saved_weights(self, tmp_path) -> None:
        """load_checkpoint must restore model state_dict equal to what was saved."""
        import torch

        from src.checkpoint import load_checkpoint, save_checkpoint
        from src.config import TrainingConfig

        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters())
        out = tmp_path / "chk.pt"
        saved = {k: v.clone() for k, v in model.state_dict().items()}

        save_checkpoint(
            str(out),
            config=TrainingConfig(),
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
        )
        result = load_checkpoint(str(out))
        loaded = result["model_state_dict"]

        for key in saved:
            assert torch.equal(saved[key], loaded[key]), f"Weight mismatch at '{key}'"

    def test_when_checkpoint_loaded_then_episode_number_is_restorable(self, tmp_path) -> None:
        """CheckpointManager payload includes episode; load restores it."""
        import torch

        from src.checkpoint import CheckpointManager, _config_to_dict
        from src.config import TrainingConfig

        out = tmp_path / "chk.pt"
        torch.save(
            {
                "episode": 42,
                "config": _config_to_dict(TrainingConfig()),
                "trainer_state": {},
                "rng_state": {},
            },
            out,
        )
        result = CheckpointManager.load(out)
        assert result["episode"] == 42

    def test_when_checkpoint_loaded_then_config_fields_are_accessible(self, tmp_path) -> None:
        """load_checkpoint returns a config that exposes ppo.lr."""
        import torch

        from src.checkpoint import load_checkpoint, save_checkpoint
        from src.config import TrainingConfig

        cfg = TrainingConfig()
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters())
        out = tmp_path / "chk.pt"

        save_checkpoint(
            str(out),
            config=cfg,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
        )
        result = load_checkpoint(str(out))
        restored = result["config"]
        assert hasattr(restored, "ppo") and hasattr(restored.ppo, "lr")


# ---------------------------------------------------------------------------
# [UNIT] Results saved as CSV + TensorBoard logs under results/
# ---------------------------------------------------------------------------


class TestResultsLogging:
    def _make_game_result(self):
        from src.multi_agent import GameResult

        return GameResult(
            winner=0,
            n_turns=50,
            territory_history=[np.zeros(6, dtype=np.int32)],
            elimination_order=[],
            card_trade_turns=[],
            action_log=[{"action_type": 5, "player": 0}],
            trajectories=[],
        )

    def test_when_logger_initialized_with_dir_then_log_dir_is_created(self, tmp_path) -> None:
        """TensorBoardLogger must create its log directory on initialisation."""
        from src.tb_logger import TensorBoardLogger

        log_dir = tmp_path / "results" / "run_seed_0"
        logger = TensorBoardLogger(log_dir=str(log_dir))
        logger.close()
        assert log_dir.exists()

    def test_when_game_result_logged_then_tensorboard_event_file_exists(self, tmp_path) -> None:
        """After log_game_result(), at least one TF event file must be written."""
        from src.tb_logger import TensorBoardLogger

        log_dir = tmp_path / "results" / "run_seed_1"
        logger = TensorBoardLogger(log_dir=str(log_dir))
        logger.log_game_result(self._make_game_result(), player_id=0, episode=1)
        logger.close()

        event_files = list(log_dir.rglob("events.out.tfevents.*"))
        assert event_files, f"No TF event file found under {log_dir}"

    def test_when_training_step_logged_then_tensorboard_event_file_exists(self, tmp_path) -> None:
        """After log_training_step(), at least one TF event file must be written."""
        from src.tb_logger import TensorBoardLogger

        log_dir = tmp_path / "results" / "run_seed_2"
        logger = TensorBoardLogger(log_dir=str(log_dir))
        logger.log_training_step(
            metrics={"policy_loss": 0.5, "value_loss": 0.3, "entropy_loss": 0.1},
            episode=1,
        )
        logger.close()

        event_files = list(log_dir.rglob("events.out.tfevents.*"))
        assert event_files, "No TF event file after log_training_step()"


# ---------------------------------------------------------------------------
# [UNIT] load_agent("llm") returns LLMOpponent;
#        load_llm_pool(path) returns one LLMOpponent per YAML profile
# ---------------------------------------------------------------------------


class TestAgentLoader:
    def test_when_load_agent_called_with_llm_then_llm_opponent_is_returned(
        self,
    ) -> None:
        """load_agent('llm') must return an LLMOpponent instance."""
        from risiko_rl.agent_loader import load_agent
        from src.agents.llm_opponent import LLMOpponent

        agent = load_agent("llm")
        assert isinstance(agent, LLMOpponent), (
            f"load_agent('llm') must return LLMOpponent, got {type(agent).__name__}"
        )

    def test_when_load_llm_pool_called_with_3_profile_yaml_then_3_opponents_returned(
        self, tmp_path
    ) -> None:
        """load_llm_pool(path) must return exactly N LLMOpponents for N profiles."""
        from risiko_rl.agent_loader import load_llm_pool
        from src.agents.llm_opponent import LLMOpponent

        profiles_yaml = tmp_path / "profiles.yaml"
        _write_player_profiles(profiles_yaml, 3)

        pool = load_llm_pool(profiles_yaml)

        assert len(pool) == 3
        for i, agent in enumerate(pool):
            assert isinstance(agent, LLMOpponent), (
                f"Pool item {i} must be LLMOpponent, got {type(agent).__name__}"
            )

    def test_when_load_llm_pool_called_with_6_profile_yaml_then_6_opponents_returned(
        self, tmp_path
    ) -> None:
        """load_llm_pool(path) with 6 profiles must return 6 LLMOpponents."""
        from risiko_rl.agent_loader import load_llm_pool

        profiles_yaml = tmp_path / "profiles6.yaml"
        _write_player_profiles(profiles_yaml, 6)
        assert len(load_llm_pool(profiles_yaml)) == 6

    @given(st.integers(min_value=1, max_value=6))
    @settings(max_examples=6)
    def test_when_load_llm_pool_called_with_n_profiles_then_n_opponents_returned(
        self, n: int
    ) -> None:
        """load_llm_pool must return exactly as many LLMOpponents as YAML profiles."""
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "profiles.yaml"
            _write_player_profiles(p, n)
            from risiko_rl.agent_loader import load_llm_pool

            pool = load_llm_pool(p)
        assert len(pool) == n


# ---------------------------------------------------------------------------
# [UNIT] CLI exposes train, evaluate, watch, benchmark;
#        train accepts --override key=value (dot notation)
# ---------------------------------------------------------------------------


class TestCli:
    def _runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    def _app(self):
        from risiko_rl.cli import app

        return app

    def test_when_help_invoked_then_train_command_is_listed(self) -> None:
        result = self._runner().invoke(self._app(), ["--help"])
        assert "train" in result.output

    def test_when_help_invoked_then_evaluate_command_is_listed(self) -> None:
        result = self._runner().invoke(self._app(), ["--help"])
        assert "evaluate" in result.output

    def test_when_help_invoked_then_watch_command_is_listed(self) -> None:
        result = self._runner().invoke(self._app(), ["--help"])
        assert "watch" in result.output

    def test_when_help_invoked_then_benchmark_command_is_listed(self) -> None:
        result = self._runner().invoke(self._app(), ["--help"])
        assert "benchmark" in result.output

    def test_when_train_help_invoked_then_override_option_is_present(self) -> None:
        """'train --help' must document the --override option."""
        result = self._runner().invoke(self._app(), ["train", "--help"])
        assert "--override" in result.output, (
            f"'--override' not found in 'train --help':\n{result.output}"
        )


# ---------------------------------------------------------------------------
# [UNIT] _build_llm_opponents() returns N-1 LLMOpponents when path is set,
#        and None fillers when it is unset
# ---------------------------------------------------------------------------


class TestBuildLlmOpponents:
    def test_when_profiles_path_set_then_5_llm_opponents_returned_for_6_players(
        self, tmp_path
    ) -> None:
        """_build_llm_opponents(6, path) must return 5 LLMOpponent objects."""
        from src.agents.llm_opponent import LLMOpponent
        from training.self_play import _build_llm_opponents

        profiles_yaml = tmp_path / "profiles.yaml"
        _write_player_profiles(profiles_yaml, 5)

        opponents = _build_llm_opponents(6, str(profiles_yaml))

        assert len(opponents) == 5
        for i, opp in enumerate(opponents):
            assert isinstance(opp, LLMOpponent), (
                f"Opponent {i} must be LLMOpponent, got {type(opp).__name__}"
            )

    def test_when_profiles_path_is_none_then_5_none_fillers_returned_for_6_players(
        self,
    ) -> None:
        """_build_llm_opponents(6, None) must return 5 None fillers."""
        from src.agents.llm_opponent import LLMOpponent
        from training.self_play import _build_llm_opponents

        opponents = _build_llm_opponents(6, None)

        assert len(opponents) == 5
        for i, opp in enumerate(opponents):
            assert not isinstance(opp, LLMOpponent), (
                f"Slot {i}: without a path, no LLMOpponent should be created"
            )

    def test_when_profiles_path_set_then_first_opponent_has_profile_temperature(
        self, tmp_path
    ) -> None:
        """First LLMOpponent must carry the temperature from profiles[0]."""
        from training.self_play import _build_llm_opponents

        profiles_yaml = tmp_path / "profiles2.yaml"
        profiles_yaml.write_text(
            yaml.dump(
                [
                    {
                        "player_id": 0,
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "strategy_hint": "greedy",
                        "model": "gpt-oss:120b",
                    },
                    {
                        "player_id": 1,
                        "temperature": 0.9,
                        "top_p": 0.8,
                        "strategy_hint": "random",
                        "model": "gpt-oss:120b",
                    },
                ]
            )
        )

        opponents = _build_llm_opponents(3, str(profiles_yaml))

        assert len(opponents) == 2
        assert pytest.approx(opponents[0]._temperature) == 0.1  # type: ignore[union-attr]
        assert pytest.approx(opponents[1]._temperature) == 0.9  # type: ignore[union-attr]

    @given(st.integers(min_value=2, max_value=6))
    @settings(max_examples=5)
    def test_when_n_players_varies_then_always_n_minus_1_opponents_returned(self, n: int) -> None:
        """_build_llm_opponents(N, path) always yields exactly N-1 items."""
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "profiles.yaml"
            _write_player_profiles(p, n - 1)

            from training.self_play import _build_llm_opponents

            opponents = _build_llm_opponents(n, str(p))

        assert len(opponents) == n - 1, (
            f"For {n} players expected {n - 1} opponents, got {len(opponents)}"
        )


# ---------------------------------------------------------------------------
# [UNIT] No BAML references in risiko_rl/, training/, config/
#        (Ollama references are expected in src/ — not scanned here)
# ---------------------------------------------------------------------------


class TestNoBamlReferences:
    @pytest.mark.parametrize("directory", ["risiko_rl", "training", "config"])
    def test_when_directory_scanned_then_no_baml_references_found(self, directory: str) -> None:
        """No file under the directory may reference BAML."""
        target = _PROJECT_ROOT / directory
        if not target.exists():
            pytest.skip(f"{directory}/ does not exist yet")
        result = subprocess.run(
            ["grep", "-ri", "baml", str(target)],
            capture_output=True,
            text=True,
        )
        assert result.stdout == "", f"BAML reference(s) found in {directory}/:\n{result.stdout}"


# ---------------------------------------------------------------------------
# [UNIT] Inert artefacts (.bak files) are not committed
# ---------------------------------------------------------------------------


class TestNoInertArtefacts:
    def test_when_git_ls_files_run_then_no_bak_files_are_tracked(self) -> None:
        """No .bak files may be tracked by git."""
        result = subprocess.run(
            ["git", "-C", str(_PROJECT_ROOT), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        )
        bak_files = [line for line in result.stdout.splitlines() if line.endswith(".bak")]
        assert bak_files == [], "Committed .bak artefacts:\n" + "\n".join(bak_files)

    def test_when_git_ls_files_run_then_dotenv_file_is_not_tracked(self) -> None:
        """The .env file (credentials) must never be committed."""
        result = subprocess.run(
            ["git", "-C", str(_PROJECT_ROOT), "ls-files", ".env"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "", (
            ".env is tracked by git — credentials must not be committed"
        )
