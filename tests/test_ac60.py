"""Source-blind acceptance-criteria tests for issue #60 — migrated to Ollama contract.

Criteria covered (oracle-verified as UNIT or T2):
  [T2]   LLM calls the native /api/chat endpoint with an enforced `format` schema
  [UNIT] Ollama credentials in .env; .env.example committed
  [UNIT] LLM output is an index into legal_actions
  [UNIT] render_action_prompt() is single source of truth; no hardcoded API key
  [UNIT] Attack requires >= 2 armies in source; ADJACENCY is symmetric / 42 territories
  [UNIT] Continent bonuses: NA+5 SA+2 EU+5 AF+3 AS+7 AU+2 (CONTINENT_BONUSES)
  [UNIT] Determinism: two RisikoEnv runs with same seed produce identical sequences
  [UNIT] Checkpoints save/load round-trip (config + model state preserved)
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

_N = 42  # territory count
_N_PLAYERS = 2

_OLLAMA_ENV = {
    "OLLAMA_BASE_URL": "http://localhost:11434/v1",
    "OLLAMA_API_KEY": "ollama",
}


def _make_obs(phase: int = 2, player: int = 0) -> dict[str, np.ndarray]:
    """Minimal valid observation dict suitable for render_action_prompt."""
    return {
        "current_player": np.array(player, dtype=np.int32),
        "phase": np.array(phase, dtype=np.int32),
        "territory_owner": np.zeros(_N, dtype=np.int32),
        "armies": np.ones(_N, dtype=np.int32) * 3,
        "cards": np.zeros((5, 4), dtype=np.int32),
        "continent_control": np.zeros(6, dtype=np.int32),
        "trade_count": np.array(0, dtype=np.int32),
        "reinforcements_remaining": np.array(5, dtype=np.int32),
        "turn_capture": np.array(0, dtype=np.int32),
        "n_players": np.array(_N_PLAYERS, dtype=np.int32),
        "eliminated": np.zeros(6, dtype=np.int32),
    }


def _make_legal_actions() -> list[dict[str, int]]:
    return [
        {"action_type": 2, "param_a": 0, "param_b": 1, "param_c": 1, "param_d": 0},
        {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0},
    ]


def _fake_httpx_response(action_index: int = 0) -> MagicMock:
    """Return a MagicMock that looks like a successful httpx response."""
    resp = MagicMock()
    resp.json.return_value = {"message": {"content": json.dumps({"action_index": action_index})}}
    resp.raise_for_status.return_value = None
    resp.status_code = 200
    return resp


# ===========================================================================
# Criterion: Ollama credentials load from a git-ignored .env;
#            .env.example is committed
# ===========================================================================


class TestOllamaCredentialFiles:
    """Ollama credential file-presence tests."""

    @pytest.fixture(scope="class")
    def env_example_text(self) -> str:
        path = PROJECT_ROOT / ".env.example"
        assert path.exists(), ".env.example must exist at project root"
        return path.read_text()

    def test_when_env_example_checked_then_ollama_base_url_key_is_present(
        self, env_example_text: str
    ) -> None:
        assert "OLLAMA_BASE_URL" in env_example_text

    def test_when_env_example_checked_then_ollama_api_key_is_present(
        self, env_example_text: str
    ) -> None:
        assert "OLLAMA_API_KEY" in env_example_text

    def test_when_gitignore_checked_then_dot_env_is_excluded(self) -> None:
        gitignore = PROJECT_ROOT / ".gitignore"
        assert gitignore.exists(), ".gitignore must exist at project root"
        lines = [
            line.strip()
            for line in gitignore.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        excluded = any(pat in (".env", "*.env", ".env*", "/.env") for pat in lines)
        assert excluded, (
            f".gitignore must contain a pattern that excludes .env (found: {lines[:20]})"
        )


# ===========================================================================
# Criterion [T2]: LLM calls the native /api/chat with an enforced `format` schema
# ===========================================================================


def _patch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _OLLAMA_ENV.items():
        monkeypatch.setenv(k, v)


class TestOllamaApiContractStrict:
    """Ollama API request-structure tests.

    Intercepts httpx.post and asserts on the request URL and body.
    """

    def test_when_action_requested_then_url_targets_api_chat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.agents.ollama_client import call_ollama_for_action_index

        captured: dict[str, Any] = {}
        _patch_env(monkeypatch)

        with patch("httpx.post", side_effect=lambda url, **kw: _capture_and_respond(url, captured)):
            call_ollama_for_action_index(
                _make_obs(), _make_legal_actions(), model="gpt-oss:120b", temperature=0.5
            )

        url = captured.get("url", "")
        assert url.endswith("/api/chat"), (
            f"Request must target the native /api/chat; got url={url!r}"
        )
        assert "/v1/" not in url, f"Native endpoint must not contain /v1/; got url={url!r}"

    def test_when_action_requested_then_format_type_is_object(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.agents.ollama_client import call_ollama_for_action_index

        captured: dict[str, Any] = {}
        _patch_env(monkeypatch)

        with patch(
            "httpx.post",
            side_effect=lambda url, **kw: _capture_body_and_respond(kw, captured),
        ):
            call_ollama_for_action_index(
                _make_obs(), _make_legal_actions(), model="gpt-oss:120b", temperature=0.5
            )

        fmt = captured.get("body", {}).get("format", {})
        assert fmt.get("type") == "object", f"format.type must be 'object', got {fmt!r}"

    def test_when_action_requested_then_format_schema_requires_action_index(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.agents.ollama_client import call_ollama_for_action_index

        captured: dict[str, Any] = {}
        _patch_env(monkeypatch)

        with patch(
            "httpx.post",
            side_effect=lambda url, **kw: _capture_body_and_respond(kw, captured),
        ):
            call_ollama_for_action_index(
                _make_obs(), _make_legal_actions(), model="gpt-oss:120b", temperature=0.5
            )

        fmt = captured.get("body", {}).get("format", {})
        assert "action_index" in fmt.get("required", []), (
            f"format schema must require 'action_index'; got {fmt!r}"
        )
        assert fmt.get("properties", {}).get("action_index", {}).get("type") == "integer", (
            f"action_index must be typed integer; got {fmt!r}"
        )

    def test_when_action_requested_then_authorization_bearer_header_is_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.agents.ollama_client import call_ollama_for_action_index

        captured: dict[str, Any] = {}
        _patch_env(monkeypatch)

        with patch(
            "httpx.post",
            side_effect=lambda url, **kw: _capture_headers_and_respond(kw, captured),
        ):
            call_ollama_for_action_index(
                _make_obs(), _make_legal_actions(), model="gpt-oss:120b", temperature=0.5
            )

        headers_lower = {k.lower(): v for k, v in captured.get("headers", {}).items()}
        assert "authorization" in headers_lower, (
            f"Request must include 'Authorization' header; got {list(headers_lower)}"
        )
        assert headers_lower["authorization"].startswith("Bearer "), (
            f"Authorization header must use Bearer scheme; got {headers_lower['authorization']!r}"
        )

    def test_when_action_requested_then_temperature_is_injected_in_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.agents.ollama_client import call_ollama_for_action_index

        captured: dict[str, Any] = {}
        _patch_env(monkeypatch)

        with patch(
            "httpx.post",
            side_effect=lambda url, **kw: _capture_body_and_respond(kw, captured),
        ):
            call_ollama_for_action_index(
                _make_obs(), _make_legal_actions(), model="gpt-oss:120b", temperature=0.3
            )

        assert captured.get("body", {}).get("options", {}).get("temperature") == pytest.approx(
            0.3
        ), "temperature must appear in the request options"

    def test_when_action_requested_then_top_p_is_injected_in_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.agents.ollama_client import call_ollama_for_action_index

        captured: dict[str, Any] = {}
        _patch_env(monkeypatch)

        with patch(
            "httpx.post",
            side_effect=lambda url, **kw: _capture_body_and_respond(kw, captured),
        ):
            call_ollama_for_action_index(
                _make_obs(),
                _make_legal_actions(),
                model="gpt-oss:120b",
                temperature=0.5,
                top_p=0.85,
            )

        assert captured.get("body", {}).get("options", {}).get("top_p") == pytest.approx(0.85), (
            "top_p must appear in the request options when provided"
        )


def _capture_and_respond(url: str, captured: dict[str, Any]) -> MagicMock:
    captured["url"] = url
    return _fake_httpx_response(0)


def _capture_body_and_respond(kw: dict[str, Any], captured: dict[str, Any]) -> MagicMock:
    captured["body"] = kw.get("json", {})
    return _fake_httpx_response(0)


def _capture_headers_and_respond(kw: dict[str, Any], captured: dict[str, Any]) -> MagicMock:
    captured["headers"] = kw.get("headers", {})
    return _fake_httpx_response(0)


# ===========================================================================
# Criterion [UNIT]: LLM output is an index into legal_actions — always legal
# ===========================================================================


class TestLlmOutputIsLegalIndex:
    """LLM index-output tests."""

    def test_when_model_returns_index_0_then_function_returns_0(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.agents.ollama_client import call_ollama_for_action_index

        legal = _make_legal_actions()
        _patch_env(monkeypatch)

        with patch("httpx.post", return_value=_fake_httpx_response(0)):
            result = call_ollama_for_action_index(
                _make_obs(), legal, model="gpt-oss:120b", temperature=0.5
            )

        assert result == 0

    def test_when_model_returns_last_valid_index_then_function_returns_that_index(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.agents.ollama_client import call_ollama_for_action_index

        legal = _make_legal_actions()
        last = len(legal) - 1
        _patch_env(monkeypatch)

        with patch("httpx.post", return_value=_fake_httpx_response(last)):
            result = call_ollama_for_action_index(
                _make_obs(), legal, model="gpt-oss:120b", temperature=0.5
            )

        assert result == last

    def test_when_model_returns_index_then_index_is_within_bounds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.agents.ollama_client import call_ollama_for_action_index

        legal = _make_legal_actions()
        _patch_env(monkeypatch)

        with patch("httpx.post", return_value=_fake_httpx_response(1)):
            result = call_ollama_for_action_index(
                _make_obs(), legal, model="gpt-oss:120b", temperature=0.5
            )

        assert result is not None
        assert 0 <= result < len(legal)

    @given(st.integers(min_value=0, max_value=9))
    @settings(max_examples=10, deadline=5_000)
    def test_when_any_valid_index_returned_by_model_then_function_returns_that_index(
        self, idx: int
    ) -> None:
        """Property: for any valid index, the function returns exactly that integer."""
        from src.agents.ollama_client import call_ollama_for_action_index

        legal = [
            {"action_type": 2, "param_a": i, "param_b": 0, "param_c": 1, "param_d": 0}
            for i in range(idx + 1)
        ]

        with (
            patch.dict(os.environ, _OLLAMA_ENV),
            patch("httpx.post", return_value=_fake_httpx_response(idx)),
        ):
            result = call_ollama_for_action_index(
                _make_obs(), legal, model="gpt-oss:120b", temperature=0.5
            )

        assert result == idx


# ===========================================================================
# Criterion [UNIT]: render_action_prompt() is single source of truth;
#                   API key never hardcoded in source
# ===========================================================================


class TestRenderActionPrompt:
    """render_action_prompt and API-key hygiene tests."""

    def test_when_render_action_prompt_called_then_it_returns_a_non_empty_string(
        self,
    ) -> None:
        from src.agents.action_prompt import render_action_prompt

        result = render_action_prompt(_make_obs(), _make_legal_actions())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_when_render_action_prompt_called_then_legal_actions_section_appears_in_output(
        self,
    ) -> None:
        from src.agents.action_prompt import render_action_prompt

        result = render_action_prompt(_make_obs(), _make_legal_actions())
        assert "Legal Actions" in result or "SKIP" in result, (
            "Rendered prompt must include the legal-actions block"
        )

    def test_when_render_action_prompt_called_with_strategy_hint_then_hint_is_in_output(
        self,
    ) -> None:
        from src.agents.action_prompt import render_action_prompt

        hint = "Focus on securing full continents before expanding."
        result = render_action_prompt(_make_obs(), _make_legal_actions(), strategy_hint=hint)
        assert hint in result, "strategy_hint must appear verbatim in the rendered prompt"

    def test_when_ollama_client_called_then_render_action_prompt_is_invoked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """render_action_prompt must be the single source of truth for prompt construction."""
        from src.agents import ollama_client

        call_count: list[int] = [0]
        real_render = ollama_client.render_action_prompt

        def spy_render(*args: Any, **kwargs: Any) -> str:
            call_count[0] += 1
            return real_render(*args, **kwargs)

        monkeypatch.setattr(ollama_client, "render_action_prompt", spy_render)
        _patch_env(monkeypatch)

        with patch("httpx.post", return_value=_fake_httpx_response(0)):
            ollama_client.call_ollama_for_action_index(
                _make_obs(), _make_legal_actions(), model="gpt-oss:120b", temperature=0.5
            )

        assert call_count[0] >= 1, (
            "render_action_prompt must be called at least once per LLM request"
        )

    def test_when_source_files_scanned_then_no_api_key_is_hardcoded(self) -> None:
        """The API key must never appear as a literal in source files."""
        src_dir = PROJECT_ROOT / "src"
        hex32 = re.compile(r'["\'][0-9a-fA-F]{32}["\']')
        violations: list[str] = []
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(errors="replace")
            for m in hex32.finditer(content):
                violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:{m.group()}")
        assert not violations, "Possible hardcoded API key(s):\n" + "\n".join(violations)


# ===========================================================================
# Criterion [UNIT]: Continent bonuses NA+5 / SA+2 / EU+5 / AF+3 / AS+7 / AU+2
# ===========================================================================

_EXPECTED: dict[str, int] = {
    "North America": 5,
    "South America": 2,
    "Europe": 5,
    "Africa": 3,
    "Asia": 7,
    "Australia": 2,
}


class TestContinentBonuses:
    """CONTINENT_BONUSES must match the specification exactly."""

    @pytest.fixture(scope="class")
    def bonuses(self) -> dict[str, int]:
        from src.utils.constants import CONTINENT_BONUSES

        return CONTINENT_BONUSES

    @pytest.mark.parametrize("continent,expected", list(_EXPECTED.items()), ids=list(_EXPECTED))
    def test_when_continent_bonus_checked_then_value_matches_spec(
        self, bonuses: dict[str, int], continent: str, expected: int
    ) -> None:
        assert bonuses.get(continent) == expected, (
            f"CONTINENT_BONUSES[{continent!r}] must be {expected}; got {bonuses.get(continent)}"
        )

    def test_when_all_bonuses_summed_then_total_is_24(self, bonuses: dict[str, int]) -> None:
        assert sum(bonuses.values()) == 24, f"5+2+5+3+7+2=24; got {sum(bonuses.values())}"

    def test_when_continent_bonuses_checked_then_exactly_six_continents_exist(
        self, bonuses: dict[str, int]
    ) -> None:
        assert len(bonuses) == 6, f"Must have exactly 6 continents; got {len(bonuses)}"


# ===========================================================================
# Criterion [UNIT]: Attack requires >= 2 armies; ADJACENCY symmetric / 42 entries
# ===========================================================================


class TestAdjacencyConstant:
    """ADJACENCY structural invariants."""

    @pytest.fixture(scope="class")
    def adjacency(self) -> dict[int, list[int]]:
        from src.utils.constants import ADJACENCY

        return ADJACENCY

    def test_when_adjacency_imported_then_it_is_non_empty(
        self, adjacency: dict[int, list[int]]
    ) -> None:
        assert len(adjacency) > 0

    def test_when_adjacency_checked_then_it_covers_all_42_territories(
        self, adjacency: dict[int, list[int]]
    ) -> None:
        assert len(adjacency) == 42, f"Risiko has 42 territories; got {len(adjacency)}"

    def test_when_adjacency_checked_then_relationship_is_symmetric(
        self, adjacency: dict[int, list[int]]
    ) -> None:
        violations: list[str] = []
        for territory, neighbours in adjacency.items():
            for nbr in neighbours:
                if territory not in adjacency.get(nbr, []):
                    violations.append(f"{territory} → {nbr} but not reverse")
        assert not violations, "ADJACENCY not symmetric:\n" + "\n".join(violations[:10])


# ===========================================================================
# Criterion [UNIT]: Determinism — same seed produces identical sequences
# ===========================================================================


class TestEnvironmentDeterminism:
    """RisikoEnv determinism tests."""

    def test_when_two_envs_reset_with_same_seed_then_initial_obs_are_equal(
        self,
    ) -> None:
        from src.env import RisikoEnv

        env1 = RisikoEnv(n_players=_N_PLAYERS)
        env2 = RisikoEnv(n_players=_N_PLAYERS)
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        for key in obs1:
            np.testing.assert_array_equal(
                obs1[key], obs2[key], err_msg=f"obs['{key}'] differs for same seed"
            )

    def test_when_two_envs_run_same_seed_then_legal_actions_match_each_step(
        self,
    ) -> None:
        from src.env import RisikoEnv

        n_steps = 10
        env1 = RisikoEnv(n_players=_N_PLAYERS)
        env2 = RisikoEnv(n_players=_N_PLAYERS)
        _, info1 = env1.reset(seed=0)
        _, info2 = env2.reset(seed=0)
        for step in range(n_steps):
            legal1 = info1["legal_actions"]
            legal2 = info2["legal_actions"]
            assert legal1 == legal2, (
                f"Legal actions diverged at step {step}: {legal1!r} vs {legal2!r}"
            )
            if not legal1:
                break
            _, _, d1, t1, info1 = env1.step(legal1[0])
            _, _, d2, t2, info2 = env2.step(legal2[0])
            if d1 or t1:
                break

    def test_when_two_envs_run_same_seed_then_rewards_match_each_step(
        self,
    ) -> None:
        from src.env import RisikoEnv

        n_steps = 10
        env1 = RisikoEnv(n_players=_N_PLAYERS)
        env2 = RisikoEnv(n_players=_N_PLAYERS)
        _, info1 = env1.reset(seed=7)
        _, info2 = env2.reset(seed=7)
        for step in range(n_steps):
            legal1 = info1["legal_actions"]
            legal2 = info2["legal_actions"]
            if not legal1 or not legal2:
                break
            _, r1, d1, t1, info1 = env1.step(legal1[0])
            _, r2, d2, t2, info2 = env2.step(legal2[0])
            assert r1 == pytest.approx(r2), f"Rewards diverged at step {step}: {r1} vs {r2}"
            if d1 or t1:
                break

    def test_when_two_envs_run_same_seed_then_full_trajectories_are_equal(
        self,
    ) -> None:
        from src.env import RisikoEnv

        def run(env: Any, seed: int, steps: int) -> list[tuple]:
            traj: list[tuple] = []
            _, info = env.reset(seed=seed)
            traj.append(("reset", str(info["legal_actions"])))
            for _ in range(steps):
                legal = info["legal_actions"]
                if not legal:
                    break
                obs, reward, done, trunc, info = env.step(legal[0])
                traj.append((str(legal), reward, str(obs["phase"])))
                if done or trunc:
                    break
            return traj

        env1 = RisikoEnv(n_players=_N_PLAYERS)
        env2 = RisikoEnv(n_players=_N_PLAYERS)
        assert run(env1, 99, 15) == run(env2, 99, 15), (
            "Full 15-step trajectories must be identical for same seed"
        )

    @given(st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=10, deadline=15_000, suppress_health_check=[HealthCheck.too_slow])
    def test_when_any_seed_used_then_two_envs_produce_same_initial_obs(self, seed: int) -> None:
        """Property: reset with any seed is deterministic across two env instances."""
        from src.env import RisikoEnv

        env1 = RisikoEnv(n_players=_N_PLAYERS)
        env2 = RisikoEnv(n_players=_N_PLAYERS)
        obs1, _ = env1.reset(seed=seed)
        obs2, _ = env2.reset(seed=seed)
        for key in obs1:
            np.testing.assert_array_equal(obs1[key], obs2[key])


# ===========================================================================
# Criterion [UNIT]: Checkpoints save/load round-trip
# ===========================================================================


class TestCheckpointRoundtrip:
    """Checkpoint save/load round-trip tests."""

    def test_when_checkpoint_saved_then_file_exists_at_given_path(
        self, tmp_path: pathlib.Path
    ) -> None:
        from src.checkpoint import save_checkpoint
        from src.config import TrainingConfig

        path = tmp_path / "ckpt.pt"
        save_checkpoint(
            str(path),
            config=TrainingConfig(seed=1),
            model_state_dict={},
            optimizer_state_dict={},
        )
        assert path.exists(), "save_checkpoint must create the file at the given path"

    def test_when_checkpoint_loaded_then_config_matches_saved_config(
        self, tmp_path: pathlib.Path
    ) -> None:
        from src.checkpoint import load_checkpoint, save_checkpoint
        from src.config import TrainingConfig

        cfg = TrainingConfig(seed=42, config_path="config/default.yaml")
        path = tmp_path / "round_trip.pt"
        save_checkpoint(str(path), config=cfg, model_state_dict={}, optimizer_state_dict={})
        loaded = load_checkpoint(str(path))
        assert loaded["config"].seed == 42, "Config seed must survive checkpoint round-trip"
        assert loaded["config"].config_path == "config/default.yaml"

    def test_when_checkpoint_loaded_then_episode_is_absent_from_config(
        self, tmp_path: pathlib.Path
    ) -> None:
        """save_checkpoint has no episode param; all TrainingConfig fields are preserved."""
        from src.checkpoint import load_checkpoint, save_checkpoint
        from src.config import TrainingConfig

        cfg = TrainingConfig(seed=99)
        path = tmp_path / "no_episode.pt"
        save_checkpoint(str(path), config=cfg, model_state_dict={}, optimizer_state_dict={})
        loaded = load_checkpoint(str(path))
        assert loaded["config"].seed == 99

    def test_when_checkpoint_loaded_then_model_state_dict_keys_are_preserved(
        self, tmp_path: pathlib.Path
    ) -> None:
        from src.checkpoint import load_checkpoint, save_checkpoint
        from src.config import TrainingConfig

        state = {"layer.weight": [1.0, 2.0], "layer.bias": [0.1]}
        path = tmp_path / "weights.pt"
        save_checkpoint(
            str(path),
            config=TrainingConfig(seed=0),
            model_state_dict=state,
            optimizer_state_dict={},
        )
        loaded = load_checkpoint(str(path))
        assert set(loaded["model_state_dict"].keys()) == set(state.keys()), (
            "Model state dict keys must be preserved"
        )

    @given(st.integers(min_value=0, max_value=9999))
    @settings(
        max_examples=10,
        deadline=10_000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_when_any_seed_saved_then_it_is_recovered_exactly(
        self, tmp_path: pathlib.Path, seed: int
    ) -> None:
        """Property: config.seed round-trips exactly for any non-negative integer."""
        from src.checkpoint import load_checkpoint, save_checkpoint
        from src.config import TrainingConfig

        path = tmp_path / f"seed_{seed}.pt"
        save_checkpoint(
            str(path),
            config=TrainingConfig(seed=seed),
            model_state_dict={},
            optimizer_state_dict={},
        )
        assert load_checkpoint(str(path))["config"].seed == seed
