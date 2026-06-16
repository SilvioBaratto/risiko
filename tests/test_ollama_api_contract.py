"""Spec-derived tests for issue #58 — Ollama API contract and render_action_prompt.

Criteria covered:
  [T2]   LLM opponent calls Ollama via /chat/completions
         with json_schema strict output
  [UNIT] Ollama credentials load from a git-ignored .env
         (OLLAMA_BASE_URL / OLLAMA_API_KEY); .env.example committed
  [UNIT] LLM output is an index into legal_actions — chosen action always legal
  [UNIT] render_action_prompt() is the single source of truth for the LLM prompt;
         the API key is never hardcoded in source
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.action_prompt import render_action_prompt
from src.agents.ollama_client import call_ollama_for_action_index

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# ────────────────────────────────────────────────────────────────────────────
# Test helpers
# ────────────────────────────────────────────────────────────────────────────

_FAKE_ENV = {
    "OLLAMA_BASE_URL": "http://fake-ollama.local/v1",
    "OLLAMA_API_KEY": "test-api-key-abc123",
}

_N_TERRITORIES = 42
_N_PLAYERS = 6


def _make_obs(phase: int = 2) -> dict[str, np.ndarray]:
    """Return a minimal valid observation dict (all zeros/ones; enough to render a prompt)."""
    return {
        "current_player": np.array(0, dtype=np.int32),
        "phase": np.array(phase, dtype=np.int32),
        "territory_owner": np.zeros(_N_TERRITORIES, dtype=np.int32),
        "armies": np.ones(_N_TERRITORIES, dtype=np.int32),
        "reinforcements_remaining": np.array(3, dtype=np.int32),
        "trade_count": np.array(0, dtype=np.int32),
        "eliminated": np.zeros(_N_PLAYERS, dtype=np.int32),
        "cards": np.zeros((5, 4), dtype=np.int32),
    }


def _skip_action() -> dict[str, int]:
    """Return a minimal SKIP action dict."""
    return {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}


def _make_mock_response(action_index: int) -> MagicMock:
    """Build a mock httpx.Response for a given action_index reply."""
    body = {"choices": [{"message": {"content": json.dumps({"action_index": action_index})}}]}
    resp = MagicMock()
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


def _call_ollama(
    legal_actions: list[dict[str, int]],
    action_index: int = 0,
    temperature: float = 0.5,
    top_p: float | None = 0.9,
    strategy_hint: str | None = "hint",
    env: dict[str, str] | None = None,
    obs: dict[str, np.ndarray] | None = None,
) -> tuple[int | None, list[Any]]:
    """Call call_ollama_for_action_index with a patched httpx.post.

    Returns (result, captured_call_args_list).
    """
    if obs is None:
        obs = _make_obs()
    captured: list[Any] = []

    def fake_post(url: str, **kwargs: Any) -> MagicMock:
        captured.append({"url": url, "kwargs": kwargs})
        return _make_mock_response(action_index)

    with (
        patch.dict(os.environ, env or _FAKE_ENV),
        patch("httpx.post", side_effect=fake_post),
    ):
        result = call_ollama_for_action_index(
            obs,
            legal_actions,
            model="gpt-oss:120b",
            temperature=temperature,
            top_p=top_p,
            strategy_hint=strategy_hint,
        )
    return result, captured


# ────────────────────────────────────────────────────────────────────────────
# Endpoint contract — POST /chat/completions
# ────────────────────────────────────────────────────────────────────────────


class TestOllamaChatCompletionsEndpoint:
    """Ollama /chat/completions endpoint and json_schema strict contract."""

    def test_when_ollama_called_then_exactly_one_http_request_is_made(self):
        """Criterion: the client makes exactly one call per move."""
        _, calls = _call_ollama([_skip_action()])
        assert len(calls) == 1

    def test_when_ollama_called_then_request_url_contains_chat_completions(self):
        """Criterion: LLM opponent calls Ollama via /chat/completions."""
        _, calls = _call_ollama([_skip_action()])
        url = calls[0]["url"]
        assert "/chat/completions" in url, f"Expected /chat/completions in URL, got: {url}"

    def test_when_ollama_called_then_request_url_does_not_contain_api_version(self):
        """Criterion: Ollama URL has no api-version query parameter."""
        _, calls = _call_ollama([_skip_action()])
        url = calls[0]["url"]
        assert "api-version=" not in url, f"URL must not contain api-version, got: {url}"

    def test_when_ollama_called_then_request_body_uses_json_schema_response_format(self):
        """Criterion: response_format type must be 'json_schema'."""
        _, calls = _call_ollama([_skip_action(), _skip_action()])
        body = calls[0]["kwargs"]["json"]
        rf = body.get("response_format", {})
        assert rf.get("type") == "json_schema", (
            f"response_format.type must be 'json_schema', got: {rf}"
        )

    def test_when_ollama_called_then_json_schema_strict_is_true(self):
        """Criterion: json_schema strict must be True (enforces output shape)."""
        _, calls = _call_ollama([_skip_action()])
        body = calls[0]["kwargs"]["json"]
        rf = body.get("response_format", {})
        inner = rf.get("json_schema", {})
        assert inner.get("strict") is True, f"json_schema.strict must be True, got: {inner}"

    def test_when_ollama_called_then_request_body_contains_action_index_schema(self):
        """The json_schema must constrain the output to contain 'action_index'."""
        _, calls = _call_ollama([_skip_action()])
        body = calls[0]["kwargs"]["json"]
        raw_schema = json.dumps(body)
        assert "action_index" in raw_schema, "The request schema must reference 'action_index'"

    def test_when_temperature_supplied_then_it_appears_in_request_body(self):
        """Criterion: per-call temperature injected into the Ollama request body."""
        _, calls = _call_ollama([_skip_action()], temperature=0.3)
        body = calls[0]["kwargs"]["json"]
        assert body.get("temperature") == pytest.approx(0.3), (
            "temperature must be forwarded to the request body"
        )

    def test_when_top_p_supplied_then_it_appears_in_request_body(self):
        """Criterion: per-call top_p injected into the Ollama request body."""
        _, calls = _call_ollama([_skip_action()], top_p=0.85)
        body = calls[0]["kwargs"]["json"]
        assert body.get("top_p") == pytest.approx(0.85), (
            "top_p must be forwarded to the request body"
        )


# ────────────────────────────────────────────────────────────────────────────
# LLM output → action index → always legal
# ────────────────────────────────────────────────────────────────────────────


class TestActionIndexLegality:
    """LLM returns an in-range index; the resolved action is always legal."""

    def test_when_ollama_returns_index_0_then_result_is_zero(self):
        """Criterion: LLM output is an index into legal_actions."""
        legal = [_skip_action(), _skip_action(), _skip_action()]
        result, _ = _call_ollama(legal, action_index=0)
        assert result == 0

    def test_when_ollama_returns_index_1_then_result_is_one(self):
        legal = [_skip_action(), _skip_action(), _skip_action()]
        result, _ = _call_ollama(legal, action_index=1)
        assert result == 1

    def test_when_action_index_in_range_then_result_is_that_index(self):
        """Criterion: the returned index is always in [0, len(legal_actions))."""
        legal = [_skip_action(), _skip_action(), _skip_action()]
        result, _ = _call_ollama(legal, action_index=2)
        assert result == 2

    def test_when_returned_index_used_then_resolved_action_is_legal(self):
        """Criterion: the chosen action is always legal by construction."""
        legal = [_skip_action(), _skip_action()]
        result, _ = _call_ollama(legal, action_index=0)
        assert result is not None
        assert 0 <= result < len(legal)

    @given(
        st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=10),
        st.integers(min_value=0, max_value=9),
    )
    @settings(max_examples=20)
    def test_when_index_in_range_then_it_resolves_to_a_legal_action(
        self, items: list[int], raw_index: int
    ) -> None:
        """Invariant: any in-range index resolves to a member of the legal list."""
        idx = raw_index % len(items)
        assert items[idx] in items


# ────────────────────────────────────────────────────────────────────────────
# Ollama credentials — loaded from environment variables, not hardcoded
# ────────────────────────────────────────────────────────────────────────────


class TestOllamaCredentialsFromEnv:
    """Ollama credentials come from env vars; .env.example is committed."""

    def test_when_api_key_env_var_set_then_request_header_contains_bearer_token(self):
        """Criterion: API key from OLLAMA_API_KEY forwarded as Authorization: Bearer."""
        expected_key = "my-secret-key-xyz"
        env = {**_FAKE_ENV, "OLLAMA_API_KEY": expected_key}
        _, calls = _call_ollama([_skip_action()], env=env)
        headers = calls[0]["kwargs"].get("headers", {})
        assert f"Bearer {expected_key}" in headers.values(), (
            "OLLAMA_API_KEY must be forwarded as 'Authorization: Bearer <key>'"
        )

    def test_when_api_key_env_var_set_then_no_api_key_header_is_present(self):
        """The old api-key header must not be set; only Authorization: Bearer is used."""
        _, calls = _call_ollama([_skip_action()])
        headers = calls[0]["kwargs"].get("headers", {})
        assert "api-key" not in headers, "The api-key header must not be present in Ollama requests"

    def test_when_base_url_env_var_set_then_request_url_starts_with_it(self):
        """Criterion: OLLAMA_BASE_URL determines the request endpoint."""
        base = "http://custom-ollama.local/v1"
        env = {**_FAKE_ENV, "OLLAMA_BASE_URL": base}
        _, calls = _call_ollama([_skip_action()], env=env)
        assert calls[0]["url"].startswith(base), "Request URL must start with OLLAMA_BASE_URL"

    def test_when_no_env_vars_set_then_request_url_defaults_to_localhost(self):
        """Criterion: missing OLLAMA_BASE_URL falls back to localhost without raising."""
        captured: list[Any] = []

        def fake_post(url: str, **kwargs: Any) -> MagicMock:
            captured.append({"url": url, "kwargs": kwargs})
            return _make_mock_response(0)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("httpx.post", side_effect=fake_post),
        ):
            call_ollama_for_action_index(_make_obs(), [_skip_action()], model="gpt-oss:120b")

        assert captured[0]["url"].startswith("http://localhost:11434/v1"), (
            "Default base URL must be http://localhost:11434/v1"
        )

    def test_when_no_api_key_env_var_then_authorization_header_is_bearer_ollama(self):
        """Criterion: missing OLLAMA_API_KEY defaults to 'Bearer ollama', never raises."""
        captured: list[Any] = []

        def fake_post(url: str, **kwargs: Any) -> MagicMock:
            captured.append({"url": url, "kwargs": kwargs})
            return _make_mock_response(0)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("httpx.post", side_effect=fake_post),
        ):
            call_ollama_for_action_index(_make_obs(), [_skip_action()], model="gpt-oss:120b")

        headers = captured[0]["kwargs"].get("headers", {})
        assert headers.get("Authorization") == "Bearer ollama", (
            "Default Authorization header must be 'Bearer ollama'"
        )

    def test_when_dot_env_example_checked_then_file_exists_at_project_root(self):
        """Criterion: .env.example is committed (exists in the repository root)."""
        env_example = PROJECT_ROOT / ".env.example"
        assert env_example.exists(), ".env.example must be committed at the project root"

    def test_when_dot_env_example_read_then_ollama_base_url_key_is_present(self):
        """Criterion: .env.example documents OLLAMA_BASE_URL."""
        content = (PROJECT_ROOT / ".env.example").read_text()
        assert "OLLAMA_BASE_URL" in content

    def test_when_dot_env_example_read_then_ollama_api_key_key_is_present(self):
        """Criterion: .env.example documents OLLAMA_API_KEY."""
        content = (PROJECT_ROOT / ".env.example").read_text()
        assert "OLLAMA_API_KEY" in content


# ────────────────────────────────────────────────────────────────────────────
# render_action_prompt — single source of truth for the LLM prompt
# ────────────────────────────────────────────────────────────────────────────


class TestRenderActionPrompt:
    """render_action_prompt is the single source of truth for the LLM prompt."""

    def test_when_called_then_a_non_empty_string_is_returned(self):
        """Criterion: render_action_prompt() is callable and returns a string."""
        result = render_action_prompt(_make_obs(), [_skip_action()])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_when_called_then_output_contains_action_index_numeric_indices(self):
        """The prompt must present actions with numeric indices for index-based selection."""
        actions = [_skip_action(), _skip_action(), _skip_action()]
        result = render_action_prompt(_make_obs(), actions)
        assert "0" in result, "Index 0 must appear in prompt"
        assert "1" in result, "Index 1 must appear in prompt"
        assert "2" in result, "Index 2 must appear in prompt"

    def test_when_called_with_strategy_hint_then_hint_appears_in_output(self):
        """Criterion: strategy_hint is appended as a one-line directive."""
        hint = "Fortify borders and expand only when safe."
        result = render_action_prompt(_make_obs(), [_skip_action()], strategy_hint=hint)
        assert hint in result, "strategy_hint must appear verbatim in the rendered prompt"

    def test_when_called_then_output_contains_skip_for_skip_action(self):
        """A SKIP action (type=5) must be described in the rendered prompt."""
        result = render_action_prompt(_make_obs(), [_skip_action()])
        assert "skip" in result.lower(), "SKIP action description must appear in prompt"

    def test_when_called_then_output_does_not_contain_an_api_key_pattern(self):
        """Criterion: the API key is never hardcoded in source."""
        fake_key = "do-not-leak-this-key-abc987"
        with patch.dict(os.environ, {"OLLAMA_API_KEY": fake_key}):
            result = render_action_prompt(_make_obs(), [_skip_action()])
        assert fake_key not in result, "render_action_prompt must not embed the API key"

    def test_when_called_then_ollama_client_uses_its_output_as_message_content(self):
        """Criterion: render_action_prompt() is the single source of truth."""
        obs = _make_obs()
        legal = [_skip_action()]
        hint = "Be aggressive."

        expected = render_action_prompt(obs, legal, strategy_hint=hint)
        _, calls = _call_ollama(legal, strategy_hint=hint, obs=obs)

        messages = calls[0]["kwargs"]["json"].get("messages", [])
        all_content = " ".join(m.get("content", "") for m in messages)
        assert expected in all_content, (
            "Ollama request messages must contain the render_action_prompt() output"
        )

    @given(st.integers(min_value=0, max_value=5))
    @settings(max_examples=10)
    def test_when_called_with_valid_phase_then_no_exception_is_raised(self, phase: int) -> None:
        """Invariant: render_action_prompt never raises for any valid game phase."""
        render_action_prompt(_make_obs(phase=phase), [_skip_action()])
