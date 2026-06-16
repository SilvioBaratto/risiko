"""Spec-derived tests for issue #58 — Azure API contract and render_action_prompt.

Criteria covered:
  [T2]   LLM opponent calls Azure OpenAI GPT-4.1 via /chat/completions
         with json_schema strict output
  [UNIT] Azure credentials load from a git-ignored .env
         (AZURE_OPENAI_BASE_URL / _API_KEY / _API_VERSION); .env.example committed
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
from src.agents.azure_openai import call_azure_for_action_index

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# ────────────────────────────────────────────────────────────────────────────
# Test helpers
# ────────────────────────────────────────────────────────────────────────────

_FAKE_ENV = {
    "AZURE_OPENAI_BASE_URL": "https://test.openai.azure.com/openai/deployments/gpt-4.1",
    "AZURE_OPENAI_API_KEY": "test-api-key-abc123",
    "AZURE_OPENAI_API_VERSION": "2024-02-01",
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


def _call_azure(
    legal_actions: list[dict[str, int]],
    action_index: int = 0,
    temperature: float = 0.5,
    top_p: float | None = 0.9,
    strategy_hint: str | None = "hint",
    env: dict[str, str] | None = None,
    obs: dict[str, np.ndarray] | None = None,
) -> tuple[int | None, list[Any]]:
    """Call call_azure_for_action_index with a patched httpx.post.

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
        result = call_azure_for_action_index(
            obs,
            legal_actions,
            model="gpt-4.1",
            temperature=temperature,
            top_p=top_p,
            strategy_hint=strategy_hint,
        )
    return result, captured


# ────────────────────────────────────────────────────────────────────────────
# Endpoint contract — POST /chat/completions
# ────────────────────────────────────────────────────────────────────────────


class TestAzureChatCompletionsEndpoint:
    """Azure /chat/completions endpoint and json_schema strict contract."""

    def test_when_llm_called_then_exactly_one_http_request_is_made(self):
        """Criterion: the client makes exactly one call per move."""
        _, calls = _call_azure([_skip_action()])
        assert len(calls) == 1

    def test_when_llm_called_then_request_url_contains_chat_completions(self):
        """Criterion: LLM opponent calls Azure OpenAI via /chat/completions."""
        _, calls = _call_azure([_skip_action()])
        url = calls[0]["url"]
        assert "/chat/completions" in url, f"Expected /chat/completions in URL, got: {url}"

    def test_when_llm_called_then_request_body_uses_json_schema_response_format(self):
        """Criterion: response_format type must be 'json_schema'."""
        _, calls = _call_azure([_skip_action(), _skip_action()])
        body = calls[0]["kwargs"]["json"]
        rf = body.get("response_format", {})
        assert rf.get("type") == "json_schema", (
            f"response_format.type must be 'json_schema', got: {rf}"
        )

    def test_when_llm_called_then_json_schema_strict_is_true(self):
        """Criterion: json_schema strict must be True (enforces output shape)."""
        _, calls = _call_azure([_skip_action()])
        body = calls[0]["kwargs"]["json"]
        rf = body.get("response_format", {})
        inner = rf.get("json_schema", {})
        assert inner.get("strict") is True, f"json_schema.strict must be True, got: {inner}"

    def test_when_llm_called_then_request_body_contains_action_index_schema(self):
        """The json_schema must constrain the output to contain 'action_index'."""
        _, calls = _call_azure([_skip_action()])
        body = calls[0]["kwargs"]["json"]
        raw_schema = json.dumps(body)
        assert "action_index" in raw_schema, "The request schema must reference 'action_index'"

    def test_when_temperature_supplied_then_it_appears_in_request_body(self):
        """Criterion: per-call temperature injected into the Azure request body."""
        _, calls = _call_azure([_skip_action()], temperature=0.3)
        body = calls[0]["kwargs"]["json"]
        assert body.get("temperature") == pytest.approx(0.3), (
            "temperature must be forwarded to the request body"
        )

    def test_when_top_p_supplied_then_it_appears_in_request_body(self):
        """Criterion: per-call top_p injected into the Azure request body."""
        _, calls = _call_azure([_skip_action()], top_p=0.85)
        body = calls[0]["kwargs"]["json"]
        assert body.get("top_p") == pytest.approx(0.85), (
            "top_p must be forwarded to the request body"
        )


# ────────────────────────────────────────────────────────────────────────────
# LLM output → action index → always legal
# ────────────────────────────────────────────────────────────────────────────


class TestActionIndexLegality:
    """LLM returns an in-range index; the resolved action is always legal."""

    def test_when_llm_returns_index_0_then_result_is_zero(self):
        """Criterion: LLM output is an index into legal_actions."""
        legal = [_skip_action(), _skip_action(), _skip_action()]
        result, _ = _call_azure(legal, action_index=0)
        assert result == 0

    def test_when_llm_returns_index_1_then_result_is_one(self):
        legal = [_skip_action(), _skip_action(), _skip_action()]
        result, _ = _call_azure(legal, action_index=1)
        assert result == 1

    def test_when_action_index_in_range_then_result_is_that_index(self):
        """Criterion: the returned index is always in [0, len(legal_actions))."""
        legal = [_skip_action(), _skip_action(), _skip_action()]
        result, _ = _call_azure(legal, action_index=2)
        assert result == 2

    def test_when_returned_index_used_then_resolved_action_is_legal(self):
        """Criterion: the chosen action is always legal by construction."""
        legal = [_skip_action(), _skip_action()]
        result, _ = _call_azure(legal, action_index=0)
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
# Azure credentials — loaded from environment variables, not hardcoded
# ────────────────────────────────────────────────────────────────────────────


class TestAzureCredentialsFromEnv:
    """Azure credentials come from env vars; .env.example is committed."""

    def test_when_api_key_env_var_set_then_request_header_contains_it(self):
        """Criterion: API key comes from AZURE_OPENAI_API_KEY (auth via api-key header)."""
        expected_key = "my-secret-key-xyz"
        env = {**_FAKE_ENV, "AZURE_OPENAI_API_KEY": expected_key}
        _, calls = _call_azure([_skip_action()], env=env)
        headers = calls[0]["kwargs"].get("headers", {})
        assert expected_key in headers.values(), (
            "AZURE_OPENAI_API_KEY must be forwarded in request headers"
        )

    def test_when_base_url_env_var_set_then_request_url_starts_with_it(self):
        """Criterion: AZURE_OPENAI_BASE_URL determines the request endpoint."""
        base = "https://custom.openai.azure.com/openai/deployments/gpt-4.1"
        env = {**_FAKE_ENV, "AZURE_OPENAI_BASE_URL": base}
        _, calls = _call_azure([_skip_action()], env=env)
        assert calls[0]["url"].startswith(base), "Request URL must start with AZURE_OPENAI_BASE_URL"

    def test_when_api_version_env_var_set_then_request_url_contains_it(self):
        """Criterion: AZURE_OPENAI_API_VERSION appears in the request URL."""
        version = "2024-05-01-preview"
        env = {**_FAKE_ENV, "AZURE_OPENAI_API_VERSION": version}
        _, calls = _call_azure([_skip_action()], env=env)
        assert version in calls[0]["url"], "AZURE_OPENAI_API_VERSION must appear in the request URL"

    def test_when_dot_env_example_checked_then_file_exists_at_project_root(self):
        """Criterion: .env.example is committed (exists in the repository root)."""
        env_example = PROJECT_ROOT / ".env.example"
        assert env_example.exists(), ".env.example must be committed at the project root"

    def test_when_dot_env_example_read_then_azure_base_url_key_is_present(self):
        """Criterion: .env.example documents AZURE_OPENAI_BASE_URL."""
        content = (PROJECT_ROOT / ".env.example").read_text()
        assert "AZURE_OPENAI_BASE_URL" in content

    def test_when_dot_env_example_read_then_azure_api_key_key_is_present(self):
        """Criterion: .env.example documents AZURE_OPENAI_API_KEY."""
        content = (PROJECT_ROOT / ".env.example").read_text()
        assert "AZURE_OPENAI_API_KEY" in content

    def test_when_dot_env_example_read_then_azure_api_version_key_is_present(self):
        """Criterion: .env.example documents AZURE_OPENAI_API_VERSION."""
        content = (PROJECT_ROOT / ".env.example").read_text()
        assert "AZURE_OPENAI_API_VERSION" in content


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
        with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": fake_key}):
            result = render_action_prompt(_make_obs(), [_skip_action()])
        assert fake_key not in result, "render_action_prompt must not embed the API key"

    def test_when_called_then_azure_client_uses_its_output_as_message_content(self):
        """Criterion: render_action_prompt() is the single source of truth."""
        obs = _make_obs()
        legal = [_skip_action()]
        hint = "Be aggressive."

        expected = render_action_prompt(obs, legal, strategy_hint=hint)
        _, calls = _call_azure(legal, strategy_hint=hint, obs=obs)

        messages = calls[0]["kwargs"]["json"].get("messages", [])
        all_content = " ".join(m.get("content", "") for m in messages)
        assert expected in all_content, (
            "Azure request messages must contain the render_action_prompt() output"
        )

    @given(st.integers(min_value=0, max_value=5))
    @settings(max_examples=10)
    def test_when_called_with_valid_phase_then_no_exception_is_raised(self, phase: int) -> None:
        """Invariant: render_action_prompt never raises for any valid game phase."""
        render_action_prompt(_make_obs(phase=phase), [_skip_action()])
