"""Acceptance tests for issue #59: LLMOpponent on Azure backend.

30 s timeout, random fallback, per-call sampling, pickle-safe.
One named test per verifiable LLM acceptance criterion.
All HTTP calls are mocked — no real network or .env access.
"""

from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.action_prompt import render_action_prompt
from src.agents.azure_openai import AzureConfigError, call_azure_for_action_index
from src.agents.base import Agent
from src.agents.llm_opponent import LLMOpponent
from src.agents.player_config import PlayerConfig

# ---------------------------------------------------------------------------
# Shared test doubles
# ---------------------------------------------------------------------------

_CREDS = {
    "AZURE_OPENAI_BASE_URL": "https://fake.openai.azure.com/openai/deployments/gpt-4.1",
    "AZURE_OPENAI_API_KEY": "test-key-00000000",
    "AZURE_OPENAI_API_VERSION": "2024-12-01",
}

_FAKE_OBS: dict = {}  # acceptable for azure-client tests (render_action_prompt mocked)


def _cfg(**overrides) -> PlayerConfig:
    return PlayerConfig(
        **{
            "player_id": 0,
            "model": "gpt-4.1",
            "temperature": 0.5,
            "top_p": 0.9,
            "strategy_hint": "Play greedily.",
            **overrides,
        }
    )


def _ok_response(idx: int = 0) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps({"action_index": idx})}}]
    }
    return resp


@pytest.fixture
def env_data():
    """Provide real obs and legal_actions from a 3-player game."""
    from src.env import RisikoEnv

    env = RisikoEnv(n_players=3)
    obs, info = env.reset(seed=42)
    return obs, info["legal_actions"]


# ---------------------------------------------------------------------------
# 1. json_schema strict
#    Criterion: LLM opponent calls Azure GPT-4.1 via /chat/completions with
#               json_schema strict output.
# ---------------------------------------------------------------------------


def test_when_choose_action_called_then_request_uses_json_schema_strict():
    """Azure request body must declare response_format.type='json_schema' with strict=True."""
    with (
        patch.dict(os.environ, _CREDS, clear=False),
        patch("src.agents.azure_openai.ensure_env_loaded"),
        patch("src.agents.azure_openai.render_action_prompt", return_value="prompt"),
        patch("src.agents.azure_openai.httpx.post", return_value=_ok_response(0)) as mock_post,
    ):
        call_azure_for_action_index(_FAKE_OBS, ["a"], model="gpt-4.1", temperature=0.5, top_p=0.9)

    body = mock_post.call_args.kwargs["json"]
    rf = body.get("response_format", {})
    assert rf.get("type") == "json_schema"
    assert rf.get("json_schema", {}).get("strict") is True


# ---------------------------------------------------------------------------
# 2. Per-call temperature / top_p in body
#    Criterion: per-call temperature/top_p reach the Azure request body.
# ---------------------------------------------------------------------------


def test_when_choose_action_called_then_temperature_and_top_p_are_in_request_body():
    """PlayerConfig temperature and top_p must appear verbatim in the POST body."""
    with (
        patch.dict(os.environ, _CREDS, clear=False),
        patch("src.agents.azure_openai.ensure_env_loaded"),
        patch("src.agents.azure_openai.render_action_prompt", return_value="prompt"),
        patch("src.agents.azure_openai.httpx.post", return_value=_ok_response(0)) as mock_post,
    ):
        call_azure_for_action_index(_FAKE_OBS, ["a"], model="gpt-4.1", temperature=0.3, top_p=0.85)

    body = mock_post.call_args.kwargs["json"]
    assert body["temperature"] == pytest.approx(0.3)
    assert body["top_p"] == pytest.approx(0.85)


@given(
    temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    top_p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=25)
def test_when_any_valid_temperature_and_top_p_then_both_reach_request_body(
    temperature: float, top_p: float
) -> None:
    """Property: for any valid (temperature, top_p), both reach the body unchanged."""
    with (
        patch.dict(os.environ, _CREDS, clear=False),
        patch("src.agents.azure_openai.ensure_env_loaded"),
        patch("src.agents.azure_openai.render_action_prompt", return_value="prompt"),
        patch("src.agents.azure_openai.httpx.post", return_value=_ok_response(0)) as mock_post,
    ):
        call_azure_for_action_index(
            _FAKE_OBS,
            ["a"],
            model="gpt-4.1",
            temperature=temperature,
            top_p=top_p,
        )

    body = mock_post.call_args.kwargs["json"]
    assert body["temperature"] == pytest.approx(temperature, abs=1e-9)
    assert body["top_p"] == pytest.approx(top_p, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. strategy_hint in rendered prompt content
#    Criterion: strategy_hint appears in the rendered prompt; precedes legal actions.
# ---------------------------------------------------------------------------


def test_when_strategy_hint_provided_then_hint_appears_in_rendered_prompt(env_data):
    """render_action_prompt() must embed strategy_hint in its output."""
    obs, legal = env_data
    hint = "Focus on securing full continents before expanding."
    result = render_action_prompt(obs, legal, strategy_hint=hint)
    assert hint in result


def test_when_strategy_hint_provided_then_hint_appears_before_legal_actions_in_prompt(
    env_data,
):
    """strategy_hint must appear before the Legal Actions section in the prompt."""
    obs, legal = env_data
    hint = "UNIQUE_HINT_MARKER_XYZ_987"
    prompt = render_action_prompt(obs, legal, strategy_hint=hint)
    hint_pos = prompt.find(hint)
    assert hint_pos != -1, "strategy_hint not found in rendered prompt"
    actions_pos = prompt.find("## Legal Actions")
    assert hint_pos < actions_pos, "strategy_hint does not precede the legal actions section"


# ---------------------------------------------------------------------------
# 4. Timeout → fallback
#    Criterion: ≤ 30s hard timeout; falls back to RandomAgent on timeout.
# ---------------------------------------------------------------------------


def test_when_llm_times_out_then_fallback_returns_legal_action(env_data):
    """On FuturesTimeoutError from _call_with_timeout, fallback action is in legal_actions."""
    obs, legal = env_data
    with patch.object(LLMOpponent, "_call_with_timeout", side_effect=FuturesTimeoutError()):
        action = LLMOpponent(player_config=_cfg()).act(obs, legal)
    assert action in legal


# ---------------------------------------------------------------------------
# 5. HTTP error → fallback
#    Criterion: falls back to RandomAgent on any HTTP error.
# ---------------------------------------------------------------------------


def test_when_http_error_occurs_then_fallback_returns_legal_action(env_data):
    """On httpx.HTTPError, LLMOpponent must fall back and return a legal action."""
    obs, legal = env_data
    with patch.object(LLMOpponent, "_call_with_timeout", side_effect=httpx.HTTPError("HTTP 500")):
        action = LLMOpponent(player_config=_cfg()).act(obs, legal)
    assert action in legal


# ---------------------------------------------------------------------------
# 6. Parse error → fallback
#    Criterion: falls back to RandomAgent on parse error.
# ---------------------------------------------------------------------------


def test_when_parse_error_occurs_then_fallback_returns_legal_action(env_data):
    """On JSON parse failure, LLMOpponent must fall back and return a legal action."""
    obs, legal = env_data
    with patch.object(
        LLMOpponent,
        "_call_with_timeout",
        side_effect=json.JSONDecodeError("bad", "", 0),
    ):
        action = LLMOpponent(player_config=_cfg()).act(obs, legal)
    assert action in legal


# ---------------------------------------------------------------------------
# 7. Out-of-range / empty index → fallback
#    Criterion: falls back to RandomAgent on out-of-range or empty index.
# ---------------------------------------------------------------------------


def test_when_out_of_range_index_returned_then_fallback_returns_legal_action(env_data):
    """When _call_with_timeout returns None (out-of-range handled internally), fallback is used."""
    obs, legal = env_data
    with patch.object(LLMOpponent, "_call_with_timeout", return_value=None):
        action = LLMOpponent(player_config=_cfg()).act(obs, legal)
    assert action in legal


# ---------------------------------------------------------------------------
# 8. Config error when credentials are unresolved
#    Criterion: config error when creds unresolved.
# ---------------------------------------------------------------------------


def test_when_credentials_not_configured_then_config_error_is_raised(monkeypatch):
    """Missing Azure credentials must raise AzureConfigError before any HTTP call."""
    monkeypatch.delenv("AZURE_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    with (
        patch("src.agents.azure_openai.ensure_env_loaded"),
        pytest.raises(AzureConfigError),
    ):
        call_azure_for_action_index(_FAKE_OBS, ["a"], model="gpt-4.1", temperature=0.5, top_p=0.9)


# ---------------------------------------------------------------------------
# 9. URL ends /chat/completions?api-version=... with api-key header
#    Criterion: URL format + api-key header.
# ---------------------------------------------------------------------------


def test_when_request_sent_url_contains_chat_completions_api_version_and_api_key_header():
    """POST URL must contain /chat/completions?api-version=...; api-key header required."""
    with (
        patch.dict(os.environ, _CREDS, clear=False),
        patch("src.agents.azure_openai.ensure_env_loaded"),
        patch("src.agents.azure_openai.render_action_prompt", return_value="prompt"),
        patch("src.agents.azure_openai.httpx.post", return_value=_ok_response(0)) as mock_post,
    ):
        call_azure_for_action_index(_FAKE_OBS, ["a"], model="gpt-4.1", temperature=0.5, top_p=0.9)

    url: str = mock_post.call_args.args[0]
    headers: dict = mock_post.call_args.kwargs["headers"]
    assert "/chat/completions" in url, f"URL {url!r} missing '/chat/completions'"
    assert "api-version=" in url, f"URL {url!r} missing 'api-version=' query param"
    assert "api-key" in headers, f"'api-key' header missing; got: {list(headers)}"


# ---------------------------------------------------------------------------
# 10. Empty legal_actions → ValueError
#     Criterion: raises ValueError on empty legal_actions.
# ---------------------------------------------------------------------------


def test_when_legal_actions_empty_then_value_error_is_raised(env_data):
    """LLMOpponent.act() must raise ValueError immediately when legal_actions is empty."""
    obs, _ = env_data
    with pytest.raises(ValueError, match="empty"):
        LLMOpponent(player_config=_cfg()).act(obs, [])


# ---------------------------------------------------------------------------
# 11. Pickle round-trip: attributes preserved, Agent protocol satisfied
#     Criterion: pickle.loads(pickle.dumps(...)) preserves _model/_temperature/_strategy_hint.
# ---------------------------------------------------------------------------


def test_when_pickle_roundtrip_then_attributes_preserved_and_agent_protocol_satisfied():
    """Pickle/unpickle must preserve _model, _temperature, _strategy_hint and Agent protocol."""
    cfg = _cfg(model="gpt-4.1", temperature=0.7, strategy_hint="unique-hint-abc")
    restored: LLMOpponent = pickle.loads(pickle.dumps(LLMOpponent(player_config=cfg)))
    assert isinstance(restored, Agent)
    assert restored._model == "gpt-4.1"
    assert restored._temperature == pytest.approx(0.7)
    assert restored._strategy_hint == "unique-hint-abc"


# ---------------------------------------------------------------------------
# 12. PlayerConfig params override scalar kwargs
#     Criterion: model/temperature/top_p/strategy_hint from PlayerConfig override scalars.
# ---------------------------------------------------------------------------


def test_when_player_config_provided_then_config_params_override_scalar_kwargs():
    """PlayerConfig values must take precedence over contradicting constructor scalar kwargs."""
    cfg = _cfg(model="gpt-4.1", temperature=0.1, top_p=0.95, strategy_hint="Config hint")
    opponent = LLMOpponent(
        player_config=cfg,
        model="old-model",
        temperature=0.99,
        top_p=0.01,
        strategy_hint="Scalar hint",
    )
    assert opponent._model == "gpt-4.1"
    assert opponent._temperature == pytest.approx(0.1)
    assert opponent._top_p == pytest.approx(0.95)
    assert opponent._strategy_hint == "Config hint"


# ---------------------------------------------------------------------------
# 13. Returned action always in legal_actions (property)
#     Criterion: fallback action always in legal_actions — invariant over any input.
# ---------------------------------------------------------------------------


@given(
    st.lists(
        st.fixed_dictionaries(
            {
                "action_type": st.integers(min_value=0, max_value=5),
                "param_a": st.integers(min_value=0, max_value=41),
                "param_b": st.integers(min_value=0, max_value=41),
                "param_c": st.integers(min_value=0, max_value=10),
                "param_d": st.integers(min_value=0, max_value=0),
            }
        ),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=20)
def test_when_fallback_triggered_then_returned_action_is_always_in_legal_actions(
    legal_actions: list[dict],
) -> None:
    """Property: for any non-empty legal_actions, the fallback action is always in it."""
    with patch.object(LLMOpponent, "_call_with_timeout", side_effect=RuntimeError("fail")):
        action = LLMOpponent(player_config=_cfg()).act({}, legal_actions)
    assert action in legal_actions


# ---------------------------------------------------------------------------
# 14. .env.example committed with required variable names
#     Criterion: Azure credentials load from git-ignored .env; .env.example committed.
# ---------------------------------------------------------------------------


def test_when_env_example_committed_then_all_required_variable_names_are_present():
    """.env.example at project root must list all three Azure credential variable names."""
    project_root = Path(__file__).resolve().parent.parent
    env_example = project_root / ".env.example"
    assert env_example.exists(), f".env.example not found at {project_root}"
    content = env_example.read_text()
    assert "AZURE_OPENAI_BASE_URL" in content
    assert "AZURE_OPENAI_API_KEY" in content
    assert "AZURE_OPENAI_API_VERSION" in content
