"""Tests for global acceptance criteria (Issue #48, oracle tier UNIT / T2).

Covered:
  [T2]   LLM opponent calls Ollama via the native /api/chat endpoint with
          an enforced JSON-schema `format` field.
  [UNIT] Ollama credentials in .env; .env.example is committed.
  [UNIT] LLM output is an index into legal_actions — always legal.
  [UNIT] render_action_prompt() is the single source of truth for the prompt.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.action_prompt import render_action_prompt
from src.agents.ollama_client import call_ollama_for_action_index
from src.utils.constants import NUM_TERRITORIES

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_PLAYERS = 3


def _make_obs(n_players: int = _N_PLAYERS) -> dict[str, np.ndarray]:
    """Minimal but real observation dict for render_action_prompt."""
    return {
        "territory_owner": np.zeros(NUM_TERRITORIES, dtype=np.int32),
        "armies": np.ones(NUM_TERRITORIES, dtype=np.int32),
        "current_player": np.int32(0),
        "phase": np.int32(1),  # REINFORCE
        "reinforcements_remaining": np.int32(3),
        "trade_count": np.int32(0),
        "eliminated": np.zeros(n_players, dtype=np.int32),
        "cards": np.zeros((5, 4), dtype=np.int32),
    }


def _reinforce(territory: int, armies: int) -> dict[str, int]:
    return {"action_type": 1, "param_a": territory, "param_b": armies, "param_c": 0, "param_d": 0}


def _skip() -> dict[str, int]:
    return {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}


def _make_ollama_response(action_index: int) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"message": {"content": json.dumps({"action_index": action_index})}}
    return resp


def _call_ollama(legal_actions, *, temperature=0.5, top_p=0.9, model="gpt-oss:120b"):
    """Call call_ollama_for_action_index with env vars stubbed out."""
    return call_ollama_for_action_index(
        obs=_make_obs(),
        legal_actions=legal_actions,
        model=model,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=temperature,
        top_p=top_p,
    )


# ---------------------------------------------------------------------------
# .env.example committed with required keys
# ---------------------------------------------------------------------------


def test_when_env_example_checked_then_file_exists_at_project_root():
    assert (PROJECT_ROOT / ".env.example").is_file(), (
        ".env.example must be present and committed to the repository"
    )


def test_when_env_example_read_then_ollama_base_url_key_is_present():
    content = (PROJECT_ROOT / ".env.example").read_text()
    assert "OLLAMA_BASE_URL" in content


def test_when_env_example_read_then_ollama_api_key_is_present():
    content = (PROJECT_ROOT / ".env.example").read_text()
    assert "OLLAMA_API_KEY" in content


# ---------------------------------------------------------------------------
# render_action_prompt — single source of truth for the LLM prompt
# ---------------------------------------------------------------------------


def test_when_render_action_prompt_called_then_result_is_a_string():
    obs = _make_obs()
    result = render_action_prompt(obs, [_skip()])
    assert isinstance(result, str)


def test_when_render_action_prompt_called_then_action_indices_appear_in_prompt():
    """The LLM picks by INDEX; every index must appear in the rendered prompt."""
    legal_actions = [_reinforce(0, 1), _reinforce(1, 1), _skip()]
    obs = _make_obs()
    prompt = render_action_prompt(obs, legal_actions)
    for i in range(len(legal_actions)):
        assert str(i) in prompt, f"Expected index {i} in rendered prompt"


def test_when_render_action_prompt_called_then_territory_names_appear_in_prompt():
    """Board section must reference territory names so the LLM can reason spatially."""
    legal_actions = [_reinforce(0, 3)]  # territory 0 = Alaska
    obs = _make_obs()
    prompt = render_action_prompt(obs, legal_actions)
    assert "Alaska" in prompt, "Territory names must appear in the LLM prompt"


def test_when_render_action_prompt_called_then_reply_instruction_is_in_prompt():
    """Prompt must include the reply instruction so the LLM knows the output format."""
    obs = _make_obs()
    prompt = render_action_prompt(obs, [_skip()])
    assert "action_index" in prompt, (
        "Prompt must contain 'action_index' so the LLM returns the right key"
    )


@given(
    n_actions=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=30)
def test_when_render_action_prompt_called_with_any_n_actions_then_no_error_is_raised(
    n_actions,
):
    """Property: render_action_prompt accepts any non-empty legal_actions list."""
    legal_actions = [_skip() for _ in range(n_actions)]
    obs = _make_obs()
    result = render_action_prompt(obs, legal_actions)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Ollama native /api/chat with an enforced `format` schema
# ---------------------------------------------------------------------------


def test_when_ollama_called_then_url_targets_api_chat():
    """Criterion: LLM opponent calls Ollama via the native /api/chat endpoint."""
    captured: dict = {}

    def _fake_post(url, **kwargs):
        captured["url"] = url
        return _make_ollama_response(0)

    with patch("httpx.post", side_effect=_fake_post):
        _call_ollama([_skip()])

    url = captured.get("url", "")
    assert url.endswith("/api/chat"), "Request URL must target the native /api/chat endpoint"
    assert "/v1/" not in url, "Native endpoint must not contain /v1/"


def test_when_ollama_called_then_format_type_is_object():
    """Criterion: request uses an object `format` JSON schema."""
    captured: dict = {}

    def _fake_post(url, **kwargs):
        captured["body"] = kwargs.get("json", {})
        return _make_ollama_response(0)

    with patch("httpx.post", side_effect=_fake_post):
        _call_ollama([_skip()])

    body = captured.get("body", {})
    assert "format" in body
    assert body["format"].get("type") == "object"


def test_when_ollama_called_then_format_schema_requires_action_index():
    """Criterion: the enforced `format` schema constrains output to 'action_index'."""
    captured: dict = {}

    def _fake_post(url, **kwargs):
        captured["body"] = kwargs.get("json", {})
        return _make_ollama_response(0)

    with patch("httpx.post", side_effect=_fake_post):
        _call_ollama([_skip()])

    fmt = captured["body"].get("format", {})
    assert "action_index" in fmt.get("required", [])
    assert fmt.get("properties", {}).get("action_index", {}).get("type") == "integer"


def test_when_ollama_called_then_temperature_is_in_request_options():
    """Criterion: per-call temperature injected into the Ollama request options."""
    captured: dict = {}

    def _fake_post(url, **kwargs):
        captured["body"] = kwargs.get("json", {})
        return _make_ollama_response(0)

    with patch("httpx.post", side_effect=_fake_post):
        _call_ollama([_skip()], temperature=0.3)

    assert captured["body"].get("options", {}).get("temperature") == pytest.approx(0.3)


def test_when_ollama_called_then_top_p_is_in_request_options():
    """Criterion: per-call top_p injected into the Ollama request options."""
    captured: dict = {}

    def _fake_post(url, **kwargs):
        captured["body"] = kwargs.get("json", {})
        return _make_ollama_response(0)

    with patch("httpx.post", side_effect=_fake_post):
        _call_ollama([_skip()], top_p=0.95)

    assert captured["body"].get("options", {}).get("top_p") == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# LLM output is an index into legal_actions — always legal by construction
# ---------------------------------------------------------------------------


def test_when_ollama_returns_action_index_then_returned_value_is_int():
    with patch("httpx.post", return_value=_make_ollama_response(0)):
        result = _call_ollama([_skip(), _reinforce(0, 1)])
    assert isinstance(result, int)


def test_when_ollama_returns_action_index_then_returned_value_is_within_legal_bounds():
    """Criterion: the chosen action is always legal by construction."""
    legal_actions = [_skip(), _reinforce(0, 1), _reinforce(1, 2)]
    with patch("httpx.post", return_value=_make_ollama_response(2)):
        index = _call_ollama(legal_actions)
    assert 0 <= index < len(legal_actions)


@given(
    chosen=st.integers(min_value=0, max_value=9),
    n_actions=st.integers(min_value=10, max_value=20),
)
@settings(max_examples=30)
def test_when_ollama_returns_any_valid_index_then_it_is_always_in_range(chosen, n_actions):
    """Property: any valid index the LLM returns must be in [0, len(legal_actions))."""
    legal_actions = [_skip()] * n_actions
    with patch("httpx.post", return_value=_make_ollama_response(chosen)):
        index = _call_ollama(legal_actions)
    assert 0 <= index < len(legal_actions)
