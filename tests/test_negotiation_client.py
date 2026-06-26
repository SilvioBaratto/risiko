"""Tests for call_ollama_for_negotiation and render_negotiation_prompt (issue #100).

HTTP is mocked with httpx throughout — no live Ollama needed.
"""

from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock, patch

import httpx
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.negotiation_prompt import render_negotiation_prompt
from src.agents.ollama_client import (
    _NEGOTIATION_SCHEMA,
    call_ollama_for_action_index,
    call_ollama_for_negotiation,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_ROOT = "http://localhost:11434"
_KEY = "test-key"
_MODEL = "gemma:7b"
_MAX_TOKENS = 64
_PROMPT = "You are player 0. Negotiate with your opponents."

_VALID_CONTENT = json.dumps(
    {
        "messages_to": {"1": "Let us not attack each other this round."},
        "propose_alliance_with": [1],
        "accept_alliance_with": [],
        "declare_war_on": [],
        "attack_priority": [],
    }
)


def _make_http_ok(content: str = _VALID_CONTENT) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"message": {"content": content}}
    return resp


def _make_http_truncated(content: str = "") -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"message": {"content": content}, "done_reason": "length"}
    return resp


def _call(**overrides):
    """Call call_ollama_for_negotiation with sensible defaults."""
    kwargs = {
        "root": _ROOT,
        "key": _KEY,
        "model": _MODEL,
        "prompt": _PROMPT,
        "max_message_tokens": _MAX_TOKENS,
    }
    kwargs.update(overrides)
    return call_ollama_for_negotiation(**kwargs)


def _minimal_board() -> dict:
    return {
        "territories": {
            "Alaska": {"owner": 0, "armies": 3},
            "Kamchatka": {"owner": 1, "armies": 5},
        },
        "reinforcements_remaining": 3,
        "trade_count": 2,
    }


# ===========================================================================
# Criterion: POSTs to {root}/api/chat with _NEGOTIATION_SCHEMA as `format`
#            and Authorization: Bearer <key> header
# ===========================================================================


def test_when_called_then_posts_to_api_chat_endpoint():
    """POSTs to the Ollama native /api/chat endpoint, not /v1/chat/completions."""
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = _make_http_ok()

        _call()

        assert mock_httpx.post.called
        call_args = mock_httpx.post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert url == f"{_ROOT}/api/chat"


def test_when_called_then_format_field_equals_negotiation_schema():
    """Request body carries _NEGOTIATION_SCHEMA as the 'format' field."""
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = _make_http_ok()

        _call()

        payload = mock_httpx.post.call_args[1].get("json", {})
        assert "format" in payload, "'format' key missing from request body"
        assert payload["format"] == _NEGOTIATION_SCHEMA


def test_when_called_then_authorization_bearer_header_is_set():
    """Authorization: Bearer <key> header is present in every request."""
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = _make_http_ok()

        _call(key="my-secret")

        headers = mock_httpx.post.call_args[1].get("headers", {})
        assert headers.get("Authorization") == "Bearer my-secret"


# ===========================================================================
# Criterion: num_predict is bounded by max_message_tokens
# ===========================================================================


def test_when_max_message_tokens_given_then_num_predict_does_not_exceed_it():
    """num_predict in Ollama options must be <= max_message_tokens."""
    limit = 37
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = _make_http_ok()

        _call(max_message_tokens=limit)

        payload = mock_httpx.post.call_args[1].get("json", {})
        num_predict = payload.get("options", {}).get("num_predict")
        assert num_predict is not None, "num_predict must appear in request options"
        assert num_predict <= limit


@given(st.integers(min_value=1, max_value=4096))
@settings(max_examples=100)
def test_when_max_message_tokens_is_any_positive_int_then_num_predict_is_bounded(limit):
    """Invariant: num_predict <= max_message_tokens for every valid positive-integer limit."""
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = _make_http_ok()

        _call(max_message_tokens=limit)

        payload = mock_httpx.post.call_args[1].get("json", {})
        num_predict = payload.get("options", {}).get("num_predict", float("inf"))
        assert num_predict <= limit


# ===========================================================================
# Criterion: Returns a parsed dict on success; returns None on any failure
# ===========================================================================


def test_when_response_is_valid_json_then_a_dict_is_returned():
    """On a well-formed 200 response, returns the parsed negotiation dict."""
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = _make_http_ok()

        result = _call()

        assert isinstance(result, dict)


def test_when_messages_to_is_absent_then_constrained_fields_still_parse():
    """Constrained integer-array fields parse successfully even without messages_to."""
    content = json.dumps(
        {
            "propose_alliance_with": [1],
            "accept_alliance_with": [],
            "declare_war_on": [3],
            "attack_priority": [3],
        }
    )
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = _make_http_ok(content=content)

        result = _call()

        assert isinstance(result, dict)
        assert "propose_alliance_with" in result


def test_when_content_is_markdown_fenced_then_json_is_recovered():
    """Cloud models wrap the object in ```json fences; the fence is stripped."""
    inner = json.dumps(
        {
            "propose_alliance_with": [1, 2],
            "accept_alliance_with": [],
            "declare_war_on": [3],
            "attack_priority": [3],
        }
    )
    content = f"```json\n{inner}\n```"
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = _make_http_ok(content=content)

        result = _call()

        assert isinstance(result, dict)
        assert result["propose_alliance_with"] == [1, 2]
        assert result["declare_war_on"] == [3]


def test_when_content_has_prose_around_object_then_json_is_recovered():
    """Prose before/after the object is ignored; the {...} span is parsed."""
    content = 'Here is my move:\n{"propose_alliance_with": [1], "attack_priority": []}\nGood luck.'
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = _make_http_ok(content=content)

        result = _call()

        assert isinstance(result, dict)
        assert result["propose_alliance_with"] == [1]


def test_when_content_is_empty_then_none_is_returned_without_exception():
    """Empty response content falls back to None (same contract as action client)."""
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = _make_http_ok(content="")

        result = _call()

        assert result is None


def test_when_content_is_garbage_json_then_none_is_returned_without_exception():
    """Unparseable JSON content falls back to None."""
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = _make_http_ok(content="not json !@# }")

        result = _call()

        assert result is None


def test_when_done_reason_is_length_then_none_is_returned_without_exception():
    """Truncated response (done_reason=length) returns None — known documented behaviour."""
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = _make_http_truncated('{"propose_alliance_with": [1, 2')

        result = _call()

        assert result is None


def test_when_request_times_out_then_none_is_returned_without_exception():
    """Timeout exception is swallowed and None is returned."""
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.side_effect = httpx.TimeoutException("timeout")

        result = _call()

        assert result is None


def test_when_http_status_error_is_raised_then_none_is_returned_without_exception():
    """HTTP 4xx/5xx from raise_for_status() is swallowed and None is returned."""
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        mock_httpx.post.return_value = resp

        result = _call()

        assert result is None


def test_when_connection_error_occurs_then_none_is_returned_without_exception():
    """Connection errors are swallowed and None is returned."""
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.side_effect = httpx.ConnectError("connection refused")

        result = _call()

        assert result is None


@given(st.text())
@settings(max_examples=200)
def test_when_content_is_any_string_then_no_exception_is_ever_raised(content):
    """Invariant: call_ollama_for_negotiation never raises for any content string."""
    with patch("src.agents.ollama_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = _make_http_ok(content=content)

        call_ollama_for_negotiation(
            root=_ROOT,
            key=_KEY,
            model=_MODEL,
            prompt=_PROMPT,
            max_message_tokens=_MAX_TOKENS,
        )


# ===========================================================================
# Criterion: call_ollama_for_action_index is left byte-identical
# ===========================================================================


def test_when_negotiation_function_exists_then_it_is_a_distinct_callable():
    """Negotiation logic must live in a separate function, not grafted onto the action client."""
    assert call_ollama_for_negotiation is not call_ollama_for_action_index


def test_when_action_index_function_inspected_then_no_diplomacy_kwargs_were_added():
    """call_ollama_for_action_index must not have gained any negotiation-related parameters."""
    sig = inspect.signature(call_ollama_for_action_index)
    param_names = set(sig.parameters.keys())
    forbidden = {"allies", "grudges", "leader", "diplomacy_state", "negotiation", "alliance"}
    added = param_names & forbidden
    assert not added, f"call_ollama_for_action_index gained unexpected diplomacy kwargs: {added}"


# ===========================================================================
# Criterion: render_negotiation_prompt includes board + allies / leader / grudges
#            and caps message length
# ===========================================================================


def test_when_render_called_then_output_includes_territory_information():
    """Prompt must include board/territory information."""
    result = render_negotiation_prompt(
        player_id=0,
        board=_minimal_board(),
        allies=[],
        leader=None,
        grudges={},
        max_chars=2000,
    )
    assert isinstance(result, str) and len(result) > 0
    assert any(tok in result for tok in ("Alaska", "Kamchatka", "territories", "territory"))


def test_when_render_called_with_allies_then_output_references_ally():
    """Prompt must reference current allies."""
    result = render_negotiation_prompt(
        player_id=0,
        board=_minimal_board(),
        allies=[2, 3],
        leader=None,
        grudges={},
        max_chars=2000,
    )
    has_id = "2" in result or "3" in result
    has_word = "ally" in result.lower() or "alliance" in result.lower()
    assert has_id or has_word, "ally ids 2 and 3 are not mentioned in the prompt"


def test_when_render_called_with_leader_then_output_references_leader():
    """Prompt must reference the current leader."""
    result = render_negotiation_prompt(
        player_id=0,
        board=_minimal_board(),
        allies=[],
        leader=4,
        grudges={},
        max_chars=2000,
    )
    has_id = "4" in result
    has_word = any(w in result.lower() for w in ("leader", "leading", "ahead", "dominant"))
    assert has_id or has_word, "leader (player 4) is not mentioned in the prompt"


def test_when_render_called_with_grudges_then_output_references_grudge():
    """Prompt must reference current grudges / betrayal history."""
    result = render_negotiation_prompt(
        player_id=0,
        board=_minimal_board(),
        allies=[],
        leader=None,
        grudges={1: ["attacked turn 3", "betrayed alliance turn 5"]},
        max_chars=2000,
    )
    has_player = "1" in result
    has_word = any(w in result.lower() for w in ("grudge", "betray", "attacked", "trust"))
    assert has_player or has_word, "grudge against player 1 is not mentioned in the prompt"


def test_when_render_called_then_output_does_not_exceed_max_chars():
    """Prompt must be truncated/capped to max_chars."""
    max_chars = 300
    result = render_negotiation_prompt(
        player_id=0,
        board=_minimal_board(),
        allies=[1, 2, 3, 4, 5],
        leader=5,
        grudges={i: ["attacked"] * 30 for i in range(1, 6)},
        max_chars=max_chars,
    )
    assert len(result) <= max_chars, f"Prompt length {len(result)} exceeded max_chars={max_chars}"


@given(
    max_chars=st.integers(min_value=10, max_value=10_000),
    n_allies=st.integers(min_value=0, max_value=5),
    n_grudge_events=st.integers(min_value=0, max_value=50),
)
@settings(max_examples=150)
def test_when_render_called_with_any_inputs_then_length_never_exceeds_max_chars(
    max_chars, n_allies, n_grudge_events
):
    """Invariant: render_negotiation_prompt never returns more than max_chars characters."""
    allies = list(range(1, n_allies + 1))
    grudges = {1: ["event"] * n_grudge_events} if n_grudge_events else {}
    result = render_negotiation_prompt(
        player_id=0,
        board=_minimal_board(),
        allies=allies,
        leader=allies[-1] if allies else None,
        grudges=grudges,
        max_chars=max_chars,
    )
    assert len(result) <= max_chars
