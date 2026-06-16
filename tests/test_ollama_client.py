"""Source-blind example tests for src.agents.ollama_client — migrated from Azure contract.

Derived exclusively from acceptance criteria; no implementation source was read.
All HTTP calls and env-loading are mocked. No real .env is read.

Criteria covered:
- call_ollama_for_action_index() POSTs to the native {root}/api/chat endpoint
- Auth uses Authorization: Bearer <key> header (NOT api-key)
- Request body sets the native `format` to the action_index JSON schema, with
  thinking enabled (think=true) so the schema mask is honoured (ollama#15260)
- Returns an index in [0, len(legal_actions)) or None on empty/unparseable/out-of-range reply
- Per-call temperature and top_p injected verbatim; top_p omitted when None
- When OLLAMA_BASE_URL / OLLAMA_API_KEY are unset, falls back to defaults (localhost + "ollama")
  — AzureConfigError no longer exists; missing creds are never an error
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ──────────────────────────────────────────────────────────────────────────────
# Shared test doubles
# ──────────────────────────────────────────────────────────────────────────────

_FAKE_BASE_URL = "http://fake-ollama.local/v1"
_FAKE_API_KEY = "test-key-never-commit-this"

_FAKE_ENV = {
    "OLLAMA_BASE_URL": _FAKE_BASE_URL,
    "OLLAMA_API_KEY": _FAKE_API_KEY,
}

_FAKE_OBS: dict = {}  # minimal placeholder; real obs built by env internally

_DEFAULT_MODEL = "gpt-oss:120b"


@pytest.fixture(autouse=True)
def _no_dotenv():
    """Prevent ensure_env_loaded and render_action_prompt from touching real I/O.

    render_action_prompt requires a fully-formed environment observation; mocking
    it keeps these tests focused purely on the HTTP-client contract.
    """
    with (
        patch("src.agents.ollama_client.ensure_env_loaded", return_value=None),
        patch("src.agents.ollama_client.render_action_prompt", return_value="fake prompt"),
    ):
        yield


def _fake_response(content: str | None) -> MagicMock:
    """Build a fake httpx.Response returning *content* as the message body."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "message": {"content": content or ""},
    }
    return resp


def _resp(action_index: int | None = 0) -> MagicMock:
    """Convenience: fake response with a well-formed action_index payload."""
    if action_index is None:
        return _fake_response("")
    return _fake_response(json.dumps({"action_index": action_index}))


def _call(
    legal_actions: list,
    *,
    temperature: float = 0.5,
    top_p: float | None = 0.9,
    resp: MagicMock | None = None,
    model: str = _DEFAULT_MODEL,
):
    """Invoke call_ollama_for_action_index() with all I/O mocked."""
    from src.agents.ollama_client import call_ollama_for_action_index

    if resp is None:
        resp = _resp(0)

    with (
        patch.dict("os.environ", _FAKE_ENV, clear=False),
        patch("src.agents.ollama_client.httpx.post", return_value=resp) as mock_post,
    ):
        result = call_ollama_for_action_index(
            _FAKE_OBS,
            legal_actions,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )
    return result, mock_post


# ──────────────────────────────────────────────────────────────────────────────
# HTTP-contract: URL, header, response_format
# ──────────────────────────────────────────────────────────────────────────────


def test_when_ollama_called_then_posts_to_native_api_chat_endpoint():
    """call_ollama_for_action_index() must POST to the native /api/chat endpoint.

    The OpenAI-compat /v1 endpoint only honours `format` as loose JSON mode and
    ignores the schema for reasoning models (ollama#10937); the native endpoint
    enforces it via XGrammar.
    """
    _, mock_post = _call(["a", "b"])

    assert mock_post.called
    url = mock_post.call_args.args[0]
    assert url.endswith("/api/chat")
    assert "/v1/" not in url


def test_when_ollama_called_then_url_does_not_contain_api_version_query_param():
    """The POST URL must NOT include an api-version= query parameter (Ollama has none)."""
    _, mock_post = _call(["a", "b"])

    url = mock_post.call_args.args[0]
    assert "api-version=" not in url


def test_when_ollama_called_then_authorization_bearer_header_equals_env_var():
    """The Authorization: Bearer header must equal OLLAMA_API_KEY from env."""
    _, mock_post = _call(["a", "b"])

    headers = mock_post.call_args.kwargs.get("headers", {})
    assert headers.get("Authorization") == f"Bearer {_FAKE_API_KEY}"


def test_when_ollama_called_then_no_api_key_header_is_set():
    """The old api-key header must NOT be present; Ollama uses Authorization: Bearer."""
    _, mock_post = _call(["a", "b"])

    headers = mock_post.call_args.kwargs.get("headers", {})
    assert "api-key" not in headers


def test_when_ollama_called_then_format_is_action_index_schema():
    """Request body must set the native `format` to the action_index JSON schema."""
    _, mock_post = _call(["a", "b"])

    body = mock_post.call_args.kwargs.get("json", {})
    fmt = body.get("format", {})
    assert fmt.get("type") == "object"
    assert fmt.get("properties", {}).get("action_index", {}).get("type") == "integer"
    assert "action_index" in fmt.get("required", [])


def test_when_ollama_called_then_thinking_is_enabled():
    """Thinking must stay ON: think=false breaks `format` enforcement (ollama#15260)."""
    _, mock_post = _call(["a", "b"])

    body = mock_post.call_args.kwargs.get("json", {})
    assert body.get("think") is True


# ──────────────────────────────────────────────────────────────────────────────
# Return-value contract: valid index, out-of-range, bad content → None
# ──────────────────────────────────────────────────────────────────────────────


def test_when_ollama_returns_index_0_then_zero_is_returned():
    """action_index=0 on a non-empty list must return the integer 0."""
    result, _ = _call(["only_action"], resp=_resp(0))
    assert result == 0


def test_when_ollama_returns_valid_mid_index_then_that_integer_is_returned():
    """A valid action_index within bounds must be returned unchanged."""
    result, _ = _call(["a", "b", "c"], resp=_resp(2))
    assert result == 2


def test_when_ollama_returns_last_valid_index_then_it_is_returned():
    """action_index == len-1 must succeed (boundary check)."""
    legal = ["x", "y", "z"]
    result, _ = _call(legal, resp=_resp(len(legal) - 1))
    assert result == len(legal) - 1


def test_when_ollama_returns_index_equal_to_len_then_none_is_returned():
    """action_index == len(legal_actions) is out-of-range → must return None."""
    legal = ["a", "b"]
    result, _ = _call(legal, resp=_resp(len(legal)))
    assert result is None


def test_when_ollama_returns_index_far_out_of_range_then_none_is_returned():
    """action_index far beyond len must silently return None, not raise."""
    result, _ = _call(["a", "b"], resp=_resp(999))
    assert result is None


def test_when_ollama_returns_negative_index_then_none_is_returned():
    """Negative action_index must return None (out-of-range by spec)."""
    result, _ = _call(["a", "b", "c"], resp=_resp(-1))
    assert result is None


def test_when_response_content_is_empty_string_then_none_is_returned():
    """Empty response content must silently return None, not raise."""
    result, _ = _call(["a"], resp=_resp(None))
    assert result is None


def test_when_response_content_is_invalid_json_then_none_is_returned():
    """Non-JSON response content must return None, not propagate a parse error."""
    result, _ = _call(["a", "b"], resp=_fake_response("not-valid-json!!!"))
    assert result is None


def test_when_response_content_lacks_action_index_field_then_none_is_returned():
    """JSON without action_index key must return None."""
    result, _ = _call(["a"], resp=_fake_response(json.dumps({"wrong_key": 0})))
    assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# temperature / top_p injection
# ──────────────────────────────────────────────────────────────────────────────


def test_when_temperature_provided_then_it_appears_in_request_body():
    """Temperature must be injected verbatim into the native `options` block."""
    _, mock_post = _call(["a"], temperature=0.3, resp=_resp(0))

    body = mock_post.call_args.kwargs.get("json", {})
    assert body.get("options", {}).get("temperature") == pytest.approx(0.3)


def test_when_top_p_provided_then_it_appears_in_request_body():
    """top_p must be injected verbatim into the native `options` block."""
    _, mock_post = _call(["a"], top_p=0.85, resp=_resp(0))

    body = mock_post.call_args.kwargs.get("json", {})
    assert body.get("options", {}).get("top_p") == pytest.approx(0.85)


def test_when_top_p_is_none_then_top_p_key_is_absent_from_body():
    """When top_p=None, the 'top_p' key must not appear in the `options` block."""
    _, mock_post = _call(["a"], top_p=None, resp=_resp(0))

    body = mock_post.call_args.kwargs.get("json", {})
    assert "top_p" not in body.get("options", {})


# ──────────────────────────────────────────────────────────────────────────────
# Default-fallback behavior when credentials are unset
# (replaces the old AzureConfigError tests — Ollama never raises on missing creds)
# ──────────────────────────────────────────────────────────────────────────────


def test_when_base_url_env_var_missing_then_defaults_to_localhost():
    """Missing OLLAMA_BASE_URL must fall back to http://localhost:11434/v1, not raise."""
    from src.agents.ollama_client import call_ollama_for_action_index

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("src.agents.ollama_client.httpx.post", return_value=_resp(0)) as mock_post,
    ):
        call_ollama_for_action_index(
            _FAKE_OBS, ["a"], model=_DEFAULT_MODEL, temperature=0.5, top_p=0.9
        )

    url = mock_post.call_args.args[0]
    assert url == "http://localhost:11434/api/chat", (
        f"Expected localhost native /api/chat fallback URL, got: {url!r}"
    )


def test_when_api_key_env_var_missing_then_defaults_to_ollama_bearer():
    """Missing OLLAMA_API_KEY must fall back to Authorization: Bearer ollama, not raise."""
    from src.agents.ollama_client import call_ollama_for_action_index

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("src.agents.ollama_client.httpx.post", return_value=_resp(0)) as mock_post,
    ):
        call_ollama_for_action_index(
            _FAKE_OBS, ["a"], model=_DEFAULT_MODEL, temperature=0.5, top_p=0.9
        )

    headers = mock_post.call_args.kwargs.get("headers", {})
    assert headers.get("Authorization") == "Bearer ollama", (
        f"Expected 'Bearer ollama' fallback, got: {headers.get('Authorization')!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Property-based tests (invariants from the criteria)
# ──────────────────────────────────────────────────────────────────────────────


@given(
    n=st.integers(min_value=1, max_value=20),
    idx=st.integers(min_value=0, max_value=19),
)
@settings(max_examples=60)
def test_when_index_in_range_then_that_exact_integer_is_always_returned(n, idx):
    """For any action_index in [0, n), the call returns that exact integer.

    Invariant: the function is a transparent pass-through for valid indices;
    no re-mapping or offset is applied.
    """
    if idx >= n:
        return  # out-of-range covered by example tests above

    from src.agents.ollama_client import call_ollama_for_action_index

    legal = [f"action_{i}" for i in range(n)]
    with (
        patch.dict("os.environ", _FAKE_ENV, clear=False),
        patch("src.agents.ollama_client.httpx.post", return_value=_resp(idx)),
    ):
        result = call_ollama_for_action_index(
            _FAKE_OBS, legal, model=_DEFAULT_MODEL, temperature=0.5, top_p=0.9
        )
    assert result == idx


@given(temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=40)
def test_when_any_valid_temperature_given_then_it_appears_verbatim_in_body(temperature):
    """Temperature is injected verbatim for all floats in [0.0, 2.0].

    Invariant: no rounding, clamping, or transformation is applied.
    """
    from src.agents.ollama_client import call_ollama_for_action_index

    with (
        patch.dict("os.environ", _FAKE_ENV, clear=False),
        patch("src.agents.ollama_client.httpx.post", return_value=_resp(0)) as mock_post,
    ):
        call_ollama_for_action_index(
            _FAKE_OBS, ["a"], model=_DEFAULT_MODEL, temperature=temperature, top_p=None
        )

    body = mock_post.call_args.kwargs.get("json", {})
    assert body.get("options", {}).get("temperature") == pytest.approx(
        temperature, rel=1e-6, abs=1e-9
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tolerant parsing: not every model honours response_format json_schema.
# Reasoning/instruct models may wrap the answer in a ```json fence or add prose;
# _extract_action_index must still recover a usable action_index.
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractActionIndex:
    """Direct unit tests for the tolerant action_index extractor."""

    def test_when_strict_json_then_index_is_extracted(self):
        from src.agents.ollama_client import _extract_action_index

        assert _extract_action_index('{"action_index": 7}') == 7

    def test_when_bare_integer_then_index_is_extracted(self):
        """Cloud models flatten the single-property schema to a bare int."""
        from src.agents.ollama_client import _extract_action_index

        assert _extract_action_index("0") == 0
        assert _extract_action_index("  42 ") == 42

    def test_when_bare_boolean_then_returns_none(self):
        """`true`/`false` are valid JSON ints-ish but must NOT parse as an index."""
        from src.agents.ollama_client import _extract_action_index

        assert _extract_action_index("true") is None

    def test_when_json_fenced_with_lang_tag_then_index_is_extracted(self):
        from src.agents.ollama_client import _extract_action_index

        assert _extract_action_index('```json\n{"action_index": 51}\n```') == 51

    def test_when_json_fenced_without_lang_tag_then_index_is_extracted(self):
        from src.agents.ollama_client import _extract_action_index

        assert _extract_action_index('```\n{"action_index": 3}\n```') == 3

    def test_when_prose_surrounds_json_then_regex_fallback_extracts_index(self):
        from src.agents.ollama_client import _extract_action_index

        assert _extract_action_index("I choose action_index: 12 because it is best.") == 12

    def test_when_extra_keys_present_then_index_is_still_extracted(self):
        from src.agents.ollama_client import _extract_action_index

        assert _extract_action_index('{"reason": "expand", "action_index": 4}') == 4

    def test_when_no_index_anywhere_then_returns_none(self):
        from src.agents.ollama_client import _extract_action_index

        assert _extract_action_index("sorry, I cannot decide") is None

    def test_when_empty_string_then_returns_none(self):
        from src.agents.ollama_client import _extract_action_index

        assert _extract_action_index("") is None


class TestTolerantParsingThroughClient:
    """The client recovers fenced/prose replies into a legal in-range index."""

    def test_when_reply_is_fenced_json_then_client_returns_in_range_index(self):
        legal = ["a", "b", "c"]
        result, _ = _call(legal, resp=_fake_response('```json\n{"action_index": 2}\n```'))
        assert result == 2

    def test_when_fenced_index_out_of_range_then_client_returns_none(self):
        legal = ["a", "b"]
        result, _ = _call(legal, resp=_fake_response('```json\n{"action_index": 9}\n```'))
        assert result is None

    def test_when_reply_has_prose_then_client_recovers_index(self):
        legal = ["a", "b", "c", "d"]
        result, _ = _call(legal, resp=_fake_response("Best move is action_index = 3."))
        assert result == 3


def test_default_max_tokens_leaves_reasoning_headroom():
    """Default token budget must be large enough for reasoning models to answer."""
    from src.agents.ollama_client import DEFAULT_MAX_TOKENS

    assert DEFAULT_MAX_TOKENS >= 2048
