"""Source-blind example tests for src.agents.azure_openai — Issue #56.

Derived exclusively from acceptance criteria; no implementation source was read.
All HTTP calls and env-loading are mocked. No real .env is read.

Criteria covered:
- call_azure_for_action_index() POSTs to /chat/completions?api-version=...
  with the api-key header and response_format json_schema strict:true
- Returns an index in [0, len(legal_actions)) or None on empty/unparseable/out-of-range reply
- Per-call temperature and top_p injected verbatim; top_p omitted when None
- Raises AzureConfigError when AZURE_OPENAI_BASE_URL or AZURE_OPENAI_API_KEY is unresolved
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

_FAKE_BASE_URL = "https://fake.openai.azure.com/openai/deployments/gpt-4.1"
_FAKE_API_KEY = "test-key-never-commit-this"
_FAKE_API_VERSION = "2024-02-15-preview"

_FAKE_ENV = {
    "AZURE_OPENAI_BASE_URL": _FAKE_BASE_URL,
    "AZURE_OPENAI_API_KEY": _FAKE_API_KEY,
    "AZURE_OPENAI_API_VERSION": _FAKE_API_VERSION,
}

_FAKE_OBS: dict = {}  # minimal placeholder; real obs built by env internally


@pytest.fixture(autouse=True)
def _no_dotenv():
    """Prevent ensure_env_loaded and render_action_prompt from touching real I/O.

    render_action_prompt requires a fully-formed environment observation; mocking
    it keeps these tests focused purely on the HTTP-client contract.
    """
    with (
        patch("src.agents.azure_openai.ensure_env_loaded", return_value=None),
        patch("src.agents.azure_openai.render_action_prompt", return_value="fake prompt"),
    ):
        yield


def _fake_response(content: str | None) -> MagicMock:
    """Build a fake httpx.Response returning *content* as the message body."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"message": {"content": content or ""}}],
    }
    return resp


def _resp(action_index: int | None = 0) -> MagicMock:
    """Convenience: fake response with a well-formed action_index payload."""
    if action_index is None:
        return _fake_response("")
    return _fake_response(json.dumps({"action_index": action_index}))


_DEFAULT_MODEL = "gpt-4.1"


def _call(
    legal_actions: list,
    *,
    temperature: float = 0.5,
    top_p: float | None = 0.9,
    resp: MagicMock | None = None,
    model: str = _DEFAULT_MODEL,
):
    """Invoke call_azure_for_action_index() with all I/O mocked."""
    from src.agents.azure_openai import call_azure_for_action_index

    if resp is None:
        resp = _resp(0)

    with (
        patch.dict("os.environ", _FAKE_ENV, clear=False),
        patch("src.agents.azure_openai.httpx.post", return_value=resp) as mock_post,
    ):
        result = call_azure_for_action_index(
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


def test_when_called_then_posts_to_url_containing_chat_completions():
    """call_azure_for_action_index() must POST to a URL containing /chat/completions."""
    _, mock_post = _call(["a", "b"])

    assert mock_post.called
    url = mock_post.call_args.args[0]
    assert "/chat/completions" in url


def test_when_called_then_url_contains_api_version_query_param():
    """The POST URL must include an api-version= query parameter."""
    _, mock_post = _call(["a", "b"])

    url = mock_post.call_args.args[0]
    assert "api-version=" in url


def test_when_called_then_api_key_header_equals_env_var():
    """The api-key request header must equal AZURE_OPENAI_API_KEY from env."""
    _, mock_post = _call(["a", "b"])

    headers = mock_post.call_args.kwargs.get("headers", {})
    assert headers.get("api-key") == _FAKE_API_KEY


def test_when_called_then_response_format_type_is_json_schema():
    """Request body must set response_format.type to 'json_schema'."""
    _, mock_post = _call(["a", "b"])

    body = mock_post.call_args.kwargs.get("json", {})
    assert body.get("response_format", {}).get("type") == "json_schema"


def test_when_called_then_json_schema_strict_is_true():
    """Request body must set response_format.json_schema.strict to True."""
    _, mock_post = _call(["a", "b"])

    body = mock_post.call_args.kwargs.get("json", {})
    schema_obj = body.get("response_format", {}).get("json_schema", {})
    assert schema_obj.get("strict") is True


# ──────────────────────────────────────────────────────────────────────────────
# Return-value contract: valid index, out-of-range, bad content → None
# ──────────────────────────────────────────────────────────────────────────────


def test_when_model_returns_index_0_then_zero_is_returned():
    """action_index=0 on a non-empty list must return the integer 0."""
    result, _ = _call(["only_action"], resp=_resp(0))
    assert result == 0


def test_when_model_returns_valid_mid_index_then_that_integer_is_returned():
    """A valid action_index within bounds must be returned unchanged."""
    result, _ = _call(["a", "b", "c"], resp=_resp(2))
    assert result == 2


def test_when_model_returns_last_valid_index_then_it_is_returned():
    """action_index == len-1 must succeed (boundary check)."""
    legal = ["x", "y", "z"]
    result, _ = _call(legal, resp=_resp(len(legal) - 1))
    assert result == len(legal) - 1


def test_when_model_returns_index_equal_to_len_then_none_is_returned():
    """action_index == len(legal_actions) is out-of-range → must return None."""
    legal = ["a", "b"]
    result, _ = _call(legal, resp=_resp(len(legal)))
    assert result is None


def test_when_model_returns_index_far_out_of_range_then_none_is_returned():
    """action_index far beyond len must silently return None, not raise."""
    result, _ = _call(["a", "b"], resp=_resp(999))
    assert result is None


def test_when_model_returns_negative_index_then_none_is_returned():
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
    """Temperature must be injected verbatim into the POST body."""
    _, mock_post = _call(["a"], temperature=0.3, resp=_resp(0))

    body = mock_post.call_args.kwargs.get("json", {})
    assert body.get("temperature") == pytest.approx(0.3)


def test_when_top_p_provided_then_it_appears_in_request_body():
    """top_p must be injected verbatim into the POST body."""
    _, mock_post = _call(["a"], top_p=0.85, resp=_resp(0))

    body = mock_post.call_args.kwargs.get("json", {})
    assert body.get("top_p") == pytest.approx(0.85)


def test_when_top_p_is_none_then_top_p_key_is_absent_from_body():
    """When top_p=None, the 'top_p' key must not appear in the POST body."""
    _, mock_post = _call(["a"], top_p=None, resp=_resp(0))

    body = mock_post.call_args.kwargs.get("json", {})
    assert "top_p" not in body


# ──────────────────────────────────────────────────────────────────────────────
# Credential error propagation
# ──────────────────────────────────────────────────────────────────────────────


def test_when_base_url_env_var_missing_then_azure_config_error_is_raised():
    """Missing AZURE_OPENAI_BASE_URL must raise AzureConfigError before any HTTP call."""
    from src.agents.azure_openai import AzureConfigError, call_azure_for_action_index

    env_no_base = {k: v for k, v in _FAKE_ENV.items() if k != "AZURE_OPENAI_BASE_URL"}
    with (
        patch.dict("os.environ", env_no_base, clear=True),
        pytest.raises(AzureConfigError),
    ):
        call_azure_for_action_index(
            _FAKE_OBS, ["a"], model=_DEFAULT_MODEL, temperature=0.5, top_p=0.9
        )


def test_when_api_key_env_var_missing_then_azure_config_error_is_raised():
    """Missing AZURE_OPENAI_API_KEY must raise AzureConfigError before any HTTP call."""
    from src.agents.azure_openai import AzureConfigError, call_azure_for_action_index

    env_no_key = {k: v for k, v in _FAKE_ENV.items() if k != "AZURE_OPENAI_API_KEY"}
    with (
        patch.dict("os.environ", env_no_key, clear=True),
        pytest.raises(AzureConfigError),
    ):
        call_azure_for_action_index(
            _FAKE_OBS, ["a"], model=_DEFAULT_MODEL, temperature=0.5, top_p=0.9
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

    from src.agents.azure_openai import call_azure_for_action_index

    legal = [f"action_{i}" for i in range(n)]
    with (
        patch.dict("os.environ", _FAKE_ENV, clear=False),
        patch("src.agents.azure_openai.httpx.post", return_value=_resp(idx)),
    ):
        result = call_azure_for_action_index(
            _FAKE_OBS, legal, model=_DEFAULT_MODEL, temperature=0.5, top_p=0.9
        )
    assert result == idx


@given(temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=40)
def test_when_any_valid_temperature_given_then_it_appears_verbatim_in_body(temperature):
    """Temperature is injected verbatim for all floats in [0.0, 2.0].

    Invariant: no rounding, clamping, or transformation is applied.
    """
    from src.agents.azure_openai import call_azure_for_action_index

    with (
        patch.dict("os.environ", _FAKE_ENV, clear=False),
        patch("src.agents.azure_openai.httpx.post", return_value=_resp(0)) as mock_post,
    ):
        call_azure_for_action_index(
            _FAKE_OBS, ["a"], model=_DEFAULT_MODEL, temperature=temperature, top_p=None
        )

    body = mock_post.call_args.kwargs.get("json", {})
    assert body.get("temperature") == pytest.approx(temperature, rel=1e-6, abs=1e-9)
