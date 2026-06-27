"""Tests for HTTP 429 rate/usage-limit wait-out in the Ollama client.

Ollama cloud throttles on per-minute, 5-hour session, and 7-day weekly quotas
with HTTP 429. The client must wait the limit out (backoff + retry) rather than
failing or playing a random move. time.sleep is patched so tests don't block.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.agents.ollama_client import (
    _post_waiting_out_rate_limits,
    _retry_after_seconds,
)


def _resp(status: int = 200, headers: dict | None = None, body: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.json.return_value = body or {}

    def _raise() -> None:
        if status >= 400:
            raise httpx.HTTPStatusError("err", request=MagicMock(), response=r)

    r.raise_for_status.side_effect = _raise
    return r


# ── _retry_after_seconds ─────────────────────────────────────────────────────


def test_when_retry_after_is_seconds_then_parsed():
    assert _retry_after_seconds(_resp(429, headers={"retry-after": "12"})) == 12.0


def test_when_retry_after_absent_then_none():
    assert _retry_after_seconds(_resp(429)) is None


# ── _post_waiting_out_rate_limits ────────────────────────────────────────────


def test_when_first_response_ok_then_returns_without_sleeping():
    ok = _resp(200, body={"x": 1})
    with (
        patch("src.agents.ollama_client.httpx.post", return_value=ok) as post,
        patch("src.agents.ollama_client.time.sleep") as sleep,
    ):
        out = _post_waiting_out_rate_limits(
            "u", headers={}, json_body={}, timeout=1.0, rate_limit_max_wait=100.0
        )
    assert out is ok
    assert post.call_count == 1
    sleep.assert_not_called()


def test_when_429_then_200_then_waits_and_retries():
    seq = [_resp(429), _resp(429), _resp(200)]
    with (
        patch("src.agents.ollama_client.httpx.post", side_effect=seq) as post,
        patch("src.agents.ollama_client.time.sleep") as sleep,
    ):
        out = _post_waiting_out_rate_limits(
            "u", headers={}, json_body={}, timeout=1.0, rate_limit_max_wait=1000.0
        )
    assert out is seq[-1]
    assert post.call_count == 3
    assert sleep.call_count == 2  # slept before each retry, not after success


def test_when_retry_after_header_present_then_that_delay_is_used():
    seq = [_resp(429, headers={"retry-after": "8"}), _resp(200)]
    with (
        patch("src.agents.ollama_client.httpx.post", side_effect=seq),
        patch("src.agents.ollama_client.time.sleep") as sleep,
    ):
        _post_waiting_out_rate_limits(
            "u", headers={}, json_body={}, timeout=1.0, rate_limit_max_wait=1000.0
        )
    sleep.assert_called_once_with(8.0)


def test_when_max_wait_is_zero_then_429_raises_immediately():
    with (
        patch("src.agents.ollama_client.httpx.post", return_value=_resp(429)),
        patch("src.agents.ollama_client.time.sleep") as sleep,
        pytest.raises(httpx.HTTPStatusError),
    ):
        _post_waiting_out_rate_limits(
            "u", headers={}, json_body={}, timeout=1.0, rate_limit_max_wait=0.0
        )
    sleep.assert_not_called()


def test_when_non_429_error_then_propagates_without_retry():
    with (
        patch("src.agents.ollama_client.httpx.post", return_value=_resp(500)) as post,
        patch("src.agents.ollama_client.time.sleep") as sleep,
        pytest.raises(httpx.HTTPStatusError),
    ):
        _post_waiting_out_rate_limits(
            "u", headers={}, json_body={}, timeout=1.0, rate_limit_max_wait=1000.0
        )
    assert post.call_count == 1
    sleep.assert_not_called()
