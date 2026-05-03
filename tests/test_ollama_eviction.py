"""Tests for Ollama model eviction and call serialisation lock."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from src.agents.ollama_eviction import OLLAMA_LOCK, _post_evict, evict


class TestOllamaLock:
    """OLLAMA_LOCK is a module-level threading.Lock."""

    def test_when_imported_lock_is_threading_lock(self) -> None:
        assert isinstance(OLLAMA_LOCK, type(threading.Lock()))


class TestPostEvict:
    """_post_evict sends the correct HTTP request synchronously."""

    def test_when_called_posts_to_api_generate_not_v1(self) -> None:
        with patch("src.agents.ollama_eviction.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            _post_evict("qwen3.5:4b", "http://localhost:11434")
        req = mock_open.call_args[0][0]
        assert req.full_url.endswith("/api/generate")
        assert "/v1" not in req.full_url

    def test_when_called_body_has_model_and_keep_alive_zero(self) -> None:
        with patch("src.agents.ollama_eviction.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            _post_evict("qwen3.5:4b", "http://localhost:11434")
        req = mock_open.call_args[0][0]
        body = json.loads(req.data)
        assert body["model"] == "qwen3.5:4b"
        assert body["keep_alive"] == 0

    def test_when_urlopen_raises_urlerror_exception_is_swallowed(self) -> None:
        with patch(
            "src.agents.ollama_eviction.urllib.request.urlopen",
            side_effect=URLError("connection refused"),
        ):
            _post_evict("model", "http://localhost:11434")  # must not raise

    def test_when_urlopen_raises_generic_exception_is_swallowed(self) -> None:
        with patch(
            "src.agents.ollama_eviction.urllib.request.urlopen",
            side_effect=OSError("network error"),
        ):
            _post_evict("model", "http://localhost:11434")  # must not raise


class TestEvict:
    """evict() is non-blocking (daemon thread)."""

    def test_when_evict_called_it_does_not_block_caller(self) -> None:
        start = time.monotonic()
        with patch(
            "src.agents.ollama_eviction._post_evict",
            side_effect=lambda m, u: time.sleep(0.5),
        ):
            evict("model")
        elapsed = time.monotonic() - start
        assert elapsed < 0.1
