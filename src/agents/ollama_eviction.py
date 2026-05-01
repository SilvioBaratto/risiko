"""Ollama model eviction and process-wide BAML call serialisation lock."""

from __future__ import annotations

import json
import threading
import urllib.request

OLLAMA_LOCK: threading.Lock = threading.Lock()


def evict(model: str, base_url: str = "http://localhost:11434") -> None:
    """Fire-and-forget POST to evict *model* from Ollama GPU RAM.

    Args:
        model: The Ollama model name to evict (e.g. ``"qwen3.5:4b"``).
        base_url: Native Ollama base URL (no ``/v1`` suffix).
    """
    thread = threading.Thread(target=_post_evict, args=(model, base_url), daemon=True)
    thread.start()


def _post_evict(model: str, base_url: str) -> None:
    """Send keep_alive=0 to Ollama synchronously; silently swallow all errors.

    Args:
        model: The Ollama model name.
        base_url: Native Ollama base URL.
    """
    try:
        url = f"{base_url}/api/generate"
        body = json.dumps({"model": model, "keep_alive": 0}).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=2):
            pass
    except Exception:
        pass
