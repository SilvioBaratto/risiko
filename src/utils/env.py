"""Lazy ``.env`` loading.

Optional settings (the Ollama base URL / cloud API key) live in a git-ignored
``.env`` at the project root. ``ensure_env_loaded()`` populates ``os.environ``
from it exactly once per process; it is safe to call from any entry point (CLI,
tests, library use) and is a no-op if python-dotenv is absent or no ``.env``
exists. Local Ollama needs no ``.env`` at all.
"""

from __future__ import annotations

__all__ = ["ensure_env_loaded"]

_loaded = False


def ensure_env_loaded() -> None:
    """Load the project ``.env`` into ``os.environ`` once (idempotent)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    load_dotenv(find_dotenv(usecwd=True))
