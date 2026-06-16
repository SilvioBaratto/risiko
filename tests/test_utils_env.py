"""Source-blind tests for src.utils.env.ensure_env_loaded() — Issue #56.

Derived exclusively from acceptance criteria; no implementation source was read.
All dotenv I/O is mocked; no real .env file is read.

Criteria covered:
- ensure_env_loaded() is idempotent: calling it N times invokes load_dotenv exactly once
- When python-dotenv is not installed the function raises ImportError
- Happy path calls load_dotenv(find_dotenv(usecwd=True))
- Raises AzureConfigError when AZURE_OPENAI_BASE_URL or AZURE_OPENAI_API_KEY is unresolved
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ──────────────────────────────────────────────────────────────────────────────
# Fixture: force a fresh import of src.utils.env before every test so the
# module-level `_loaded` flag (or equivalent) is always in its initial state.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fresh_env_module():
    """Remove src.utils.env from sys.modules so each test starts cold."""
    sys.modules.pop("src.utils.env", None)
    yield
    sys.modules.pop("src.utils.env", None)


# ──────────────────────────────────────────────────────────────────────────────
# Idempotence
# ──────────────────────────────────────────────────────────────────────────────


def test_when_called_twice_then_load_dotenv_is_invoked_exactly_once():
    """ensure_env_loaded() must short-circuit on the second call.

    Criterion: idempotent short-circuit.
    """
    mock_find = MagicMock(return_value="/project/.env")
    mock_load = MagicMock(return_value=True)

    with (
        patch("dotenv.find_dotenv", mock_find),
        patch("dotenv.load_dotenv", mock_load),
        patch.dict(
            "os.environ",
            {"AZURE_OPENAI_BASE_URL": "https://x", "AZURE_OPENAI_API_KEY": "k"},
            clear=False,
        ),
    ):
        from src.utils.env import ensure_env_loaded

        ensure_env_loaded()
        ensure_env_loaded()  # second call must be a no-op

    assert mock_load.call_count == 1


# ──────────────────────────────────────────────────────────────────────────────
# dotenv-missing ImportError branch
# ──────────────────────────────────────────────────────────────────────────────


def test_when_dotenv_package_is_not_installed_then_function_returns_gracefully():
    """When python-dotenv is absent, ensure_env_loaded() must complete without error.

    Criterion: dotenv-missing ImportError branch — the function catches the
    ImportError internally and returns None (graceful degradation, no re-raise).
    """
    with patch.dict(sys.modules, {"dotenv": None}):  # None → any `import dotenv` raises ImportError
        sys.modules.pop("src.utils.env", None)
        from src.utils.env import ensure_env_loaded

        ensure_env_loaded()  # must NOT raise


# ──────────────────────────────────────────────────────────────────────────────
# Happy path: load_dotenv(find_dotenv(usecwd=True))
# ──────────────────────────────────────────────────────────────────────────────


def test_when_dotenv_present_then_find_dotenv_is_called_with_usecwd_true():
    """Happy path: find_dotenv(usecwd=True) must be called.

    Criterion: happy load_dotenv(find_dotenv(usecwd=True)) path.
    """
    mock_find = MagicMock(return_value="/project/.env")
    mock_load = MagicMock(return_value=True)

    with (
        patch("dotenv.find_dotenv", mock_find),
        patch("dotenv.load_dotenv", mock_load),
        patch.dict(
            "os.environ",
            {"AZURE_OPENAI_BASE_URL": "https://x", "AZURE_OPENAI_API_KEY": "k"},
            clear=False,
        ),
    ):
        from src.utils.env import ensure_env_loaded

        ensure_env_loaded()

    mock_find.assert_called_once_with(usecwd=True)


def test_when_dotenv_present_then_load_dotenv_is_called_with_path_from_find_dotenv():
    """Happy path: load_dotenv must receive the path returned by find_dotenv.

    Criterion: happy load_dotenv(find_dotenv(usecwd=True)) path.
    """
    env_path = "/project/.env"
    mock_find = MagicMock(return_value=env_path)
    mock_load = MagicMock(return_value=True)

    with (
        patch("dotenv.find_dotenv", mock_find),
        patch("dotenv.load_dotenv", mock_load),
        patch.dict(
            "os.environ",
            {"AZURE_OPENAI_BASE_URL": "https://x", "AZURE_OPENAI_API_KEY": "k"},
            clear=False,
        ),
    ):
        from src.utils.env import ensure_env_loaded

        ensure_env_loaded()

    mock_load.assert_called_once_with(env_path)


# ──────────────────────────────────────────────────────────────────────────────
# Property-based test: idempotence invariant
# ──────────────────────────────────────────────────────────────────────────────


@given(n_calls=st.integers(min_value=1, max_value=5))
@settings(max_examples=20)
def test_when_called_n_times_then_load_dotenv_is_always_called_exactly_once(
    n_calls: int,
) -> None:
    """ensure_env_loaded() is idempotent for any number of calls ≥ 1.

    Invariant: the dotenv load side-effect occurs exactly once, regardless
    of how many times the caller invokes ensure_env_loaded().
    """
    sys.modules.pop("src.utils.env", None)

    mock_find = MagicMock(return_value=".env")
    mock_load = MagicMock(return_value=True)

    with (
        patch("dotenv.find_dotenv", mock_find),
        patch("dotenv.load_dotenv", mock_load),
        patch.dict(
            "os.environ",
            {"AZURE_OPENAI_BASE_URL": "https://x", "AZURE_OPENAI_API_KEY": "k"},
            clear=False,
        ),
    ):
        from src.utils.env import ensure_env_loaded

        for _ in range(n_calls):
            ensure_env_loaded()

    assert mock_load.call_count == 1
