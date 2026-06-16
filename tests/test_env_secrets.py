"""Source-blind tests for .env.example, git-ignore, and render_action_prompt() — Issue #56.

Criteria covered:
- .env.example keys exactly match OLLAMA_BASE_URL / OLLAMA_API_KEY
- git check-ignore .env confirms .env is ignored
- render_action_prompt() returns a non-empty string, embeds legal action indices,
  prepends strategy_hint before the actions list, and never leaks the API key.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ──────────────────────────────────────────────────────────────────────────────
# .env.example — key presence
# ──────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent  # tests/ → repo root
_ENV_EXAMPLE = _REPO_ROOT / ".env.example"

_REQUIRED_KEYS = {
    "OLLAMA_BASE_URL",
    "OLLAMA_API_KEY",
}


@pytest.fixture(scope="module")
def env_example_lines() -> list[str]:
    """Read .env.example lines, skipping blanks and comments."""
    assert _ENV_EXAMPLE.exists(), ".env.example must be committed at the project root"
    return [
        line.strip()
        for line in _ENV_EXAMPLE.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


@pytest.fixture(scope="module")
def env_example_keys(env_example_lines) -> set[str]:
    """Extract variable names from non-comment .env.example lines."""
    return {line.split("=", 1)[0].strip() for line in env_example_lines if "=" in line}


def test_env_example_contains_ollama_base_url(env_example_keys):
    """Criterion: .env.example must declare OLLAMA_BASE_URL."""
    assert "OLLAMA_BASE_URL" in env_example_keys


def test_env_example_contains_ollama_api_key(env_example_keys):
    """Criterion: .env.example must declare OLLAMA_API_KEY."""
    assert "OLLAMA_API_KEY" in env_example_keys


def test_env_example_keys_exactly_match_required_set(env_example_keys):
    """Criterion: .env.example keys exactly match the two required env vars.

    No extra undeclared keys and no missing keys.
    """
    assert env_example_keys == _REQUIRED_KEYS


# ──────────────────────────────────────────────────────────────────────────────
# .env must be git-ignored
# ──────────────────────────────────────────────────────────────────────────────


def test_when_git_check_ignore_run_on_dot_env_then_it_is_ignored():
    """Criterion: git check-ignore .env confirms .env is git-ignored."""
    result = subprocess.run(
        ["git", "check-ignore", "-v", ".env"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"'git check-ignore .env' exited {result.returncode}; "
        f".env is NOT git-ignored. stdout={result.stdout!r} stderr={result.stderr!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# render_action_prompt() — single source of truth for the LLM prompt
# ──────────────────────────────────────────────────────────────────────────────


def _make_obs() -> dict:
    """Minimal obs dict matching the shape render_action_prompt() expects."""
    n_t = 42
    n_p = 6
    n_sym = 4
    return {
        "current_player": np.array(0, dtype=np.int32),
        "phase": np.array(4, dtype=np.int32),  # FORTIFY phase
        "territory_owner": np.zeros(n_t, dtype=np.int32),
        "armies": np.ones(n_t, dtype=np.int32),
        "reinforcements_remaining": np.array(0, dtype=np.int32),
        "trade_count": np.array(0, dtype=np.int32),
        "cards": np.zeros((5, n_sym), dtype=np.int32),
        "eliminated": np.zeros(n_p, dtype=np.int32),
    }


def _skip_action() -> dict:
    """Legal SKIP action — exercises no per-territory lookups."""
    return {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}


_FAKE_OBS = _make_obs()
_FAKE_LEGAL = [_skip_action(), _skip_action()]
_STRATEGY_HINT = "Focus on securing full continents before expanding."


def test_when_render_action_prompt_is_called_then_a_non_empty_string_is_returned():
    """Criterion: render_action_prompt() is the single source of truth for the LLM prompt."""
    from src.agents.action_prompt import render_action_prompt

    result = render_action_prompt(_FAKE_OBS, _FAKE_LEGAL)
    assert isinstance(result, str)
    assert len(result) > 0


def test_when_render_action_prompt_is_called_then_legal_actions_are_referenced():
    """Legal action indices must appear in the prompt output.

    Confirms the action list is embedded so the LLM can choose an action_index.
    Criterion: render_action_prompt() is the single LLM prompt source.
    """
    from src.agents.action_prompt import render_action_prompt

    legal = [_skip_action() for _ in range(3)]
    result = render_action_prompt(_FAKE_OBS, legal)
    assert "0" in result  # index 0 must be represented
    assert str(len(legal) - 1) in result  # last index must be represented


def test_when_strategy_hint_provided_then_it_appears_in_prompt_output():
    """Strategy_hint must be appended to the prompt as a directive.

    Criterion: strategy_hint is appended as a one-line directive before the
    legal actions list.
    """
    from src.agents.action_prompt import render_action_prompt

    result = render_action_prompt(_FAKE_OBS, _FAKE_LEGAL, strategy_hint=_STRATEGY_HINT)
    assert _STRATEGY_HINT in result


def test_when_strategy_hint_provided_then_it_appears_before_legal_actions_section():
    """Strategy_hint must precede the legal-actions enumeration in the prompt.

    Criterion: strategy_hint is a one-line directive BEFORE the legal actions list.
    """
    from src.agents.action_prompt import render_action_prompt

    result = render_action_prompt(_FAKE_OBS, _FAKE_LEGAL, strategy_hint=_STRATEGY_HINT)
    hint_pos = result.find(_STRATEGY_HINT)
    action_section_pos = result.find("0", hint_pos)
    assert hint_pos != -1, "strategy_hint not found in prompt"
    assert action_section_pos > hint_pos, (
        "strategy_hint must appear before the legal-actions enumeration"
    )


def test_when_render_action_prompt_is_called_then_ollama_api_key_literal_is_absent():
    """The prompt output must never contain a hardcoded Ollama API key value.

    Criterion: the API key is never hardcoded in source.
    """
    from src.agents.action_prompt import render_action_prompt

    known_key_patterns = [
        "OLLAMA_API_KEY=",
        "Authorization:",
    ]
    result = render_action_prompt(_FAKE_OBS, _FAKE_LEGAL)
    for pattern in known_key_patterns:
        assert pattern not in result, (
            f"render_action_prompt() output contains forbidden pattern {pattern!r}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Property-based test: render_action_prompt is total over its stated domain
# ──────────────────────────────────────────────────────────────────────────────


@given(n_actions=st.integers(min_value=1, max_value=20))
@settings(max_examples=20)
def test_when_any_non_empty_legal_actions_given_then_prompt_is_non_empty(
    n_actions: int,
) -> None:
    """Render_action_prompt() returns a non-empty string for any non-empty legal_actions.

    Invariant (never-raises-for-valid-input): total over its stated domain.
    """
    from src.agents.action_prompt import render_action_prompt

    legal = [_skip_action() for _ in range(n_actions)]
    result = render_action_prompt(_make_obs(), legal)
    assert isinstance(result, str)
    assert len(result) > 0
