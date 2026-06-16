"""Example tests for verifiable global acceptance criteria — Issue #49.

Covers:
  - .env.example is committed with the two required Ollama variable names
  - LLM output is an index into legal_actions (action always legal)
  - render_action_prompt() is the single source of truth for the LLM prompt

Skipped criteria (not runtime-verifiable per oracle report):
  - LLM calls non-blocking / time-bounded (no concrete unit check inferable)
  - Environment step deterministic (no concrete unit check inferable)
  - All hyperparameters via YAML/CLI (no concrete unit check inferable)
  - Baseline win rates (empirical, not a unit assertion)
"""

import pathlib
import unittest.mock as mock

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# ── Helpers for building structured observations ──────────────────────────────

_N_TERRITORIES = 42
_N_PLAYERS = 2


def _minimal_obs() -> dict:
    """Return a minimal env-compatible observation with numpy arrays."""
    return {
        "current_player": np.int32(0),
        "phase": np.int32(2),  # ATTACK
        "territory_owner": np.zeros(_N_TERRITORIES, dtype=np.int32),
        "armies": np.ones(_N_TERRITORIES, dtype=np.int32),
        "reinforcements_remaining": np.int32(0),
        "trade_count": np.int32(0),
        "cards": np.zeros((5, 4), dtype=np.int32),
        "eliminated": np.zeros(_N_PLAYERS, dtype=np.int32),
        "continent_control": np.zeros(6, dtype=np.int32),
        "turn_capture": np.int32(0),
        "n_players": np.int32(_N_PLAYERS),
    }


def _skip_action() -> dict:
    """Return a minimal legal SKIP action dict."""
    return {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}


# ── .env.example committed with required credentials ─────────────────────────
# Criterion: Ollama credentials load from a git-ignored .env
#            (OLLAMA_BASE_URL / OLLAMA_API_KEY); .env.example is committed.


class TestEnvExample:
    """Criterion: .env.example committed with both Ollama credential variable names."""

    def test_when_env_example_checked_file_exists_at_project_root(self):
        assert (PROJECT_ROOT / ".env.example").is_file()

    def test_when_env_example_read_ollama_base_url_variable_is_present(self):
        content = (PROJECT_ROOT / ".env.example").read_text()
        assert "OLLAMA_BASE_URL" in content

    def test_when_env_example_read_ollama_api_key_variable_is_present(self):
        content = (PROJECT_ROOT / ".env.example").read_text()
        assert "OLLAMA_API_KEY" in content

    def test_when_gitignore_present_dotenv_file_is_excluded_from_version_control(self):
        gitignore = PROJECT_ROOT / ".gitignore"
        if not gitignore.is_file():
            pytest.skip(".gitignore not found — cannot verify .env exclusion")
        content = gitignore.read_text()
        assert ".env" in content


# ── LLM output is an index into legal_actions ────────────────────────────────
# Criterion: LLM output is an index into legal_actions — the chosen action is
#            always legal by construction.
#
# call_ollama_for_action_index(obs, legal_actions, *, model, base_url, api_key, ...)
#   → int | None   (the index, not the action element)


class TestLLMOutputIsIndex:
    """Mock the HTTP layer so tests are deterministic and network-free."""

    def _fake_http_response(self, action_index: int) -> mock.MagicMock:
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.raise_for_status = mock.MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": f'{{"action_index": {action_index}}}'}}]
        }
        return resp

    def _call_with_mocked_http(self, action_index: int, legal_actions: list) -> object:
        fake_resp = self._fake_http_response(action_index)
        obs = _minimal_obs()
        with (
            mock.patch("httpx.post", return_value=fake_resp),
            mock.patch("src.agents.ollama_client.ensure_env_loaded"),
        ):
            from src.agents.ollama_client import call_ollama_for_action_index

            return call_ollama_for_action_index(
                obs=obs,
                legal_actions=legal_actions,
                model="gpt-oss:120b",
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                temperature=0.5,
                top_p=0.9,
            )

    def test_when_ollama_returns_index_1_result_is_1(self):
        legal_actions = [_skip_action(), _skip_action(), _skip_action()]
        result = self._call_with_mocked_http(action_index=1, legal_actions=legal_actions)
        assert result == 1

    def test_when_ollama_returns_index_0_result_is_0(self):
        legal_actions = [_skip_action(), _skip_action(), _skip_action()]
        result = self._call_with_mocked_http(action_index=0, legal_actions=legal_actions)
        assert result == 0

    def test_when_ollama_returns_last_valid_index_result_is_in_range(self):
        legal_actions = [_skip_action()] * 4
        result = self._call_with_mocked_http(action_index=3, legal_actions=legal_actions)
        assert result == 3
        assert 0 <= result < len(legal_actions)

    def test_when_ollama_returns_out_of_range_index_result_is_none(self):
        """Out-of-range action_index must be rejected (returns None)."""
        legal_actions = [_skip_action(), _skip_action()]
        result = self._call_with_mocked_http(action_index=99, legal_actions=legal_actions)
        assert result is None


# ── render_action_prompt is the single prompt source ─────────────────────────
# Criterion: render_action_prompt() is the single source of truth for the LLM
#            prompt; the API key is never hardcoded in source.


class TestRenderActionPrompt:
    """Verify render_action_prompt is callable and produces a structured string."""

    def test_when_called_render_action_prompt_returns_a_string(self):
        from src.agents.action_prompt import render_action_prompt

        result = render_action_prompt(_minimal_obs(), [_skip_action()])
        assert isinstance(result, str)

    def test_when_called_render_action_prompt_returns_non_empty_string(self):
        from src.agents.action_prompt import render_action_prompt

        result = render_action_prompt(_minimal_obs(), [_skip_action()])
        assert len(result.strip()) > 0

    def test_when_legal_actions_provided_render_action_prompt_includes_index_0(self):
        from src.agents.action_prompt import render_action_prompt

        result = render_action_prompt(_minimal_obs(), [_skip_action()])
        assert "0" in result  # action index 0 must appear

    def test_when_strategy_hint_provided_render_action_prompt_includes_it(self):
        """Criterion: strategy_hint is a one-line directive included in the prompt."""
        from src.agents.action_prompt import render_action_prompt

        hint = "PLAY_AGGRESSIVELY_HINT"
        result = render_action_prompt(_minimal_obs(), [_skip_action()], strategy_hint=hint)
        assert hint in result
