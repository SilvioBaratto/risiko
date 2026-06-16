"""Source-blind example tests for issue #47: GameResult and Transition dataclasses.

Authored from acceptance criteria text only — no implementation source was read.
Every test is in the Red phase: it must fail until the criterion is genuinely met.
"""

from __future__ import annotations

import dataclasses
import math
import os
import unittest.mock
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

# ─── Factories ────────────────────────────────────────────────────────────────


def _obs() -> dict[str, np.ndarray]:
    """Full observation dict matching every key used by render_action_prompt."""
    return {
        "territory_owner": np.zeros(42, dtype=np.int32),
        "armies": np.ones(42, dtype=np.int32),
        "current_player": np.array(0, dtype=np.int32),
        "phase": np.array(1, dtype=np.int32),
        "reinforcements_remaining": np.array(3, dtype=np.int32),
        "trade_count": np.array(0, dtype=np.int32),
        "eliminated": np.zeros(6, dtype=np.int32),
        "cards": np.zeros((5, 4), dtype=np.int32),
    }


def _action() -> dict[str, int]:
    """Proper game action format: SKIP action (action_type=5)."""
    return {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}


def _transition(**kw):
    from src.multi_agent import Transition

    defaults: dict[str, Any] = {"obs": _obs(), "action": _action(), "reward": 0.0}
    defaults.update(kw)
    return Transition(**defaults)


def _game_result(**kw):
    from src.multi_agent import GameResult

    defaults: dict[str, Any] = {
        "winner": 0,
        "n_turns": 10,
        "territory_history": [np.zeros(42, dtype=np.float32)],
        "elimination_order": [1, 2, 3, 4, 5],
        "card_trade_turns": [],
        "action_log": [],
        "trajectories": [],
    }
    defaults.update(kw)
    return GameResult(**defaults)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _fake_http_response(action_index: int) -> unittest.mock.MagicMock:
    resp = unittest.mock.MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "message": {"content": f'{{"action_index": {action_index}}}'}
    }
    return resp


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ENV_EXAMPLE = os.path.join(_PROJECT_ROOT, ".env.example")

_FAKE_OLLAMA_ENV = {
    "OLLAMA_BASE_URL": "http://localhost:11434/v1",
    "OLLAMA_API_KEY": "fake-key-00000000000000000000000000000000",
}


# ══════════════════════════════════════════════════════════════════════════════
# Import: both dataclasses importable as `from src.multi_agent import ...`
# ══════════════════════════════════════════════════════════════════════════════


def test_when_transition_is_imported_then_it_is_available_from_src_multi_agent():
    from src.multi_agent import Transition  # noqa: F401

    assert Transition is not None


def test_when_game_result_is_imported_then_it_is_available_from_src_multi_agent():
    from src.multi_agent import GameResult  # noqa: F401

    assert GameResult is not None


# ══════════════════════════════════════════════════════════════════════════════
# Transition — required fields
# Criterion: obs: dict[str, np.ndarray], action: dict[str, int], reward: float
# ══════════════════════════════════════════════════════════════════════════════


def test_when_transition_created_with_obs_then_obs_is_stored():
    from src.multi_agent import Transition

    obs = _obs()
    t = Transition(obs=obs, action=_action(), reward=0.0)
    assert t.obs is obs


def test_when_transition_created_with_action_then_action_is_stored():
    from src.multi_agent import Transition

    action = _action()
    t = Transition(obs=_obs(), action=action, reward=0.0)
    assert t.action is action


def test_when_transition_created_with_reward_then_reward_is_stored():
    from src.multi_agent import Transition

    t = Transition(obs=_obs(), action=_action(), reward=-1.5)
    assert t.reward == -1.5


# ══════════════════════════════════════════════════════════════════════════════
# Transition — default fields
# Criterion: log_prob=math.nan, value=math.nan, legal_actions=[]
# ══════════════════════════════════════════════════════════════════════════════


def test_when_transition_created_without_log_prob_then_log_prob_defaults_to_nan():
    t = _transition()
    assert math.isnan(t.log_prob)


def test_when_transition_created_without_value_then_value_defaults_to_nan():
    t = _transition()
    assert math.isnan(t.value)


def test_when_transition_created_without_legal_actions_then_legal_actions_defaults_to_empty_list():
    t = _transition()
    assert t.legal_actions == []


def test_when_transition_created_with_explicit_legal_actions_then_they_are_stored():
    legal = [{"territory": 3, "armies": 2}, {"territory": 7, "armies": 1}]
    t = _transition(legal_actions=legal)
    assert t.legal_actions == legal


def test_when_two_transitions_created_without_legal_actions_then_list_is_not_shared():
    """Default factory must produce a new list per instance (field(default_factory=list))."""
    t1 = _transition()
    t2 = _transition()
    assert t1.legal_actions is not t2.legal_actions


# ══════════════════════════════════════════════════════════════════════════════
# Transition — frozen (dataclass(frozen=True))
# ══════════════════════════════════════════════════════════════════════════════


def test_when_transition_obs_is_reassigned_then_frozen_error_is_raised():
    t = _transition()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        t.obs = _obs()


def test_when_transition_action_is_reassigned_then_frozen_error_is_raised():
    t = _transition()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        t.action = _action()


def test_when_transition_reward_is_reassigned_then_frozen_error_is_raised():
    t = _transition()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        t.reward = 99.0


def test_when_transition_log_prob_is_reassigned_then_frozen_error_is_raised():
    t = _transition(log_prob=-0.3)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        t.log_prob = -0.1


def test_when_transition_value_is_reassigned_then_frozen_error_is_raised():
    t = _transition(value=1.2)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        t.value = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Transition — __iter__
# Criterion: yields (obs, action, reward) so `obs, action, reward = t` works
# ══════════════════════════════════════════════════════════════════════════════


def test_when_transition_is_iterated_then_exactly_three_values_are_yielded():
    t = _transition()
    assert len(list(t)) == 3


def test_when_transition_is_unpacked_then_first_element_is_obs():
    obs = _obs()
    t = _transition(obs=obs)
    first, *_ = t
    assert first is obs


def test_when_transition_is_unpacked_then_second_element_is_action():
    action = _action()
    t = _transition(action=action)
    _, second, _ = t
    assert second is action


def test_when_transition_is_unpacked_then_third_element_is_reward():
    t = _transition(reward=3.14)
    _, _, third = t
    assert third == pytest.approx(3.14)


def test_when_transition_is_triple_unpacked_then_all_three_bindings_are_correct():
    """The stated use-case: `obs, action, reward = transition` must work."""
    obs = _obs()
    action = _action()
    t = _transition(obs=obs, action=action, reward=2.0)
    o, a, r = t
    assert o is obs
    assert a is action
    assert r == pytest.approx(2.0)


# ──── Property: iter ordering invariant ───────────────────────────────────────


@given(st.floats(allow_nan=False, allow_infinity=True))
def test_when_transition_is_iterated_then_reward_is_always_third_for_any_finite_value(
    reward: float,
) -> None:
    """Invariant (ordering): __iter__ always yields reward as the third element."""
    t = _transition(reward=reward)
    _, _, third = t
    assert third == pytest.approx(reward)


# ──── Property: frozen invariant ─────────────────────────────────────────────


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_when_transition_reward_is_reassigned_then_frozen_error_is_always_raised(
    reward: float,
) -> None:
    """Invariant (frozen): mutating reward on a Transition always raises."""
    t = _transition(reward=reward)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        t.reward = reward + 1.0


# ══════════════════════════════════════════════════════════════════════════════
# GameResult — required fields
# Criterion: winner, n_turns, territory_history, elimination_order,
#            card_trade_turns, action_log, trajectories
# ══════════════════════════════════════════════════════════════════════════════


def test_when_game_result_created_with_int_winner_then_winner_is_stored():
    gr = _game_result(winner=3)
    assert gr.winner == 3


def test_when_game_result_created_with_none_winner_then_winner_is_none():
    """Criterion: winner: int | None — None must be a valid value."""
    gr = _game_result(winner=None)
    assert gr.winner is None


def test_when_game_result_created_with_n_turns_then_n_turns_is_stored():
    gr = _game_result(n_turns=77)
    assert gr.n_turns == 77


def test_when_game_result_created_with_territory_history_then_history_is_stored():
    hist = [np.zeros(42, dtype=np.float32), np.ones(42, dtype=np.float32)]
    gr = _game_result(territory_history=hist)
    assert gr.territory_history is hist


def test_when_game_result_created_with_elimination_order_then_order_is_stored():
    order = [5, 2, 4, 1, 3]
    gr = _game_result(elimination_order=order)
    assert gr.elimination_order == order


def test_when_game_result_created_with_card_trade_turns_then_turns_are_stored():
    turns = [3, 8, 14]
    gr = _game_result(card_trade_turns=turns)
    assert gr.card_trade_turns == turns


def test_when_game_result_created_with_action_log_then_log_is_stored():
    log: list[dict[str, Any]] = [{"phase": "reinforce", "territory": 0, "armies": 3}]
    gr = _game_result(action_log=log)
    assert gr.action_log is log


def test_when_game_result_created_with_trajectories_then_trajectories_are_stored():
    traj = [_transition()]
    gr = _game_result(trajectories=traj)
    assert gr.trajectories is traj


# ══════════════════════════════════════════════════════════════════════════════
# GameResult — default field
# Criterion: card_trade_hand_sizes: list[int] = field(default_factory=list)
# ══════════════════════════════════════════════════════════════════════════════


def test_when_game_result_created_without_card_trade_hand_sizes_then_it_defaults_to_empty_list():
    gr = _game_result()
    assert gr.card_trade_hand_sizes == []


def test_when_game_result_created_with_card_trade_hand_sizes_then_sizes_are_stored():
    sizes = [3, 5, 3]
    gr = _game_result(card_trade_hand_sizes=sizes)
    assert gr.card_trade_hand_sizes == sizes


def test_when_two_game_results_created_without_hand_sizes_then_list_is_not_shared():
    """Default factory must produce a new list per instance."""
    gr1 = _game_result()
    gr2 = _game_result()
    assert gr1.card_trade_hand_sizes is not gr2.card_trade_hand_sizes


# ══════════════════════════════════════════════════════════════════════════════
# GameResult — frozen
# ══════════════════════════════════════════════════════════════════════════════


def test_when_game_result_winner_is_reassigned_then_frozen_error_is_raised():
    gr = _game_result()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        gr.winner = 99


def test_when_game_result_n_turns_is_reassigned_then_frozen_error_is_raised():
    gr = _game_result()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        gr.n_turns = 999


def test_when_game_result_elimination_order_is_reassigned_then_frozen_error_is_raised():
    gr = _game_result()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        gr.elimination_order = []


# ──── Property: winner accepts int | None for all valid player indices ────────


@given(st.one_of(st.none(), st.integers(min_value=0, max_value=5)))
def test_when_game_result_winner_is_any_valid_value_then_it_is_always_stored_correctly(
    winner: int | None,
) -> None:
    """Invariant (never-raises): winner=None or winner in 0..5 always accepted."""
    gr = _game_result(winner=winner)
    assert gr.winner == winner


# ──── Property: GameResult frozen invariant ───────────────────────────────────


@given(st.integers(min_value=1, max_value=10_000))
def test_when_game_result_is_frozen_then_n_turns_mutation_always_raises(
    n_turns: int,
) -> None:
    """Invariant (frozen): any attempt to overwrite n_turns always raises."""
    gr = _game_result(n_turns=n_turns)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        gr.n_turns = 0


# ══════════════════════════════════════════════════════════════════════════════
# Global: .env.example committed
# Criterion: Ollama credentials load from .env; .env.example is committed
# ══════════════════════════════════════════════════════════════════════════════


def test_when_project_root_is_checked_then_env_example_file_is_present():
    """Criterion: .env.example is committed (present on disk at project root)."""
    assert os.path.isfile(_ENV_EXAMPLE), f".env.example not found at {_ENV_EXAMPLE}"


def test_when_env_example_is_read_then_ollama_base_url_variable_is_declared():
    with open(_ENV_EXAMPLE) as fh:
        content = fh.read()
    assert "OLLAMA_BASE_URL" in content


def test_when_env_example_is_read_then_ollama_api_key_variable_is_declared():
    with open(_ENV_EXAMPLE) as fh:
        content = fh.read()
    assert "OLLAMA_API_KEY" in content


# ══════════════════════════════════════════════════════════════════════════════
# Global: render_action_prompt is the single source of truth for the LLM prompt
# ══════════════════════════════════════════════════════════════════════════════


def test_when_render_action_prompt_is_called_then_a_non_empty_string_is_returned():
    """Criterion: render_action_prompt() is the single LLM prompt source."""
    from src.agents.action_prompt import render_action_prompt

    result = render_action_prompt(_obs(), [_action()])
    assert isinstance(result, str)
    assert result.strip() != ""


def test_when_render_action_prompt_is_called_then_legal_actions_are_represented_in_prompt():
    """The prompt must expose the legal action list indexed so index 0 appears."""
    from src.agents.action_prompt import render_action_prompt

    # Use real REINFORCE action (action_type=1) targeting territory 17
    legal = [{"action_type": 1, "param_a": 17, "param_b": 2, "param_c": 0, "param_d": 0}]
    prompt = render_action_prompt(_obs(), legal)
    # Index 0 must appear (the action list is indexed from 0)
    assert "0" in prompt


def test_when_render_action_prompt_is_called_then_ollama_api_key_literal_is_absent_from_output():
    """Criterion: API key never hardcoded — prompt must not contain the key literal."""
    from src.agents.action_prompt import render_action_prompt

    prompt = render_action_prompt(_obs(), [_action()])
    assert "OLLAMA_API_KEY" not in prompt


# ══════════════════════════════════════════════════════════════════════════════
# Global: LLM output is index into legal_actions — always legal by construction
# Criterion: call_ollama_for_action_index returns an action from legal_actions
# ══════════════════════════════════════════════════════════════════════════════


_LEGAL_SKIP = [
    {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0},
    {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0},
]

_SKIP_ACTION = {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}


def test_when_ollama_returns_action_index_0_then_result_is_zero():
    """Criterion: index 0 maps to the first element; function returns the index (int)."""
    from src.agents.ollama_client import call_ollama_for_action_index

    legal = [_SKIP_ACTION, _SKIP_ACTION]
    with (
        unittest.mock.patch.dict(os.environ, _FAKE_OLLAMA_ENV),
        unittest.mock.patch("httpx.post", return_value=_fake_http_response(0)),
    ):
        result = call_ollama_for_action_index(_obs(), legal, model="gpt-oss:120b")
    assert result == 0


def test_when_ollama_returns_action_index_1_then_result_is_one():
    """Criterion: index 1 maps to the second element; function returns the index (int)."""
    from src.agents.ollama_client import call_ollama_for_action_index

    legal = [_SKIP_ACTION, _SKIP_ACTION]
    with (
        unittest.mock.patch.dict(os.environ, _FAKE_OLLAMA_ENV),
        unittest.mock.patch("httpx.post", return_value=_fake_http_response(1)),
    ):
        result = call_ollama_for_action_index(_obs(), legal, model="gpt-oss:120b")
    assert result == 1


def test_when_ollama_returns_last_valid_index_then_result_equals_last_index():
    """Criterion: index at the end of legal_actions is returned as-is."""
    from src.agents.ollama_client import call_ollama_for_action_index

    legal = [_SKIP_ACTION] * 5
    last_idx = len(legal) - 1
    with (
        unittest.mock.patch.dict(os.environ, _FAKE_OLLAMA_ENV),
        unittest.mock.patch("httpx.post", return_value=_fake_http_response(last_idx)),
    ):
        result = call_ollama_for_action_index(_obs(), legal, model="gpt-oss:120b")
    assert result == last_idx


# ──── Property: any valid index is always in-range for legal_actions ─────────


@given(
    st.integers(min_value=1, max_value=10),  # list length
    st.integers(min_value=0, max_value=9),  # raw index
)
def test_when_ollama_returns_any_valid_index_then_result_is_always_in_range(
    n_actions: int, raw_index: int
) -> None:
    """Return value is always a valid index into legal_actions (int, in-range).

    Derived from: 'LLM output is an index into legal_actions — always legal.'
    """
    from src.agents.ollama_client import call_ollama_for_action_index

    legal = [_SKIP_ACTION] * n_actions
    index = raw_index % n_actions
    with (
        unittest.mock.patch.dict(os.environ, _FAKE_OLLAMA_ENV),
        unittest.mock.patch("httpx.post", return_value=_fake_http_response(index)),
    ):
        result = call_ollama_for_action_index(_obs(), legal, model="gpt-oss:120b")
    assert result is not None
    assert 0 <= result < n_actions


# ══════════════════════════════════════════════════════════════════════════════
# Global [T2]: LLM calls the native /api/chat with an enforced `format` schema
# Criterion: LLM opponent calls Ollama via /api/chat with a JSON-schema `format`
#            (XGrammar-enforced) rather than the loose /v1 response_format, which
#            Ollama ignores for reasoning models (ollama#10937, #15260).
# Black-box: capture httpx.post call args and inspect the outgoing request body.
# ══════════════════════════════════════════════════════════════════════════════


def test_when_ollama_client_is_called_then_url_is_native_api_chat():
    """URL sent to httpx.post must be the native /api/chat endpoint, not /v1."""
    from src.agents.ollama_client import call_ollama_for_action_index

    legal = [_SKIP_ACTION]
    with (
        unittest.mock.patch.dict(os.environ, _FAKE_OLLAMA_ENV),
        unittest.mock.patch("httpx.post", return_value=_fake_http_response(0)) as mock_post,
    ):
        call_ollama_for_action_index(_obs(), legal, model="gpt-oss:120b")

    call_args = mock_post.call_args
    url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
    assert str(url).endswith("/api/chat"), (
        f"Expected URL to end with '/api/chat', got: {url!r}"
    )
    assert "/v1/" not in str(url)


def test_when_ollama_client_is_called_then_request_body_includes_format():
    """Request body sent to Ollama must include a native 'format' key."""
    from src.agents.ollama_client import call_ollama_for_action_index

    legal = [_SKIP_ACTION]
    with (
        unittest.mock.patch.dict(os.environ, _FAKE_OLLAMA_ENV),
        unittest.mock.patch("httpx.post", return_value=_fake_http_response(0)) as mock_post,
    ):
        call_ollama_for_action_index(_obs(), legal, model="gpt-oss:120b")

    call_args = mock_post.call_args
    body = call_args.kwargs.get("json") or {}
    assert "format" in body, "Ollama request body must include native 'format' schema"


def test_when_ollama_client_is_called_then_format_type_is_object():
    """format.type must equal 'object' (a raw JSON schema, not OpenAI nesting)."""
    from src.agents.ollama_client import call_ollama_for_action_index

    legal = [_SKIP_ACTION]
    with (
        unittest.mock.patch.dict(os.environ, _FAKE_OLLAMA_ENV),
        unittest.mock.patch("httpx.post", return_value=_fake_http_response(0)) as mock_post,
    ):
        call_ollama_for_action_index(_obs(), legal, model="gpt-oss:120b")

    call_args = mock_post.call_args
    body = call_args.kwargs.get("json") or {}
    fmt = body.get("format", {})
    assert fmt.get("type") == "object", (
        f"format.type must be 'object', got: {fmt.get('type')!r}"
    )


def test_when_ollama_client_is_called_then_thinking_is_enabled():
    """Think must be True: think=false breaks `format` enforcement (ollama#15260)."""
    from src.agents.ollama_client import call_ollama_for_action_index

    legal = [_SKIP_ACTION]
    with (
        unittest.mock.patch.dict(os.environ, _FAKE_OLLAMA_ENV),
        unittest.mock.patch("httpx.post", return_value=_fake_http_response(0)) as mock_post,
    ):
        call_ollama_for_action_index(_obs(), legal, model="gpt-oss:120b")

    body = mock_post.call_args.kwargs.get("json") or {}
    assert body.get("think") is True, "thinking must stay enabled for format masking"


def test_when_ollama_client_is_called_then_schema_enforces_action_index_field():
    """Format properties must include 'action_index' so the LLM picks an index."""
    from src.agents.ollama_client import call_ollama_for_action_index

    legal = [_SKIP_ACTION]
    with (
        unittest.mock.patch.dict(os.environ, _FAKE_OLLAMA_ENV),
        unittest.mock.patch("httpx.post", return_value=_fake_http_response(0)) as mock_post,
    ):
        call_ollama_for_action_index(_obs(), legal, model="gpt-oss:120b")

    call_args = mock_post.call_args
    body = call_args.kwargs.get("json") or {}
    properties = body.get("format", {}).get("properties", {})
    assert "action_index" in properties, (
        "format schema must declare 'action_index' as a required property"
    )


# ──── Property: request body format is stable across any non-empty legal_actions ─


@given(st.integers(min_value=1, max_value=10))
def test_when_ollama_is_called_with_any_n_actions_then_format_type_is_object(
    n_actions: int,
) -> None:
    """format.type is always 'object' regardless of legal_actions size.

    Derived from: 'calls Ollama via /api/chat with an enforced JSON-schema format'.
    """
    from src.agents.ollama_client import call_ollama_for_action_index

    legal = [_SKIP_ACTION] * n_actions
    with (
        unittest.mock.patch.dict(os.environ, _FAKE_OLLAMA_ENV),
        unittest.mock.patch("httpx.post", return_value=_fake_http_response(0)) as mock_post,
    ):
        call_ollama_for_action_index(_obs(), legal, model="gpt-oss:120b")

    body = mock_post.call_args.kwargs.get("json") or {}
    assert body.get("format", {}).get("type") == "object"
