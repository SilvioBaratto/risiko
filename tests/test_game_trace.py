"""Tests for LLM call tracing (visualization/full-game replay).

HTTP is mocked so no live Ollama is needed; the real client functions run so the
trace-emit paths are exercised end to end.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import src.agents.ollama_client as oc
from src.env import RisikoEnv


def _http_ok(content: str, thinking: str | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    message: dict = {"content": content}
    if thinking is not None:
        message["thinking"] = thinking
    resp.json.return_value = {"message": message, "eval_count": 7}
    return resp


def _obs_and_actions():
    env = RisikoEnv(n_players=6)
    obs, _ = env.reset(seed=1)
    return obs, env.get_legal_actions()


def test_when_tracing_active_then_action_call_records_prompt_response_thinking():
    obs, legal = _obs_and_actions()
    oc.start_call_trace()
    try:
        with patch(
            "src.agents.ollama_client.httpx.post",
            return_value=_http_ok('{"action_index": 0}', thinking="hold Australia"),
        ):
            idx = oc.call_ollama_for_action_index(
                obs, legal, model="m:cloud", base_url="http://x/v1", api_key="k"
            )
    finally:
        trace = oc.stop_call_trace()

    assert idx == 0
    assert trace is not None and len(trace) == 1
    rec = trace[0]
    assert rec["kind"] == "action"
    assert rec["model"] == "m:cloud"
    assert rec["prompt"]  # non-empty rendered prompt
    assert rec["response"] == '{"action_index": 0}'
    assert rec["thinking"] == "hold Australia"
    assert rec["parsed_index"] == 0
    assert rec["chosen_action"] == legal[0]


def test_when_tracing_inactive_then_no_records_and_calls_unaffected():
    obs, legal = _obs_and_actions()
    oc.stop_call_trace()  # ensure sink is off
    with patch(
        "src.agents.ollama_client.httpx.post",
        return_value=_http_ok('{"action_index": 0}'),
    ):
        idx = oc.call_ollama_for_action_index(
            obs, legal, model="m:cloud", base_url="http://x/v1", api_key="k"
        )
    assert idx == 0
    # A freshly started trace is empty → nothing leaked while inactive.
    assert oc.start_call_trace() == []
    oc.stop_call_trace()


def test_when_tracing_active_then_negotiation_call_records_thinking_and_parsed():
    content = (
        '{"propose_alliance_with": [1], "accept_alliance_with": [], '
        '"declare_war_on": [], "attack_priority": []}'
    )
    oc.start_call_trace()
    try:
        with patch(
            "src.agents.ollama_client.httpx.post",
            return_value=_http_ok(content, thinking="ally with player 1"),
        ):
            result = oc.call_ollama_for_negotiation(
                root="http://x",
                key="k",
                model="n:cloud",
                prompt="negotiate now",
                max_message_tokens=64,
            )
    finally:
        trace = oc.stop_call_trace()

    assert isinstance(result, dict)
    assert trace is not None and len(trace) == 1
    rec = trace[0]
    assert rec["kind"] == "negotiation"
    assert rec["model"] == "n:cloud"
    assert rec["prompt"] == "negotiate now"
    assert rec["thinking"] == "ally with player 1"
    assert rec["parsed"]["propose_alliance_with"] == [1]
