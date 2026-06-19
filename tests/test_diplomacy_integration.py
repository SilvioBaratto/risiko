"""Scripted-pool integration test for issue #105.

Drives the full alliance → betrayal → grudge arc end-to-end with mocked LLM
responses keyed on (speaker, round_index) for reproducibility. Reports the RL
learner's win-rate vs the socially-coordinating pool via run_social_eval.

All tests carry @pytest.mark.integration; run with:
    pytest -m integration tests/test_diplomacy_integration.py
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Scripted call_fn factory — keyed on (speaker, round_index)
# ---------------------------------------------------------------------------


class _ScriptedCallFn:
    """Deterministic call_fn whose responses are keyed on (speaker, call_number).

    Each call to __call__ increments the per-speaker counter so successive
    rounds receive different scripted payloads.
    """

    def __init__(self, script: dict[tuple[int, int], dict[str, Any]]) -> None:
        self._script = script
        self._counters: dict[int, int] = defaultdict(int)
        self.calls: list[tuple[int, int]] = []

    def __call__(self, speaker: int, prompt: str) -> dict[str, Any] | None:
        idx = self._counters[speaker]
        self._counters[speaker] += 1
        self.calls.append((speaker, idx))
        return self._script.get((speaker, idx))


def _idle_response() -> dict:
    return {
        "propose_alliance_with": [],
        "accept_alliance_with": [],
        "declare_war_on": [],
        "attack_priority": [],
    }


def _make_profiles(player_ids: list[int]):
    from src.agents.player_config import PlayerConfig

    return [
        PlayerConfig(
            player_id=pid,
            temperature=0.5,
            top_p=0.9,
            strategy_hint="scripted test agent",
        )
        for pid in player_ids
    ]


# ---------------------------------------------------------------------------
# Alliance arc
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_when_scripted_pool_proposes_and_accepts_alliance_then_state_records_it():
    """
    End-to-end arc step 1 — alliance formation:
    P1 proposes alliance with P2 and P2 accepts → are_allied(1, 2) is True.
    P0 is the learner and is excluded from speakers.
    """
    from src.agents.diplomacy import DiplomacyState
    from src.agents.negotiation import NegotiationOrchestrator
    from src.agents.reputation import ReputationBook
    from src.config import DiplomacyConfig

    state = DiplomacyState()
    script = {
        (1, 0): {
            "propose_alliance_with": [2],
            "accept_alliance_with": [],
            "declare_war_on": [],
            "attack_priority": [],
        },
        (2, 0): {
            "propose_alliance_with": [],
            "accept_alliance_with": [1],
            "declare_war_on": [],
            "attack_priority": [],
        },
    }
    call_fn = _ScriptedCallFn(script)

    orchestrator = NegotiationOrchestrator(
        state=state,
        reputation=ReputationBook(),
        cfg=DiplomacyConfig(enabled=True, n_rounds=1),
        profiles=_make_profiles([0, 1, 2]),
        call_fn=call_fn,
        learner_slot=0,
    )
    orchestrator.run_round(obs={}, speakers=[1, 2])

    assert state.are_allied(1, 2), "Alliance must form when both sides express intent"
    assert state.are_allied(2, 1), "Alliance is symmetric"


# ---------------------------------------------------------------------------
# Betrayal arc
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_when_scripted_speaker_declares_war_on_ally_then_betrayal_arc_completes():
    """
    End-to-end arc steps 2 and 3 — betrayal and grudge:
    P1 declares war on P2 (an established ally).
    Post-arc: alliance dissolved, P2 holds a grudge toward P1.
    """
    from src.agents.diplomacy import DiplomacyState
    from src.agents.negotiation import NegotiationOrchestrator
    from src.agents.reputation import ReputationBook
    from src.config import DiplomacyConfig

    state = DiplomacyState()
    state.form_alliance(1, 2)  # pre-existing alliance

    # Round 0: P1 betrays P2 via war declaration
    script = {
        (1, 0): {
            "propose_alliance_with": [],
            "accept_alliance_with": [],
            "declare_war_on": [2],
            "attack_priority": [],
        },
        (2, 0): _idle_response(),
    }
    call_fn = _ScriptedCallFn(script)

    orchestrator = NegotiationOrchestrator(
        state=state,
        reputation=ReputationBook(),
        cfg=DiplomacyConfig(enabled=True, n_rounds=1),
        profiles=_make_profiles([0, 1, 2]),
        call_fn=call_fn,
        learner_slot=0,
    )
    orchestrator.run_round(obs={}, speakers=[1, 2])

    # Arc complete
    assert not state.are_allied(1, 2), "Alliance must be dissolved after war declaration"
    assert state.has_grudge(victim=2, aggressor=1), "P2 must hold a grudge toward P1"


# ---------------------------------------------------------------------------
# Full arc in sequence: alliance → betrayal → grudge across two rounds
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_when_scripted_pool_drives_full_arc_then_all_state_transitions_are_correct():
    """
    Full alliance → betrayal → grudge arc across two negotiation rounds:
      Round 0: P1 proposes, P2 accepts → alliance formed.
      Round 1: P1 declares war on P2 → alliance broken, grudge recorded.

    The call_fn is keyed on (speaker, round_index) so the two rounds produce
    different, deterministic outcomes.
    """
    from src.agents.diplomacy import DiplomacyState
    from src.agents.negotiation import NegotiationOrchestrator
    from src.agents.reputation import ReputationBook
    from src.config import DiplomacyConfig

    state = DiplomacyState()
    script = {
        # Round 0
        (1, 0): {
            "propose_alliance_with": [2],
            "accept_alliance_with": [],
            "declare_war_on": [],
            "attack_priority": [],
        },
        (2, 0): {
            "propose_alliance_with": [],
            "accept_alliance_with": [1],
            "declare_war_on": [],
            "attack_priority": [],
        },
        # Round 1
        (1, 1): {
            "propose_alliance_with": [],
            "accept_alliance_with": [],
            "declare_war_on": [2],
            "attack_priority": [],
        },
        (2, 1): _idle_response(),
    }
    call_fn = _ScriptedCallFn(script)

    orchestrator = NegotiationOrchestrator(
        state=state,
        reputation=ReputationBook(),
        cfg=DiplomacyConfig(enabled=True, n_rounds=2),
        profiles=_make_profiles([0, 1, 2]),
        call_fn=call_fn,
        learner_slot=0,
    )

    # Round 0: alliance formation
    orchestrator.run_round(obs={}, speakers=[1, 2])
    assert state.are_allied(1, 2), "Alliance must form after round 0"

    # Round 1: betrayal
    orchestrator.run_round(obs={}, speakers=[1, 2])
    assert not state.are_allied(1, 2), "Alliance must dissolve after round 1 war declaration"
    assert state.has_grudge(victim=2, aggressor=1), "Grudge must persist after betrayal"

    # call_fn was called exactly 4 times (2 speakers × 2 rounds)
    assert len(call_fn.calls) == 4


# ---------------------------------------------------------------------------
# RL win-rate reporting
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_when_social_eval_runs_then_learner_win_rate_is_reported():
    """
    run_social_eval returns a SocialEvalResult whose learner_win_rate is a
    float in [0.0, 1.0]. The MultiAgentRunner is mocked to return scripted
    GameResults so no real game is played and the result is deterministic.
    """
    from src.agents.random_agent import RandomAgent
    from training.social_eval import run_social_eval

    def _stub_result(winner):
        r = MagicMock()
        r.winner = winner
        r.n_turns = 8
        return r

    with patch("training.social_eval.MultiAgentRunner") as mock_runner:
        # 2 learner wins, 1 draw across 3 games
        mock_runner.return_value.run.side_effect = [
            _stub_result(0),
            _stub_result(None),
            _stub_result(0),
        ]
        result = run_social_eval(
            learner=RandomAgent(),
            llm_pool=[RandomAgent() for _ in range(5)],
            n_games=3,
            seed=42,
        )

    assert 0.0 <= result.learner_win_rate <= 1.0, (
        f"learner_win_rate must be in [0, 1], got {result.learner_win_rate}"
    )
    assert result.learner_win_rate == pytest.approx(2 / 3)
