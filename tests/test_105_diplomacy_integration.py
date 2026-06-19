"""
Issue #105 — scripted-pool integration test: alliance → betrayal → grudge arc.

Criterion: "A scripted-pool integration test (@pytest.mark.integration) drives an
end-to-end alliance → betrayal → grudge arc and reports the RL win-rate vs the pool"
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# DiplomacyState — unit-level state-machine tests (preconditions for the arc)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_when_alliance_is_formed_then_players_are_allied():
    """form_alliance(0, 1) must make are_allied(0, 1) and are_allied(1, 0) True."""
    from src.agents.diplomacy import DiplomacyState

    state = DiplomacyState()
    state.form_alliance(0, 1)

    assert state.are_allied(0, 1), "P0 and P1 should be allied after form_alliance"
    assert state.are_allied(1, 0), "Alliance must be symmetric"


@pytest.mark.integration
def test_when_allied_player_attacks_ally_then_grudge_is_recorded():
    """
    After P0 attacks P1 (a current ally):
      - has_grudge(victim=1, aggressor=0) is True
      - has_grudge(victim=0, aggressor=1) is False (asymmetric)
    """
    from src.agents.diplomacy import DiplomacyState

    state = DiplomacyState()
    state.form_alliance(0, 1)
    state.record_attack(offender=0, victim=1, turn=0, base_severity=1.0)

    assert state.has_grudge(victim=1, aggressor=0), (
        "P1 must hold a grudge toward P0 after P0 attacked"
    )
    assert not state.has_grudge(victim=0, aggressor=1), (
        "P0 should not automatically hold a grudge toward its own victim"
    )


@pytest.mark.integration
def test_when_grudge_exists_then_it_persists_after_additional_attacks():
    """Grudges are append-only — they persist as new events are added."""
    from src.agents.diplomacy import DiplomacyState

    state = DiplomacyState()
    state.form_alliance(0, 1)
    state.record_attack(offender=0, victim=1, turn=0, base_severity=1.0)

    # A second event by someone else must not erase the existing grudge
    state.record_attack(offender=2, victim=3, turn=1, base_severity=0.5)

    assert state.has_grudge(victim=1, aggressor=0), (
        "Grudge must persist after subsequent unrelated events"
    )


@pytest.mark.integration
def test_when_betrayal_happens_then_alliance_is_dissolved():
    """declare_war_on breaks the alliance between attacker and target."""
    from src.agents.diplomacy import DiplomacyState

    state = DiplomacyState()
    state.form_alliance(0, 1)
    state.declare_war_on(attacker=0, target=1, turn=0)

    assert not state.are_allied(0, 1), "Alliance must be dissolved after declare_war_on"
    assert state.has_grudge(victim=1, aggressor=0), "A grudge must be recorded for the target"


# ---------------------------------------------------------------------------
# NegotiationOrchestrator — scripted call_fn drives DiplomacyState
# ---------------------------------------------------------------------------


def _make_profiles(player_ids):
    from src.agents.player_config import PlayerConfig

    return [
        PlayerConfig(
            player_id=pid,
            temperature=0.5,
            top_p=0.9,
            strategy_hint="test profile",
        )
        for pid in player_ids
    ]


@pytest.mark.integration
def test_when_orchestrator_runs_with_mutual_proposal_then_alliance_is_formed():
    """
    NegotiationOrchestrator.run_round with scripted call_fn that has both P1 and P2
    express interest in each other must result in DiplomacyState.are_allied(1, 2).

    P0 is the learner and is excluded from speakers.
    """
    from src.agents.diplomacy import DiplomacyState
    from src.agents.negotiation import NegotiationOrchestrator
    from src.agents.reputation import ReputationBook
    from src.config import DiplomacyConfig

    state = DiplomacyState()
    scripted = {
        1: {
            "propose_alliance_with": [2],
            "accept_alliance_with": [],
            "declare_war_on": [],
            "attack_priority": [],
        },
        2: {
            "propose_alliance_with": [],
            "accept_alliance_with": [1],
            "declare_war_on": [],
            "attack_priority": [],
        },
    }

    def call_fn(speaker: int, prompt: str):
        return scripted.get(speaker)

    orchestrator = NegotiationOrchestrator(
        state=state,
        reputation=ReputationBook(),
        cfg=DiplomacyConfig(enabled=True, n_rounds=1),
        profiles=_make_profiles([0, 1, 2]),
        call_fn=call_fn,
        learner_slot=0,
    )
    orchestrator.run_round(obs={}, speakers=[1, 2])

    assert state.are_allied(1, 2), (
        "Scripted mutual proposal/accept must result in alliance between P1 and P2"
    )


@pytest.mark.integration
def test_when_orchestrator_runs_with_war_declaration_then_grudge_is_recorded():
    """
    When a speaker declares war on another, the orchestrator must record a grudge
    in DiplomacyState via declare_war_on.
    """
    from src.agents.diplomacy import DiplomacyState
    from src.agents.negotiation import NegotiationOrchestrator
    from src.agents.reputation import ReputationBook
    from src.config import DiplomacyConfig

    state = DiplomacyState()
    state.form_alliance(1, 2)  # pre-existing alliance

    def call_fn(speaker: int, prompt: str):
        if speaker == 1:
            return {
                "propose_alliance_with": [],
                "accept_alliance_with": [],
                "declare_war_on": [2],
                "attack_priority": [],
            }
        return {
            "propose_alliance_with": [],
            "accept_alliance_with": [],
            "declare_war_on": [],
            "attack_priority": [],
        }

    orchestrator = NegotiationOrchestrator(
        state=state,
        reputation=ReputationBook(),
        cfg=DiplomacyConfig(enabled=True, n_rounds=1),
        profiles=_make_profiles([0, 1, 2]),
        call_fn=call_fn,
        learner_slot=0,
    )
    orchestrator.run_round(obs={}, speakers=[1, 2])

    assert state.has_grudge(victim=2, aggressor=1), (
        "P2 must have a grudge toward P1 after war declaration"
    )
    assert not state.are_allied(1, 2), "Alliance must be broken after war declaration"


# ---------------------------------------------------------------------------
# Full eval harness — RL win-rate reporting
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_when_social_eval_runs_then_learner_win_rate_is_in_valid_range():
    """
    run_social_eval with mocked MultiAgentRunner must return a result whose
    learner_win_rate is a float in [0.0, 1.0].
    """
    from src.agents.random_agent import RandomAgent
    from training.social_eval import run_social_eval

    def _stub_result(winner):
        r = MagicMock()
        r.winner = winner
        r.n_turns = 5
        return r

    with patch(
        "training.social_eval.MultiAgentRunner",
    ) as mock_runner:
        mock_runner.return_value.run.side_effect = [
            _stub_result(0),
            _stub_result(1),
            _stub_result(0),
        ]
        result = run_social_eval(
            learner=RandomAgent(),
            llm_pool=[RandomAgent() for _ in range(5)],
            n_games=3,
            seed=42,
        )

    assert isinstance(result.learner_win_rate, float), (
        f"learner_win_rate must be a float, got {type(result.learner_win_rate)}"
    )
    assert 0.0 <= result.learner_win_rate <= 1.0, (
        f"learner_win_rate must be in [0, 1], got {result.learner_win_rate}"
    )


@pytest.mark.integration
def test_when_all_games_won_by_learner_then_win_rate_is_one():
    """Win rate is 1.0 when all games are won by the learner (slot 0)."""
    from src.agents.random_agent import RandomAgent
    from training.social_eval import run_social_eval

    def _stub_result(winner):
        r = MagicMock()
        r.winner = winner
        r.n_turns = 5
        return r

    with patch("training.social_eval.MultiAgentRunner") as mock_runner:
        mock_runner.return_value.run.side_effect = [_stub_result(0), _stub_result(0)]
        result = run_social_eval(
            learner=RandomAgent(),
            llm_pool=[RandomAgent() for _ in range(5)],
            n_games=2,
            seed=0,
        )

    assert result.learner_win_rate == pytest.approx(1.0)
