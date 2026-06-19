"""
Issue #106 — DiplomacyRunner never fires negotiation: orchestrator unwired,
call_count pinned at 0.

Required tests (from issue body):
1. Negotiation fires per the chosen cadence under a mocked call_fn
   (assert call_count > 0 and bounded by n_speakers per player-turn).
2. End-to-end at the runner level: scripted pool forms an alliance via
   negotiation, then a betrayal during run_game records the grudge /
   breaks the alliance / updates reputation.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _idle_response() -> dict[str, Any]:
    """Negotiation response that proposes nothing (silent no-op)."""
    return {
        "propose_alliance_with": [],
        "accept_alliance_with": [],
        "declare_war_on": [],
        "attack_priority": [],
    }


def _idle_call_fn(player_id: int, prompt: str) -> dict[str, Any] | None:
    """call_fn that always returns an idle response for any speaker."""
    return _idle_response()


def _n_players() -> int:
    return 3


def _make_runner(*, enabled: bool, call_fn=None, max_turns: int = 5):
    """Build a DiplomacyRunner with RandomAgents and an injectable call_fn."""
    from src.agents.random_agent import RandomAgent
    from src.config import DiplomacyConfig
    from src.env import RisikoEnv
    from src.multi_agent import DiplomacyRunner

    n = _n_players()
    env = RisikoEnv(n_players=n)
    agents = [RandomAgent(seed=i) for i in range(n)]
    cfg = DiplomacyConfig(enabled=enabled, n_rounds=1)
    return DiplomacyRunner(
        env,
        agents,
        max_turns=max_turns,
        diplomacy_cfg=cfg,
        call_fn=call_fn,
    )


# ---------------------------------------------------------------------------
# Criterion: call_count > 0 after a game with enabled + mocked call_fn
# ---------------------------------------------------------------------------


class TestNegotiationFiresWhenEnabled:
    def test_when_enabled_with_idle_call_fn_then_call_count_positive_after_game(self):
        """Negotiation must be called at least once per game when enabled=True."""
        runner = _make_runner(enabled=True, call_fn=_idle_call_fn)
        runner.run_game(seed=42)
        assert runner.call_count > 0, (
            "DiplomacyRunner.call_count must be > 0 after a game with "
            "diplomacy enabled and a mocked call_fn.  The orchestrator is "
            "currently unwired — negotiation never fires."
        )

    def test_when_enabled_then_call_count_is_zero_before_game(self):
        """call_count starts at 0 (no negotiation before run_game)."""
        runner = _make_runner(enabled=True, call_fn=_idle_call_fn)
        assert runner.call_count == 0

    def test_when_disabled_then_call_count_stays_zero_after_game(self):
        """Disabled runner must not call any negotiation LLM (regression guard)."""
        runner = _make_runner(enabled=False)
        runner.run_game(seed=42)
        assert runner.call_count == 0

    def test_when_enabled_with_different_seed_then_call_count_positive(self):
        """call_count > 0 for multiple seeds — not a seed-specific fluke."""
        for seed in (0, 7, 99):
            runner = _make_runner(enabled=True, call_fn=_idle_call_fn, max_turns=3)
            runner.run_game(seed=seed)
            assert runner.call_count > 0, f"call_count must be > 0 (seed={seed}, enabled=True)"


# ---------------------------------------------------------------------------
# Criterion: call_count bounded by n_speakers per player-turn
# ---------------------------------------------------------------------------


class TestNegotiationCallCountBounded:
    def test_when_enabled_then_call_count_bounded_by_n_speakers_times_n_turns(self):
        """
        Negotiation fires at most once per player-turn boundary.
        With n_players=3 (learner + 2 LLM speakers) and max_turns=5,
        call_count must not exceed 2 speakers × 5 player-turns = 10.
        """
        runner = _make_runner(enabled=True, call_fn=_idle_call_fn, max_turns=5)
        runner.run_game(seed=0)
        n_speakers = _n_players() - 1  # exclude learner (slot 0)
        max_turns = 5
        assert runner.call_count <= n_speakers * max_turns, (
            f"call_count={runner.call_count} exceeds the per-player-turn bound "
            f"of {n_speakers} speakers × {max_turns} turns."
        )

    def test_when_enabled_then_call_count_is_multiple_of_n_speakers(self):
        """
        Each negotiation round calls exactly one LLM per speaker.
        call_count must be a non-negative multiple of the speaker count
        (= non-learner, non-eliminated players).
        """
        runner = _make_runner(enabled=True, call_fn=_idle_call_fn, max_turns=3)
        runner.run_game(seed=1)
        n_speakers = _n_players() - 1
        # call_count = k × n_speakers for some k >= 1
        assert runner.call_count % n_speakers == 0, (
            f"call_count={runner.call_count} is not a multiple of n_speakers={n_speakers}."
        )

    def test_when_enabled_then_call_count_increments_each_turn(self):
        """call_count after 3 turns ≥ call_count after 1 turn (monotone)."""
        runner_short = _make_runner(enabled=True, call_fn=_idle_call_fn, max_turns=1)
        runner_long = _make_runner(enabled=True, call_fn=_idle_call_fn, max_turns=3)
        runner_short.run_game(seed=5)
        runner_long.run_game(seed=5)
        assert runner_long.call_count >= runner_short.call_count


# ---------------------------------------------------------------------------
# Criterion: end-to-end alliance formation via runner negotiation
# ---------------------------------------------------------------------------


class _ScriptedAllianceFn:
    """
    call_fn that makes P1 propose an alliance with P2 and P2 accept it.
    P0 is the learner and is never called.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def __call__(self, player_id: int, prompt: str) -> dict[str, Any] | None:
        self.calls.append((player_id, prompt))
        if player_id == 1:
            return {
                "propose_alliance_with": [2],
                "accept_alliance_with": [],
                "declare_war_on": [],
                "attack_priority": [],
            }
        if player_id == 2:
            return {
                "propose_alliance_with": [],
                "accept_alliance_with": [1],
                "declare_war_on": [],
                "attack_priority": [],
            }
        return _idle_response()


class TestRunnerLevelAllianceFormation:
    def test_when_scripted_alliance_call_fn_then_state_shows_alliance_after_run(self):
        """
        A scripted call_fn (P1 proposes, P2 accepts) must cause DiplomacyState
        to record an alliance between P1 and P2 during the runner's run_game.
        """
        from src.agents.diplomacy import DiplomacyState
        from src.agents.random_agent import RandomAgent
        from src.agents.reputation import ReputationBook
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        state = DiplomacyState()
        scripted = _ScriptedAllianceFn()

        env = RisikoEnv(n_players=3)
        runner = DiplomacyRunner(
            env,
            [RandomAgent(seed=i) for i in range(3)],
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=True, n_rounds=1),
            state=state,
            reputation=ReputationBook(),
            call_fn=scripted,
        )
        runner.run_game(seed=42)

        assert scripted.calls, "call_fn must have been called at least once"
        assert state.are_allied(1, 2), (
            "DiplomacyState must record an alliance between P1 and P2 after "
            "the scripted call_fn makes them mutually propose/accept."
        )
        assert state.are_allied(2, 1), "Alliance must be symmetric"

    def test_when_scripted_alliance_call_fn_then_learner_never_called(self):
        """The learner (slot 0) is excluded from negotiation speakers."""
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        calls: list[int] = []

        def tracking_fn(player_id: int, prompt: str) -> dict | None:
            calls.append(player_id)
            return _idle_response()

        env = RisikoEnv(n_players=3)
        DiplomacyRunner(
            env,
            [RandomAgent(seed=i) for i in range(3)],
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=True, n_rounds=1),
            call_fn=tracking_fn,
        ).run_game(seed=42)

        assert calls, "call_fn must be called for non-learner speakers"
        assert 0 not in calls, (
            f"Learner (slot 0) must never appear as a negotiation speaker; got calls for: {calls}"
        )


# ---------------------------------------------------------------------------
# Criterion: end-to-end betrayal at runner level records grudge
# ---------------------------------------------------------------------------


class _ScriptedWarFn:
    """call_fn that makes P1 declare war on P2 (the established ally)."""

    def __call__(self, player_id: int, prompt: str) -> dict[str, Any] | None:
        if player_id == 1:
            return {
                "propose_alliance_with": [],
                "accept_alliance_with": [],
                "declare_war_on": [2],
                "attack_priority": [],
            }
        return _idle_response()


class TestRunnerLevelBetrayalDetection:
    def test_when_scripted_war_breaks_pre_existing_alliance_then_grudge_recorded(self):
        """
        Pre-existing alliance P1–P2; scripted call_fn has P1 declare war on P2.
        After run_game, alliance must be broken and P2 must hold a grudge.
        """
        from src.agents.diplomacy import DiplomacyState
        from src.agents.random_agent import RandomAgent
        from src.agents.reputation import ReputationBook
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        state = DiplomacyState()
        state.form_alliance(1, 2)

        env = RisikoEnv(n_players=3)
        runner = DiplomacyRunner(
            env,
            [RandomAgent(seed=i) for i in range(3)],
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=True, n_rounds=1),
            state=state,
            reputation=ReputationBook(),
            call_fn=_ScriptedWarFn(),
        )
        runner.run_game(seed=42)

        assert not state.are_allied(1, 2), (
            "Alliance must be dissolved when P1 declares war on P2 in negotiation."
        )
        assert state.has_grudge(victim=2, aggressor=1), (
            "P2 must hold a grudge toward P1 after war declaration."
        )

    def test_when_ally_attacks_ally_in_mechanical_play_then_betrayal_is_recorded(self):
        """
        Mechanical betrayal (ally attacks ally via env action) in _after_step must
        also record a grudge and break the alliance even when negotiation is enabled.
        This test uses a pre-formed alliance and verifies the _after_step hook.
        """
        import numpy as np

        from src.agents.diplomacy import DiplomacyState
        from src.agents.reputation import ReputationBook
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        state = DiplomacyState()
        state.form_alliance(1, 2)

        # Simulate _after_step for an attack action from player 1 targeting
        # a territory owned by player 2. Use the runner's hook directly.
        env = RisikoEnv(n_players=3)
        runner = DiplomacyRunner(
            env,
            # agents don't matter for direct hook testing
            [object() for _ in range(3)],  # type: ignore[arg-type]
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=True, n_rounds=1),
            state=state,
            reputation=ReputationBook(),
            call_fn=_idle_call_fn,
        )
        # Minimal obs: territory 5 owned by player 2
        territory_owner = np.zeros(42, dtype=np.int32)
        territory_owner[5] = 2
        pre_obs = {"territory_owner": territory_owner}
        attack_action = {
            "action_type": 2,
            "param_a": 0,
            "param_b": 5,
            "param_c": 1,
            "param_d": 0,
        }
        runner._after_step(player=1, action=attack_action, pre_obs=pre_obs, new_obs={})

        assert not state.are_allied(1, 2), "Alliance broken by mechanical betrayal"
        assert state.has_grudge(victim=2, aggressor=1), (
            "Grudge must be recorded when an ally attacks another ally."
        )


# ---------------------------------------------------------------------------
# Criterion: disabled-equivalence regression guard
# ---------------------------------------------------------------------------


class TestDisabledEquivalenceRegression:
    """Ensures the disabled path remains byte-identical to MultiAgentRunner."""

    def test_when_disabled_then_action_log_identical_to_multi_agent_runner(self):
        """Regression: DiplomacyRunner(disabled) must not alter mechanical play."""
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner, MultiAgentRunner

        seed, max_turns, n = 42, 5, 3
        env_a = RisikoEnv(n_players=n)
        result_a = MultiAgentRunner(
            env_a, [RandomAgent(seed=i) for i in range(n)], max_turns=max_turns
        ).run_game(seed=seed)

        env_b = RisikoEnv(n_players=n)
        result_b = DiplomacyRunner(
            env_b,
            [RandomAgent(seed=i) for i in range(n)],
            max_turns=max_turns,
            diplomacy_cfg=DiplomacyConfig(enabled=False),
        ).run_game(seed=seed)

        assert result_a.action_log == result_b.action_log, (
            "DiplomacyRunner(disabled) must produce the same action_log as "
            "MultiAgentRunner under the same seed."
        )

    def test_when_enabled_call_fn_provided_then_disabled_still_zero_calls(self):
        """Even with a call_fn provided, disabled runner must not call it."""
        called: list[int] = []

        def spy_fn(player_id: int, prompt: str) -> dict | None:
            called.append(player_id)
            return _idle_response()

        runner = _make_runner(enabled=False, call_fn=spy_fn)
        runner.run_game(seed=0)
        assert runner.call_count == 0
        assert not called, "call_fn must not be invoked when diplomacy is disabled"
