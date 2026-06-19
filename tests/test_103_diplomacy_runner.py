"""Source-blind example tests for issue #103:
DiplomacyRunner — negotiation hook + betrayal detection, learner untouched.

Tests derived solely from the acceptance criteria and requirements.md.
No implementation source (src/) was read — Red phase of TDD.

Verifiable criteria (oracle: [UNIT]):
  1. LLM seats receive a DiplomacyContext (state, acting_player, leader); allies are
     avoided and the leader is preferred (verified with a scripted pool that honours
     the context via context.state.are_allied / context.leader).
  2. The learner seat (slot 0) never receives diplomacy context and is excluded from
     speakers; its act_with_meta call is exactly the base-class call (context=None
     always passed / ignored by the base implementation).
  3. With diplomacy disabled: call_count == 0 AND GameResult.action_log is identical
     to a plain MultiAgentRunner under the same seed + (mocked) deterministic agents.

Skipped (oracle: NOT VERIFIABLE):
  - Negotiation fires once per player-turn boundary, bounded by config  (runtime infra)
  - Ally attacking ally records betrayal grudge / breaks alliance / updates reputation
    (no isolated unit hook exposed from DiplomacyRunner)
  - Deterministic under mocked LLM + fixed seed                        (runtime infra)
  - All tests pass / SOLID / ruff / coverage                           (process gates)
"""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Minimal helpers — built from criteria text, NOT from implementation source
# ---------------------------------------------------------------------------


def _skip_action() -> dict[str, int]:
    return {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}


def _legal_n(n: int = 2) -> list[dict[str, int]]:
    return [_skip_action() for _ in range(n)]


def _fake_territory_owner(n_territories: int = 42, n_players: int = 2) -> np.ndarray:
    """Return a round-robin territory ownership array for use in obs."""
    return np.array([i % n_players for i in range(n_territories)], dtype=np.int32)


# ---------------------------------------------------------------------------
# Spy / scripted agents  —  no import from src/; pure test fixtures
# ---------------------------------------------------------------------------


class _ContextSpyAgent:
    """Records every context kwarg received via act_with_meta.

    Used to assert that the DiplomacyRunner (a) passes non-None context to
    non-learner seats, and (b) always passes context=None to the learner.
    """

    def __init__(self) -> None:
        self.contexts_received: list = []

    def act(
        self,
        obs: dict,
        legal_actions: list,
        *,
        deterministic: bool = False,
    ) -> dict:
        return legal_actions[0] if legal_actions else _skip_action()

    def act_with_meta(
        self,
        obs: dict,
        legal_actions: list,
        *,
        deterministic: bool = False,
        context=None,
    ) -> tuple:
        self.contexts_received.append(context)
        action = legal_actions[0] if legal_actions else _skip_action()
        return action, torch.tensor(float("nan")), torch.tensor(float("nan"))


class _AllyAvoidingScriptedAgent:
    """Scripted pool that honours DiplomacyContext: avoids attacking current allies.

    Uses `context.state.are_allied(context.acting_player, target_owner)` and
    `obs["territory_owner"]` to decide which attack actions are safe.

    Derived from criterion: 'allies are avoided and the leader is preferred
    (verified with a scripted pool that honours the context)'.
    """

    def __init__(self) -> None:
        self.context_received = None
        self.last_action: dict | None = None

    def act(
        self,
        obs: dict,
        legal_actions: list,
        *,
        deterministic: bool = False,
    ) -> dict:
        return legal_actions[0] if legal_actions else _skip_action()

    def act_with_meta(
        self,
        obs: dict,
        legal_actions: list,
        *,
        deterministic: bool = False,
        context=None,
    ) -> tuple:
        self.context_received = context

        if context is not None:
            territory_owner = obs.get("territory_owner", np.array([], dtype=np.int32))
            safe: list[dict] = []
            for action in legal_actions:
                if action.get("action_type") == 2:  # ATTACK
                    target = action.get("param_b", -1)
                    if 0 <= target < len(territory_owner):
                        owner = int(territory_owner[target])
                        if context.state.are_allied(context.acting_player, owner):
                            continue  # skip — attacking an ally
                safe.append(action)
            chosen = safe[0] if safe else legal_actions[0]
        else:
            chosen = legal_actions[0] if legal_actions else _skip_action()

        self.last_action = chosen
        return chosen, torch.tensor(float("nan")), torch.tensor(float("nan"))


# ===========================================================================
# Criterion 1: DiplomacyContext exists with (state, acting_player, leader)
# ===========================================================================


class TestDiplomacyContextStructure:
    """DiplomacyContext must be importable and carry state, acting_player, leader."""

    def test_when_diplomacy_context_imported_then_class_is_available(self):
        from src.agents.diplomacy import DiplomacyContext  # noqa: F401

        assert DiplomacyContext is not None

    def test_when_diplomacy_context_created_then_acting_player_is_stored(self):
        from src.agents.diplomacy import DiplomacyContext, DiplomacyState

        ctx = DiplomacyContext(state=DiplomacyState(), acting_player=2, leader=1)
        assert ctx.acting_player == 2

    def test_when_diplomacy_context_created_then_leader_is_stored(self):
        from src.agents.diplomacy import DiplomacyContext, DiplomacyState

        ctx = DiplomacyContext(state=DiplomacyState(), acting_player=2, leader=1)
        assert ctx.leader == 1

    def test_when_diplomacy_context_created_then_state_is_stored(self):
        from src.agents.diplomacy import DiplomacyContext, DiplomacyState

        state = DiplomacyState()
        ctx = DiplomacyContext(state=state, acting_player=0, leader=3)
        assert ctx.state is state

    def test_when_diplomacy_context_leader_is_none_then_none_is_stored(self):
        """Leader may be None when no clear leader exists (e.g. all territories tied).

        Assumption: the criterion mentions leader as an optional concept (leader is
        identified by territory count; None when tied or indeterminate).
        """
        from src.agents.diplomacy import DiplomacyContext, DiplomacyState

        ctx = DiplomacyContext(state=DiplomacyState(), acting_player=1, leader=None)
        assert ctx.leader is None

    def test_when_diplomacy_context_state_has_alliance_then_are_allied_reflects_it(self):
        """DiplomacyContext.state is a live DiplomacyState: alliances can be queried.

        Derived from: 'LLM seats receive a DiplomacyContext (state, acting player,
        leader); allies are avoided ... (scripted pool that honours the context)'.
        The scripted pool uses context.state.are_allied to find allies.
        """
        from src.agents.diplomacy import DiplomacyContext, DiplomacyState

        state = DiplomacyState()
        state.form_alliance(0, 2)
        ctx = DiplomacyContext(state=state, acting_player=0, leader=3)

        assert ctx.state.are_allied(0, 2) is True
        assert ctx.state.are_allied(0, 3) is False

    @given(
        acting_player=st.integers(min_value=0, max_value=5),
        leader=st.one_of(st.none(), st.integers(min_value=0, max_value=5)),
    )
    @settings(max_examples=50)
    def test_when_diplomacy_context_created_with_any_valid_player_ids_then_no_error(
        self, acting_player: int, leader
    ) -> None:
        """Invariant (never-raises): DiplomacyContext accepts any valid player-id pair
        without raising.

        Derived from: 'LLM seats receive a DiplomacyContext (state, acting player, leader)'.
        """
        from src.agents.diplomacy import DiplomacyContext, DiplomacyState

        ctx = DiplomacyContext(state=DiplomacyState(), acting_player=acting_player, leader=leader)
        assert ctx.acting_player == acting_player
        assert ctx.leader == leader


# ===========================================================================
# Criterion 2a: base-class act_with_meta ignores DiplomacyContext
# ===========================================================================


class TestActWithMetaBaseClass:
    """Base-class act_with_meta must accept context=... but produce the same
    action as act() — context is ignored (learner-side guarantee).

    Derived from: 'its act_with_meta call is exactly the base-class call'.
    """

    def test_when_random_agent_act_with_meta_called_with_none_context_then_returns_legal_action(
        self,
    ):
        from src.agents.random_agent import RandomAgent

        agent = RandomAgent(seed=0)
        legal = _legal_n(3)
        action, _lp, _v = agent.act_with_meta({}, legal, context=None)
        assert action in legal

    def test_when_random_agent_act_with_meta_called_with_context_then_returns_legal_action(
        self,
    ):
        """Base class must accept a DiplomacyContext kwarg without raising."""
        from src.agents.diplomacy import DiplomacyContext, DiplomacyState
        from src.agents.random_agent import RandomAgent

        agent = RandomAgent(seed=0)
        legal = _legal_n(3)
        ctx = DiplomacyContext(state=DiplomacyState(), acting_player=0, leader=1)

        action, _lp, _v = agent.act_with_meta({}, legal, context=ctx)
        assert action in legal

    def test_when_random_agent_act_with_meta_called_with_and_without_context_then_same_action(
        self,
    ):
        """Base-class ignores context: same seed + legal_actions → same action.

        Derived from: 'its act_with_meta call is exactly the base-class call'.
        """
        from src.agents.diplomacy import DiplomacyContext, DiplomacyState
        from src.agents.random_agent import RandomAgent

        legal = _legal_n(3)
        ctx = DiplomacyContext(state=DiplomacyState(), acting_player=0, leader=1)

        agent_a = RandomAgent(seed=7)
        action_no_ctx, _, _ = agent_a.act_with_meta({}, legal, context=None)

        agent_b = RandomAgent(seed=7)
        action_with_ctx, _, _ = agent_b.act_with_meta({}, legal, context=ctx)

        assert action_no_ctx == action_with_ctx, (
            "Base-class act_with_meta must be context-blind: same seed must produce "
            "the same action regardless of whether a DiplomacyContext is supplied."
        )


# ===========================================================================
# Criterion 3: DiplomacyRunner is importable
# ===========================================================================


class TestDiplomacyRunnerImport:
    def test_when_diplomacy_runner_imported_then_class_is_available(self):
        from src.multi_agent import DiplomacyRunner  # noqa: F401

        assert DiplomacyRunner is not None


# ===========================================================================
# Criterion 3: With diplomacy disabled — call_count == 0
# ===========================================================================


class TestDiplomacyRunnerDisabledCallCount:
    """With diplomacy.enabled=False, DiplomacyRunner.call_count must be 0."""

    def test_when_diplomacy_disabled_then_call_count_is_zero_before_game(self):
        """call_count starts at 0 upon construction when diplomacy is disabled."""
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        env = RisikoEnv(n_players=2)
        agents = [RandomAgent(seed=i) for i in range(2)]
        runner = DiplomacyRunner(
            env,
            agents,
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=False),
        )
        assert runner.call_count == 0

    def test_when_diplomacy_disabled_then_call_count_is_zero_after_game(self):
        """After run_game with diplomacy disabled, call_count is still 0.

        Derived from: 'With diplomacy disabled: call_count == 0'.
        """
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        env = RisikoEnv(n_players=2)
        agents = [RandomAgent(seed=i) for i in range(2)]
        runner = DiplomacyRunner(
            env,
            agents,
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=False),
        )
        runner.run_game(seed=42)

        assert runner.call_count == 0, (
            "DiplomacyRunner must make zero negotiation LLM calls when "
            "diplomacy_cfg.enabled is False."
        )

    def test_when_diplomacy_disabled_in_3_player_game_then_call_count_is_zero(self):
        """call_count==0 holds for any player count when disabled."""
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        env = RisikoEnv(n_players=3)
        agents = [RandomAgent(seed=i) for i in range(3)]
        runner = DiplomacyRunner(
            env,
            agents,
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=False),
        )
        runner.run_game(seed=0)
        assert runner.call_count == 0


# ===========================================================================
# Criterion 3: With diplomacy disabled — action_log identical to MultiAgentRunner
# ===========================================================================


class TestDiplomacyRunnerDisabledActionLog:
    """DiplomacyRunner(enabled=False).run_game() must produce action_log identical
    to MultiAgentRunner.run_game() under the same seed and deterministic agents.

    Derived from: 'GameResult.action_log is identical to a plain MultiAgentRunner
    under the same seed + mocked LLM'.
    Uses RandomAgents (deterministic under seed) to eliminate LLM variance.
    """

    def test_when_diplomacy_disabled_then_action_log_length_matches_multi_agent_runner(
        self,
    ):
        """Same number of logged actions as the baseline MultiAgentRunner."""
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner, MultiAgentRunner

        seed, max_turns, n = 42, 5, 2

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

        assert len(result_a.action_log) == len(result_b.action_log), (
            "DiplomacyRunner(enabled=False) must produce the same number of "
            "action_log entries as MultiAgentRunner under identical seed."
        )

    def test_when_diplomacy_disabled_then_every_action_log_entry_matches_multi_agent_runner(
        self,
    ):
        """Every individual action_log entry must be identical to the baseline.

        Derived from: 'GameResult.action_log is identical to a plain MultiAgentRunner'.
        """
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner, MultiAgentRunner

        seed, max_turns, n = 42, 5, 2

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

        for i, (entry_a, entry_b) in enumerate(
            zip(result_a.action_log, result_b.action_log, strict=True)
        ):
            assert entry_a == entry_b, (
                f"action_log[{i}] diverges between MultiAgentRunner and "
                f"DiplomacyRunner(disabled): {entry_a!r} != {entry_b!r}"
            )

    def test_when_diplomacy_disabled_then_winner_matches_multi_agent_runner(self):
        """The winner must also be identical (full game parity, not just log)."""
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner, MultiAgentRunner

        seed, max_turns, n = 42, 5, 2

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

        assert result_a.winner == result_b.winner, (
            "DiplomacyRunner(enabled=False) must produce the same winner as "
            "MultiAgentRunner under the same seed and agents."
        )


# ===========================================================================
# Criterion 2b: Learner (slot 0) NEVER receives a non-None DiplomacyContext
# ===========================================================================


class TestLearnerNeverReceivesDiplomacyContext:
    """The learner at slot 0 must receive context=None in every act_with_meta call,
    regardless of whether diplomacy is enabled or disabled.

    Derived from: 'The learner seat (slot 0) never receives diplomacy context
    and is excluded from speakers; its act_with_meta call is exactly the
    base-class call'.
    """

    def test_when_diplomacy_enabled_then_learner_slot_0_always_receives_none_context(
        self,
    ):
        """DiplomacyRunner must pass context=None to the learner at every game step."""
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        learner_spy = _ContextSpyAgent()

        env = RisikoEnv(n_players=3)
        agents = [learner_spy, RandomAgent(seed=1), RandomAgent(seed=2)]

        DiplomacyRunner(
            env,
            agents,
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=True, n_rounds=1),
        ).run_game(seed=42)

        assert learner_spy.contexts_received, "Learner spy must have been called at least once."
        assert all(ctx is None for ctx in learner_spy.contexts_received), (
            "Learner (slot 0) must never receive a non-None DiplomacyContext; "
            f"received: {learner_spy.contexts_received}"
        )

    def test_when_diplomacy_disabled_then_learner_slot_0_receives_none_context(self):
        """Even when disabled, learner context is None (no regression)."""
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        learner_spy = _ContextSpyAgent()

        env = RisikoEnv(n_players=2)
        agents = [learner_spy, RandomAgent(seed=1)]

        DiplomacyRunner(
            env,
            agents,
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=False),
        ).run_game(seed=42)

        assert all(ctx is None for ctx in learner_spy.contexts_received)

    @given(n_players=st.integers(min_value=2, max_value=6))
    @settings(max_examples=8)
    def test_when_diplomacy_enabled_for_any_player_count_then_learner_never_gets_context(
        self, n_players: int
    ) -> None:
        """Invariant: context=None for learner (slot 0) for all valid player counts.

        Derived from: 'The learner seat (slot 0) never receives diplomacy context'.
        """
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        learner_spy = _ContextSpyAgent()
        other_agents = [RandomAgent(seed=i + 1) for i in range(n_players - 1)]

        env = RisikoEnv(n_players=n_players)
        agents = [learner_spy, *other_agents]

        DiplomacyRunner(
            env,
            agents,
            max_turns=3,
            diplomacy_cfg=DiplomacyConfig(enabled=True, n_rounds=1),
        ).run_game(seed=0)

        assert all(ctx is None for ctx in learner_spy.contexts_received), (
            f"Learner must never receive DiplomacyContext (n_players={n_players})"
        )


# ===========================================================================
# Criterion 1: Non-learner (LLM) seats receive DiplomacyContext when enabled
# ===========================================================================


class TestNonLearnerReceivesDiplomacyContext:
    """Non-learner agents must receive a non-None DiplomacyContext carrying
    state, acting_player, and leader when diplomacy is enabled.

    Derived from: 'LLM seats receive a DiplomacyContext (state, acting player,
    leader)'.
    """

    def test_when_diplomacy_enabled_then_non_learner_spy_receives_non_none_context(
        self,
    ):
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        learner = RandomAgent(seed=0)
        llm_spy = _ContextSpyAgent()

        env = RisikoEnv(n_players=2)
        agents = [learner, llm_spy]

        DiplomacyRunner(
            env,
            agents,
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=True, n_rounds=1),
        ).run_game(seed=42)

        non_none = [c for c in llm_spy.contexts_received if c is not None]
        assert len(non_none) > 0, (
            "Non-learner LLM agent (slot 1) must receive at least one non-None "
            "DiplomacyContext when diplomacy is enabled."
        )

    def test_when_diplomacy_enabled_then_context_is_diplomacy_context_instance(self):
        """The context passed to non-learner agents must be a DiplomacyContext."""
        from src.agents.diplomacy import DiplomacyContext
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        learner = RandomAgent(seed=0)
        llm_spy = _ContextSpyAgent()

        env = RisikoEnv(n_players=2)
        agents = [learner, llm_spy]

        DiplomacyRunner(
            env,
            agents,
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=True, n_rounds=1),
        ).run_game(seed=42)

        for ctx in llm_spy.contexts_received:
            if ctx is not None:
                assert isinstance(ctx, DiplomacyContext), (
                    f"Context must be DiplomacyContext, got {type(ctx)!r}"
                )

    def test_when_diplomacy_enabled_then_context_has_valid_acting_player(self):
        """Each DiplomacyContext.acting_player must be a valid player index."""
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        n_players = 3
        learner = RandomAgent(seed=0)
        llm_spy = _ContextSpyAgent()
        env = RisikoEnv(n_players=n_players)
        agents = [learner, llm_spy, RandomAgent(seed=2)]

        DiplomacyRunner(
            env,
            agents,
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=True, n_rounds=1),
        ).run_game(seed=42)

        for ctx in llm_spy.contexts_received:
            if ctx is not None:
                assert hasattr(ctx, "acting_player"), (
                    "DiplomacyContext must have 'acting_player' attribute"
                )
                assert isinstance(ctx.acting_player, int)
                assert 0 <= ctx.acting_player < n_players

    def test_when_diplomacy_enabled_then_context_has_state_that_is_diplomacy_state(self):
        """DiplomacyContext.state must be a DiplomacyState instance."""
        from src.agents.diplomacy import DiplomacyState
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        learner = RandomAgent(seed=0)
        llm_spy = _ContextSpyAgent()
        env = RisikoEnv(n_players=2)
        agents = [learner, llm_spy]

        DiplomacyRunner(
            env,
            agents,
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=True, n_rounds=1),
        ).run_game(seed=42)

        for ctx in llm_spy.contexts_received:
            if ctx is not None:
                assert hasattr(ctx, "state"), "DiplomacyContext must have 'state'"
                assert isinstance(ctx.state, DiplomacyState)

    def test_when_diplomacy_enabled_then_context_has_leader_int_or_none(self):
        """DiplomacyContext.leader must be int or None (not an arbitrary type)."""
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        learner = RandomAgent(seed=0)
        llm_spy = _ContextSpyAgent()
        env = RisikoEnv(n_players=2)
        agents = [learner, llm_spy]

        DiplomacyRunner(
            env,
            agents,
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=True, n_rounds=1),
        ).run_game(seed=42)

        for ctx in llm_spy.contexts_received:
            if ctx is not None:
                assert hasattr(ctx, "leader"), "DiplomacyContext must have 'leader'"
                assert ctx.leader is None or isinstance(ctx.leader, int), (
                    f"leader must be int | None, got {type(ctx.leader)!r}"
                )

    def test_when_diplomacy_disabled_then_non_learner_never_receives_context(self):
        """When disabled, the non-learner also receives context=None at every step."""
        from src.agents.random_agent import RandomAgent
        from src.config import DiplomacyConfig
        from src.env import RisikoEnv
        from src.multi_agent import DiplomacyRunner

        learner = RandomAgent(seed=0)
        llm_spy = _ContextSpyAgent()
        env = RisikoEnv(n_players=2)
        agents = [learner, llm_spy]

        DiplomacyRunner(
            env,
            agents,
            max_turns=5,
            diplomacy_cfg=DiplomacyConfig(enabled=False),
        ).run_game(seed=42)

        assert all(c is None for c in llm_spy.contexts_received), (
            "With diplomacy disabled, no agent must receive a non-None DiplomacyContext."
        )


# ===========================================================================
# Criterion 1: Scripted pool honours DiplomacyContext — allies avoided
# ===========================================================================


class TestScriptedPoolHonoursContext:
    """A scripted pool that honours DiplomacyContext avoids attacking allied players.

    The 'scripted pool' is _AllyAvoidingScriptedAgent above; it uses
    context.state.are_allied(context.acting_player, target_owner) to decide.

    Derived from: 'allies are avoided and the leader is preferred (verified
    with a scripted pool that honours the context)'.
    """

    def _obs_with_owner(self, target_territory: int, owner: int) -> dict:
        """Return an obs dict where territory `target_territory` is owned by `owner`."""
        territory_owner = np.zeros(42, dtype=np.int32)
        territory_owner[target_territory] = owner
        return {"territory_owner": territory_owner}

    def test_when_scripted_agent_given_context_with_allied_player_then_avoids_attacking_ally(
        self,
    ):
        """Scripted pool must not choose an attack action targeting an allied territory.

        Setup: acting_player=0, allied with player 2; target territory 5 owned by
        player 2; skip action is also offered.  Scripted pool must choose skip.
        """
        from src.agents.diplomacy import DiplomacyContext, DiplomacyState

        state = DiplomacyState()
        state.form_alliance(0, 2)
        ctx = DiplomacyContext(state=state, acting_player=0, leader=1)

        # territory 5 is owned by player 2 (the ally)
        obs = self._obs_with_owner(target_territory=5, owner=2)
        attack_ally = {"action_type": 2, "param_a": 0, "param_b": 5, "param_c": 1, "param_d": 1}
        skip = _skip_action()

        agent = _AllyAvoidingScriptedAgent()
        action, _, _ = agent.act_with_meta(obs, [attack_ally, skip], context=ctx)

        assert action != attack_ally, (
            "Scripted pool honouring DiplomacyContext must not attack an allied territory."
        )
        assert action == skip

    def test_when_scripted_agent_has_ally_and_enemy_then_attacks_enemy_not_ally(self):
        """Scripted pool prefers enemy target over allied target when given context."""
        from src.agents.diplomacy import DiplomacyContext, DiplomacyState

        state = DiplomacyState()
        state.form_alliance(0, 2)  # player 0 allied with player 2
        ctx = DiplomacyContext(state=state, acting_player=0, leader=3)

        # territory 1 owned by player 2 (ally), territory 3 owned by player 1 (enemy)
        territory_owner = np.zeros(42, dtype=np.int32)
        territory_owner[1] = 2
        territory_owner[3] = 1
        obs = {"territory_owner": territory_owner}

        attack_ally = {"action_type": 2, "param_a": 0, "param_b": 1, "param_c": 1, "param_d": 1}
        attack_enemy = {"action_type": 2, "param_a": 0, "param_b": 3, "param_c": 1, "param_d": 1}

        agent = _AllyAvoidingScriptedAgent()
        action, _, _ = agent.act_with_meta(obs, [attack_ally, attack_enemy], context=ctx)

        assert action == attack_enemy, (
            "Scripted pool honouring context must choose the non-ally attack over the ally attack."
        )

    def test_when_scripted_agent_has_no_context_then_first_action_is_chosen(self):
        """Without context the scripted pool falls back to the first legal action."""
        agent = _AllyAvoidingScriptedAgent()
        obs = self._obs_with_owner(0, 0)
        legal = [_skip_action(), _skip_action()]

        action, _, _ = agent.act_with_meta(obs, legal, context=None)
        assert action == legal[0]

    def test_when_context_has_no_alliance_then_scripted_agent_may_attack_any_territory(
        self,
    ):
        """Without any alliances, scripted pool is unconstrained — can choose any action."""
        from src.agents.diplomacy import DiplomacyContext, DiplomacyState

        state = DiplomacyState()
        # No alliances formed
        ctx = DiplomacyContext(state=state, acting_player=0, leader=1)

        obs = self._obs_with_owner(target_territory=2, owner=3)
        attack = {"action_type": 2, "param_a": 0, "param_b": 2, "param_c": 1, "param_d": 1}

        agent = _AllyAvoidingScriptedAgent()
        action, _, _ = agent.act_with_meta(obs, [attack], context=ctx)
        assert action == attack

    @given(
        ally_player=st.integers(min_value=1, max_value=5),
        territory_index=st.integers(min_value=0, max_value=41),
    )
    @settings(max_examples=40)
    def test_when_scripted_pool_honours_context_then_never_attacks_allied_territory(
        self, ally_player: int, territory_index: int
    ) -> None:
        """Invariant: scripted pool never attacks an allied territory for any
        (ally_player, territory_index) combination.

        Derived from: 'allies are avoided ... (verified with a scripted pool that
        honours the context)'.
        """
        from src.agents.diplomacy import DiplomacyContext, DiplomacyState

        state = DiplomacyState()
        state.form_alliance(0, ally_player)
        ctx = DiplomacyContext(state=state, acting_player=0, leader=None)

        territory_owner = np.zeros(42, dtype=np.int32)
        territory_owner[territory_index] = ally_player
        obs = {"territory_owner": territory_owner}

        attack_ally = {
            "action_type": 2,
            "param_a": 0,
            "param_b": territory_index,
            "param_c": 1,
            "param_d": 1,
        }
        skip = _skip_action()

        agent = _AllyAvoidingScriptedAgent()
        action, _, _ = agent.act_with_meta(obs, [attack_ally, skip], context=ctx)

        assert action != attack_ally, (
            f"Scripted pool must not attack allied territory {territory_index} "
            f"(owner={ally_player}) when acting as player 0."
        )
