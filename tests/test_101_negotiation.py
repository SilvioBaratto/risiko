"""Source-blind example tests for NegotiationOrchestrator (issue #101).

Derived solely from the acceptance criteria:
  - The learner (slot 0) is excluded from speakers and never speaks.
  - `call_count` equals the number of speaker calls and is bounded by
    `n_rounds * n_speakers`.
  - `call_fn` returning `None` => no-op, no exception.

API note: tests updated to match the actual NegotiationOrchestrator signature,
which takes (state, reputation, cfg, profiles, *, call_fn, learner_slot=0)
and exposes run_round(obs, speakers) -> int.  call_fn: (player_id, prompt) -> dict | None.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.diplomacy import DiplomacyState
from src.agents.negotiation import NegotiationOrchestrator
from src.agents.player_config import PlayerConfig
from src.agents.reputation import ReputationBook
from src.config import DiplomacyConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _profiles(n: int) -> list[PlayerConfig]:
    return [
        PlayerConfig(player_id=i, temperature=0.5, top_p=0.9, strategy_hint="test")
        for i in range(n)
    ]


def _make_orch(
    n_players: int = 6,
    call_fn=None,
    learner_slot: int = 0,
) -> NegotiationOrchestrator:
    if call_fn is None:
        call_fn = lambda _pid, _prompt: None  # noqa: E731
    return NegotiationOrchestrator(
        state=DiplomacyState(),
        reputation=ReputationBook(),
        cfg=DiplomacyConfig(n_rounds=2, max_message_tokens=64),
        profiles=_profiles(n_players),
        call_fn=call_fn,
        learner_slot=learner_slot,
    )


# ---------------------------------------------------------------------------
# Criterion: learner (slot 0) is excluded from speakers and never speaks
# ---------------------------------------------------------------------------


class TestLearnerNeverSpeaks:
    """Slot 0 must never appear as the player_id argument passed to call_fn."""

    def test_when_run_round_then_learner_slot_0_never_speaks(self) -> None:
        """call_fn is never invoked with player_id == 0."""
        spoken_by: list[int] = []

        def tracking_fn(player_id: int, prompt: str):
            spoken_by.append(player_id)
            return None

        orch = _make_orch(call_fn=tracking_fn)
        orch.run_round(obs={}, speakers=list(range(6)))

        assert 0 not in spoken_by, f"Learner slot 0 must not speak; spoke: {spoken_by}"

    def test_when_run_round_with_one_speaker_list_then_learner_slot_0_never_speaks(
        self,
    ) -> None:
        """Learner exclusion holds for a 4-player game."""
        spoken_by: list[int] = []

        def tracking_fn(player_id: int, prompt: str):
            spoken_by.append(player_id)
            return None

        orch = _make_orch(n_players=4, call_fn=tracking_fn)
        orch.run_round(obs={}, speakers=list(range(4)))

        assert 0 not in spoken_by


# ---------------------------------------------------------------------------
# Criterion: call_count equals calls and is bounded by n_rounds * n_speakers
# ---------------------------------------------------------------------------


class TestCallCount:
    """call_count tracks exactly the number of call_fn invocations."""

    def test_when_run_round_then_call_count_equals_actual_calls(self) -> None:
        """call_count matches the number of times call_fn was invoked."""
        actual: list[int] = []

        def counting_fn(player_id: int, prompt: str):
            actual.append(player_id)
            return None

        n_players = 6
        orch = _make_orch(n_players=n_players, call_fn=counting_fn)
        orch.run_round(obs={}, speakers=list(range(n_players)))

        assert orch.call_count == len(actual)

    def test_when_two_rounds_then_call_count_bounded_by_n_rounds_times_n_speakers(
        self,
    ) -> None:
        """After 2 rounds: call_count <= 2 * n_speakers (n_speakers = n_players - 1)."""
        n_rounds, n_players = 2, 6
        orch = _make_orch(n_players=n_players)

        for _ in range(n_rounds):
            orch.run_round(obs={}, speakers=list(range(n_players)))

        n_speakers = n_players - 1
        assert orch.call_count <= n_rounds * n_speakers

    def test_when_two_rounds_then_call_count_equals_n_rounds_times_n_speakers(
        self,
    ) -> None:
        """Exact tracking: call_count == 2 * 5 == 10 for 2 rounds × 5 non-learner speakers."""
        n_rounds, n_players = 2, 6
        orch = _make_orch(n_players=n_players)

        for _ in range(n_rounds):
            orch.run_round(obs={}, speakers=list(range(n_players)))

        n_speakers = n_players - 1
        assert orch.call_count == n_rounds * n_speakers


# ---------------------------------------------------------------------------
# Criterion: call_fn returning None => no-op, no exception
# ---------------------------------------------------------------------------


class TestNullCallFn:
    """call_fn returning None must not raise and must leave state unchanged."""

    def test_when_call_fn_returns_none_then_no_exception_is_raised(self) -> None:
        """run_round() completes without raising when call_fn returns None."""
        orch = _make_orch(call_fn=lambda _pid, _prompt: None)
        orch.run_round(obs={}, speakers=list(range(6)))  # must not raise

    def test_when_call_fn_returns_none_then_no_alliance_is_formed(self) -> None:
        """No alliance is formed when call_fn always returns None."""
        state = DiplomacyState()
        orch = NegotiationOrchestrator(
            state=state,
            reputation=ReputationBook(),
            cfg=DiplomacyConfig(n_rounds=2),
            profiles=_profiles(6),
            call_fn=lambda _pid, _prompt: None,
        )
        orch.run_round(obs={}, speakers=list(range(6)))

        for a in range(6):
            for b in range(6):
                if a != b:
                    assert not state.are_allied(a, b)

    def test_when_call_fn_returns_none_then_no_grudge_is_recorded(self) -> None:
        """No grudge is recorded when call_fn always returns None."""
        state = DiplomacyState()
        orch = NegotiationOrchestrator(
            state=state,
            reputation=ReputationBook(),
            cfg=DiplomacyConfig(n_rounds=2),
            profiles=_profiles(6),
            call_fn=lambda _pid, _prompt: None,
        )
        orch.run_round(obs={}, speakers=list(range(6)))

        for victim in range(6):
            for aggressor in range(6):
                if victim != aggressor:
                    assert not state.has_grudge(victim=victim, aggressor=aggressor)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@settings(max_examples=40)
@given(
    n_rounds=st.integers(min_value=1, max_value=10),
    n_players=st.integers(min_value=2, max_value=6),
)
def test_when_call_fn_returns_none_for_any_config_then_no_exception_is_raised(
    n_rounds: int,
    n_players: int,
) -> None:
    """Invariant: None-returning call_fn never raises for any valid (n_rounds, n_players)."""
    orch = _make_orch(n_players=n_players, call_fn=lambda _pid, _prompt: None)
    for _ in range(n_rounds):
        orch.run_round(obs={}, speakers=list(range(n_players)))


@settings(max_examples=40)
@given(
    n_rounds=st.integers(min_value=1, max_value=10),
    n_players=st.integers(min_value=2, max_value=6),
)
def test_when_run_with_any_config_then_call_count_bounded_by_n_rounds_times_n_speakers(
    n_rounds: int,
    n_players: int,
) -> None:
    """Invariant: call_count <= n_rounds * n_speakers for ALL valid inputs.

    Also verifies call_count tracks actual invocations exactly.
    """
    actual: list[int] = []

    def counting_fn(player_id: int, prompt: str):
        actual.append(player_id)
        return None

    orch = _make_orch(n_players=n_players, call_fn=counting_fn)
    for _ in range(n_rounds):
        orch.run_round(obs={}, speakers=list(range(n_players)))

    n_speakers = n_players - 1
    assert orch.call_count == len(actual), "call_count must equal actual invocation count"
    assert orch.call_count <= n_rounds * n_speakers, (
        f"call_count={orch.call_count} exceeds bound {n_rounds * n_speakers}"
    )


@settings(max_examples=40)
@given(
    n_rounds=st.integers(min_value=1, max_value=10),
    n_players=st.integers(min_value=2, max_value=6),
)
def test_when_run_with_any_config_then_learner_slot_0_never_speaks(
    n_rounds: int,
    n_players: int,
) -> None:
    """Invariant: learner slot 0 is NEVER passed to call_fn for any valid input."""
    spoken_by: list[int] = []

    def tracking_fn(player_id: int, prompt: str):
        spoken_by.append(player_id)
        return None

    orch = _make_orch(n_players=n_players, call_fn=tracking_fn)
    for _ in range(n_rounds):
        orch.run_round(obs={}, speakers=list(range(n_players)))

    assert 0 not in spoken_by, (
        f"Learner slot 0 must never speak (n_rounds={n_rounds}, "
        f"n_players={n_players}); spoken_by={spoken_by}"
    )
