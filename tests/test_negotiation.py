"""Tests for NegotiationOrchestrator and NegotiationMessage (issue #101).

Covers all acceptance criteria with mocked call_fn (no HTTP, no Ollama).
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.diplomacy import DiplomacyState
from src.agents.negotiation import (
    LEARNER_SLOT,
    NegotiationMessage,
    NegotiationOrchestrator,
    _apply_war_declarations,
    _form_mutual_alliances,
)
from src.agents.player_config import PlayerConfig
from src.agents.reputation import ReputationBook
from src.config import DiplomacyConfig

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _profiles(n: int) -> list[PlayerConfig]:
    return [
        PlayerConfig(player_id=i, temperature=0.5, top_p=0.9, strategy_hint="test")
        for i in range(n)
    ]


def _make_orch(
    state: DiplomacyState | None = None,
    n_players: int = 6,
    call_fn=None,
    learner_slot: int = LEARNER_SLOT,
    max_message_tokens: int = 64,
) -> NegotiationOrchestrator:
    if state is None:
        state = DiplomacyState()
    if call_fn is None:
        call_fn = lambda _pid, _prompt: None  # noqa: E731
    return NegotiationOrchestrator(
        state=state,
        reputation=ReputationBook(),
        cfg=DiplomacyConfig(n_rounds=1, max_message_tokens=max_message_tokens),
        profiles=_profiles(n_players),
        call_fn=call_fn,
        learner_slot=learner_slot,
    )


def _msg(
    speaker: int,
    *,
    propose: tuple[int, ...] = (),
    accept: tuple[int, ...] = (),
    war: tuple[int, ...] = (),
) -> NegotiationMessage:
    return NegotiationMessage(
        speaker=speaker,
        propose_alliance_with=propose,
        accept_alliance_with=accept,
        declare_war_on=war,
        attack_priority=(),
    )


# ===========================================================================
# Criterion: schema validation — drop out-of-range/self ids, dedupe lists
# ===========================================================================


class TestSchemaValidation:
    """NegotiationMessage.from_raw validates and sanitizes raw LLM output."""

    def test_when_raw_contains_out_of_range_id_then_it_is_dropped(self) -> None:
        raw = {"propose_alliance_with": [1, 99, -1, 2]}
        msg = NegotiationMessage.from_raw(speaker=0, raw=raw, n_players=6)
        assert 99 not in msg.propose_alliance_with
        assert -1 not in msg.propose_alliance_with
        assert 1 in msg.propose_alliance_with
        assert 2 in msg.propose_alliance_with

    def test_when_raw_contains_self_reference_then_it_is_dropped(self) -> None:
        raw = {"propose_alliance_with": [0, 1, 2]}  # speaker=0 self-ref
        msg = NegotiationMessage.from_raw(speaker=0, raw=raw, n_players=6)
        assert 0 not in msg.propose_alliance_with

    def test_when_raw_contains_duplicate_ids_then_they_are_deduped(self) -> None:
        raw = {"propose_alliance_with": [1, 1, 2, 2, 2]}
        msg = NegotiationMessage.from_raw(speaker=0, raw=raw, n_players=6)
        assert list(msg.propose_alliance_with).count(1) == 1
        assert list(msg.propose_alliance_with).count(2) == 1

    def test_when_raw_field_is_missing_then_result_is_empty_tuple(self) -> None:
        msg = NegotiationMessage.from_raw(speaker=1, raw={}, n_players=6)
        assert msg.propose_alliance_with == ()
        assert msg.accept_alliance_with == ()
        assert msg.declare_war_on == ()
        assert msg.attack_priority == ()

    def test_when_raw_field_contains_non_integer_then_it_is_dropped(self) -> None:
        raw = {"propose_alliance_with": ["1", None, 2.0, 3]}
        msg = NegotiationMessage.from_raw(speaker=0, raw=raw, n_players=6)
        assert msg.propose_alliance_with == (3,)

    @given(
        speaker=st.integers(min_value=0, max_value=5),
        ids=st.lists(st.integers(min_value=-5, max_value=10), max_size=15),
    )
    @settings(max_examples=100)
    def test_when_raw_has_any_ids_then_all_validated_ids_are_in_range_and_not_self(
        self, speaker: int, ids: list[int]
    ) -> None:
        """Invariant: from_raw always produces ids in [0, n_players) and != speaker."""
        raw = {"propose_alliance_with": ids}
        msg = NegotiationMessage.from_raw(speaker=speaker, raw=raw, n_players=6)
        for pid in msg.propose_alliance_with:
            assert 0 <= pid < 6
            assert pid != speaker

    @given(ids=st.lists(st.integers(min_value=0, max_value=5), max_size=20))
    @settings(max_examples=60)
    def test_when_raw_has_any_ids_then_result_has_no_duplicates(self, ids: list[int]) -> None:
        """Invariant: from_raw always dedupes the output."""
        raw = {"propose_alliance_with": ids}
        msg = NegotiationMessage.from_raw(speaker=0, raw=raw, n_players=6)
        assert len(msg.propose_alliance_with) == len(set(msg.propose_alliance_with))


# ===========================================================================
# Criterion: alliance forms only on mutual consent; order-independent
# ===========================================================================


class TestMutualConsentAlliances:
    """_form_mutual_alliances only allies pairs that both expressed intent."""

    def test_when_both_propose_each_other_then_alliance_forms(self) -> None:
        state = DiplomacyState()
        messages = [
            _msg(1, propose=(2,)),
            _msg(2, propose=(1,)),
        ]
        _form_mutual_alliances(messages, state)
        assert state.are_allied(1, 2)

    def test_when_one_proposes_but_other_does_not_then_alliance_does_not_form(
        self,
    ) -> None:
        state = DiplomacyState()
        messages = [_msg(1, propose=(2,))]  # player 2 says nothing
        _form_mutual_alliances(messages, state)
        assert not state.are_allied(1, 2)

    def test_when_both_accept_each_other_then_alliance_forms(self) -> None:
        state = DiplomacyState()
        messages = [
            _msg(1, accept=(2,)),
            _msg(2, accept=(1,)),
        ]
        _form_mutual_alliances(messages, state)
        assert state.are_allied(1, 2)

    def test_when_one_proposes_and_other_accepts_then_alliance_forms(self) -> None:
        state = DiplomacyState()
        messages = [
            _msg(3, propose=(4,)),
            _msg(4, accept=(3,)),
        ]
        _form_mutual_alliances(messages, state)
        assert state.are_allied(3, 4)

    def test_when_order_of_messages_changes_then_result_is_identical(self) -> None:
        """Order-independence: A, B vs B, A yields same alliances."""
        state_ab = DiplomacyState()
        _form_mutual_alliances([_msg(1, propose=(2,)), _msg(2, propose=(1,))], state_ab)

        state_ba = DiplomacyState()
        _form_mutual_alliances([_msg(2, propose=(1,)), _msg(1, propose=(2,))], state_ba)

        assert state_ab.are_allied(1, 2) == state_ba.are_allied(1, 2)

    def test_when_three_players_form_chain_then_correct_pairs_are_allied(self) -> None:
        """1↔2 and 2↔3 both form; 1 and 3 are not allied (no mutual expression)."""
        state = DiplomacyState()
        messages = [
            _msg(1, propose=(2,)),
            _msg(2, propose=(1, 3)),
            _msg(3, propose=(2,)),
        ]
        _form_mutual_alliances(messages, state)
        assert state.are_allied(1, 2)
        assert state.are_allied(2, 3)
        assert not state.are_allied(1, 3)


# ===========================================================================
# Criterion: declare_war_on on an ally breaks alliance and records grudge
# ===========================================================================


class TestWarDeclaration:
    """_apply_war_declarations breaks alliances and records grudges."""

    def test_when_war_declared_on_ally_then_alliance_is_broken(self) -> None:
        state = DiplomacyState()
        state.form_alliance(1, 2)
        _apply_war_declarations([_msg(1, war=(2,))], state)
        assert not state.are_allied(1, 2)

    def test_when_war_declared_then_grudge_is_recorded(self) -> None:
        state = DiplomacyState()
        _apply_war_declarations([_msg(3, war=(4,))], state)
        assert state.has_grudge(victim=4, aggressor=3)

    def test_when_war_declared_on_ally_then_both_alliance_broken_and_grudge_recorded(
        self,
    ) -> None:
        state = DiplomacyState()
        state.form_alliance(0, 5)
        _apply_war_declarations([_msg(0, war=(5,))], state)
        assert not state.are_allied(0, 5)
        assert state.has_grudge(victim=5, aggressor=0)

    def test_when_multiple_war_declarations_then_all_are_processed(self) -> None:
        state = DiplomacyState()
        messages = [_msg(1, war=(2, 3)), _msg(4, war=(5,))]
        _apply_war_declarations(messages, state)
        assert state.has_grudge(victim=2, aggressor=1)
        assert state.has_grudge(victim=3, aggressor=1)
        assert state.has_grudge(victim=5, aggressor=4)


# ===========================================================================
# Criterion: learner (slot 0) excluded from speakers; never speaks
# ===========================================================================


class TestLearnerExclusion:
    """NegotiationOrchestrator never calls call_fn with player_id == 0."""

    def test_when_run_round_includes_slot_0_then_it_is_never_called(self) -> None:
        spoken_by: list[int] = []

        def tracking(pid: int, prompt: str) -> None:
            spoken_by.append(pid)

        orch = _make_orch(call_fn=tracking)
        orch.run_round(obs={}, speakers=list(range(6)))

        assert 0 not in spoken_by

    def test_when_all_speakers_are_slot_0_then_zero_calls_are_made(self) -> None:
        """If only the learner is listed, nothing is called."""
        call_count = [0]

        def counting(pid: int, prompt: str) -> None:
            call_count[0] += 1

        orch = _make_orch(call_fn=counting)
        result = orch.run_round(obs={}, speakers=[0])

        assert result == 0
        assert call_count[0] == 0

    def test_when_learner_slot_customised_then_that_slot_is_excluded(self) -> None:
        """Non-default learner_slot is also excluded."""
        spoken_by: list[int] = []

        def tracking(pid: int, prompt: str) -> None:
            spoken_by.append(pid)

        orch = _make_orch(call_fn=tracking, learner_slot=2)
        orch.run_round(obs={}, speakers=list(range(6)))

        assert 2 not in spoken_by


# ===========================================================================
# Criterion: call_count equals #calls; bounded by n_rounds * n_speakers
# ===========================================================================


class TestCallCount:
    """call_count tracks total LLM invocations accurately."""

    def test_when_one_round_then_call_count_equals_n_speakers(self) -> None:
        n_players = 6
        orch = _make_orch(n_players=n_players)
        orch.run_round(obs={}, speakers=list(range(n_players)))
        assert orch.call_count == n_players - 1  # slot 0 excluded

    def test_when_two_rounds_then_call_count_accumulates(self) -> None:
        n_players = 6
        orch = _make_orch(n_players=n_players)
        orch.run_round(obs={}, speakers=list(range(n_players)))
        orch.run_round(obs={}, speakers=list(range(n_players)))
        assert orch.call_count == 2 * (n_players - 1)

    def test_when_run_round_then_call_count_matches_actual_invocations(self) -> None:
        actual: list[int] = []

        def counting(pid: int, prompt: str) -> None:
            actual.append(pid)

        orch = _make_orch(call_fn=counting)
        orch.run_round(obs={}, speakers=list(range(6)))

        assert orch.call_count == len(actual)


# ===========================================================================
# Criterion: call_fn returning None => no-op, no exception
# ===========================================================================


class TestNullCallFnIsNoOp:
    """None-returning call_fn must not raise and must not mutate state."""

    def test_when_call_fn_returns_none_then_run_round_does_not_raise(self) -> None:
        orch = _make_orch()
        orch.run_round(obs={}, speakers=list(range(6)))  # must not raise

    def test_when_call_fn_returns_none_then_no_new_alliances_formed(self) -> None:
        state = DiplomacyState()
        orch = _make_orch(state=state)
        orch.run_round(obs={}, speakers=list(range(6)))

        for a in range(6):
            for b in range(a + 1, 6):
                assert not state.are_allied(a, b)

    def test_when_call_fn_returns_none_then_no_new_grudges_recorded(self) -> None:
        state = DiplomacyState()
        orch = _make_orch(state=state)
        orch.run_round(obs={}, speakers=list(range(6)))

        assert state.grudges == []


# ===========================================================================
# Criterion: deterministic — identical mocked input => identical state
# ===========================================================================


class TestDeterminism:
    """Same mocked call_fn inputs must always produce the same DiplomacyState."""

    def _scripted_call_fn(self, script: dict[int, dict[str, Any]]):
        """Return a call_fn that maps player_id -> fixed raw dict."""
        return lambda pid, _prompt: script.get(pid)

    def test_when_same_mocked_inputs_applied_twice_then_states_are_identical(
        self,
    ) -> None:
        script = {
            1: {
                "propose_alliance_with": [2],
                "accept_alliance_with": [],
                "declare_war_on": [],
                "attack_priority": [],
            },
            2: {
                "propose_alliance_with": [1],
                "accept_alliance_with": [],
                "declare_war_on": [],
                "attack_priority": [],
            },
            3: {
                "propose_alliance_with": [],
                "accept_alliance_with": [],
                "declare_war_on": [4],
                "attack_priority": [],
            },
            4: {
                "propose_alliance_with": [],
                "accept_alliance_with": [],
                "declare_war_on": [],
                "attack_priority": [],
            },
            5: {
                "propose_alliance_with": [],
                "accept_alliance_with": [],
                "declare_war_on": [],
                "attack_priority": [],
            },
        }

        def _build_state() -> DiplomacyState:
            s = DiplomacyState()
            orch = _make_orch(state=s, call_fn=self._scripted_call_fn(script))
            orch.run_round(obs={}, speakers=list(range(1, 6)))  # skip learner
            return s

        s1, s2 = _build_state(), _build_state()
        assert s1.are_allied(1, 2) == s2.are_allied(1, 2)
        assert s1.has_grudge(victim=4, aggressor=3) == s2.has_grudge(victim=4, aggressor=3)

    def test_when_speaker_order_in_script_differs_then_state_is_unchanged(self) -> None:
        """Alliance mutual-consent is order-independent (checked by sorted iteration)."""
        script_ab = {
            1: {
                "propose_alliance_with": [2],
                "accept_alliance_with": [],
                "declare_war_on": [],
                "attack_priority": [],
            },
            2: {
                "propose_alliance_with": [1],
                "accept_alliance_with": [],
                "declare_war_on": [],
                "attack_priority": [],
            },
        }
        script_ba = {
            2: {
                "propose_alliance_with": [1],
                "accept_alliance_with": [],
                "declare_war_on": [],
                "attack_priority": [],
            },
            1: {
                "propose_alliance_with": [2],
                "accept_alliance_with": [],
                "declare_war_on": [],
                "attack_priority": [],
            },
        }

        def _run(script: dict) -> DiplomacyState:
            s = DiplomacyState()
            orch = _make_orch(state=s, n_players=3, call_fn=lambda pid, _p: script.get(pid))
            orch.run_round(obs={}, speakers=[1, 2])
            return s

        s1, s2 = _run(script_ab), _run(script_ba)
        assert s1.are_allied(1, 2) == s2.are_allied(1, 2)


# ===========================================================================
# Integration: full round with scripted LLM outputs
# ===========================================================================


class TestFullRound:
    """Integration tests using scripted call_fn (no HTTP)."""

    def test_when_two_players_mutually_propose_then_alliance_is_in_state(self) -> None:
        state = DiplomacyState()
        script = {
            1: {
                "propose_alliance_with": [2],
                "accept_alliance_with": [],
                "declare_war_on": [],
                "attack_priority": [],
            },
            2: {
                "propose_alliance_with": [1],
                "accept_alliance_with": [],
                "declare_war_on": [],
                "attack_priority": [],
            },
            3: {
                "propose_alliance_with": [],
                "accept_alliance_with": [],
                "declare_war_on": [],
                "attack_priority": [],
            },
            4: {
                "propose_alliance_with": [],
                "accept_alliance_with": [],
                "declare_war_on": [],
                "attack_priority": [],
            },
            5: {
                "propose_alliance_with": [],
                "accept_alliance_with": [],
                "declare_war_on": [],
                "attack_priority": [],
            },
        }
        orch = _make_orch(state=state, call_fn=lambda pid, _p: script.get(pid))
        orch.run_round(obs={}, speakers=list(range(6)))

        assert state.are_allied(1, 2)
        assert not state.are_allied(1, 3)

    def test_when_player_declares_war_on_ally_then_alliance_broken_and_grudge_in_state(
        self,
    ) -> None:
        state = DiplomacyState()
        state.form_alliance(3, 4)
        script = {
            3: {
                "propose_alliance_with": [],
                "accept_alliance_with": [],
                "declare_war_on": [4],
                "attack_priority": [],
            },
        }
        orch = _make_orch(state=state, call_fn=lambda pid, _p: script.get(pid))
        orch.run_round(obs={}, speakers=[3])

        assert not state.are_allied(3, 4)
        assert state.has_grudge(victim=4, aggressor=3)

    def test_when_run_round_returns_int_then_value_equals_calls_made(self) -> None:
        n_players = 6
        orch = _make_orch(n_players=n_players)
        result = orch.run_round(obs={}, speakers=list(range(n_players)))
        assert isinstance(result, int)
        assert result == n_players - 1  # slot 0 excluded
