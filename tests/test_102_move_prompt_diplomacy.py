"""
Source-blind example tests for issue #102:
  feat(agents): move-prompt conditioning — diplomacy note + LLMOpponent threading

All tests are derived from the acceptance-criteria text and the product
requirements doc only.  No implementation source (src/) was read.  These
represent the Red phase of TDD — they fail today and pass only once the
relevant criterion is genuinely satisfied.

Verifiable criteria covered here (oracle classification in brackets):
  [UNIT]  render_action_prompt(..., diplomacy_note=None) is byte-identical to
          the current output (golden / backward-compat test).
  [T3]    A diplomacy_note section lists allies ("do not attack"), the leader
          ("prefer"), and standing grudges — rendered deterministically (sorted).
  [UNIT]  LLMOpponent.act(..., diplomacy=None) calls
          call_ollama_for_action_index with exactly today's kwargs (call_args).
  [UNIT]  With a diplomacy context the prompt-affecting kwarg is populated;
          the LLM still selects an index into the unchanged legal_actions.
  [UNIT]  Fallback/timeout behaviour unchanged; LLMOpponent still pickles.

Skipped (oracle: NOT VERIFIABLE or already covered by test_101_*):
  - forced-trade / PHASE_TRADE invariants  (test_101_env_phase_trade.py)
  - game-resolution integration check      (not verifiable)
  - DEFERRED territory-card +2 bonus       (deferred)
  - SOLID / coverage / ruff / process gates (not unit-verifiable)
"""

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Minimal fixtures — built from the criteria, not from implementation source
# ---------------------------------------------------------------------------

# A synthetic observation compatible with render_action_prompt (all numpy arrays
# with the correct keys).  Used for prompt-content tests that actually render.
_SYNTH_OBS: dict = {
    "territory_owner": np.zeros(42, dtype=np.int32),
    "armies": np.ones(42, dtype=np.int32) * 3,
    "phase": np.array(2, dtype=np.int32),  # ATTACK phase
    "current_player": np.array(0, dtype=np.int32),
    "cards": np.zeros((5, 4), dtype=np.int32),
    "trade_count": np.array(0, dtype=np.int32),
    "reinforcements_remaining": np.array(0, dtype=np.int32),
    "eliminated": np.zeros(6, dtype=np.int32),
}


def _synth_obs() -> dict:
    """Proper obs dict for render_action_prompt tests (numpy-backed)."""
    return _SYNTH_OBS


def _mock_obs() -> MagicMock:
    """Lightweight MagicMock obs for LLMOpponent tests where the Ollama client
    is patched and render_action_prompt is never actually invoked.  MagicMock
    identity comparisons work correctly in call_args equality checks.
    """
    return MagicMock()


def _legal_actions(n: int = 3) -> list[dict]:
    """Return n SKIP action dicts with the correct keys expected by the env."""
    return [
        {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0} for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# 1 – render_action_prompt backward-compatibility (golden test)
#
# Criterion: render_action_prompt(..., diplomacy_note=None) is byte-identical
# to the current output (the kwarg must be additive and default-transparent).
# ---------------------------------------------------------------------------


class TestRenderActionPromptGolden:
    def test_when_diplomacy_note_is_none_then_output_is_byte_identical_to_no_kwarg(self):
        """Passing diplomacy_note=None must produce the exact same string as omitting
        the argument entirely — the existing prompt contract is fully preserved.
        """
        from src.agents.action_prompt import render_action_prompt

        obs = _synth_obs()
        legal = _legal_actions()

        baseline = render_action_prompt(obs, legal)
        with_none = render_action_prompt(obs, legal, diplomacy_note=None)

        assert baseline == with_none, (
            "render_action_prompt(..., diplomacy_note=None) must be byte-identical "
            "to render_action_prompt(...) with no diplomacy_note kwarg."
        )


# ---------------------------------------------------------------------------
# 2 – diplomacy_note section: content and deterministic ordering
#
# Criterion: the rendered section lists allies ("do not attack"), the leader
# ("prefer"), and standing grudges — all sorted deterministically.
# ---------------------------------------------------------------------------


class TestDiplomacyNoteContent:
    """Tests that the diplomacy_note block appears with the right copy when each
    field (allies / leader / grudges) is populated.
    """

    @pytest.fixture(scope="class")
    def note_cls(self):
        """Return the DiplomacyNote class (imported lazily so the fixture is the Red indicator)."""
        from src.agents.diplomacy import DiplomacyNote

        return DiplomacyNote

    # -- allies --------------------------------------------------------------

    def test_when_allies_present_then_do_not_attack_appears_in_prompt(self, note_cls):
        from src.agents.action_prompt import render_action_prompt

        note = note_cls(allies=[2, 4], leader=None, grudges=[])
        result = render_action_prompt(_synth_obs(), _legal_actions(), diplomacy_note=note)

        assert "do not attack" in result.lower(), (
            "The rendered prompt must contain 'do not attack' when allies are listed."
        )

    def test_when_specific_ally_id_present_then_that_id_appears_in_prompt(self, note_cls):
        from src.agents.action_prompt import render_action_prompt

        note = note_cls(allies=[7], leader=None, grudges=[])
        result = render_action_prompt(_synth_obs(), _legal_actions(), diplomacy_note=note)

        assert "7" in result, "The ally's player id must be visible in the rendered prompt."

    # -- leader --------------------------------------------------------------

    def test_when_leader_present_then_prefer_appears_in_prompt(self, note_cls):
        from src.agents.action_prompt import render_action_prompt

        note = note_cls(allies=[], leader=3, grudges=[])
        result = render_action_prompt(_synth_obs(), _legal_actions(), diplomacy_note=note)

        assert "prefer" in result.lower(), (
            "The rendered prompt must contain 'prefer' when a leader is identified."
        )

    def test_when_leader_id_given_then_leader_id_appears_in_prompt(self, note_cls):
        from src.agents.action_prompt import render_action_prompt

        note = note_cls(allies=[], leader=3, grudges=[])
        result = render_action_prompt(_synth_obs(), _legal_actions(), diplomacy_note=note)

        assert "3" in result, "The leader's player id must be visible in the rendered prompt."

    # -- grudges -------------------------------------------------------------

    def test_when_grudges_present_then_grudge_ids_appear_in_prompt(self, note_cls):
        from src.agents.action_prompt import render_action_prompt

        note = note_cls(allies=[], leader=None, grudges=[1, 5])
        result = render_action_prompt(_synth_obs(), _legal_actions(), diplomacy_note=note)

        assert "1" in result and "5" in result, (
            "Both grudge player ids must be visible in the rendered prompt."
        )

    # -- deterministic ordering ----------------------------------------------

    def test_when_allies_supplied_in_descending_order_then_prompt_equals_ascending_version(
        self, note_cls
    ):
        """Allies must be sorted before rendering so that the prompt is
        deterministic regardless of the order in which ids were added.
        """
        from src.agents.action_prompt import render_action_prompt

        obs, legal = _synth_obs(), _legal_actions()
        note_asc = note_cls(allies=[1, 2, 4], leader=None, grudges=[])
        note_desc = note_cls(allies=[4, 2, 1], leader=None, grudges=[])

        assert render_action_prompt(obs, legal, diplomacy_note=note_asc) == render_action_prompt(
            obs, legal, diplomacy_note=note_desc
        ), "Ally ids must be sorted before rendering so the output is deterministic."

    def test_when_grudges_supplied_in_descending_order_then_prompt_equals_ascending_version(
        self, note_cls
    ):
        """Grudges must be sorted before rendering so that the prompt is
        deterministic regardless of the order in which ids were added.
        """
        from src.agents.action_prompt import render_action_prompt

        obs, legal = _synth_obs(), _legal_actions()
        note_asc = note_cls(allies=[], leader=None, grudges=[1, 3, 5])
        note_desc = note_cls(allies=[], leader=None, grudges=[5, 3, 1])

        assert render_action_prompt(obs, legal, diplomacy_note=note_asc) == render_action_prompt(
            obs, legal, diplomacy_note=note_desc
        ), "Grudge ids must be sorted before rendering so the output is deterministic."


# ---------------------------------------------------------------------------
# 2b – Property-based: sorted invariant holds for all valid id combinations
#
# Criterion: "rendered deterministically (sorted)" — reversing any ally or
# grudge list must always produce the same prompt (ordering invariant).
# ---------------------------------------------------------------------------


class TestDiplomacyNoteOrdering:
    @given(
        allies=st.lists(st.integers(min_value=0, max_value=5), max_size=5, unique=True),
        grudges=st.lists(st.integers(min_value=0, max_value=5), max_size=5, unique=True),
    )
    @settings(max_examples=60)
    def test_when_ally_and_grudge_id_lists_are_reversed_then_prompt_is_identical(
        self, allies: list[int], grudges: list[int]
    ):
        """For any combination of player-id lists the rendered prompt must be the
        same regardless of which order those ids were provided — confirming the
        sorted/deterministic rendering invariant stated in the acceptance criteria.
        """
        from src.agents.action_prompt import render_action_prompt
        from src.agents.diplomacy import DiplomacyNote

        obs, legal = _synth_obs(), _legal_actions()

        note_forward = DiplomacyNote(allies=allies, leader=None, grudges=grudges)
        note_reversed = DiplomacyNote(
            allies=list(reversed(allies)),
            leader=None,
            grudges=list(reversed(grudges)),
        )

        assert render_action_prompt(
            obs, legal, diplomacy_note=note_forward
        ) == render_action_prompt(obs, legal, diplomacy_note=note_reversed)


# ---------------------------------------------------------------------------
# 3 – LLMOpponent.act(diplomacy=None) backward compatibility
#
# Criterion: LLMOpponent.act(..., diplomacy=None) calls
# call_ollama_for_action_index with *exactly* the same kwargs as today
# (asserted via call_args).
#
# Patch target: the name as imported inside llm_opponent.py — adjust if
# the module uses a different import style (e.g. module-level import).
# ---------------------------------------------------------------------------

_OLLAMA_PATCH = "src.agents.ollama_client.call_ollama_for_action_index"


class TestLLMOpponentActBackwardCompat:
    def test_when_diplomacy_is_none_then_ollama_call_args_match_no_kwarg_baseline(self):
        """act(state, legal, diplomacy=None) must forward exactly the same positional
        and keyword arguments to call_ollama_for_action_index as act(state, legal)
        without the diplomacy kwarg — the backward-compatibility guarantee.
        """
        from src.agents.llm_opponent import LLMOpponent

        # Use MagicMock obs (not a numpy dict) so that call_args identity comparison
        # works correctly without triggering numpy array truth-value errors.
        state = _mock_obs()
        legal = _legal_actions()

        with patch(_OLLAMA_PATCH, return_value=0) as mock_fn:
            opp_a = LLMOpponent(model="test-model", base_url="http://localhost:11434/v1")
            opp_a.act(state, legal)
            call_args_baseline = mock_fn.call_args

        with patch(_OLLAMA_PATCH, return_value=0) as mock_fn:
            opp_b = LLMOpponent(model="test-model", base_url="http://localhost:11434/v1")
            opp_b.act(state, legal, diplomacy=None)
            call_args_with_none = mock_fn.call_args

        assert call_args_baseline == call_args_with_none, (
            "act(diplomacy=None) must call call_ollama_for_action_index with identical "
            "args to act() without the diplomacy kwarg — no regressions allowed."
        )


# ---------------------------------------------------------------------------
# 4 – LLMOpponent.act with a real diplomacy context
#
# Criterion (a): with a diplomacy context the prompt-affecting kwarg is
#               populated (call differs from the no-diplomacy baseline).
# Criterion (b): the LLM still selects an index into the *unchanged*
#               legal_actions — no new actions are introduced.
# ---------------------------------------------------------------------------


class TestLLMOpponentActWithDiplomacy:
    @pytest.fixture
    def sample_note(self):
        from src.agents.diplomacy import DiplomacyNote

        return DiplomacyNote(allies=[1], leader=2, grudges=[3])

    # (a) prompt-affecting kwarg must be different from the no-diplomacy call ------

    def test_when_diplomacy_context_provided_then_ollama_call_args_differ_from_baseline(
        self, sample_note
    ):
        """When a real DiplomacyNote is passed, call_ollama_for_action_index must
        receive at least one different argument compared to the no-diplomacy call —
        confirming that the prompt-affecting kwarg is populated.
        """
        from src.agents.llm_opponent import LLMOpponent

        state = _mock_obs()
        legal = _legal_actions()

        with patch(_OLLAMA_PATCH, return_value=0) as mock_fn:
            opp_a = LLMOpponent(model="test-model", base_url="http://localhost:11434/v1")
            opp_a.act(state, legal, diplomacy=None)
            call_args_no_dipl = mock_fn.call_args

        with patch(_OLLAMA_PATCH, return_value=0) as mock_fn:
            opp_b = LLMOpponent(model="test-model", base_url="http://localhost:11434/v1")
            opp_b.act(state, legal, diplomacy=sample_note)
            call_args_with_dipl = mock_fn.call_args

        assert call_args_no_dipl != call_args_with_dipl, (
            "Supplying a DiplomacyNote must change at least one kwarg forwarded to "
            "call_ollama_for_action_index (the prompt-affecting kwarg must be populated)."
        )

    # (b) legal_actions list is unchanged — action returned is from original list --

    def test_when_diplomacy_provided_then_returned_action_is_from_original_legal_list(
        self, sample_note
    ):
        """Diplomacy must never add or remove actions from legal_actions.  When the
        LLM returns index 1, act() must return legal[1] — the original action, not
        a synthetic one injected by the diplomacy layer.
        """
        from src.agents.llm_opponent import LLMOpponent

        state = _mock_obs()
        legal = _legal_actions(n=4)

        with patch(_OLLAMA_PATCH, return_value=1):  # LLM picks index 1
            opp = LLMOpponent(model="test-model", base_url="http://localhost:11434/v1")
            result = opp.act(state, legal, diplomacy=sample_note)

        assert result == legal[1], (
            "act() must return legal[index] from the original legal_actions list "
            "when a diplomacy context is provided — no new actions are introduced."
        )

    def test_when_diplomacy_provided_then_act_returns_a_value_from_legal_list(self, sample_note):
        """The value returned by act() must always belong to the original legal_actions
        list, regardless of the diplomacy context (no new synthetic actions).
        """
        from src.agents.llm_opponent import LLMOpponent

        state = _mock_obs()
        legal = _legal_actions(n=3)

        with patch(_OLLAMA_PATCH, return_value=2):  # LLM picks index 2
            opp = LLMOpponent(model="test-model", base_url="http://localhost:11434/v1")
            result = opp.act(state, legal, diplomacy=sample_note)

        assert result in legal, (
            "The action returned by act() must be one of the original legal_actions "
            "entries — diplomacy must not introduce actions outside that list."
        )


# ---------------------------------------------------------------------------
# 5 – LLMOpponent pickle round-trip
#
# Criterion: Fallback/timeout behaviour unchanged; LLMOpponent still pickles.
# (Required for multiprocessing workers in the self-play training loop.)
# ---------------------------------------------------------------------------


class TestLLMOpponentPickle:
    def test_when_llm_opponent_pickled_and_unpickled_then_type_is_preserved(self):
        """LLMOpponent must survive a pickle round-trip — a requirement for
        multiprocessing-based self-play workers that fork agent instances.
        """
        from src.agents.llm_opponent import LLMOpponent

        opp = LLMOpponent(model="test-model", base_url="http://localhost:11434/v1")
        raw = pickle.dumps(opp)
        restored = pickle.loads(raw)  # noqa: S301  (safe: test-only round-trip)

        assert isinstance(restored, LLMOpponent), (
            "Unpickling an LLMOpponent must return an LLMOpponent instance."
        )

    def test_when_llm_opponent_with_api_key_pickled_then_unpickling_succeeds(self):
        """LLMOpponent configured with an API key (cloud mode) must also survive
        a pickle round-trip — the ThreadPoolExecutor must not block serialisation.
        """
        from src.agents.llm_opponent import LLMOpponent

        opp = LLMOpponent(
            model="test-model",
            base_url="https://ollama.com/v1",
            api_key="sk-test-key",
        )
        raw = pickle.dumps(opp)
        restored = pickle.loads(raw)  # noqa: S301

        assert isinstance(restored, LLMOpponent)
