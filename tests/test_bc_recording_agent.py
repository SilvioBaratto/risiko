"""
Tests for training/bc_recording_agent.py — issue #78.

Covers acceptance criteria:
- RecordingAgent satisfies the Agent protocol and wraps a HeuristicAgent.
- DecisionRecord captures obs (copy), legal_actions, label_action, label_index, acting_player.
- explore_eps=0.0 → executed == heuristic == label (many seeds).
- explore_eps=1.0 → executed is a random legal action; label is still the heuristic's choice.
- Exactly one DecisionRecord per act/act_with_meta call; recorded obs is immutable.
- legal_actions[label_index] == label_action and acting_player == obs["current_player"].
- BCConfig is a frozen dataclass with explore_eps field.
- data/bc/ is in .gitignore.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

# ── Minimal Agent-protocol stub (NOT from src/) ──────────────────────────────


class _FixedAgent:
    """Stub that always returns legal_actions[0]; satisfies the Agent protocol."""

    def act(self, obs, legal_actions, *, deterministic: bool = False):  # noqa: ARG002
        return legal_actions[0]

    def act_with_meta(self, obs, legal_actions, *, deterministic: bool = False):  # noqa: ARG002
        nan = float("nan")
        return legal_actions[0], nan, nan


def _obs(player: int = 0) -> dict:
    """Minimal obs dict; criterion: acting_player == obs['current_player']."""
    return {"current_player": player}


# Three distinct legal actions — enough to observe ε-greedy divergence.
_LEGAL = [
    {"action_type": 1, "param_a": 0, "param_b": 1, "param_c": 0, "param_d": 0},
    {"action_type": 2, "param_a": 0, "param_b": 1, "param_c": 1, "param_d": 0},
    {"action_type": 3, "param_a": 0, "param_b": 2, "param_c": 0, "param_d": 0},
]


# ── Module existence ─────────────────────────────────────────────────────────


def test_when_module_imported_recording_agent_is_accessible():
    """Criterion: RecordingAgent lives in training/bc_recording_agent.py."""
    from training.bc_recording_agent import RecordingAgent  # noqa: F401


def test_when_module_imported_decision_record_is_accessible():
    from training.bc_recording_agent import DecisionRecord  # noqa: F401


# ── DecisionRecord shape ─────────────────────────────────────────────────────


def test_when_decision_record_is_inspected_it_is_a_dataclass():
    from training.bc_recording_agent import DecisionRecord

    assert dataclasses.is_dataclass(DecisionRecord)


def test_when_decision_record_is_inspected_all_required_fields_are_present():
    """
    Criterion: DecisionRecord captures obs copy, legal_actions, label_action,
    label_index, and acting_player.
    """
    from training.bc_recording_agent import DecisionRecord

    names = {f.name for f in dataclasses.fields(DecisionRecord)}
    required = {"obs", "legal_actions", "label_action", "label_index", "acting_player"}
    missing = required - names
    assert not missing, f"DecisionRecord is missing fields: {missing}"


# ── Agent protocol compliance ─────────────────────────────────────────────────


def test_when_act_is_called_it_returns_a_value_from_legal_actions():
    """Criterion: RecordingAgent satisfies the Agent protocol."""
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=0.0)
    result = agent.act(_obs(), list(_LEGAL), deterministic=False)
    assert result in _LEGAL


# ── explore_eps = 0.0 — executed == heuristic == label ──────────────────────


def test_when_explore_eps_is_zero_executed_equals_heuristic_preferred():
    """Criterion: explore_eps=0.0 → executed action == heuristic action."""
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=0.0)
    legal = list(_LEGAL)
    executed = agent.act(_obs(), legal, deterministic=False)
    assert executed == legal[0]


def test_when_explore_eps_is_zero_recorded_label_equals_executed():
    """Criterion: explore_eps=0.0 → recorded label == executed action."""
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=0.0)
    legal = list(_LEGAL)
    executed = agent.act(_obs(), legal, deterministic=False)
    assert agent.records[-1].label_action == executed


def test_when_explore_eps_is_zero_many_seeds_all_agree():
    """Criterion: explore_eps=0.0 → execute==heuristic==label for many seeds."""
    from training.bc_recording_agent import RecordingAgent

    for seed in range(25):
        rng = np.random.default_rng(seed)
        agent = RecordingAgent(_FixedAgent(), explore_eps=0.0, rng=rng)
        legal = list(_LEGAL)
        executed = agent.act(_obs(), legal, deterministic=False)
        record = agent.records[-1]
        assert executed == legal[0], f"seed={seed}: executed={executed!r}"
        assert record.label_action == legal[0], f"seed={seed}: label={record.label_action!r}"


# ── explore_eps = 1.0 — random execute, heuristic label ─────────────────────


def test_when_explore_eps_is_one_executed_is_in_legal_actions():
    """Criterion: explore_eps=1.0 → executed action is a legal random action."""
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=1.0, rng=np.random.default_rng(42))
    executed = agent.act(_obs(), list(_LEGAL), deterministic=False)
    assert executed in _LEGAL


def test_when_explore_eps_is_one_label_is_still_heuristic_preferred():
    """
    Criterion: explore_eps=1.0 → recorded label is the heuristic's preferred action,
    not the random one that was executed.
    """
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=1.0, rng=np.random.default_rng(0))
    legal = list(_LEGAL)
    agent.act(_obs(), legal, deterministic=False)
    assert agent.records[-1].label_action == legal[0], (
        "label_action must be the heuristic's choice even under full exploration"
    )


def test_when_explore_eps_is_one_label_index_dereferences_to_heuristic_action():
    """
    Criterion: label_index dereferences (in the pre-step legal list)
    to the heuristic's preferred action.
    """
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=1.0, rng=np.random.default_rng(7))
    legal = list(_LEGAL)
    agent.act(_obs(), legal, deterministic=False)
    rec = agent.records[-1]
    assert legal[rec.label_index] == rec.label_action


def test_when_explore_eps_is_one_executed_sometimes_differs_from_heuristic():
    """
    Sanity: with 3 distinct legal actions and explore_eps=1.0, over 30 seeds
    at least one execution must diverge from the heuristic choice.
    """
    from training.bc_recording_agent import RecordingAgent

    legal = list(_LEGAL)
    diverged = any(
        RecordingAgent(_FixedAgent(), explore_eps=1.0, rng=np.random.default_rng(s)).act(
            _obs(), legal, deterministic=False
        )
        != legal[0]
        for s in range(30)
    )
    assert diverged, "explore_eps=1.0 never diverged from heuristic over 30 seeds"


# ── Exactly one DecisionRecord per act / act_with_meta call ─────────────────


def test_when_act_called_once_exactly_one_record_is_appended():
    """Criterion: exactly one DecisionRecord is appended per act call."""
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=0.0)
    assert len(agent.records) == 0
    agent.act(_obs(), list(_LEGAL), deterministic=False)
    assert len(agent.records) == 1


def test_when_act_called_five_times_five_records_are_present():
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=0.0)
    for _ in range(5):
        agent.act(_obs(), list(_LEGAL), deterministic=False)
    assert len(agent.records) == 5


def test_when_act_with_meta_called_exactly_one_record_is_appended():
    """Criterion: exactly one DecisionRecord is appended per act_with_meta call."""
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=0.0)
    assert len(agent.records) == 0
    agent.act_with_meta(_obs(), list(_LEGAL), deterministic=False)
    assert len(agent.records) == 1


# ── Recorded obs is a copy ────────────────────────────────────────────────────


def test_when_obs_mutated_after_act_recorded_obs_is_unchanged():
    """Criterion: recorded obs is a COPY; not mutated by later steps."""
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=0.0)
    obs = _obs(player=3)
    agent.act(obs, list(_LEGAL), deterministic=False)
    record = agent.records[0]
    snapshot = copy.deepcopy(record.obs)
    obs["current_player"] = 99
    obs["injected"] = "pollution"
    assert record.obs == snapshot, "Recorded obs changed after original was mutated"


def test_when_act_called_recorded_obs_equals_input_at_call_time():
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=0.0)
    obs = _obs(player=1)
    snapshot = copy.deepcopy(obs)
    agent.act(obs, list(_LEGAL), deterministic=False)
    assert agent.records[0].obs == snapshot


# ── label_index and acting_player invariants ──────────────────────────────────


def test_when_act_called_legal_at_label_index_equals_label_action():
    """Criterion: legal_actions[label_index] == label_action."""
    from training.bc_recording_agent import RecordingAgent

    legal = list(_LEGAL)
    agent = RecordingAgent(_FixedAgent(), explore_eps=0.0)
    agent.act(_obs(), legal, deterministic=False)
    rec = agent.records[0]
    assert legal[rec.label_index] == rec.label_action


def test_when_explore_eps_one_label_index_still_resolves_correctly():
    """Criterion: legal_actions[label_index] == label_action, even under full exploration."""
    from training.bc_recording_agent import RecordingAgent

    legal = list(_LEGAL)
    agent = RecordingAgent(_FixedAgent(), explore_eps=1.0, rng=np.random.default_rng(5))
    agent.act(_obs(), legal, deterministic=False)
    rec = agent.records[0]
    assert legal[rec.label_index] == rec.label_action


def test_when_act_called_for_each_player_acting_player_matches_current_player():
    """Criterion: acting_player == obs['current_player']."""
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=0.0)
    for player in range(6):
        agent.act(_obs(player=player), list(_LEGAL), deterministic=False)
    for idx, rec in enumerate(agent.records):
        assert rec.acting_player == idx, (
            f"Record {idx}: acting_player={rec.acting_player!r}, expected {idx}"
        )


# ── Global: .gitignore excludes data/bc/ ────────────────────────────────────


def test_when_gitignore_read_data_bc_is_excluded():
    """Criterion: data/bc/ is added to .gitignore."""
    gitignore = Path(".gitignore")
    assert gitignore.exists(), ".gitignore does not exist"
    lines = {ln.strip() for ln in gitignore.read_text().splitlines()}
    assert lines & {"data/bc", "data/bc/"}, (
        "Expected 'data/bc' or 'data/bc/' in .gitignore — neither found."
    )


# ── Global: BCConfig is a frozen dataclass with explore_eps ─────────────────


def test_when_bc_config_imported_it_is_a_dataclass():
    """Criterion: every threshold/weight lives in BCConfig (no magic numbers)."""
    from src.config import BCConfig

    assert dataclasses.is_dataclass(BCConfig)


def test_when_bc_config_instantiated_mutation_raises_frozen_error():
    """Criterion: BCConfig follows the frozen-config convention."""
    from src.config import BCConfig

    cfg = BCConfig()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        cfg.explore_eps = 0.99  # type: ignore[misc]


def test_when_bc_config_inspected_explore_eps_field_is_present():
    """Criterion: explore_eps declared in BCConfig, not hardcoded."""
    from src.config import BCConfig

    names = {f.name for f in dataclasses.fields(BCConfig)}
    assert "explore_eps" in names


# ── Property-based tests — invariants from acceptance criteria ────────────────


@given(st.integers(min_value=0, max_value=5))
def test_property_acting_player_always_equals_current_player(player):
    """Invariant: acting_player == obs['current_player'] for all valid player indices."""
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=0.0)
    agent.act(_obs(player=player), list(_LEGAL), deterministic=False)
    assert agent.records[-1].acting_player == player


@given(
    st.lists(
        st.fixed_dictionaries(
            {
                "action_type": st.integers(min_value=0, max_value=5),
                "param_a": st.integers(min_value=0, max_value=41),
                "param_b": st.integers(min_value=0, max_value=41),
                "param_c": st.integers(min_value=0, max_value=3),
                "param_d": st.integers(min_value=0, max_value=41),
            }
        ),
        min_size=1,
        max_size=12,
    )
)
def test_property_label_index_always_dereferences_to_label_action(legal):
    """Invariant: legal_actions[label_index] == label_action for all non-empty lists."""
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=0.0)
    agent.act(_obs(), legal, deterministic=False)
    rec = agent.records[-1]
    assert legal[rec.label_index] == rec.label_action


@given(st.integers(min_value=0, max_value=20))
def test_property_record_count_equals_act_call_count(n):
    """Invariant: len(records) == number of act calls."""
    from training.bc_recording_agent import RecordingAgent

    agent = RecordingAgent(_FixedAgent(), explore_eps=0.0)
    for _ in range(n):
        agent.act(_obs(), list(_LEGAL), deterministic=False)
    assert len(agent.records) == n
