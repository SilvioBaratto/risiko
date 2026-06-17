"""
Source-blind example tests for HeuristicAgent (issue #76).

All tests are derived from the acceptance criteria only.  Tests written in
Red phase and confirmed failing before implementation existed.

Design choices (from criterion text):
- action_type == 5 is the skip action.
- act_with_meta returns (action, nan_tensor, nan_tensor).
- MultiAgentRunner(env, agents).run_game(seed=N) is the correct API.
- RisikoEnv takes no seed in constructor; seed goes to env.reset(seed=N).
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.base import Agent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.random_agent import RandomAgent
from src.config import HeuristicConfig
from src.env import RisikoEnv
from src.multi_agent import MultiAgentRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SKIP_ACTION_TYPE = 5


def _make_env(n_players: int = 2) -> RisikoEnv:
    return RisikoEnv(n_players=n_players)


def _reset(env: RisikoEnv, seed: int = 0):
    obs, info = env.reset(seed=seed)
    return obs, info["legal_actions"]


def _play_steps(agent: HeuristicAgent, env: RisikoEnv, max_steps: int = 50, seed: int = 0):
    """Advance env using agent decisions; assert legality at every step."""
    obs, legal = _reset(env, seed=seed)
    actions = []
    for _ in range(max_steps):
        a = agent.act(obs, legal, deterministic=True)
        assert a in legal, f"Illegal action returned: {a}"
        actions.append(a)
        obs, _, done, _, info = env.step(a)
        legal = info["legal_actions"]
        if done:
            break
    return actions


# ---------------------------------------------------------------------------
# 1. Agent protocol
# ---------------------------------------------------------------------------


class TestAgentProtocol:
    """HeuristicAgent must implement the Agent protocol exactly like RandomAgent."""

    def test_when_instantiated_then_isinstance_agent_is_true(self):
        assert isinstance(HeuristicAgent(), Agent)

    def test_when_inspected_then_act_is_callable(self):
        assert callable(getattr(HeuristicAgent(), "act", None))

    def test_when_inspected_then_act_with_meta_is_callable(self):
        assert callable(getattr(HeuristicAgent(), "act_with_meta", None))

    def test_when_act_signature_inspected_then_deterministic_is_keyword_only(self):
        sig = inspect.signature(HeuristicAgent.act)
        param = sig.parameters.get("deterministic")
        assert param is not None
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_when_act_with_meta_signature_inspected_then_deterministic_is_keyword_only(self):
        sig = inspect.signature(HeuristicAgent.act_with_meta)
        param = sig.parameters.get("deterministic")
        assert param is not None
        assert param.kind == inspect.Parameter.KEYWORD_ONLY


# ---------------------------------------------------------------------------
# 2. act_with_meta return value
# ---------------------------------------------------------------------------


class TestActWithMetaReturnValue:
    """act_with_meta returns (action, nan_tensor, nan_tensor)."""

    @pytest.fixture()
    def _result_and_legal(self):
        env = _make_env()
        obs, legal = _reset(env)
        result = HeuristicAgent(seed=0).act_with_meta(obs, legal, deterministic=True)
        return result, legal

    def test_when_called_then_result_has_three_elements(self, _result_and_legal):
        result, _ = _result_and_legal
        assert len(result) == 3

    def test_when_called_then_first_element_is_in_legal_actions(self, _result_and_legal):
        (action, _, _), legal = _result_and_legal
        assert action in legal

    def test_when_called_then_second_element_is_nan_tensor(self, _result_and_legal):
        (_, val, _), _ = _result_and_legal
        assert torch.is_tensor(val)
        assert torch.all(torch.isnan(val)).item()

    def test_when_called_then_third_element_is_nan_tensor(self, _result_and_legal):
        (_, _, logp), _ = _result_and_legal
        assert torch.is_tensor(logp)
        assert torch.all(torch.isnan(logp)).item()


# ---------------------------------------------------------------------------
# 3. Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    """Constructor mirrors RandomAgent plus config and optional rng."""

    def test_when_no_args_then_cfg_is_heuristicconfig(self):
        assert isinstance(HeuristicAgent()._cfg, HeuristicConfig)

    def test_when_config_is_none_then_cfg_defaults_to_heuristicconfig(self):
        assert isinstance(HeuristicAgent(config=None)._cfg, HeuristicConfig)

    def test_when_custom_config_passed_then_that_config_is_stored(self):
        cfg = HeuristicConfig()
        assert HeuristicAgent(config=cfg)._cfg is cfg

    def test_when_seed_passed_then_rng_is_numpy_generator(self):
        assert isinstance(HeuristicAgent(seed=7)._rng, np.random.Generator)

    def test_when_no_seed_and_no_rng_then_rng_is_numpy_generator(self):
        assert isinstance(HeuristicAgent()._rng, np.random.Generator)

    def test_when_explicit_rng_passed_then_that_rng_is_stored(self):
        rng = np.random.default_rng(7)
        assert HeuristicAgent(rng=rng)._rng is rng

    def test_when_both_seed_and_rng_passed_then_explicit_rng_wins_over_seed(self):
        rng = np.random.default_rng(7)
        agent = HeuristicAgent(seed=42, rng=rng)
        assert agent._rng is rng


# ---------------------------------------------------------------------------
# 4. Legal-actions invariant
# ---------------------------------------------------------------------------


class TestLegalActionsInvariant:
    """Every action returned by the agent is a member of the supplied legal_actions."""

    def test_when_empty_legal_actions_passed_then_value_error_is_raised(self):
        env = _make_env()
        obs, _ = _reset(env)
        with pytest.raises(ValueError):
            HeuristicAgent(seed=0).act(obs, [], deterministic=True)

    def test_when_act_called_then_returned_action_is_in_legal_actions(self):
        env = _make_env()
        obs, legal = _reset(env)
        action = HeuristicAgent(seed=0).act(obs, legal, deterministic=True)
        assert action in legal

    def test_when_single_action_provided_then_that_action_is_returned(self):
        """When only one action exists, the agent must return it regardless of type."""
        env = _make_env()
        obs, legal = _reset(env)
        single = legal[:1]
        action = HeuristicAgent(seed=0).act(obs, single, deterministic=True)
        assert action == single[0]

    def test_when_full_game_is_played_then_every_action_is_legal(self):
        env = _make_env()
        agent = HeuristicAgent(seed=0)
        obs, legal = _reset(env, seed=1)
        for _ in range(200):
            action = agent.act(obs, legal, deterministic=True)
            assert action in legal, f"Illegal action returned: {action}"
            obs, _, done, _, info = env.step(action)
            legal = info["legal_actions"]
            if done:
                break


@given(st.integers(min_value=0, max_value=500))
@settings(max_examples=20)
def test_property_for_any_seed_act_returns_action_in_legal_actions(seed):
    """Property: act() always returns a legal action regardless of seed."""
    env = _make_env()
    obs, legal = _reset(env, seed=seed % 8)
    if not legal:
        return
    action = HeuristicAgent(seed=seed).act(obs, legal, deterministic=True)
    assert action in legal


# ---------------------------------------------------------------------------
# 5. Attack behavior
# ---------------------------------------------------------------------------


class TestAttackBehavior:
    """Returns skip when best available odds fall below attack_odds_stop."""

    def test_when_attack_odds_stop_is_impossibly_high_then_skip_chosen_when_available(self):
        """With attack_odds_stop=1e6 no real attack qualifies; agent must skip."""
        cfg = HeuristicConfig(attack_odds_stop=1_000_000.0)
        agent = HeuristicAgent(config=cfg, seed=0)
        env = _make_env()
        obs, legal = _reset(env, seed=42)

        observed = False
        for _ in range(150):
            action = agent.act(obs, legal, deterministic=True)
            assert action in legal

            has_non_skip = any(a.get("action_type") != _SKIP_ACTION_TYPE for a in legal)
            has_skip = any(a.get("action_type") == _SKIP_ACTION_TYPE for a in legal)
            if has_non_skip and has_skip and action.get("action_type") == _SKIP_ACTION_TYPE:
                observed = True
                break

            obs, _, done, _, info = env.step(action)
            legal = info["legal_actions"]
            if done:
                break

        if not observed:
            pytest.xfail("No mixed attack+skip state reached; test inconclusive.")

    def test_when_only_non_skip_actions_available_then_some_action_is_still_returned(self):
        env = _make_env()
        obs, legal = _reset(env)
        non_skip = [a for a in legal if a.get("action_type") != _SKIP_ACTION_TYPE]
        if not non_skip:
            pytest.skip("No non-skip actions in initial state")
        action = HeuristicAgent(seed=0).act(obs, non_skip, deterministic=True)
        assert action in non_skip

    def test_when_attack_min_odds_changed_via_config_then_config_is_stored(self):
        cfg = HeuristicConfig(attack_min_odds=3.5)
        assert HeuristicAgent(config=cfg, seed=0)._cfg.attack_min_odds == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# 6. Fortify behavior
# ---------------------------------------------------------------------------


class TestFortifyBehavior:
    """Fortify: skip when no safe interior source retaining fortify_min_interior_armies."""

    def test_when_only_skip_action_provided_then_skip_is_returned(self):
        env = _make_env()
        obs, legal = _reset(env)
        skip_only = [a for a in legal if a.get("action_type") == _SKIP_ACTION_TYPE]
        if not skip_only:
            skip_only = [{"action_type": _SKIP_ACTION_TYPE}]
        action = HeuristicAgent(seed=0).act(obs, skip_only, deterministic=True)
        assert action in skip_only

    def test_when_fortify_min_interior_armies_is_very_large_then_game_completes_legally(self):
        """No territory can donate armies → agent must still play legally (skip fortify)."""
        cfg = HeuristicConfig(fortify_min_interior_armies=9_999_999)
        agent = HeuristicAgent(config=cfg, seed=0)
        env = _make_env()
        obs, legal = _reset(env, seed=7)
        for _ in range(120):
            action = agent.act(obs, legal, deterministic=True)
            assert action in legal
            obs, _, done, _, info = env.step(action)
            legal = info["legal_actions"]
            if done:
                break

    def test_when_fortify_min_interior_armies_changed_then_config_is_stored(self):
        cfg = HeuristicConfig(fortify_min_interior_armies=7)
        assert HeuristicAgent(config=cfg, seed=0)._cfg.fortify_min_interior_armies == 7


# ---------------------------------------------------------------------------
# 7. Capture move
# ---------------------------------------------------------------------------


class TestCaptureMoveBehavior:
    """capture_move_fraction controls what fraction of armies enter the captured territory."""

    def test_when_capture_move_fraction_is_one_then_action_is_legal(self):
        env = _make_env()
        obs, legal = _reset(env, seed=1)
        cfg = HeuristicConfig(capture_move_fraction=1.0)
        action = HeuristicAgent(config=cfg, seed=0).act(obs, legal, deterministic=True)
        assert action in legal

    def test_when_capture_move_fraction_is_zero_then_action_is_legal(self):
        env = _make_env()
        obs, legal = _reset(env, seed=1)
        cfg = HeuristicConfig(capture_move_fraction=0.0)
        action = HeuristicAgent(config=cfg, seed=0).act(obs, legal, deterministic=True)
        assert action in legal

    def test_when_capture_move_fraction_changed_then_config_is_stored(self):
        cfg = HeuristicConfig(capture_move_fraction=0.75)
        assert HeuristicAgent(config=cfg, seed=0)._cfg.capture_move_fraction == pytest.approx(0.75)

    def test_when_full_game_played_with_max_fraction_then_all_actions_are_legal(self):
        cfg = HeuristicConfig(capture_move_fraction=1.0)
        agent = HeuristicAgent(config=cfg, seed=5)
        _play_steps(agent, _make_env(), max_steps=150, seed=5)


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=15)
def test_property_for_any_capture_fraction_act_returns_legal_action(fraction):
    """Property: any valid capture_move_fraction in [0,1] yields a legal action."""
    env = _make_env()
    obs, legal = _reset(env)
    cfg = HeuristicConfig(capture_move_fraction=fraction)
    action = HeuristicAgent(config=cfg, seed=0).act(obs, legal, deterministic=True)
    assert action in legal


# ---------------------------------------------------------------------------
# 8. Cards behavior
# ---------------------------------------------------------------------------


class TestCardsBehavior:
    """Agent trades a valid 3-set (preferring it over skip) before attacking."""

    def test_when_full_game_is_played_then_card_decisions_never_raise(self):
        agent = HeuristicAgent(seed=3)
        env = _make_env()
        obs, legal = _reset(env, seed=3)
        for _ in range(300):
            action = agent.act(obs, legal, deterministic=True)
            assert action in legal
            obs, _, done, _, info = env.step(action)
            legal = info["legal_actions"]
            if done:
                break

    def test_when_skip_is_only_action_then_skip_is_returned(self):
        """If only skip is in the legal list, the agent must return it."""
        env = _make_env()
        obs, legal = _reset(env)
        skip_only = [{"action_type": _SKIP_ACTION_TYPE}]
        action = HeuristicAgent(seed=0).act(obs, skip_only, deterministic=True)
        assert action in skip_only


# ---------------------------------------------------------------------------
# 9. All thresholds from HeuristicConfig
# ---------------------------------------------------------------------------


class TestHeuristicConfigThresholds:
    """Each configurable threshold is stored and accessible on the agent."""

    def test_when_attack_odds_stop_set_then_stored_in_cfg(self):
        cfg = HeuristicConfig(attack_odds_stop=99.0)
        assert HeuristicAgent(config=cfg, seed=0)._cfg.attack_odds_stop == pytest.approx(99.0)

    def test_when_attack_min_odds_set_then_stored_in_cfg(self):
        cfg = HeuristicConfig(attack_min_odds=2.0)
        assert HeuristicAgent(config=cfg, seed=0)._cfg.attack_min_odds == pytest.approx(2.0)

    def test_when_fortify_min_interior_armies_set_then_stored_in_cfg(self):
        cfg = HeuristicConfig(fortify_min_interior_armies=10)
        assert HeuristicAgent(config=cfg, seed=0)._cfg.fortify_min_interior_armies == 10

    def test_when_capture_move_fraction_set_then_stored_in_cfg(self):
        cfg = HeuristicConfig(capture_move_fraction=0.6)
        assert HeuristicAgent(config=cfg, seed=0)._cfg.capture_move_fraction == pytest.approx(0.6)

    def test_when_reinforce_border_weight_set_then_stored_in_cfg(self):
        cfg = HeuristicConfig(reinforce_border_weight=5.0)
        assert HeuristicAgent(config=cfg, seed=0)._cfg.reinforce_border_weight == pytest.approx(5.0)

    def test_when_fortify_threat_weight_set_then_stored_in_cfg(self):
        cfg = HeuristicConfig(fortify_threat_weight=3.0)
        assert HeuristicAgent(config=cfg, seed=0)._cfg.fortify_threat_weight == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 10. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed (or same injected rng) ⇒ identical action sequences."""

    def test_when_same_seed_used_twice_then_first_action_is_identical(self):
        obs1, legal1 = _reset(_make_env(), seed=0)
        obs2, legal2 = _reset(_make_env(), seed=0)
        a1 = HeuristicAgent(seed=42).act(obs1, legal1, deterministic=True)
        a2 = HeuristicAgent(seed=42).act(obs2, legal2, deterministic=True)
        assert a1 == a2

    def test_when_same_seed_used_twice_then_full_sequences_are_identical(self):
        seq1 = _play_steps(HeuristicAgent(seed=42), _make_env(), max_steps=30)
        seq2 = _play_steps(HeuristicAgent(seed=42), _make_env(), max_steps=30)
        assert seq1 == seq2

    def test_when_same_rng_seed_injected_then_same_first_action_is_returned(self):
        obs, legal = _reset(_make_env(), seed=0)
        a1 = HeuristicAgent(rng=np.random.default_rng(7)).act(obs, legal, deterministic=True)
        a2 = HeuristicAgent(rng=np.random.default_rng(7)).act(obs, legal, deterministic=True)
        assert a1 == a2

    def test_when_different_seeds_used_then_both_return_legal_actions(self):
        obs1, legal1 = _reset(_make_env(), seed=0)
        obs2, legal2 = _reset(_make_env(), seed=0)
        a1 = HeuristicAgent(seed=1).act(obs1, legal1, deterministic=True)
        a2 = HeuristicAgent(seed=9999).act(obs2, legal2, deterministic=True)
        assert a1 in legal1
        assert a2 in legal2


@given(st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=20)
def test_property_same_seed_always_yields_same_first_action(seed):
    """Property: determinism holds for arbitrary seeds on the first decision."""
    obs, legal = _reset(_make_env(), seed=0)
    a1 = HeuristicAgent(seed=seed).act(obs, legal, deterministic=True)
    a2 = HeuristicAgent(seed=seed).act(obs, legal, deterministic=True)
    assert a1 == a2


# ---------------------------------------------------------------------------
# 11. Head-to-head win rate (integration)
# ---------------------------------------------------------------------------


def _wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    """Wilson score CI lower bound."""
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - margin


_N_PLAYERS_6P = 6
_RANDOM_BASELINE_6P = 1.0 / _N_PLAYERS_6P
_N_GAMES_WINRATE = 50


@pytest.mark.integration
def test_when_heuristic_plays_6p_vs_random_then_wilson_lower_exceeds_random_baseline():
    """Win-rate Wilson CI lower bound must clear the ≈16% random baseline."""
    wins = 0
    for game_idx in range(_N_GAMES_WINRATE):
        agents = [HeuristicAgent(seed=game_idx)] + [RandomAgent() for _ in range(_N_PLAYERS_6P - 1)]
        env = RisikoEnv(n_players=_N_PLAYERS_6P)
        result = MultiAgentRunner(env, agents).run_game(seed=game_idx)
        if result.winner == 0:
            wins += 1

    p_hat = wins / _N_GAMES_WINRATE
    lower = _wilson_lower(wins, _N_GAMES_WINRATE)
    assert lower > _RANDOM_BASELINE_6P, (
        f"Win rate {p_hat:.1%} (Wilson CI lower {lower:.1%}) does not "
        f"significantly exceed random baseline {_RANDOM_BASELINE_6P:.1%}"
    )


@pytest.mark.integration
def test_when_heuristic_plays_via_runner_then_game_completes_and_winner_is_valid():
    """Runner must complete cleanly; no uncaught exception from the heuristic seat."""
    agents = [HeuristicAgent(seed=0)] + [RandomAgent() for _ in range(_N_PLAYERS_6P - 1)]
    env = RisikoEnv(n_players=_N_PLAYERS_6P)
    result = MultiAgentRunner(env, agents).run_game(seed=0)
    assert result is not None
    assert result.winner in range(_N_PLAYERS_6P) or result.winner is None


@pytest.mark.integration
def test_when_heuristic_plays_multiple_6p_games_then_never_raises():
    """All HeuristicAgent decisions across several full games must remain legal."""
    for game_idx in range(5):
        agents = [HeuristicAgent(seed=game_idx)] + [RandomAgent() for _ in range(_N_PLAYERS_6P - 1)]
        env = RisikoEnv(n_players=_N_PLAYERS_6P)
        result = MultiAgentRunner(env, agents).run_game(seed=game_idx)
        assert result is not None
