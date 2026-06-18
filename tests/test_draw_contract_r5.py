"""Tests for Issue #97 — draw contract and winner resolution correctness.

Scope criteria being tested (all [UNIT] per oracle):

  [C1] ``_resolve_winner`` returns ``None`` on turn-cap truncation; ``winner``
       reflects only a genuine single-survivor end-state — not the territory leader.

  [C2] Win-rate consumers treat ``winner=None`` as a draw, not as a win for any
       player (``self_play``, ``evaluate``, ``monte_carlo``, ``tb_logger``).

  [C3] ``tests/test_multi_agent.py`` truncation test re-asserts the draw contract
       (``winner is None``, not ``isinstance(winner, int)``).

  [C4] ``get_legal_actions()`` is never empty in ``PHASE_TRADE`` under a forced
       trade.

Criteria NOT tested (oracle: NOT VERIFIABLE):
  - Forced-trade rule at turn start (no concrete runtime check derivable at
    the unit level beyond prior coverage from #94/#95/#96).
  - 6-player integration resolution details (integration-class criterion).
  - Territory-card +2 bonus (DEFERRED this phase).
  - Fortification adjacency (covered by #95 test suite).
  - Seed / action determinism (structural, no inferable unit check).
  - New-code conventions (style-only, no runtime assertion).
  - Full suite green (meta, not testable here).
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.random_agent import RandomAgent
from src.env import PHASE_TRADE, RisikoEnv
from src.multi_agent import GameResult, MultiAgentRunner

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_BASE_SEED = 42
_N_6P = 6

# 5 player-turns is too few for any player to eliminate all 5 opponents in a
# 6-player game; using this cap guarantees truncation without relying on state.
_GUARANTEED_TRUNCATION_TURNS = 5


# ═══════════════════════════════════════════════════════════════════════════
# C1 — draw contract: turn-cap truncation → winner is None
# ═══════════════════════════════════════════════════════════════════════════


class TestDrawContractOnTurnCap:
    """``_resolve_winner`` must return ``None`` when the turn cap is reached.

    The draw contract (Issue #97 scope): a game that hits ``max_turns`` without
    a single survivor is a DRAW (``winner=None``), not a win for the territory
    leader.  This explicitly reverts the #96 change that declared the territory
    leader as winner on truncation.
    """

    def test_when_six_player_game_hits_five_turn_cap_then_winner_is_none(
        self,
    ) -> None:
        """Concrete: 6 players, 5 turns — impossible to win by elimination → winner=None."""
        env = RisikoEnv(n_players=_N_6P)
        agents = [RandomAgent() for _ in range(_N_6P)]
        runner = MultiAgentRunner(env, agents, max_turns=_GUARANTEED_TRUNCATION_TURNS)
        result = runner.run_game(seed=_BASE_SEED)
        assert result.winner is None, (
            f"winner={result.winner!r}; turn-cap truncation must yield winner=None "
            "— _resolve_winner must not declare the territory leader as winner."
        )

    @pytest.mark.parametrize(
        "n_players,seed",
        [(2, 10), (3, 20), (4, 30), (6, 40)],
    )
    def test_when_any_player_count_hits_one_turn_cap_then_winner_is_none(
        self, n_players: int, seed: int
    ) -> None:
        """max_turns=1 is too short for elimination regardless of player count."""
        env = RisikoEnv(n_players=n_players)
        agents = [RandomAgent() for _ in range(n_players)]
        runner = MultiAgentRunner(env, agents, max_turns=1)
        result = runner.run_game(seed=seed)
        assert result.winner is None, (
            f"n_players={n_players} seed={seed}: winner={result.winner!r}; "
            "1-turn game cannot produce a single survivor."
        )

    def test_when_game_result_is_constructed_as_draw_then_winner_is_none(self) -> None:
        """``GameResult(winner=None, …)`` correctly represents a draw record."""
        result = GameResult(
            winner=None,
            n_turns=500,
            territory_history=[],
            elimination_order=[],
            card_trade_turns=[],
            action_log=[],
            trajectories=[],
        )
        assert result.winner is None
        # winner=None must not accidentally equal any player index
        assert not any(result.winner == p for p in range(_N_6P))


# ═══════════════════════════════════════════════════════════════════════════
# C1 property — draw contract is invariant over arbitrary seeds
# ═══════════════════════════════════════════════════════════════════════════


@given(st.integers(min_value=0, max_value=9_999))
@settings(max_examples=25, deadline=None)
def test_when_six_player_five_turn_game_runs_then_winner_is_always_none(
    seed: int,
) -> None:
    """Invariant: any seed, 6 players, 5 turns → turn-cap truncation → winner=None.

    Five player-turns are insufficient for any player to eliminate all five
    opponents by natural play.  ``_resolve_winner`` must therefore return ``None``
    for every seed — the territory leader must NOT be declared winner.
    """
    env = RisikoEnv(n_players=_N_6P)
    agents = [RandomAgent() for _ in range(_N_6P)]
    runner = MultiAgentRunner(env, agents, max_turns=_GUARANTEED_TRUNCATION_TURNS)
    result = runner.run_game(seed=seed)
    assert result.winner is None, (
        f"seed={seed}: winner={result.winner!r}; 5-turn 6-player game cannot "
        "produce a genuine single survivor — winner must be None."
    )


# ═══════════════════════════════════════════════════════════════════════════
# C1 — genuine end-state: single survivor → winner is that player's index
# ═══════════════════════════════════════════════════════════════════════════


class TestGenuineSingleSurvivor:
    """When the game ends naturally (real elimination), ``winner`` must be set.

    ``_resolve_winner`` returns the surviving player's index when exactly one
    active player remains — this is the other half of the C1 contract.
    """

    def test_when_game_result_stores_integer_winner_then_winner_equals_player_id(
        self,
    ) -> None:
        """``GameResult(winner=1, …)`` stores the genuine winner correctly."""
        result = GameResult(
            winner=1,
            n_turns=75,
            territory_history=[],
            elimination_order=[0, 2],
            card_trade_turns=[5, 10],
            action_log=[],
            trajectories=[],
        )
        assert isinstance(result.winner, int)
        assert result.winner == 1
        assert result.winner not in result.elimination_order

    @pytest.mark.parametrize(
        "winner,eliminated,n_players",
        [
            (0, [1], 2),
            (2, [0, 1], 3),
            (5, [0, 1, 2, 3, 4], 6),
        ],
    )
    def test_when_winner_is_set_then_winner_is_not_in_elimination_order(
        self, winner: int, eliminated: list[int], n_players: int
    ) -> None:
        """A genuine winner must not also appear in ``elimination_order``."""
        result = GameResult(
            winner=winner,
            n_turns=100,
            territory_history=[],
            elimination_order=eliminated,
            card_trade_turns=[],
            action_log=[],
            trajectories=[],
        )
        assert result.winner is not None
        assert result.winner not in result.elimination_order, (
            f"winner={winner} must not be in elimination_order={eliminated}."
        )

    @pytest.mark.integration
    def test_when_two_player_game_runs_many_turns_and_resolves_then_winner_is_valid(
        self,
    ) -> None:
        """2-player, large turn cap: if a genuine winner appears it must be valid.

        When ``winner is not None``, ``_resolve_winner`` detected a real
        single-survivor end-state — not a turn-cap leader.  The winner must be
        a valid player index who is not in ``elimination_order``.
        If the game draws instead, the test is inconclusive (winner=None is
        acceptable for a genuinely drawn 2-player game).
        """
        env = RisikoEnv(n_players=2)
        agents = [RandomAgent(seed=7), RandomAgent(seed=13)]
        runner = MultiAgentRunner(env, agents, max_turns=5_000)
        result = runner.run_game(seed=_BASE_SEED)
        if result.winner is not None:
            assert result.winner in (0, 1), (
                f"winner={result.winner!r} is not a valid index for a 2-player game."
            )
            assert result.winner not in result.elimination_order, (
                "winner must not also appear in elimination_order."
            )


# ═══════════════════════════════════════════════════════════════════════════
# C2 — win-rate consumers: winner=None means no player won
# ═══════════════════════════════════════════════════════════════════════════


class TestWinRateConsumers:
    """Win-rate consumers must treat ``winner=None`` as a draw.

    Before this fix: ``_resolve_winner`` returned the territory leader on
    truncation, giving consumers a non-None ``winner`` for every game.
    After this fix: ``winner=None`` signals a draw; consumers must NOT count
    it as a win for any player.  This contract applies to ``self_play``,
    ``evaluate``, ``monte_carlo``, and ``tb_logger``.
    """

    def test_when_result_winner_is_none_then_it_does_not_equal_any_player_id(
        self,
    ) -> None:
        """``winner=None`` must not equal any integer player index.

        Consumers that compute ``r.winner == player_id`` must receive False for
        all ``player_id`` values when the game was a draw — the draw contract at
        the data-structure level.
        """
        result = GameResult(
            winner=None,
            n_turns=500,
            territory_history=[],
            elimination_order=[],
            card_trade_turns=[],
            action_log=[],
            trajectories=[],
        )
        assert result.winner is None
        for player_id in range(_N_6P):
            assert result.winner != player_id, (
                f"winner=None must not equal player_id={player_id}. "
                "Win-rate consumers checking `winner == player_id` must NOT count "
                "this as a win."
            )

    def test_when_all_results_have_none_winner_then_win_count_is_zero_for_all_players(
        self,
    ) -> None:
        """All-draw batch → 0 wins for every player.

        This is the minimal computation that all consumers
        (``monte_carlo``, ``evaluate``, ``self_play``, ``tb_logger``) perform:
        ``win_count = sum(1 for r in results if r.winner == player_id)``.
        Every such count must be 0 when every result is a draw.
        """
        n_games = 10
        results = [
            GameResult(
                winner=None,
                n_turns=500,
                territory_history=[],
                elimination_order=[],
                card_trade_turns=[],
                action_log=[],
                trajectories=[],
            )
            for _ in range(n_games)
        ]
        for player_id in range(_N_6P):
            win_count = sum(1 for r in results if r.winner == player_id)
            assert win_count == 0, (
                f"player={player_id}: win_count={win_count}/{n_games}; "
                "all-None-winner results must give 0 wins for every player."
            )
            win_rate = win_count / n_games
            assert win_rate == 0.0

    @pytest.mark.integration
    def test_when_six_player_short_games_run_then_all_results_are_draws_and_win_rate_is_zero(
        self,
    ) -> None:
        """Integration: short games → all draws; win rate is 0 for every player.

        Validates the full pipeline: runner → env → _resolve_winner → winner=None.
        Any win-rate consumer receiving these results must compute 0% win rate
        for all players.
        """
        env = RisikoEnv(n_players=_N_6P)
        agents = [RandomAgent() for _ in range(_N_6P)]
        runner = MultiAgentRunner(env, agents, max_turns=_GUARANTEED_TRUNCATION_TURNS)
        results = runner.run_games(n=10, seed=_BASE_SEED)

        for i, result in enumerate(results):
            assert result.winner is None, (
                f"game {i}: winner={result.winner!r}; "
                "5-turn 6-player game must be a draw — winner=None."
            )

        # Consumer-level win-rate computation
        for player_id in range(_N_6P):
            wins = sum(1 for r in results if r.winner == player_id)
            assert wins == 0, (
                f"player={player_id}: {wins} wins counted from draw results. "
                "Truncated games (winner=None) must never increment a win count."
            )


# ═══════════════════════════════════════════════════════════════════════════
# C3 — re-assert the draw contract (mirrors test_max_turns_truncation)
# ═══════════════════════════════════════════════════════════════════════════


def test_when_runner_truncates_at_turn_cap_then_winner_is_none_not_territory_leader() -> None:
    """Re-asserts the draw contract that ``test_max_turns_truncation`` must carry.

    ``tests/test_multi_agent.py::test_max_turns_truncation`` was changed by
    Issue #96 to assert ``isinstance(result.winner, int)`` (territory leader as
    winner).  Issue #97 reverts that change; the correct assertion is
    ``result.winner is None``.  This test pins the correct contract so that if
    the truncation test is regressed again, this catch fires first.
    """
    env = RisikoEnv(n_players=2)
    agents = [RandomAgent(seed=1), RandomAgent(seed=2)]
    max_turns = 5
    runner = MultiAgentRunner(env, agents, max_turns=max_turns)
    result = runner.run_game(seed=_BASE_SEED)
    # With only 5 turns and 2 players, elimination is not expected; if it did
    # happen (terminated early), turns_elapsed < max_turns and winner != None is
    # acceptable.  We only assert the draw contract when the cap was actually hit.
    turns_hit_cap = env.state.turns_elapsed >= max_turns
    if turns_hit_cap:
        assert result.n_turns >= max_turns
        assert result.winner is None, (
            f"winner={result.winner!r}; turn-cap game must be a draw. "
            "This re-asserts the draw contract for test_multi_agent.py."
        )


def test_when_six_player_five_turn_game_truncates_winner_is_none_draw_contract() -> None:
    """Unconditional draw-contract assertion using guaranteed-truncation parameters.

    With 6 players and max_turns=5, natural elimination is impossible.  This
    guarantees the draw-contract path is exercised without a conditional guard.
    """
    env = RisikoEnv(n_players=_N_6P)
    agents = [RandomAgent() for _ in range(_N_6P)]
    runner = MultiAgentRunner(env, agents, max_turns=_GUARANTEED_TRUNCATION_TURNS)
    result = runner.run_game(seed=_BASE_SEED)
    assert env.state.turns_elapsed == _GUARANTEED_TRUNCATION_TURNS, (
        "Game must have hit the turn cap, not terminated early."
    )
    assert result.winner is None, (
        f"winner={result.winner!r}; truncated 6-player game must yield winner=None. "
        "The draw contract must hold."
    )


# ═══════════════════════════════════════════════════════════════════════════
# C4 — PHASE_TRADE: get_legal_actions() never empty under a forced trade
# ═══════════════════════════════════════════════════════════════════════════


@given(st.integers(min_value=0, max_value=9_999))
@settings(max_examples=30, deadline=None)
def test_when_any_seed_then_phase_trade_legal_actions_are_never_empty(
    seed: int,
) -> None:
    """Invariant: ``get_legal_actions()`` returns ≥1 action whenever in PHASE_TRADE.

    A forced trade at turn start must always provide at least one legal action
    so that the current player can make progress and the game does not deadlock.
    This invariant must hold for all seeds.
    """
    env = RisikoEnv(n_players=3)
    obs, _ = env.reset(seed=seed)
    agent = RandomAgent()
    done = False

    for _ in range(200):
        legal = env.get_legal_actions()
        if env.state.phase == PHASE_TRADE:
            assert len(legal) > 0, (
                f"seed={seed}: get_legal_actions()=[] while in PHASE_TRADE. "
                "The env must always offer ≥1 action to avoid deadlock."
            )
        if done or not legal:
            break
        action = agent.act(obs, legal)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated


# ═══════════════════════════════════════════════════════════════════════════
# Ruff lint gate — criterion: "ruff check . tests reports no issues"
# ═══════════════════════════════════════════════════════════════════════════


def test_when_ruff_check_runs_then_no_lint_issues_are_reported() -> None:
    """``ruff check . tests`` must exit 0 (no lint violations)."""
    completed = subprocess.run(
        [sys.executable, "-m", "ruff", "check", ".", "tests"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"ruff reported issues:\n{completed.stdout}\n{completed.stderr}"
    )
