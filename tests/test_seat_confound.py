"""Guards against the three design bugs that made the first tournament uninterpretable.

All three were silent: nothing crashed, nothing logged, and the numbers looked fine.
They were only visible by cross-tabulating the published ledger and reading the prompts
the models actually received.

1. **Strategy was pinned to a seat.** The tournament permuted models across strategies
   but never permuted strategies across seats, so seat *i* played ``strategies[i]`` in
   all 100 games. "Win rate by strategy" and "win rate by seat" were then the same
   number, and no analysis could separate them.

2. **A tie invented a leader.** ``DiplomacyState.leader`` broke ties by lowest player
   id. At turn 0 every player holds an equal share, so the prompt announced
   ``Leader: Player 0`` in every game — and the models act on that line. Measured on an
   identical board, adding it moved the share of turn-0 war declarations aimed at seat 0
   from 1-in-6 to 4-in-6. Since seat 0 was always the same strategy (bug 1), one
   strategy was ganged up on at the start of every single game.

3. **A player could be told it was its own leader.** The action prompt always guarded
   this; the negotiation prompt did not. One traced negotiation in six named the reader
   as the leader. The coalition strategy's entire instruction is "gang up on the leader"
   — pointed at itself, it is degenerate, and the logs show it: the diplomat, told it
   led, declared war on a random player and proposed no alliances at all; the aggressor,
   in the same spot, declared war on a player and offered it an alliance in the same
   reply. Found by a viewer, not by us.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.agents.diplomacy import DiplomacyState
from training.tournament import (
    assign_strategies_to_seats,
    build_game_plan,
    build_seats,
    load_tournament_config,
)

CONFIG = Path("config/tournament.yaml")

pytestmark = pytest.mark.skipif(not CONFIG.is_file(), reason="tournament config not on disk")


# ── Bug 1: strategy must not be pinned to a seat ─────────────────────────────
def test_every_strategy_visits_every_seat_across_a_run() -> None:
    config = load_tournament_config(CONFIG)
    n_games = 100

    seen: dict[str, set[int]] = {s: set() for s in config.strategies}
    for game in range(n_games):
        for seat, strategy in enumerate(build_game_plan(config, game).seat_order):
            seen[strategy].add(seat)

    for strategy, seats in seen.items():
        assert seats == set(range(6)), f"{strategy} never sat in seats {set(range(6)) - seats}"


def test_seat_assignment_is_roughly_uniform() -> None:
    """No strategy may be over-represented in any seat — that is the whole point."""
    config = load_tournament_config(CONFIG)
    n_games = 600
    expected = n_games / 6  # 100 per (strategy, seat) cell

    counts: Counter[tuple[str, int]] = Counter()
    for game in range(n_games):
        for seat, strategy in enumerate(build_game_plan(config, game).seat_order):
            counts[(strategy, seat)] += 1

    for (strategy, seat), n in counts.items():
        # Generous band: this catches "pinned to a seat", not sampling noise.
        assert 0.6 * expected < n < 1.4 * expected, (
            f"{strategy} sat in seat {seat} {n} times out of {n_games} (expected ~{expected:.0f})"
        )


def test_seat_order_is_a_permutation_not_a_resample() -> None:
    order = assign_strategies_to_seats(["a", "b", "c", "d", "e", "f"], seed=3)
    assert sorted(order) == ["a", "b", "c", "d", "e", "f"]


def test_seat_order_is_deterministic_for_a_seed() -> None:
    strategies = ["a", "b", "c", "d", "e", "f"]
    assert assign_strategies_to_seats(strategies, seed=9) == assign_strategies_to_seats(
        strategies, seed=9
    )


def test_seat_order_is_independent_of_the_model_assignment() -> None:
    """Both permutations derive from the same per-game seed; they must not move together.

    If they shared an RNG stream, a strategy landing on seat 0 would also always draw the
    same model — trading one confound for another.
    """
    config = load_tournament_config(CONFIG)
    pairs = set()
    for game in range(60):
        plan = build_game_plan(config, game)
        pairs.add((plan.seat_order[0], plan.assignment[plan.seat_order[0]]))
    # Seat 0's strategy is paired with many different models across the run.
    assert len(pairs) > 12, f"seat 0 strategy/model pairs are too correlated: {len(pairs)}"


def test_build_seats_honours_the_per_game_seat_order() -> None:
    config = load_tournament_config(CONFIG)
    plan = build_game_plan(config, 0)

    seats = build_seats(plan.assignment, config, seat_order=plan.seat_order)

    for i, strategy in enumerate(plan.seat_order):
        assert seats[i].model == plan.assignment[strategy]
        assert seats[i].player_id == i


# ── Bug 2: a tie has no leader ───────────────────────────────────────────────
def test_a_tied_board_has_no_leader() -> None:
    """Turn 0 is always a tie. Crowning seat 0 there is an artefact, not a fact."""
    state = DiplomacyState()
    tied = np.array([p for p in range(6) for _ in range(7)])  # 7 territories each

    assert state.leader(tied) is None


def test_a_clear_leader_is_still_reported() -> None:
    state = DiplomacyState()
    owners = np.array([2] * 20 + [p for p in range(6) for _ in range(3)] + [0, 1, 3, 4])

    assert state.leader(owners) == 2


# ── Bug 3: nobody is ever told that the leader is themselves ─────────────────
def test_a_player_is_never_told_it_is_its_own_leader() -> None:
    """The negotiation prompt must not name the reader as the leader.

    The action prompt has always guarded this; the negotiation prompt did not, so one
    traced negotiation in six read ``Leader (most territories): Player N`` with N being
    the reader's own seat. The coalition strategy is "gang up on the leader" — aimed at
    itself it is degenerate, and the logs bear that out: the diplomat, told it led,
    declared war on a random player and proposed no alliances at all.
    """
    from src.agents.negotiation import NegotiationOrchestrator
    from src.agents.player_config import DEFAULT_6P_PROFILES
    from src.agents.reputation import ReputationBook
    from src.config import DiplomacyConfig

    # Seat 3 leads the board outright.
    owners = np.array([3] * 20 + [p for p in range(6) for _ in range(3)] + [0, 1, 4, 5])
    obs = {"territory_owner": owners, "armies": np.ones(42, dtype=int), "n_players": 6}

    orchestrator = NegotiationOrchestrator(
        state=DiplomacyState(),
        reputation=ReputationBook(),
        cfg=DiplomacyConfig(enabled=True, n_rounds=1),
        profiles=DEFAULT_6P_PROFILES,
    )

    leader_prompt = orchestrator._prompt_for(3, obs)  # the leader itself
    other_prompt = orchestrator._prompt_for(0, obs)  # everyone else

    assert "Leader" not in leader_prompt, "the leader was told it is the leader"
    assert "Leader (most territories): Player 3" in other_prompt, "others must still see it"
