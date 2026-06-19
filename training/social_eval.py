"""Social evaluation harness: RL learner win-rate vs coordinating LLM pool.

One ReputationBook is shared across the n_games loop (cross-game reputation
accumulates) and is created fresh per run_social_eval call (reset between
sessions).  Negotiation cost is observed from DiplomacyRunner.call_count —
never estimated by a formula — so a run with no actual negotiation reports 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.agents.base import Agent
from src.agents.player_config import PlayerConfig
from src.agents.random_agent import RandomAgent
from src.agents.reputation import ReputationBook
from src.config import DiplomacyConfig
from src.env import RisikoEnv
from src.multi_agent import DiplomacyRunner, MultiAgentRunner
from src.utils.log import get_logger
from training.monte_carlo import wilson_ci

_log = get_logger("social_eval")

_LEARNER_SLOT: int = 0


@dataclass(frozen=True)
class SocialEvalResult:
    """Aggregate statistics from one social-eval run."""

    learner_win_rate: float
    ci_low: float
    ci_high: float
    draw_rate: float
    avg_length: float
    total_negotiation_calls: int
    avg_calls_per_game: float


def run_social_eval(
    learner: Agent,
    llm_pool: list[Agent],
    n_games: int,
    n_rounds: int = 1,
    *,
    seed: int = 42,
    n_players: int = 6,
    max_turns: int = 1000,
    profiles: Sequence[PlayerConfig] | None = None,
    cfg: DiplomacyConfig | None = None,
) -> SocialEvalResult:
    """Evaluate *learner* against *llm_pool* over *n_games*.

    A single ReputationBook is shared across all games so betrayals in game i
    affect trust in game i+1.  The book is fresh for every call.

    ``total_negotiation_calls`` is the *observed* sum of ``runner.call_count``
    across all games, not an a-priori formula.  A run with diplomacy disabled
    or with a failing ``call_fn`` reports 0.

    Args:
        learner: RL agent under evaluation (player slot 0).
        llm_pool: LLM opponents (fills slots 1..n_players-1; padded with random).
        n_games: Number of games to run.
        n_rounds: Negotiation rounds per game; folded into *cfg* when no *cfg*
            is supplied.  Ignored when *cfg* is given explicitly.
        seed: Base RNG seed; game i uses seed+i.
        n_players: Total players per game.
        max_turns: Turn cap per game.
        profiles: Per-player LLM sampling profiles forwarded to DiplomacyRunner.
        cfg: Diplomacy layer config.  When ``None`` a default
            ``DiplomacyConfig(n_rounds=n_rounds)`` is used (diplomacy disabled).

    Returns:
        SocialEvalResult with win-rate, Wilson CI, and *observed* negotiation
        call cost.
    """
    if n_games == 0:
        return _empty_result()
    _cfg = cfg if cfg is not None else DiplomacyConfig(n_rounds=n_rounds)
    book = ReputationBook()
    agents = _seat_agents(learner, llm_pool, n_players)
    wins, draws, total_turns, total_calls = _play_games(
        agents, book, n_games, seed, n_players, max_turns, profiles, _cfg
    )
    return _build_result(wins, draws, n_games, total_turns, total_calls)


# ---------------------------------------------------------------------------
# Helpers — each under 10 lines
# ---------------------------------------------------------------------------


def _seat_agents(learner: Agent, pool: list[Agent], n_players: int) -> list[Agent]:
    seats: list[Agent] = [learner, *pool[: n_players - 1]]
    while len(seats) < n_players:
        seats.append(RandomAgent())
    return seats


def _build_runner(
    agents: list[Agent],
    n_players: int,
    max_turns: int,
    book: ReputationBook,
    profiles: Sequence[PlayerConfig] | None,
    cfg: DiplomacyConfig,
) -> DiplomacyRunner | MultiAgentRunner:
    env = RisikoEnv(n_players=n_players)
    if cfg.enabled:
        return DiplomacyRunner(
            env,
            agents,
            max_turns=max_turns,
            diplomacy_cfg=cfg,
            reputation=book,
            profiles=profiles,
        )
    return MultiAgentRunner(env, agents, max_turns=max_turns)


def _play_games(
    agents: list[Agent],
    book: ReputationBook,
    n_games: int,
    seed: int,
    n_players: int,
    max_turns: int,
    profiles: Sequence[PlayerConfig] | None,
    cfg: DiplomacyConfig,
) -> tuple[int, int, int, int]:
    wins = draws = total_turns = total_calls = 0
    for i in range(n_games):
        runner = _build_runner(agents, n_players, max_turns, book, profiles, cfg)
        result = runner.run(seed=seed + i)
        wins += int(result.winner == _LEARNER_SLOT)
        draws += int(result.winner is None)
        total_turns += result.n_turns
        total_calls += runner.call_count if cfg.enabled else 0
        book.update(result)
    return wins, draws, total_turns, total_calls


def _build_result(
    wins: int,
    draws: int,
    n_games: int,
    total_turns: int,
    total_calls: int,
) -> SocialEvalResult:
    ci = wilson_ci(wins, n_games)
    _log.info(
        "social_eval — learner_win_rate=%.3f ci=[%.3f,%.3f] total_negotiation_calls=%d n_games=%d",
        wins / n_games,
        ci[0],
        ci[1],
        total_calls,
        n_games,
    )
    return SocialEvalResult(
        learner_win_rate=wins / n_games,
        ci_low=ci[0],
        ci_high=ci[1],
        draw_rate=draws / n_games,
        avg_length=total_turns / n_games,
        total_negotiation_calls=total_calls,
        avg_calls_per_game=float(total_calls / n_games),
    )


def _empty_result() -> SocialEvalResult:
    _log.info("social_eval — n_games=0, total_negotiation_calls=0")
    return SocialEvalResult(
        learner_win_rate=0.0,
        ci_low=0.0,
        ci_high=0.0,
        draw_rate=0.0,
        avg_length=0.0,
        total_negotiation_calls=0,
        avg_calls_per_game=0.0,
    )


__all__ = ["SocialEvalResult", "run_social_eval"]
