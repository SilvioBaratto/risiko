"""
Issue #105 — disabled-equivalence regression.

Criterion: "diplomacy-disabled runner yields call_count == 0 and an action_log
identical to a plain MultiAgentRunner (fixed seed + mocked LLM)"

Uses DiplomacyRunner with DiplomacyConfig(enabled=False) — the proper API for
disabling diplomacy at the runner level.
"""

from __future__ import annotations

_FIXED_SEED = 42
_N_PLAYERS = 2
_MAX_TURNS = 10


def _random_agents(n: int):
    from src.agents.random_agent import RandomAgent

    return [RandomAgent(seed=i) for i in range(n)]


# ---------------------------------------------------------------------------
# call_count == 0 when diplomacy is disabled
# ---------------------------------------------------------------------------


def test_when_diplomacy_disabled_then_call_count_is_zero_before_run():
    """DiplomacyRunner.call_count starts at 0 when diplomacy is disabled."""
    from src.config import DiplomacyConfig
    from src.env import RisikoEnv
    from src.multi_agent import DiplomacyRunner

    env = RisikoEnv(n_players=_N_PLAYERS)
    runner = DiplomacyRunner(
        env,
        _random_agents(_N_PLAYERS),
        max_turns=_MAX_TURNS,
        diplomacy_cfg=DiplomacyConfig(enabled=False),
    )
    assert runner.call_count == 0


def test_when_diplomacy_disabled_then_call_count_is_zero_after_game():
    """After a full game with diplomacy disabled, call_count must still be 0."""
    from src.config import DiplomacyConfig
    from src.env import RisikoEnv
    from src.multi_agent import DiplomacyRunner

    env = RisikoEnv(n_players=_N_PLAYERS)
    runner = DiplomacyRunner(
        env,
        _random_agents(_N_PLAYERS),
        max_turns=_MAX_TURNS,
        diplomacy_cfg=DiplomacyConfig(enabled=False),
    )
    runner.run_game(seed=_FIXED_SEED)

    assert runner.call_count == 0, (
        f"DiplomacyRunner must make zero negotiation LLM calls when disabled; "
        f"got {runner.call_count}"
    )


# ---------------------------------------------------------------------------
# action_log equivalence
# ---------------------------------------------------------------------------


def test_when_diplomacy_disabled_then_action_log_length_matches_plain_runner():
    """DiplomacyRunner(enabled=False) must log the same number of actions as MultiAgentRunner."""
    from src.config import DiplomacyConfig
    from src.env import RisikoEnv
    from src.multi_agent import DiplomacyRunner, MultiAgentRunner

    env_a = RisikoEnv(n_players=_N_PLAYERS)
    result_plain = MultiAgentRunner(
        env_a, _random_agents(_N_PLAYERS), max_turns=_MAX_TURNS
    ).run_game(seed=_FIXED_SEED)

    env_b = RisikoEnv(n_players=_N_PLAYERS)
    result_disabled = DiplomacyRunner(
        env_b,
        _random_agents(_N_PLAYERS),
        max_turns=_MAX_TURNS,
        diplomacy_cfg=DiplomacyConfig(enabled=False),
    ).run_game(seed=_FIXED_SEED)

    assert len(result_plain.action_log) == len(result_disabled.action_log), (
        "Action log length must be identical between plain runner and disabled diplomacy runner"
    )


def test_when_diplomacy_disabled_then_action_log_matches_plain_runner():
    """Every action_log entry must be identical between plain and diplomacy-disabled runner."""
    from src.config import DiplomacyConfig
    from src.env import RisikoEnv
    from src.multi_agent import DiplomacyRunner, MultiAgentRunner

    env_a = RisikoEnv(n_players=_N_PLAYERS)
    result_plain = MultiAgentRunner(
        env_a, _random_agents(_N_PLAYERS), max_turns=_MAX_TURNS
    ).run_game(seed=_FIXED_SEED)

    env_b = RisikoEnv(n_players=_N_PLAYERS)
    result_disabled = DiplomacyRunner(
        env_b,
        _random_agents(_N_PLAYERS),
        max_turns=_MAX_TURNS,
        diplomacy_cfg=DiplomacyConfig(enabled=False),
    ).run_game(seed=_FIXED_SEED)

    assert result_plain.action_log == result_disabled.action_log, (
        "action_log must be identical — diplomacy scaffolding must not alter mechanical play"
    )


def test_when_diplomacy_disabled_then_winner_matches_plain_runner():
    """Same seed + same agents must produce the same winner with or without DiplomacyRunner."""
    from src.config import DiplomacyConfig
    from src.env import RisikoEnv
    from src.multi_agent import DiplomacyRunner, MultiAgentRunner

    env_a = RisikoEnv(n_players=_N_PLAYERS)
    result_plain = MultiAgentRunner(
        env_a, _random_agents(_N_PLAYERS), max_turns=_MAX_TURNS
    ).run_game(seed=_FIXED_SEED)

    env_b = RisikoEnv(n_players=_N_PLAYERS)
    result_disabled = DiplomacyRunner(
        env_b,
        _random_agents(_N_PLAYERS),
        max_turns=_MAX_TURNS,
        diplomacy_cfg=DiplomacyConfig(enabled=False),
    ).run_game(seed=_FIXED_SEED)

    assert result_plain.winner == result_disabled.winner, (
        f"Game winner differed: plain={result_plain.winner}, disabled={result_disabled.winner}"
    )
