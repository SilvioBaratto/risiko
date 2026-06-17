"""BC dataset generation orchestrator.

Runs ``cfg.bc.n_games`` self-play games using ``HeuristicAgent`` (all seats)
wrapped in ``RecordingAgent``, then writes (obs, label, mc_return) transitions
to sharded ``.npz`` files via ``ShardWriter``.

Reward-attribution note
-----------------------
``compute_per_player_returns`` receives the POST-step ``current_player`` as the
reward recipient (from ``result.trajectories[t].obs['current_player']``).  On
turn-ending steps the env calls ``_next_player()`` before computing the reward,
so ``trajectories[t].reward`` is credited to the NEXT player — not the
fortifying agent.  ``acting_player`` in the shard still records the PRE-step
decision-maker (policy supervision).  The value target carries a small,
documented bias (~1 step per player-turn) on those steps; this is deliberate
(option a from the reviewer comment).

ε-greedy note
-------------
With ``explore_eps > 0`` the executed action may be random, but the recorded
``label_index`` is always the heuristic's choice.  The ``mc_return`` is computed
from the executed reward stream, making the value target mildly off-policy on
explore steps.  This aids state coverage while keeping policy labels clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.agents.heuristic_agent import HeuristicAgent
from src.env import RisikoEnv
from src.multi_agent import GameResult, MultiAgentRunner
from src.utils.log import get_logger
from src.utils.seed import set_global_seeds
from training.bc_recording_agent import DecisionRecord, RecordingAgent
from training.bc_returns import compute_per_player_returns
from training.bc_shard_writer import ShardWriter, encode_obs

_log = get_logger("bc_dataset")


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class DatasetStats:
    """Aggregate statistics returned by ``generate_bc_dataset``."""

    total_rows: int
    n_games: int
    mean_game_length: float
    win_distribution: dict[int | None, int]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_bc_dataset(cfg: Any) -> DatasetStats:
    """Run n_games HeuristicAgent self-play games and write sharded .npz dataset.

    Args:
        cfg: Config with ``cfg.bc`` (BCConfig or duck-typed) and ``cfg.ppo.gamma``.

    Returns:
        ``DatasetStats`` with aggregate metrics.
    """
    set_global_seeds(cfg.bc.seed)
    base_rng = np.random.default_rng(cfg.bc.seed)
    _log.info(
        "generate_bc_dataset: n_games=%d seed=%d dir=%s",
        cfg.bc.n_games,
        cfg.bc.seed,
        cfg.bc.dataset_dir,
    )
    agents, shared_records = _make_agents(cfg, base_rng)
    runner = _build_runner(cfg, agents)
    stats = _run_all_games(cfg, runner, shared_records)
    _log.info(
        "done: rows=%d games=%d mean_len=%.1f wins=%s",
        stats.total_rows,
        stats.n_games,
        stats.mean_game_length,
        stats.win_distribution,
    )
    return stats


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def _make_agents(
    cfg: Any, base_rng: np.random.Generator
) -> tuple[list[RecordingAgent], list[DecisionRecord]]:
    """Create n_players ``RecordingAgent``s sharing a single records list."""
    hcfg = getattr(cfg.bc, "heuristic", None)
    shared: list[DecisionRecord] = []
    agents = [
        RecordingAgent(
            HeuristicAgent(config=hcfg, rng=np.random.default_rng(base_rng.integers(2**31))),
            explore_eps=cfg.bc.explore_eps,
            rng=np.random.default_rng(base_rng.integers(2**31)),
            records=shared,
        )
        for _ in range(cfg.bc.n_players)
    ]
    return agents, shared


def _build_runner(cfg: Any, agents: list[RecordingAgent]) -> MultiAgentRunner:
    """Create a ``MultiAgentRunner`` with the given agents."""
    reward_config = getattr(cfg, "reward", None)
    env = RisikoEnv(n_players=cfg.bc.n_players, reward_config=reward_config)
    return MultiAgentRunner(env, agents, max_turns=cfg.bc.max_turns)


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------


def _run_all_games(
    cfg: Any,
    runner: MultiAgentRunner,
    shared_records: list[DecisionRecord],
) -> DatasetStats:
    """Run all games, writing rows to ShardWriter; return aggregate stats."""
    total_rows, total_turns = 0, 0
    win_dist: dict[int | None, int] = {}
    log_interval = max(1, cfg.bc.n_games // 10)
    with ShardWriter(cfg.bc) as writer:
        for i in range(cfg.bc.n_games):
            rows, turns, winner = _run_one_game(
                runner, shared_records, writer, cfg.bc.seed + i, cfg.ppo.gamma
            )
            total_rows += rows
            total_turns += turns
            win_dist[winner] = win_dist.get(winner, 0) + 1
            if (i + 1) % log_interval == 0:
                _log.info("progress: %d/%d games rows=%d", i + 1, cfg.bc.n_games, total_rows)
    n = cfg.bc.n_games
    return DatasetStats(total_rows, n, total_turns / n if n else 0.0, win_dist)


def _run_one_game(
    runner: MultiAgentRunner,
    records: list[DecisionRecord],
    writer: ShardWriter,
    game_seed: int,
    gamma: float,
) -> tuple[int, int, int | None]:
    """Run one game, write its rows, return (rows_written, turns, winner)."""
    records.clear()
    result = runner.run_game(seed=game_seed)
    rows = _write_game_rows(writer, records, result, gamma)
    return rows, result.n_turns, result.winner


# ---------------------------------------------------------------------------
# Row writing
# ---------------------------------------------------------------------------


def _write_game_rows(
    writer: ShardWriter,
    records: list[DecisionRecord],
    result: GameResult,
    gamma: float,
) -> int:
    """Encode and write all decision rows for one game; return count."""
    n = len(records)
    n_traj = len(result.trajectories)
    assert n == n_traj, (
        f"records/trajectories length mismatch: {n} records vs {n_traj} trajectories"
    )
    if n == 0:
        return 0
    returns = _compute_returns(result, gamma)
    for i, rec in enumerate(records):
        writer.write(encode_obs(rec.obs), _row_labels(rec, returns[i]))
    return n


def _compute_returns(result: GameResult, gamma: float) -> list[float]:
    """Compute per-step MC returns using POST-step reward attribution."""
    rewards = [float(t.reward) for t in result.trajectories]
    players = [int(t.obs["current_player"]) for t in result.trajectories]
    return compute_per_player_returns(rewards, players, gamma)


def _row_labels(rec: DecisionRecord, mc_return: float) -> dict:
    """Build the label dict for one dataset row."""
    return {
        "action_type": rec.label_action["action_type"],
        "param_a": rec.label_action["param_a"],
        "param_b": rec.label_action["param_b"],
        "param_c": rec.label_action["param_c"],
        "param_d": rec.label_action["param_d"],
        "label_index": rec.label_index,
        "acting_player": rec.acting_player,
        "mc_return": mc_return,
    }
