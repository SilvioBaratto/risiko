"""Run one LLM tournament game and save a full JSON trace.

Captures, for every LLM call in a single game, the exact prompt sent, the raw
model response, the model's `thinking` trace (when it emits one), and the parsed
action / negotiation — plus the final game result. Useful for visualization,
debugging, and inspecting *why* a strategy won.

Usage:
    python -m visualization.trace_game                      # game 0, full game
    python -m visualization.trace_game --game-index 3
    python -m visualization.trace_game --max-turns 20       # short trace (cheap)
    python -m visualization.trace_game --out visualization/my_game.json

Output: visualization/game_trace_<run>_g<index>.json (unless --out given).

Note: this runs a real game against Ollama and consumes quota. Use --max-turns
to cap a quick sample; the default plays a full game (can be long + quota-heavy).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import src.agents.ollama_client as _oc
from src.utils.env import ensure_env_loaded
from src.utils.log import get_logger
from training.tournament import (
    _play_one_game,
    _to_serialisable,
    build_game_plan,
    load_tournament_config,
)

_log = get_logger("trace_game")


def trace_one_game(
    config_path: Path,
    game_index: int = 0,
    max_turns: int | None = None,
) -> dict[str, Any]:
    """Play one game with call tracing on; return a JSON-serialisable trace dict."""
    ensure_env_loaded()
    cfg = load_tournament_config(config_path)
    if max_turns is not None:
        cfg = dataclasses.replace(cfg, max_turns=max_turns)
    plan = build_game_plan(cfg, game_index)

    _log.info(
        "tracing game %d (seed=%d, max_turns=%d) — assignment=%s",
        game_index,
        plan.seed,
        cfg.max_turns,
        plan.assignment,
    )

    _oc.start_call_trace()
    try:
        record = _play_one_game(plan, cfg)
    finally:
        calls = _oc.stop_call_trace() or []

    return {
        "run_id": cfg.name,
        "game_index": game_index,
        "seed": plan.seed,
        "max_turns": cfg.max_turns,
        "n_rounds": cfg.n_rounds,
        "negotiation_cadence": cfg.negotiation_cadence,
        "draw_resolution": cfg.draw_resolution,
        "models": list(cfg.models),
        "strategies": list(cfg.strategies),
        "assignment": dict(plan.assignment),
        "result": _to_serialisable(record.__dict__),
        "n_calls": len(calls),
        "calls": _to_serialisable(calls),
    }


def _default_out(trace: dict[str, Any]) -> Path:
    return Path("visualization") / f"game_trace_{trace['run_id']}_g{trace['game_index']}.json"


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: trace one game and write the JSON to disk."""
    parser = argparse.ArgumentParser(description="Save a full JSON trace of one LLM game.")
    parser.add_argument("--config", type=Path, default=Path("config/tournament.yaml"))
    parser.add_argument("--game-index", type=int, default=0)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override the per-game turn cap (use a small value for a quick, cheap sample).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path.")
    args = parser.parse_args(argv)

    trace = trace_one_game(args.config, args.game_index, args.max_turns)
    out = args.out or _default_out(trace)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=2, ensure_ascii=False))

    n_think = sum(1 for c in trace["calls"] if c.get("thinking"))
    _log.info(
        "saved %d calls (%d with thinking) → %s | winner=%s (%s)",
        trace["n_calls"],
        n_think,
        out,
        trace["result"].get("winner_strategy"),
        trace["result"].get("winner_model"),
    )
    print(f"Saved game trace → {out}  ({trace['n_calls']} LLM calls, {n_think} with thinking)")


if __name__ == "__main__":
    main()
