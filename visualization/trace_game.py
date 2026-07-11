"""Run one LLM tournament game and save a full JSON trace.

Captures, for every LLM call in a single game, the exact prompt sent, the raw
model response, the model's `thinking` trace (when it emits one), and the parsed
action / negotiation — plus a snapshot of the board after every env step and the
final game result. Useful for visualization, debugging, and inspecting *why* a
strategy won.

The board snapshots are what ``visualization/render_game.py`` replays into an
animated GIF: territory owners, army counts, and the action that produced them.

Usage:
    python -m visualization.trace_game                      # game 0, full game
    python -m visualization.trace_game --game-index 3
    python -m visualization.trace_game --max-turns 20       # short trace (cheap)
    python -m visualization.trace_game --think              # capture reasoning traces
    python -m visualization.trace_game --gif figures/partita.gif
    python -m visualization.trace_game --out visualization/my_game.json

Output: visualization/game_trace_<run>_g<index>.json (unless --out given).

Note: this runs a real game against Ollama and consumes quota. Use --max-turns
to cap a quick sample; the default plays a full game (can be long + quota-heavy).
``--think`` multiplies both latency and quota — it is off in the tournament.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import threading
from pathlib import Path
from typing import Any

import src.agents.ollama_client as _oc
import src.multi_agent as _ma
from src.utils.env import ensure_env_loaded
from src.utils.log import get_logger
from training.tournament import (
    _play_one_game,
    _to_serialisable,
    build_game_plan,
    load_tournament_config,
)
from visualization import replay as _replay

_log = get_logger("trace_game")

# A deliberating model takes far longer than the tournament's 120s per action. The
# watchdog scores an overrun as a random fallback, so tracing with the tournament's
# timeout would capture exactly the moves the model never got to make.
THINK_ACTION_TIMEOUT = 900.0

# How often the running trace is flushed to disk when --out is given.
PROGRESS_FLUSH_SECONDS = 60.0


def trace_one_game(
    config_path: Path,
    game_index: int = 0,
    max_turns: int | None = None,
    *,
    think: bool = False,
    action_timeout: float | None = None,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    """Play one game with call + board tracing on; return a JSON-serialisable trace.

    Args:
        config_path: Tournament YAML to draw the roster and seeding from.
        game_index: Which game of the plan to replay (fixes the seed and the
            strategy-to-model assignment).
        max_turns: Override the per-game turn cap. Small values keep the quota cost down.
        think: Ask the models to deliberate, so the ``thinking`` field is populated.
        action_timeout: Override the per-call HTTP timeout. Deliberation routinely blows
            past the tournament's 120s — and a timeout is scored as a random fallback,
            which is exactly the move a trace is trying to capture. Defaults to
            ``THINK_ACTION_TIMEOUT`` when *think* is on.
        progress_path: Where to flush the trace-so-far every
            ``PROGRESS_FLUSH_SECONDS``, and again if the game aborts. Without it a
            crash after an hour of LLM calls leaves nothing behind.

    Returns:
        The trace: config, assignment, every LLM call, every board snapshot, and the
        game record.
    """
    ensure_env_loaded()
    cfg = load_tournament_config(config_path)
    overrides: dict[str, Any] = {}
    if max_turns is not None:
        overrides["max_turns"] = max_turns
    if think:
        overrides["think"] = True
    timeout = action_timeout or (THINK_ACTION_TIMEOUT if think else None)
    if timeout is not None:
        overrides["action_timeout"] = float(timeout)
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    plan = build_game_plan(cfg, game_index)

    _log.info(
        "tracing game %d (seed=%d, max_turns=%d, think=%s) — assignment=%s",
        game_index,
        plan.seed,
        cfg.max_turns,
        cfg.think,
        plan.assignment,
    )

    calls = _oc.start_call_trace()
    snapshots = _ma.start_board_trace()

    # A traced game runs for an hour or more and writes nothing until it returns, so a
    # crash — or an impatient Ctrl-C — throws away every LLM call it paid for. Flush what
    # exists to disk on a timer, and once more on the way out however it exits.
    stop_flush = threading.Event()
    flusher: threading.Thread | None = None
    if progress_path is not None:

        def _flush_periodically() -> None:
            while not stop_flush.wait(PROGRESS_FLUSH_SECONDS):
                _write_partial(progress_path, cfg, plan, game_index, calls, snapshots)

        flusher = threading.Thread(target=_flush_periodically, daemon=True)
        flusher.start()

    try:
        record = _play_one_game(plan, cfg)
    except BaseException:
        if progress_path is not None:
            _write_partial(progress_path, cfg, plan, game_index, calls, snapshots)
            _log.warning("game aborted — partial trace kept at %s", progress_path)
        raise
    finally:
        stop_flush.set()
        if flusher is not None:
            flusher.join(timeout=5)
        _oc.stop_call_trace()
        _ma.stop_board_trace()

    trace = _header(cfg, plan, game_index)
    trace["result"] = _to_serialisable(record.__dict__)
    trace["n_calls"] = len(calls)
    trace["calls"] = _to_serialisable(calls)
    trace["n_snapshots"] = len(snapshots)
    trace["board_snapshots"] = _to_serialisable(snapshots)
    return trace


def _header(cfg: Any, plan: Any, game_index: int) -> dict[str, Any]:
    """Return the config half of a trace — everything known before the game runs."""
    return {
        "run_id": cfg.name,
        "game_index": game_index,
        "seed": plan.seed,
        "max_turns": cfg.max_turns,
        "n_rounds": cfg.n_rounds,
        "negotiation_cadence": cfg.negotiation_cadence,
        "draw_resolution": cfg.draw_resolution,
        "think": cfg.think,
        "action_timeout": cfg.action_timeout,
        "models": list(cfg.models),
        "strategies": list(cfg.strategies),
        "assignment": dict(plan.assignment),
    }


def _write_partial(
    path: Path,
    cfg: Any,
    plan: Any,
    game_index: int,
    calls: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
) -> None:
    """Dump the trace collected so far, atomically, while the game is still running.

    The sinks are appended to by the game thread, so they are copied by slicing before
    serialising — a live list would otherwise mutate underneath ``json.dumps``.
    """
    trace = _header(cfg, plan, game_index)
    calls_now, snapshots_now = calls[:], snapshots[:]
    trace["partial"] = True
    trace["result"] = None
    trace["n_calls"] = len(calls_now)
    trace["calls"] = _to_serialisable(calls_now)
    trace["n_snapshots"] = len(snapshots_now)
    trace["board_snapshots"] = _to_serialisable(snapshots_now)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(trace, indent=2, ensure_ascii=False))
    tmp.replace(path)  # atomic: a reader never sees a half-written file
    _log.info(
        "partial trace: %d calls, %d snapshots → %s",
        len(calls_now),
        len(snapshots_now),
        path,
    )


def export_gif(
    trace: dict[str, Any],
    path: Path,
    every: int = 3,
    duration_ms: int = 900,
) -> Path:
    """Replay a traced game into an animated GIF.

    Thin delegate to :mod:`visualization.replay`, which owns the frame design. Kept here
    so a caller that already has a trace does not need to know where the renderer lives.

    Args:
        trace: A finished trace from :func:`trace_one_game`.
        path: Destination ``.gif``.
        every: Keep one board snapshot out of every N.
        duration_ms: How long each frame is held.

    Returns:
        The written path.

    Raises:
        ValueError: If the trace carries no board snapshots.
    """
    return _replay.export_gif(trace, path, every=every, duration_ms=duration_ms)


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
    parser.add_argument(
        "--think",
        action="store_true",
        help="Let the models deliberate, so the `thinking` field is populated (slow, quota-heavy).",
    )
    parser.add_argument(
        "--action-timeout",
        type=float,
        default=None,
        help=(
            "Per-call HTTP timeout (s). Defaults to the config value, or "
            f"{int(THINK_ACTION_TIMEOUT)}s with --think."
        ),
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        help="Also replay the board snapshots into an animated GIF at this path.",
    )
    parser.add_argument(
        "--gif-every",
        type=int,
        default=10,
        help="Keep one board snapshot out of every N when building the GIF (default 10).",
    )
    parser.add_argument(
        "--from-trace",
        type=Path,
        default=None,
        help=(
            "Rebuild the GIF from an existing trace JSON instead of playing a game. "
            "Offline and instant — no Ollama call, no quota."
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path.")
    args = parser.parse_args(argv)

    if args.from_trace:
        _rebuild_gif(args)
        return

    out = args.out or Path("visualization") / f"game_trace_g{args.game_index}.json"
    trace = trace_one_game(
        args.config,
        args.game_index,
        args.max_turns,
        think=args.think,
        action_timeout=args.action_timeout,
        progress_path=out,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=2, ensure_ascii=False))

    n_think = sum(1 for c in trace["calls"] if c.get("thinking"))
    _log.info(
        "saved %d calls (%d with thinking), %d board snapshots → %s | winner=%s (%s)",
        trace["n_calls"],
        n_think,
        trace["n_snapshots"],
        out,
        trace["result"].get("winner_strategy"),
        trace["result"].get("winner_model"),
    )
    print(
        f"Saved game trace → {out}  ({trace['n_calls']} LLM calls, {n_think} with thinking, "
        f"{trace['n_snapshots']} board snapshots)"
    )

    if args.gif:
        gif = export_gif(trace, args.gif, every=args.gif_every)
        print(f"Saved replay → {gif}")


def _rebuild_gif(args: argparse.Namespace) -> None:
    """Redraw the replay from a saved trace — no game, no LLM, no quota."""
    if not args.gif:
        raise SystemExit("--from-trace needs --gif <path> to write to")

    trace = json.loads(args.from_trace.read_text())
    gif = export_gif(trace, args.gif, every=args.gif_every)
    print(f"Rebuilt replay from {args.from_trace} → {gif} ({trace['n_snapshots']} snapshots)")


if __name__ == "__main__":
    main()
