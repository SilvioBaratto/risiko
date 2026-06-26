"""CLI helper primitives for the tournament command (issue #129).

Pure, side-effect-free functions (except logging) that keep `risiko_rl/cli.py`
thin by centralising run-id resolution, path composition, and structured
run logging.

Do NOT call setup_logging() here — the CLI command owns that.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utils.log import get_logger

if TYPE_CHECKING:
    import logging

_log = get_logger("tournament_cli")

_DEFAULT_BASE = Path("results/tournament")
_LEDGER_FILENAME = "games.jsonl"


def build_run_id(
    config_name: str,
    run_id: str | None,
    *,
    now: datetime | None = None,
) -> str:
    """Return *run_id* verbatim when given; otherwise derive a stable id.

    Production default (``run_id=None, now=None``): returns *config_name*
    unchanged, giving a stable directory that resumes automatically across
    invocations — resume-by-default is the headline cycle requirement.

    When *now* is injected the result is
    ``f"{config_name}_{now:%Y%m%d_%H%M%S}"`` — useful in tests that need a
    unique, deterministic id without touching the production path.
    """
    if run_id is not None:
        return run_id
    if now is not None:
        return f"{config_name}_{now:%Y%m%d_%H%M%S}"
    return config_name


def tournament_run_dir(
    run_id: str,
    base: Path = _DEFAULT_BASE,
) -> Path:
    """Return ``base / run_id`` — the canonical run directory."""
    return base / run_id


def ledger_path(run_dir: Path) -> Path:
    """Return the games ledger path ``run_dir / 'games.jsonl'``."""
    return run_dir / _LEDGER_FILENAME


def log_run_header(
    log: logging.Logger,
    config: Any,
    seed: int,
    run_dir: Path,
) -> None:
    """Emit a structured header log before any game runs.

    Logs seed, full model roster, and run directory so each run is fully
    reproducible from its log alone.
    """
    log.info("tournament run starting — seed=%d run_dir=%s", seed, run_dir)
    log.info("model roster: %s", list(config.models))


def log_run_summary(log: logging.Logger, summary: Any) -> None:
    """Emit a structured summary log after all games complete."""
    log.info(
        "run complete — games_this_run=%d total_in_ledger=%d total_llm_calls=%d",
        summary.games_completed_this_run,
        summary.total_games_in_ledger,
        summary.total_llm_calls,
    )


__all__ = [
    "build_run_id",
    "tournament_run_dir",
    "ledger_path",
    "log_run_header",
    "log_run_summary",
]
