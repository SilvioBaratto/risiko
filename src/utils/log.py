"""Centralized logging setup for risiko-rl.

Usage
-----
From any module::

    from src.utils.log import get_logger
    _log = get_logger("env")        # → logger named "risiko.env"
    _log.debug("...")
    _log.info("...")

Call ``setup_logging()`` once at process startup (done in cli.py).
Control verbosity with the ``RISIKO_LOG_LEVEL`` env var (default INFO).
Set ``RISIKO_LOG_LEVEL=DEBUG`` for full per-step traces.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

__all__ = ["get_logger", "setup_logging", "DEFAULT_LOG_FILE"]

DEFAULT_LOG_FILE = Path("logs/risiko_debug.log")


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'risiko' root namespace."""
    return logging.getLogger(f"risiko.{name}")


def setup_logging(
    level: str | None = None,
    log_file: Path | None = None,
) -> None:
    """Configure the root 'risiko' logger with console + file handlers.

    Levels:
      * file handler:    argument > RISIKO_LOG_LEVEL env var > INFO
      * console handler: RISIKO_CONSOLE_LEVEL env var > same as file handler

    Set ``RISIKO_CONSOLE_LEVEL=WARNING`` (or ERROR) to silence the terminal
    while keeping full DEBUG/INFO output in the file (useful for clean
    progress-bar-only training runs).
    """
    level_str = (level or os.getenv("RISIKO_LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, level_str, logging.INFO)

    console_level_str = os.getenv("RISIKO_CONSOLE_LEVEL", level_str).upper()
    console_level = getattr(logging, console_level_str, numeric_level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d — %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger("risiko")
    root.setLevel(min(numeric_level, console_level))
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file is None:
        log_file = DEFAULT_LOG_FILE
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    root.info(
        "logging initialised — file=%s level=%s console=%s",
        log_file,
        level_str,
        console_level_str,
    )
