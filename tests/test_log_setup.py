"""Tests for the dual console/file logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.utils import log as log_module
from src.utils.log import get_logger, setup_logging


def _split_handlers(
    handlers: list[logging.Handler],
) -> tuple[list[logging.FileHandler], list[logging.StreamHandler]]:
    """Partition handlers into file handlers and (non-file) console handlers."""
    files = [h for h in handlers if isinstance(h, logging.FileHandler)]
    consoles = [
        h
        for h in handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    return files, consoles


@pytest.fixture
def log_file(tmp_path: Path) -> Path:
    """A log file path inside a not-yet-created subdirectory."""
    return tmp_path / "nested" / "dir" / "risiko.log"


class TestGetLogger:
    """get_logger namespacing."""

    def test_when_named_then_child_of_risiko_namespace(self) -> None:
        """get_logger('env') returns the 'risiko.env' logger."""
        assert get_logger("env").name == "risiko.env"


class TestSetupLogging:
    """setup_logging handler installation and idempotency."""

    def test_when_setup_called_then_exactly_two_handlers(self, log_file: Path) -> None:
        """Console + file handlers are installed (exactly two)."""
        setup_logging(log_file=log_file)
        root = logging.getLogger("risiko")
        assert len(root.handlers) == 2

    def test_when_setup_called_twice_then_no_duplicate_handlers(self, log_file: Path) -> None:
        """Idempotent: a second call must not accumulate handlers."""
        setup_logging(log_file=log_file)
        setup_logging(log_file=log_file)
        root = logging.getLogger("risiko")
        assert len(root.handlers) == 2

    def test_when_setup_called_then_one_file_and_one_console_handler(self, log_file: Path) -> None:
        """Handlers are one FileHandler and one console StreamHandler."""
        setup_logging(log_file=log_file)
        files, consoles = _split_handlers(logging.getLogger("risiko").handlers)
        assert len(files) == 1
        assert len(consoles) == 1

    def test_when_parent_dir_missing_then_created(self, log_file: Path) -> None:
        """The log-file parent directory is auto-created."""
        assert not log_file.parent.exists()
        setup_logging(log_file=log_file)
        assert log_file.parent.exists()

    def test_when_env_levels_set_then_console_and_file_levels_independent(
        self, log_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RISIKO_LOG_LEVEL and RISIKO_CONSOLE_LEVEL configure handlers independently."""
        monkeypatch.setenv("RISIKO_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("RISIKO_CONSOLE_LEVEL", "WARNING")
        setup_logging(log_file=log_file)
        files, consoles = _split_handlers(logging.getLogger("risiko").handlers)
        assert files[0].level == logging.DEBUG
        assert consoles[0].level == logging.WARNING


class TestModuleExports:
    """Public API surface of the logging module."""

    def test_when_imported_then_all_lists_public_api(self) -> None:
        """__all__ exports get_logger, setup_logging, DEFAULT_LOG_FILE."""
        assert set(log_module.__all__) == {"get_logger", "setup_logging", "DEFAULT_LOG_FILE"}

    def test_when_imported_then_default_log_file_is_path(self) -> None:
        """DEFAULT_LOG_FILE is a Path under logs/."""
        expected = Path("logs/risiko_debug.log")
        assert expected == log_module.DEFAULT_LOG_FILE
