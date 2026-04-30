"""Tests that runtime dependencies are listed in requirements.txt."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestRequirements:
    """Critical dependencies must be declared."""

    @pytest.fixture
    def requirements(self) -> str:
        path = Path(__file__).parent.parent / "requirements.txt"
        return path.read_text()

    def test_baml_py_listed(self, requirements: str) -> None:
        """baml-py must be in requirements.txt."""
        assert "baml-py" in requirements

    def test_baml_py_version_pinned(self, requirements: str) -> None:
        """baml-py must have a minimum version pin."""
        assert "baml-py>=" in requirements

    def test_baml_py_importable(self) -> None:
        """baml_py must be importable at runtime."""
        import baml_py

        assert baml_py is not None
