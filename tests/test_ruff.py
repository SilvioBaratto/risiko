"""Lint gate for issue #88: `ruff check . tests` must report no issues.

Source-blind: derived from the acceptance criterion
  [UNIT] `ruff check . tests` reports no issues.
"""

from __future__ import annotations

import subprocess


def test_when_ruff_check_is_run_then_no_issues_are_reported() -> None:
    """Ruff check . tests must exit 0 (no lint errors) across the full codebase."""
    result = subprocess.run(
        ["ruff", "check", ".", "tests"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ruff check reported issues:\n{result.stdout}\n{result.stderr}"
