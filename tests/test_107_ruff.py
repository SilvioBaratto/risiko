"""
Issue #107 — Acceptance criterion (UNIT):
    ruff check . tests reports no issues.

Source-blind: this test is mechanically derived from the criterion text.
It runs ruff as a subprocess so the check is authoritative and format-agnostic.
"""

import subprocess
import sys


def test_ruff_check_reports_no_issues():
    """Ruff check . tests must exit with code 0 (no lint errors)."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", ".", "tests"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ruff reported issues:\n{result.stdout}\n{result.stderr}"
