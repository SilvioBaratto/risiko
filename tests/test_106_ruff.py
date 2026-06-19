"""
Issue #106 — Acceptance criterion: ``ruff check . tests`` reports no issues.

Oracle classification: UNIT

Invokes ruff as a subprocess using the same Python interpreter that is running
pytest.  The test fails if ruff exits with a non-zero status (i.e. any lint
or format issue is found).

Note: this test is intentionally slow (subprocess + full project scan) and
should be run in the normal ``pytest -m "not integration"`` suite.  It is *not*
marked ``integration`` because it exercises a pure-code property (linting) that
does not require network access or external services.
"""

from __future__ import annotations

import subprocess
import sys


def test_when_ruff_check_runs_on_source_and_tests_then_no_issues_reported():
    """
    ``ruff check . tests`` must exit with code 0.
    Any lint issue reported by ruff causes this test to fail with the ruff
    output included in the assertion message so the cause is immediately visible.
    """
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", ".", "tests"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ruff check reported issues (exit code {result.returncode}):\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
