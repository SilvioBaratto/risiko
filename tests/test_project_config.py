"""Smoke tests for project configuration and test infrastructure."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_toml_exists_and_is_valid():
    """pyproject.toml must exist and be valid TOML."""
    path = REPO_ROOT / "pyproject.toml"
    assert path.exists(), "pyproject.toml not found"
    with open(path, "rb") as f:
        data = tomllib.load(f)
    assert data, "pyproject.toml is empty"


def test_pyproject_has_build_system():
    """pyproject.toml must declare a build system."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    assert "build-system" in data, "missing [build-system]"
    assert "requires" in data["build-system"], "missing build-system.requires"


def test_pyproject_has_project_metadata():
    """pyproject.toml must contain package metadata."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    assert "project" in data, "missing [project]"
    project = data["project"]
    for key in ("name", "version", "description", "requires-python"):
        assert key in project, f"missing project.{key}"


def test_pyproject_has_scripts_entry_point():
    """pyproject.toml must define the CLI entry point."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    scripts = data.get("project", {}).get("scripts", {})
    assert "risiko-rl" in scripts, "missing 'risiko-rl' console script"


def test_pyproject_has_pytest_config():
    """pyproject.toml must configure pytest."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    assert "tool" in data, "missing [tool]"
    assert "pytest" in data["tool"], "missing [tool.pytest]"


def test_pyproject_has_coverage_config():
    """pyproject.toml must configure coverage with env.py target."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    cov = data.get("tool", {}).get("coverage", {})
    run = cov.get("run", {})
    assert "source" in run, "missing coverage.run.source"
    assert "src" in run["source"], "src not in coverage sources"


def test_pyproject_has_ruff_config():
    """pyproject.toml must configure ruff."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    ruff = data.get("tool", {}).get("ruff", {})
    assert "target-version" in ruff, "missing ruff.target-version"


def test_requirements_has_gymnasium():
    """requirements.txt must include gymnasium."""
    req = (REPO_ROOT / "requirements.txt").read_text()
    assert "gymnasium" in req, "gymnasium missing from requirements.txt"


def test_requirements_has_typer():
    """requirements.txt must include typer."""
    req = (REPO_ROOT / "requirements.txt").read_text()
    assert "typer" in req, "typer missing from requirements.txt"


def test_tests_directory_exists():
    """tests/ directory must exist with __init__.py."""
    tests_dir = REPO_ROOT / "tests"
    assert tests_dir.exists(), "tests/ directory not found"
    assert tests_dir.is_dir(), "tests/ is not a directory"
    assert (tests_dir / "__init__.py").exists(), "tests/__init__.py not found"


def test_package_is_editable_installable():
    """Package must install in editable mode without error."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"editable install failed: {result.stderr}"


def test_pytest_runs_from_repo_root():
    """Pytest --version must succeed from repo root."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"pytest --version failed: {result.stderr}"


def test_ruff_check_runs():
    """Ruff check . must exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ruff check failed: {result.stderr}"


def test_ruff_format_runs():
    """Ruff format . must exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ruff format failed: {result.stderr}"
