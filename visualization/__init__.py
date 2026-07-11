"""Figures and game rendering for the Risiko project.

Submodules are re-exported so ``from visualization import data, plots`` resolves for
type checkers as well as at runtime.
"""

from __future__ import annotations

from . import data, map_layout, plots, theme

__all__ = ["data", "map_layout", "plots", "theme"]
