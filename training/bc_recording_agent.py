"""Per-decision capture wrapper around HeuristicAgent for BC dataset generation.

Every ``act``/``act_with_meta`` call:
1. Asks the inner agent for its preferred action (the BC label).
2. With probability ``explore_eps`` executes a uniformly random legal action
   (ε-greedy state coverage) while still recording the heuristic label.
3. Appends exactly one ``DecisionRecord`` to ``records``.

Determinism: all randomness flows through an injected ``np.random.Generator``.
The orchestrator derives a child generator per slot from ``cfg.bc.seed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.agents.base import Agent

# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    """Snapshot of a single agent decision at dataset-generation time."""

    obs: dict[str, Any]
    legal_actions: list[dict[str, int]]
    label_action: dict[str, int]
    label_index: int
    acting_player: int


# ---------------------------------------------------------------------------
# Helpers (pure, module-level)
# ---------------------------------------------------------------------------


def _find_label_index(legal: list[dict[str, int]], label: dict[str, int]) -> int:
    """Return the position of ``label`` in ``legal`` by value equality."""
    for i, a in enumerate(legal):
        if a == label:
            return i
    raise ValueError(
        f"label action {label!r} not found in legal_actions list of length {len(legal)}"
    )


def _random_action(legal: list[dict[str, int]], rng: np.random.Generator) -> dict[str, int]:
    """Return a uniformly random legal action using the injected generator."""
    return legal[int(rng.integers(len(legal)))]


# ---------------------------------------------------------------------------
# RecordingAgent
# ---------------------------------------------------------------------------


class RecordingAgent:
    """Agent-protocol wrapper that records every heuristic decision.

    Args:
        inner:       Any ``Agent``-protocol object supplying BC labels;
                     intended to be a ``HeuristicAgent``.
        explore_eps: Probability of executing a random legal action instead
                     of the heuristic's preferred action (ε-greedy coverage).
                     Must be in [0, 1].  0.0 = pure heuristic.
        rng:         Injected ``np.random.Generator`` for reproducibility.
                     Defaults to an unseeded generator when ``None``.
        records:     Shared list to accumulate ``DecisionRecord``s.
                     Callers may inject one list across all agent wrappers so
                     that records are index-aligned 1:1 with game trajectories.
    """

    def __init__(
        self,
        inner: Agent,
        *,
        explore_eps: float,
        rng: np.random.Generator | None = None,
        records: list[DecisionRecord] | None = None,
    ) -> None:
        """Create a RecordingAgent wrapping *inner* with ε-greedy exploration."""
        self._inner = inner
        self._explore_eps = explore_eps
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
        self.records: list[DecisionRecord] = records if records is not None else []

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    def act(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> dict[str, int]:
        """Return the executed action and append one ``DecisionRecord``."""
        executed, _, _ = self.act_with_meta(obs, legal_actions, deterministic=deterministic)
        return executed

    def act_with_meta(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
        """Return ``(executed_action, nan, nan)`` and append one ``DecisionRecord``."""
        label = self._inner.act(obs, legal_actions, deterministic=deterministic)
        label_idx = _find_label_index(legal_actions, label)
        executed = self._pick_action(legal_actions, label)
        self.records.append(
            DecisionRecord(
                obs=obs.copy(),
                legal_actions=list(legal_actions),
                label_action=label,
                label_index=label_idx,
                acting_player=int(obs["current_player"]),
            )
        )
        nan = torch.tensor(float("nan"))
        return executed, nan, nan

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pick_action(
        self,
        legal: list[dict[str, int]],
        label: dict[str, int],
    ) -> dict[str, int]:
        """Apply ε-greedy: explore with probability ``explore_eps``."""
        if self._explore_eps > 0.0 and self._rng.random() < self._explore_eps:
            return _random_action(legal, self._rng)
        return label
