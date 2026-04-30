"""Random agent that uniformly samples from legal actions."""

from __future__ import annotations

import numpy as np

from src.agents.base import Agent


class RandomAgent(Agent):
    """Agent that uniformly samples a legal action."""

    def __init__(self, seed: int | None = None) -> None:
        """Create a random agent with an optional seed for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def act(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> dict[str, int]:
        """Return a uniformly random legal action."""
        self._validate(legal_actions, deterministic)
        idx = int(self._rng.integers(len(legal_actions)))
        return legal_actions[idx]

    @staticmethod
    def _validate(legal_actions: list[dict[str, int]], deterministic: bool) -> None:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        if deterministic:
            raise ValueError("RandomAgent does not support deterministic mode")
