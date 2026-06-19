"""Agent protocol definition."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Agent(Protocol):
    """Protocol for all Risiko agents."""

    def act(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> dict[str, int]:
        """Select an action given observation and legal actions."""
        ...

    def act_with_meta(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
        context=None,
    ) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
        """Select an action given observation and legal actions.

        Args:
            obs: Environment observation dict (numpy arrays).
            legal_actions: List of valid action dicts for the current state.
            deterministic: When ``True``, select greedily (used for evaluation).
            context: Optional DiplomacyContext injected by DiplomacyRunner for
                eval-only social conditioning. Training paths leave it ``None``.

        Returns:
            Selected action as a dict with keys ``action_type``, ``param_a``,
            ``param_b``, ``param_c``, ``param_d``.
        """
        ...
