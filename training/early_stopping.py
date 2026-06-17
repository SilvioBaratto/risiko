"""Plateau-detection early stopping for self-play training.

A small, pure-logic helper: feed it a monitored metric (win-rate vs a fixed
baseline) at each evaluation and it reports when the smoothed metric has
stopped improving for ``patience`` consecutive evaluations.

Design follows the self-play / sparse-reward literature:

* **Smooth before comparing.** RL evaluation metrics are noisy, so the decision
  is made on a trailing moving average (``window``), not the raw value.
* **Require a real gain.** Only an improvement greater than ``min_delta`` over
  the running best resets the patience counter (maximisation).
* **Track the best.** ``is_best`` flags the evaluation that set a new best so
  the caller can snapshot a "restore-best" checkpoint, since self-play agents
  often degrade past their peak.
"""

from __future__ import annotations

__all__ = ["EarlyStopper"]


class EarlyStopper:
    """Detects a plateau in a maximised, noisy metric.

    Call :meth:`update` once per evaluation with the latest metric value; it
    returns ``True`` once the smoothed metric has failed to improve by more
    than ``min_delta`` for ``patience`` consecutive evaluations.
    """

    def __init__(self, patience: int, min_delta: float = 0.0, window: int = 1) -> None:
        """Initialise the stopper.

        Args:
            patience: Evaluations without improvement tolerated before stopping.
            min_delta: Smallest increase in the smoothed metric counted as an
                improvement.
            window: Trailing window size for the moving average; ``1`` disables
                smoothing.
        """
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if window < 1:
            raise ValueError("window must be >= 1")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")
        self._patience = patience
        self._min_delta = min_delta
        self._window = window
        self._history: list[float] = []
        self._best: float = float("-inf")
        self._best_step: int = -1
        self._num_bad: int = 0
        self._is_best: bool = False

    def __call__(self, value: float) -> bool:
        """Convenience wrapper — ``stopper(metric)`` delegates to :meth:`update`."""
        return self.update(value)

    def update(self, value: float) -> bool:
        """Record an evaluation and report whether training should stop.

        Args:
            value: The latest raw metric value (higher is better).

        Returns:
            ``True`` if the smoothed metric has not improved by ``min_delta``
            for ``patience`` consecutive evaluations.
        """
        self._history.append(value)
        smoothed = self.smoothed_value
        if smoothed > self._best + self._min_delta:
            self._best = smoothed
            self._best_step = len(self._history) - 1
            self._num_bad = 0
            self._is_best = True
        else:
            self._is_best = False
            self._num_bad += 1
        return self.should_stop

    @property
    def smoothed_value(self) -> float:
        """Trailing moving average of the last ``window`` recorded values."""
        recent = self._history[-self._window :]
        return sum(recent) / len(recent)

    @property
    def is_best(self) -> bool:
        """Whether the most recent :meth:`update` set a new smoothed best."""
        return self._is_best

    @property
    def best_value(self) -> float:
        """Best smoothed value seen so far (``-inf`` before any update)."""
        return self._best

    @property
    def best_step(self) -> int:
        """Zero-based index of the evaluation that set the current best."""
        return self._best_step

    @property
    def num_bad_evals(self) -> int:
        """Consecutive evaluations since the last improvement."""
        return self._num_bad

    @property
    def should_stop(self) -> bool:
        """Whether the patience budget is exhausted."""
        return self._num_bad >= self._patience
