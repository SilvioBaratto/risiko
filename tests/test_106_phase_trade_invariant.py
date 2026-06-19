"""
Issue #106 — Acceptance criterion:
  ``get_legal_actions()`` is never empty in ``PHASE_TRADE`` under a forced trade.

Oracle classification: UNIT

Strategy
--------
Run a short deterministic episode (seed + first-legal-action policy).  Every
time the env reports ``phase == PHASE_TRADE``, assert that ``get_legal_actions()``
is non-empty.  The spec guarantees there is always at least one tradeable set
whenever a forced trade is required, so the list must never be empty in that
phase.

A Hypothesis property-based variant sweeps many seeds to catch seeds where
PHASE_TRADE is first reached, verifying the invariant is not seed-specific.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _collect_phase_trade_legals(env, seed: int, max_steps: int = 2000) -> list[list]:
    """
    Step through one episode and return a list whose entries are the
    ``get_legal_actions()`` snapshots taken while the env is in ``PHASE_TRADE``.
    Always picks the first legal action so the episode is deterministic.
    """
    from src.env import PHASE_TRADE  # noqa: PLC0415

    obs, _ = env.reset(seed=seed)
    snapshots: list[list] = []
    for _ in range(max_steps):
        legal = env.get_legal_actions()
        if obs.get("phase") == PHASE_TRADE:
            snapshots.append(list(legal))
        if not legal:
            break
        obs, _, done, _, _ = env.step(legal[0])
        if done:
            break
    return snapshots


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


def test_when_phase_trade_reached_with_seed_0_then_legal_actions_not_empty():
    """
    Whenever ``PHASE_TRADE`` is entered (forced trade active), at least one
    legal action must exist.  Seed 0 is the canonical deterministic example.
    """
    from src.env import RisikoEnv

    env = RisikoEnv(n_players=2)
    snapshots = _collect_phase_trade_legals(env, seed=0)

    for i, legal in enumerate(snapshots):
        assert len(legal) > 0, (
            f"get_legal_actions() returned an empty list at PHASE_TRADE "
            f"occurrence #{i} (seed=0).  At least one tradeable set must "
            f"always be available when a forced trade is required."
        )


def test_when_phase_trade_reached_with_seed_7_then_legal_actions_not_empty():
    """Second deterministic seed provides independent coverage of the invariant."""
    from src.env import RisikoEnv

    env = RisikoEnv(n_players=2)
    snapshots = _collect_phase_trade_legals(env, seed=7)

    for i, legal in enumerate(snapshots):
        assert len(legal) > 0, f"Empty legal-action list at PHASE_TRADE occurrence #{i} (seed=7)."


def test_when_phase_trade_reached_then_legal_actions_not_empty_across_multiple_seeds():
    """
    Three hand-picked seeds cover early, mid, and late-game PHASE_TRADE entries.
    All must yield non-empty legal-action lists in PHASE_TRADE.
    """
    from src.env import RisikoEnv

    for seed in (0, 42, 99):
        env = RisikoEnv(n_players=2)
        snapshots = _collect_phase_trade_legals(env, seed=seed)
        for i, legal in enumerate(snapshots):
            assert len(legal) > 0, (
                f"get_legal_actions() empty in PHASE_TRADE at occurrence #{i} (seed={seed})."
            )


# ---------------------------------------------------------------------------
# Property-based test — invariant over all seeds
# ---------------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=2**14 - 1))
@settings(max_examples=20)
def test_when_any_seed_enters_phase_trade_then_legal_actions_never_empty(seed: int):
    """
    Invariant: for every reachable ``PHASE_TRADE`` state across any seed in
    ``[0, 2^14)``, ``get_legal_actions()`` must be non-empty.

    This property is implied directly by the acceptance criterion: "a player
    holding any valid set at turn start is forced to trade, and
    ``get_legal_actions()`` is never empty in ``PHASE_TRADE`` under a forced
    trade."  Equivalently: a forced trade cannot be impossible to resolve.
    """
    from src.env import RisikoEnv

    env = RisikoEnv(n_players=2)
    for i, legal in enumerate(_collect_phase_trade_legals(env, seed=seed, max_steps=500)):
        assert len(legal) > 0, (
            f"seed={seed}: empty legal-action list at PHASE_TRADE occurrence #{i}."
        )
