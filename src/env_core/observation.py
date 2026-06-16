"""Raw 11-key Dict observation builder for the Risiko environment (Issue #52)."""

from __future__ import annotations

import numpy as np

from src.env_core.state import MAX_CARDS, GameState
from src.utils.constants import CONTINENTS

__all__ = ["build_obs", "build_info", "CONTINENTS"]


def _build_card_matrix(state: GameState) -> np.ndarray:
    """Return (MAX_CARDS, 4) int32 one-hot matrix for the current player's hand.

    Cards in state.cards[player] are symbol integers 0-3 (Infantry/Cavalry/Artillery/Wildcard).
    """
    matrix = np.zeros((MAX_CARDS, 4), dtype=np.int32)
    player_hand = state.cards[state.current_player]
    for i, sym in enumerate(player_hand[:MAX_CARDS]):
        matrix[i, int(sym)] = 1
    return matrix


def _build_continent_control(state: GameState) -> np.ndarray:
    """Return (6,) int32 array: 1 if current player owns the whole continent."""
    return np.array(
        [
            int(all(state.territory_owner[t] == state.current_player for t in ids))
            for ids in CONTINENTS.values()
        ],
        dtype=np.int32,
    )


def _pad_eliminated(state: GameState) -> np.ndarray:
    """Return (6,) int32 eliminated flags, padding unused slots with 1."""
    return np.pad(
        state.eliminated.copy(),
        (0, 6 - state.n_players),
        mode="constant",
        constant_values=1,
    )


def build_obs(state: GameState) -> dict[str, np.ndarray]:
    """Build the 11-key raw Dict observation from GameState.

    Returns copies of all arrays so later state mutation does not alias the observation.
    """
    return {
        "territory_owner": state.territory_owner.copy(),
        "armies": state.armies.copy(),
        "phase": np.array(state.phase, dtype=np.int32),
        "current_player": np.array(state.current_player, dtype=np.int32),
        "cards": _build_card_matrix(state),
        "continent_control": _build_continent_control(state),
        "trade_count": np.array(state.trade_count, dtype=np.int32),
        "reinforcements_remaining": np.array(state.reinforcements_remaining, dtype=np.int32),
        "turn_capture": np.array(state.turn_capture, dtype=np.int32),
        "n_players": np.array(state.n_players, dtype=np.int32),
        "eliminated": _pad_eliminated(state),
    }


def build_info(state: GameState, legal_actions: list) -> dict:
    """Return the info payload with the legal-action list."""
    return {"legal_actions": legal_actions}
