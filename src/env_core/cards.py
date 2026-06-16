"""Card system for Risiko: set validation, draw, escalating trade, forced trade, transfer.

Cards in state.cards[player] are stored as symbol integers:
  0 = Infantry, 1 = Cavalry, 2 = Artillery, 3 = Wildcard.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from src.env_core.state import MAX_CARDS
from src.utils.constants import get_trade_value

if TYPE_CHECKING:
    from src.env_core.state import GameState

__all__ = [
    "MAX_CARDS",
    "get_trade_value",
    "forms_valid_set",
    "is_valid_trade",
    "has_tradeable_cards",
    "draw_card",
    "apply_trade",
    "forced_trade",
    "transfer_cards",
]

_WILDCARD = 3


def forms_valid_set(symbols: list[int]) -> bool:
    """True for all-same, all-different, or any set containing a wildcard (3)."""
    if _WILDCARD in symbols:
        return True
    unique = set(symbols)
    return len(unique) == 1 or len(unique) == 3


def _pop_cards(hand: list[int], indices: list[int]) -> list[int]:
    """Remove cards at given indices (descending) and return them."""
    return [hand.pop(i) for i in sorted(indices, reverse=True)]


def is_valid_trade(state: GameState, indices: list[int], player: int) -> bool:
    """Reject duplicate/OOB indices; delegate symbol check to forms_valid_set."""
    hand: list[int] = state.cards[player]
    if len(set(indices)) != 3:
        return False
    if any(i < 0 or i >= len(hand) for i in indices):
        return False
    return forms_valid_set([hand[i] for i in indices])


def has_tradeable_cards(state: GameState, player: int) -> bool:
    """True iff some 3-card combination of the hand forms a valid set."""
    hand: list[int] = state.cards[player]
    if len(hand) < 3:
        return False
    return any(
        forms_valid_set([hand[i] for i in combo]) for combo in combinations(range(len(hand)), 3)
    )


def draw_card(state: GameState, player: int) -> None:
    """Pop one card from deck into hand; reshuffle discard into deck when deck is empty."""
    if not state.deck:
        if not state.discard_pile:
            return
        state.rng.shuffle(state.discard_pile)
        state.deck = state.discard_pile
        state.discard_pile = []
    state.cards[player].append(state.deck.pop())


def apply_trade(state: GameState, indices: list[int], player: int) -> bool:
    """Increment trade_count, award armies, move cards to discard; False on invalid set."""
    if not is_valid_trade(state, indices, player):
        return False
    prev_count = state.trade_count
    state.trade_count += 1
    state.reinforcements_remaining += get_trade_value(prev_count)
    state.discard_pile.extend(_pop_cards(state.cards[player], indices))
    return True


def forced_trade(state: GameState, player: int) -> None:
    """Apply the first valid 3-card combo; no-op without error when none exists."""
    hand: list[int] = state.cards[player]
    for combo in combinations(range(len(hand)), 3):
        if forms_valid_set([hand[i] for i in combo]):
            apply_trade(state, list(combo), player)
            return


def transfer_cards(state: GameState, from_player: int, to_player: int) -> None:
    """Move all cards to to_player; force-trade while hand exceeds MAX_CARDS."""
    state.cards[to_player].extend(state.cards[from_player])
    state.cards[from_player] = []
    while len(state.cards[to_player]) > MAX_CARDS:
        before = len(state.cards[to_player])
        forced_trade(state, to_player)
        if len(state.cards[to_player]) >= before:
            break  # termination guard: no valid combo to reduce hand
