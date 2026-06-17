"""Per-player discounted Monte-Carlo return helper for BC dataset generation.

CONTRACT
--------
``players[t]`` MUST be the player to whom ``rewards[t]`` was attributed by
the environment (the post-step recipient), which is NOT always the agent who
made the decision at step *t*. In ``RisikoEnv``, turn-ending steps rotate
``current_player`` before computing the reward, so the immediate reward on
those steps is credited to the NEXT player, not the fortifying agent.
Aligning ``players`` with the actual reward recipient is the caller's
responsibility; this function only performs the per-player discount arithmetic.
"""

from __future__ import annotations


def compute_per_player_returns(
    rewards: list[float],
    players: list[int],
    gamma: float,
) -> list[float]:
    """Compute discounted Monte-Carlo returns per decision.

    Discounts within each player's own decision subsequence. Each entry
    ``returns[t]`` is the discounted sum of all future rewards attributed to
    ``players[t]``, counting only that player's OWN intervening decisions
    against the discount exponent.

    Algorithm: single reverse pass, O(T).

    Args:
        rewards: Shaped immediate rewards, one per env step, in order.
        players: Player index to whom each reward was attributed (same length).
        gamma: Discount factor in [0, 1].

    Returns:
        List of per-step discounted returns, same length as inputs.
    """
    assert len(rewards) == len(players), (
        f"rewards and players must have the same length; got {len(rewards)} and {len(players)}"
    )
    n = len(rewards)
    returns: list[float] = [0.0] * n
    running: dict[int, float] = {}
    for t in reversed(range(n)):
        p = players[t]
        g = rewards[t] + gamma * running.get(p, 0.0)
        returns[t] = float(g)
        running[p] = float(g)
    return returns
