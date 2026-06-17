"""Scripted heuristic Risk demonstrator implementing the Agent protocol."""

from __future__ import annotations

import numpy as np
import torch

from src.agents.base import Agent
from src.config import HeuristicConfig
from src.env import (
    PHASE_ATTACK,
    PHASE_CAPTURE_MOVE,
    PHASE_FORTIFY,
    PHASE_REINFORCE,
    PHASE_TRADE,
)
from src.utils.constants import ADJACENCY, CONTINENTS, territory_to_continent

_SKIP_TYPE = 5
Obs = dict[str, np.ndarray]
Legal = list[dict[str, int]]

# ---------------------------------------------------------------------------
# Board topology helpers (pure functions, no side-effects)
# ---------------------------------------------------------------------------


def _enemy_neighbors(obs: Obs, terr: int, player: int) -> list[int]:
    return [n for n in ADJACENCY[terr] if obs["territory_owner"][n] != player]


def _border_threat(obs: Obs, terr: int, player: int) -> float:
    """Ratio of strongest adjacent enemy army to own army count."""
    enemies = _enemy_neighbors(obs, terr, player)
    if not enemies:
        return 0.0
    return max(int(obs["armies"][e]) for e in enemies) / max(1, int(obs["armies"][terr]))


def _continent_progress(obs: Obs, terr: int, player: int) -> float:
    """Fraction of terr's continent already owned by player."""
    cont = territory_to_continent(terr)
    if cont is None:
        return 0.0
    terrs = CONTINENTS[cont]
    return sum(1 for t in terrs if obs["territory_owner"][t] == player) / len(terrs)


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------


def _argmax_with_tiebreak(scores: dict, rng: np.random.Generator):
    """Return the key with the highest score; rng breaks ties uniformly."""
    best = max(scores.values())
    tied = [k for k, v in scores.items() if v == best]
    return tied[int(rng.integers(len(tied)))]


def _skip_action(legal: Legal) -> dict[str, int]:
    """Return the skip entry from legal_actions, or legal[0] as fallback."""
    for a in legal:
        if a["action_type"] == _SKIP_TYPE:
            return a
    return legal[0]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _reinforce_score(obs: Obs, terr: int, player: int, cfg: HeuristicConfig) -> float:
    return (
        cfg.reinforce_border_weight * len(_enemy_neighbors(obs, terr, player))
        + cfg.reinforce_continent_weight * _continent_progress(obs, terr, player)
        + cfg.reinforce_threat_weight * _border_threat(obs, terr, player)
    )


def _score_reinforce_territories(
    obs: Obs, candidates: Legal, player: int, cfg: HeuristicConfig
) -> dict[int, float]:
    seen: set[int] = set()
    scores: dict[int, float] = {}
    for a in candidates:
        t = a["param_a"]
        if t not in seen:
            seen.add(t)
            scores[t] = _reinforce_score(obs, t, player, cfg)
    return scores


def _pair_score(obs: Obs, att: int, dfn: int, player: int, cfg: HeuristicConfig) -> float:
    odds = int(obs["armies"][att]) / max(1, int(obs["armies"][dfn]))
    mask = obs["territory_owner"] == int(obs["territory_owner"][dfn])
    def_total = float(np.sum(obs["armies"][mask]))
    return (
        odds
        + cfg.attack_continent_weight * _continent_progress(obs, dfn, player)
        + cfg.attack_weakest_enemy_weight / max(1.0, def_total)
    )


def _score_attack_pairs(
    obs: Obs, attacks: Legal, player: int, cfg: HeuristicConfig
) -> dict[tuple, float]:
    scores: dict[tuple, float] = {}
    for a in attacks:
        pair = (a["param_a"], a["param_b"])
        if pair in scores:
            continue
        odds = int(obs["armies"][pair[0]]) / max(1, int(obs["armies"][pair[1]]))
        if odds < cfg.attack_min_odds:
            continue
        scores[pair] = _pair_score(obs, pair[0], pair[1], player, cfg)
    return scores


def _apply_dice_policy(attacks: Legal, pair: tuple, policy: str) -> dict[str, int]:
    pair_acts = [a for a in attacks if (a["param_a"], a["param_b"]) == pair]
    if policy == "max":
        return max(pair_acts, key=lambda a: a["param_c"])
    return min(pair_acts, key=lambda a: a["param_c"])


def _safe_fortify_moves(legal: Legal, obs: Obs, cfg: HeuristicConfig) -> Legal:
    return [
        a
        for a in legal
        if a["action_type"] != _SKIP_TYPE
        and int(obs["armies"][a["param_a"]]) - a["param_c"] >= cfg.fortify_min_interior_armies
    ]


# ---------------------------------------------------------------------------
# Per-phase choosers (module-level; each ≤ 10 lines)
# ---------------------------------------------------------------------------


def _choose_skip(
    obs: Obs, legal: Legal, cfg: HeuristicConfig, rng: np.random.Generator
) -> dict[str, int]:
    return _skip_action(legal)


def _choose_trade(
    obs: Obs, legal: Legal, cfg: HeuristicConfig, rng: np.random.Generator
) -> dict[str, int]:
    """Prefer any valid trade over skip; weakest-player targeting via attack flow."""
    trades = [a for a in legal if a["action_type"] != _SKIP_TYPE]
    return trades[0] if trades else _skip_action(legal)


def _choose_reinforce(
    obs: Obs, legal: Legal, cfg: HeuristicConfig, rng: np.random.Generator
) -> dict[str, int]:
    candidates = [a for a in legal if a["action_type"] != _SKIP_TYPE]
    if not candidates:
        return _skip_action(legal)
    player = int(obs["current_player"])
    terr_scores = _score_reinforce_territories(obs, candidates, player, cfg)
    best_terr = _argmax_with_tiebreak(terr_scores, rng)
    terr_acts = [a for a in candidates if a["param_a"] == best_terr]
    return max(terr_acts, key=lambda a: a["param_b"])


def _choose_attack(
    obs: Obs, legal: Legal, cfg: HeuristicConfig, rng: np.random.Generator
) -> dict[str, int]:
    attacks = [a for a in legal if a["action_type"] != _SKIP_TYPE]
    if not attacks:
        return _skip_action(legal)
    player = int(obs["current_player"])
    pair_scores = _score_attack_pairs(obs, attacks, player, cfg)
    if not pair_scores:
        return _skip_action(legal)
    best_pair = _argmax_with_tiebreak(pair_scores, rng)
    best_odds = int(obs["armies"][best_pair[0]]) / max(1, int(obs["armies"][best_pair[1]]))
    if best_odds < cfg.attack_odds_stop:
        return _skip_action(legal)
    return _apply_dice_policy(attacks, best_pair, cfg.attack_dice_policy)


def _choose_capture_move(
    obs: Obs, legal: Legal, cfg: HeuristicConfig, rng: np.random.Generator
) -> dict[str, int]:
    lo = min(a["param_a"] for a in legal)
    hi = max(a["param_a"] for a in legal)
    target = round(cfg.capture_move_fraction * (hi - lo) + lo)
    return min(legal, key=lambda a: abs(a["param_a"] - target))


def _choose_fortify(
    obs: Obs, legal: Legal, cfg: HeuristicConfig, rng: np.random.Generator
) -> dict[str, int]:
    safe = _safe_fortify_moves(legal, obs, cfg)
    if not safe:
        return _skip_action(legal)
    player = int(obs["current_player"])
    scores = {
        i: _border_threat(obs, a["param_b"], player) * cfg.fortify_threat_weight
        for i, a in enumerate(safe)
    }
    return safe[_argmax_with_tiebreak(scores, rng)]


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_PHASE_DISPATCH = {
    PHASE_TRADE: _choose_trade,
    PHASE_REINFORCE: _choose_reinforce,
    PHASE_ATTACK: _choose_attack,
    PHASE_CAPTURE_MOVE: _choose_capture_move,
    PHASE_FORTIFY: _choose_fortify,
}


# ---------------------------------------------------------------------------
# Public agent class
# ---------------------------------------------------------------------------


class HeuristicAgent(Agent):
    """Scripted Risk demonstrator; implements the Agent protocol."""

    def __init__(
        self,
        config: HeuristicConfig | None = None,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Create a heuristic agent with optional config, seed, or explicit rng."""
        self._cfg = config or HeuristicConfig()
        self._rng = rng if rng is not None else np.random.default_rng(seed)

    def act(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> dict[str, int]:
        """Select the highest-scoring legal action for the current phase."""
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        phase = int(obs["phase"])
        chooser = _PHASE_DISPATCH.get(phase, _choose_skip)
        return chooser(obs, legal_actions, self._cfg, self._rng)

    def act_with_meta(
        self,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        *,
        deterministic: bool = False,
    ) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
        """Delegate to act; return NaN tensors for value/logprob (no network)."""
        action = self.act(obs, legal_actions, deterministic=deterministic)
        nan = torch.tensor(float("nan"))
        return action, nan, nan
