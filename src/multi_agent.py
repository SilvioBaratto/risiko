"""Multi-agent runner and game result dataclass."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.agents.base import Agent
from src.agents.random_agent import RandomAgent
from src.env import RisikoEnv
from src.utils.log import get_logger

_log = get_logger("runner")


@dataclass(frozen=True)
class Transition:
    """One environment step's data, including PPO meta if available.

    For agents that don't compute log_prob/value (RandomAgent, LLMOpponent),
    those fields hold ``nan`` and the trainer ignores them. The PPO learner
    (slot 0 by convention) populates them via ``act_with_meta``.

    ``legal_actions`` carries the action mask context the consumer needs to
    rebuild a per-head mask at update time so log-probs are computed under
    the same constraint that was sampled.
    """

    obs: dict[str, np.ndarray]
    action: dict[str, int]
    reward: float
    log_prob: float = math.nan
    value: float = math.nan
    legal_actions: list[dict[str, int]] = field(default_factory=list)

    # Backwards-compatible 3-tuple unpacking: ``for obs, action, reward in trajectories``.
    def __iter__(self):
        """Yield (obs, action, reward) so legacy 3-tuple consumers still work."""
        yield self.obs
        yield self.action
        yield self.reward


@dataclass(frozen=True)
class GameResult:
    """Structured outcome of a single Risiko game."""

    winner: int | None
    n_turns: int
    territory_history: list[np.ndarray]
    elimination_order: list[int]
    card_trade_turns: list[int]
    action_log: list[dict[str, Any]]
    trajectories: list[Transition]
    card_trade_hand_sizes: list[int] = field(default_factory=list)


class MultiAgentRunner:
    """Coordinates multiple agents against a single RisikoEnv."""

    def __init__(
        self,
        env: RisikoEnv,
        agents: Sequence[Agent],
        max_turns: int = 10_000,
    ) -> None:
        """Initialise runner with environment and agent list."""
        self._env = env
        self._agents = agents
        self._n_players = env.n_players
        self._max_turns = max_turns
        self._validate_agents()

    def run_game(self, seed: int | None = None) -> GameResult:
        """Play one full game and return structured results."""
        _log.debug(
            "game start — n_players=%d seed=%s max_turns=%d",
            self._n_players,
            seed,
            self._max_turns,
        )
        obs, info = self._env.reset(seed=seed)
        recorder = _GameRecorder(self._env, self._n_players)
        turn = 0
        while turn < self._max_turns:
            player = int(obs["current_player"])
            legal = info["legal_actions"]
            action, log_prob, value = self._pick_action(player, obs, legal)
            recorder.record_action(player, action)
            prev_trade_count = self._env.state.trade_count
            obs, reward, terminated, truncated, info = self._env.step(action)
            recorder.record_step(obs, action, reward, log_prob, value, legal)
            recorder.detect_trade(prev_trade_count, turn)
            turn += 1
            if terminated or truncated:
                _log.info(
                    "game ended early — terminated=%s truncated=%s turn=%d",
                    terminated,
                    truncated,
                    turn,
                )
                break
        result = recorder.build_result()
        _log.info(
            "game over — winner=%s turns=%d eliminations=%s",
            result.winner,
            result.n_turns,
            result.elimination_order,
        )
        return result

    def run_games(self, n: int, seed: int) -> list[GameResult]:
        """Play *n* games with sequential seeds and return results."""
        return [self.run_game(seed=seed + i) for i in range(n)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_agents(self) -> None:
        if len(self._agents) != self._n_players:
            raise ValueError(f"Expected {self._n_players} agents, got {len(self._agents)}")

    def _pick_action(
        self,
        player: int,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
    ) -> tuple[dict[str, int], float, float]:
        """Return (action, log_prob, value); meta is nan for non-PPO agents."""
        if not legal_actions:
            return self._skip_action(), math.nan, math.nan
        agent = self._agents[player]
        try:
            action, log_prob, value = agent.act_with_meta(obs, legal_actions)
            return action, float(log_prob), float(value)
        except Exception:
            fallback = RandomAgent().act({}, legal_actions)
            return fallback, math.nan, math.nan

    @staticmethod
    def _skip_action() -> dict[str, int]:
        return {"action_type": 5, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}


class _GameRecorder:
    """Collects per-turn data during a single game."""

    def __init__(self, env: RisikoEnv, n_players: int) -> None:
        self._env = env
        self._n_players = n_players
        self._territory_history: list[np.ndarray] = []
        self._elimination_order: list[int] = []
        self._card_trade_turns: list[int] = []
        self._card_trade_hand_sizes: list[int] = []
        self._action_log: list[dict[str, Any]] = []
        self._trajectories: list[Transition] = []
        self._pending_trade_hand_size: int | None = None
        self._prev_eliminated: np.ndarray = np.zeros(6, dtype=np.int32)

    def record_action(self, player: int, action: dict[str, int]) -> None:
        self._action_log.append({**action, "player": player})
        self._pending_trade_hand_size = None
        if action["action_type"] == 0:
            self._pending_trade_hand_size = len(self._env.state.cards[player])

    def record_step(
        self,
        obs: dict[str, np.ndarray],
        action: dict[str, int],
        reward: float,
        log_prob: float = math.nan,
        value: float = math.nan,
        legal_actions: list[dict[str, int]] | None = None,
    ) -> None:
        self._territory_history.append(self._territory_counts(obs))
        self._trajectories.append(
            Transition(
                obs=obs.copy(),
                action=action.copy(),
                reward=reward,
                log_prob=log_prob,
                value=value,
                legal_actions=list(legal_actions or []),
            )
        )
        self._detect_eliminations(obs)

    def detect_trade(self, prev_trade_count: int, turn: int) -> None:
        if self._env.state.trade_count > prev_trade_count:
            self._card_trade_turns.append(turn)
            if self._pending_trade_hand_size is not None:
                self._card_trade_hand_sizes.append(self._pending_trade_hand_size)

    def build_result(self) -> GameResult:
        winner = self._resolve_winner()
        return GameResult(
            winner=winner,
            n_turns=len(self._trajectories),
            territory_history=self._territory_history,
            elimination_order=self._elimination_order,
            card_trade_turns=self._card_trade_turns,
            action_log=self._action_log,
            trajectories=self._trajectories,
            card_trade_hand_sizes=self._card_trade_hand_sizes,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _territory_counts(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        owner = obs["territory_owner"]
        return np.bincount(owner, minlength=self._n_players)

    def _resolve_winner(self) -> int | None:
        s = self._env.state
        active = [
            p
            for p in range(s.n_players)
            if not s.eliminated[p] and np.sum(s.territory_owner == p) > 0
        ]
        if len(active) == 1:
            return active[0]
        return None

    def _detect_eliminations(self, obs: dict[str, np.ndarray]) -> None:
        curr = obs["eliminated"]
        for p in range(self._n_players):
            if self._prev_eliminated[p] == 0 and curr[p] == 1:
                self._elimination_order.append(p)
        self._prev_eliminated = curr.copy()
