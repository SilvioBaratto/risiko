"""Multi-agent runner and game result dataclass."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.agents.base import Agent
from src.agents.diplomacy import DiplomacyContext, DiplomacyState
from src.agents.negotiation import NegotiationOrchestrator
from src.agents.player_config import DEFAULT_6P_PROFILES, PlayerConfig
from src.agents.random_agent import RandomAgent
from src.agents.reputation import ReputationBook
from src.config import DiplomacyConfig
from src.env import RisikoEnv
from src.utils.log import get_logger

_log = get_logger("runner")

# A single player-turn expands into several env steps (trade, reinforce, one
# step per attack, capture-move, fortify). ``max_turns`` caps *player-turns*,
# so this backstops a pathological policy that never ends its turn from
# looping forever: the game also aborts after ``max_turns * _STEPS_PER_TURN_CAP``
# env steps. Empirically ~6 steps/turn under random play; LLM bots can stall in
# an attack cycle, so keep the backstop tight (was 200 → 30k-step games).
_STEPS_PER_TURN_CAP = 50


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
        max_turns: int = 1000,
        stop_on_eliminated: int | None = None,
    ) -> None:
        """Initialise runner with environment and agent list.

        Args:
            env: The Risiko environment the agents play in.
            agents: One agent per player slot, in player-id order.
            max_turns: Player-turn cap; the game aborts as a draw past it.
            stop_on_eliminated: If set, end the game the moment this player is
                eliminated. For RL training the learner's MDP ends at its own
                death — continuing to simulate the surviving opponents adds no
                learner transitions and can loop for thousands of steps when
                bots stall in an attack cycle that never ends a turn.
        """
        self._env = env
        self._agents = agents
        self._n_players = env.n_players
        self._max_turns = max_turns
        self._stop_on_eliminated = stop_on_eliminated
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
        step = 0
        step_ceiling = self._max_turns * _STEPS_PER_TURN_CAP
        # ``max_turns`` caps player-turns, not env steps. A turn spans several
        # phase-actions, so we advance until the env has completed that many
        # turns (``state.turns_elapsed``), with a hard env-step backstop.
        while self._env.state.turns_elapsed < self._max_turns and step < step_ceiling:
            player = int(obs["current_player"])
            legal = info["legal_actions"]
            pre_obs = obs
            self._on_player_turn_start(player, obs, info)
            action, log_prob, value = self._pick_action(player, obs, legal)
            recorder.record_action(player, action)
            prev_trade_count = self._env.state.trade_count
            obs, reward, terminated, truncated, info = self._env.step(action)
            self._after_step(player, action, pre_obs, obs)
            recorder.record_step(obs, action, reward, log_prob, value, legal)
            recorder.detect_trade(prev_trade_count, self._env.state.turns_elapsed)
            step += 1
            if terminated or truncated:
                _log.info(
                    "game ended early — terminated=%s truncated=%s turns=%d steps=%d",
                    terminated,
                    truncated,
                    self._env.state.turns_elapsed,
                    step,
                )
                break
            if (
                self._stop_on_eliminated is not None
                and self._env.state.eliminated[self._stop_on_eliminated]
            ):
                _log.info(
                    "game stopped — focus player=%d eliminated, turns=%d steps=%d",
                    self._stop_on_eliminated,
                    self._env.state.turns_elapsed,
                    step,
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

    def run(self, seed: int | None = None) -> GameResult:
        """Single-game alias for run_game; used by the social-eval harness."""
        return self.run_game(seed=seed)

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

    def _on_player_turn_start(self, player: int, obs: dict, info: dict) -> None:
        """Hook called before each action pick. Override in subclasses."""

    def _after_step(self, player: int, action: dict, pre_obs: dict, new_obs: dict) -> None:
        """Hook called after env.step. Override in subclasses."""

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


class DiplomacyRunner(MultiAgentRunner):
    """Eval-only subclass that adds the diplomacy layer to MultiAgentRunner.

    Fires a bounded negotiation round at each player-turn boundary, injects a
    ``DiplomacyContext`` into every non-learner seat's ``act_with_meta`` call,
    and detects betrayal (ally attacking ally) deterministically after each
    env step.

    The RL learner (slot ``learner_id``, default 0) is never passed a context
    and is excluded from negotiation speakers — its training path is unchanged.

    When ``diplomacy_cfg.enabled`` is ``False`` the game is byte-identical to a
    plain ``MultiAgentRunner`` run.
    """

    def __init__(
        self,
        env: RisikoEnv,
        agents: Sequence[Agent],
        max_turns: int = 1000,
        stop_on_eliminated: int | None = None,
        *,
        diplomacy_cfg: DiplomacyConfig | None = None,
        state: DiplomacyState | None = None,
        reputation: ReputationBook | None = None,
        learner_id: int = 0,
        profiles: Sequence[PlayerConfig] | None = None,
        call_fn: Callable[[int, str], dict | None] | None = None,
    ) -> None:
        """Initialise runner with environment, agents, and diplomacy config.

        Args:
            env: The Risiko environment the agents play in.
            agents: One agent per player slot, in player-id order.
            max_turns: Player-turn cap; the game aborts as a draw past it.
            stop_on_eliminated: If set, end the game when this player is eliminated.
            diplomacy_cfg: Diplomacy layer config; disabled by default.
            state: Shared DiplomacyState instance; a fresh one is created if None.
            reputation: Cross-game reputation tracker; a fresh one if None.
            learner_id: Player-slot of the RL learner (excluded from negotiation).
            profiles: Per-player LLM sampling profiles; defaults to the first
                ``n_players`` entries of ``DEFAULT_6P_PROFILES``.
            call_fn: Injectable ``(player_id, prompt) -> dict | None`` for
                negotiation.  ``None`` uses a no-op (no LLM calls) so the
                disabled-equivalence guarantee is preserved by default.  Pass
                a real Ollama wrapper or a mocked function for real/test use.
        """
        super().__init__(env, agents, max_turns, stop_on_eliminated)
        self._cfg = diplomacy_cfg if diplomacy_cfg is not None else DiplomacyConfig()
        self._state = state if state is not None else DiplomacyState()
        self._reputation = reputation if reputation is not None else ReputationBook()
        self._learner_id = learner_id
        self.call_count: int = 0
        self._last_negotiation_turn: int = -1
        self._orchestrator: NegotiationOrchestrator | None = None
        if self._cfg.enabled:
            _profiles = (
                profiles if profiles is not None else list(DEFAULT_6P_PROFILES[: self._n_players])
            )
            self._orchestrator = NegotiationOrchestrator(
                state=self._state,
                reputation=self._reputation,
                cfg=self._cfg,
                profiles=_profiles,
                call_fn=call_fn,
                learner_slot=learner_id,
            )

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _pick_action(
        self,
        player: int,
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
    ) -> tuple[dict[str, int], float, float]:
        if not self._cfg.enabled or player == self._learner_id:
            return super()._pick_action(player, obs, legal_actions)
        if not legal_actions:
            return self._skip_action(), math.nan, math.nan
        context = self._build_context(player, obs)
        agent = self._agents[player]
        try:
            action, log_prob, value = agent.act_with_meta(obs, legal_actions, context=context)
            return action, float(log_prob), float(value)
        except Exception:
            fallback = RandomAgent().act({}, legal_actions)
            return fallback, math.nan, math.nan

    def _on_player_turn_start(self, player: int, obs: dict, info: dict) -> None:
        if not self._cfg.enabled or self._orchestrator is None:
            return
        current_turn = self._env.state.turns_elapsed
        if current_turn <= self._last_negotiation_turn:
            return
        self._last_negotiation_turn = current_turn
        speakers = self._active_non_learner_speakers(obs)
        if speakers:
            n_calls = self._orchestrator.run_round(obs, speakers)
            self.call_count += n_calls

    def _active_non_learner_speakers(self, obs: dict) -> list[int]:
        """Non-learner, non-eliminated player ids eligible to negotiate."""
        eliminated = obs.get("eliminated", np.zeros(self._n_players, dtype=np.int32))
        return [p for p in range(self._n_players) if p != self._learner_id and not eliminated[p]]

    def _after_step(self, player: int, action: dict, pre_obs: dict, new_obs: dict) -> None:
        if not self._cfg.enabled:
            return
        if action.get("action_type") != 2:  # only ATTACK can be betrayal
            return
        territory_owner = pre_obs.get("territory_owner", [])
        target = action.get("param_b", -1)
        if not (0 <= target < len(territory_owner)):
            return
        victim = int(territory_owner[target])
        if victim == player or not self._state.are_allied(player, victim):
            return
        turn = self._env.state.turns_elapsed
        self._state.declare_war_on(player, victim, turn=turn)
        self._reputation.record_betrayal(player)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_context(self, player: int, obs: dict[str, np.ndarray]) -> DiplomacyContext:
        territory_owner = obs.get("territory_owner", np.array([], dtype=np.int32))
        leader: int | None = None
        if len(territory_owner) > 0:
            leader = int(self._state.leader(territory_owner))
        return DiplomacyContext(state=self._state, acting_player=player, leader=leader)
