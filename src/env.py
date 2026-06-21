"""Gymnasium-compatible Risiko (Risk) game environment.

A multi-player turn-based war game where the current player acts via
`step(action)` and the environment internally rotates to the next
player when a turn ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utils.constants import (
    ADJACENCY,
    CARD_SYMBOLS,
    CONTINENT_BONUSES,
    CONTINENTS,
    FORTIFY_ADJACENT_ONLY_DEFAULT,
    MAX_ARMIES_PARAM,
    NUM_TERRITORIES,
    STARTING_ARMIES,
    get_trade_value,
)
from src.utils.log import get_logger
from src.utils.reward_config import RewardConfig

_log = get_logger("env")

PHASE_TRADE = 0
PHASE_REINFORCE = 1
PHASE_ATTACK = 2
PHASE_CAPTURE_MOVE = 3
PHASE_FORTIFY = 4

MAX_CARDS = 5


@dataclass
class GameState:
    """Mutable game state container."""

    territory_owner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    armies: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    cards: list[list[int]] = field(default_factory=list)
    deck: list[int] = field(default_factory=list)
    discard_pile: list[int] = field(default_factory=list)
    trade_count: int = 0
    current_player: int = 0
    phase: int = PHASE_REINFORCE
    reinforcements_remaining: int = 0
    turn_capture: int = 0
    turns_elapsed: int = 0
    eliminated: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    n_players: int = 0
    last_capture_dice: int = 0
    last_attacker: int = -1
    last_defender: int = -1
    rng: np.random.Generator = field(default_factory=np.random.default_rng)


class RisikoEnv(gym.Env):
    """Risiko board game as a Gymnasium environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_players: int = 3,
        reward_config: RewardConfig | None = None,
        fortify_adjacent_only: bool | None = None,
    ):
        """Initialize the environment.

        Args:
            n_players: Number of players (2–6).
            reward_config: Reward shaping configuration.
            fortify_adjacent_only: When True (default), fortification only moves
                armies into a directly adjacent own territory (official Risk rule).
                When False, any own territory reachable via a connected chain of own
                territories is a legal destination (legacy house-rule).  Defaults to
                ``FORTIFY_ADJACENT_ONLY_DEFAULT`` from ``src/utils/constants.py``.
        """
        super().__init__()
        if not 2 <= n_players <= 6:
            raise ValueError("n_players must be 2-6")
        self.n_players = n_players
        self.reward_config = reward_config or RewardConfig()
        self.fortify_adjacent_only = (
            FORTIFY_ADJACENT_ONLY_DEFAULT
            if fortify_adjacent_only is None
            else fortify_adjacent_only
        )
        self.state: GameState = GameState()
        # Optional reverse-curriculum start state: when set to
        # ``{"advantaged_player": id, "opponent_territories": k}`` the board is
        # seeded near-won for that player (it owns all but ``k`` territories per
        # opponent, opponents start with 1 army each). Lets a from-scratch agent
        # actually reach + experience wins, then the trainer anneals ``k`` up.
        self.curriculum: dict[str, float] | None = None
        self.observation_space = spaces.Dict(
            {
                "territory_owner": spaces.Box(0, 5, (NUM_TERRITORIES,), dtype=np.int32),
                "armies": spaces.Box(0, 999, (NUM_TERRITORIES,), dtype=np.int32),
                "phase": spaces.Discrete(5),
                "current_player": spaces.Discrete(6),
                "cards": spaces.Box(0, 1, (MAX_CARDS, 4), dtype=np.int32),
                "continent_control": spaces.Box(0, 1, (6,), dtype=np.int32),
                "trade_count": spaces.Discrete(1000),
                "reinforcements_remaining": spaces.Discrete(1000),
                "turn_capture": spaces.Discrete(2),
                "n_players": spaces.Discrete(7),
                "eliminated": spaces.Box(0, 1, (6,), dtype=np.int32),
            }
        )
        self.action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(6),
                "param_a": spaces.Discrete(MAX_ARMIES_PARAM + 1),
                "param_b": spaces.Discrete(MAX_ARMIES_PARAM + 1),
                "param_c": spaces.Discrete(MAX_ARMIES_PARAM + 1),
                "param_d": spaces.Discrete(1),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.state = self._new_state(seed)
        self._setup_board()
        return self._get_obs(), self._get_info()

    def step(
        self,
        action: dict[str, int],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one action for current player."""
        if not self._is_valid_action(action):
            return self._invalid_step()
        return self._dispatch_action(action)

    def _get_obs(self) -> dict[str, np.ndarray]:
        s = self.state
        card_matrix = np.zeros((MAX_CARDS, 4), dtype=np.int32)
        player_cards = s.cards[s.current_player]
        for i, cid in enumerate(player_cards[:MAX_CARDS]):
            sym = CARD_SYMBOLS.get(cid, 0)
            card_matrix[i, sym] = 1
        continent_control = np.array(
            [
                int(all(s.territory_owner[t] == s.current_player for t in ids))
                for ids in CONTINENTS.values()
            ],
            dtype=np.int32,
        )
        return {
            "territory_owner": s.territory_owner.copy(),
            "armies": s.armies.copy(),
            "phase": np.array(s.phase, dtype=np.int32),
            "current_player": np.array(s.current_player, dtype=np.int32),
            "cards": card_matrix,
            "continent_control": continent_control,
            "trade_count": np.array(s.trade_count, dtype=np.int32),
            "reinforcements_remaining": np.array(s.reinforcements_remaining, dtype=np.int32),
            "turn_capture": np.array(s.turn_capture, dtype=np.int32),
            "n_players": np.array(s.n_players, dtype=np.int32),
            "eliminated": np.pad(
                s.eliminated.copy(),
                (0, 6 - s.n_players),
                mode="constant",
                constant_values=1,
            ),
        }

    def _get_info(self) -> dict[str, Any]:
        return {"legal_actions": self._get_legal_actions()}

    def get_legal_actions(self) -> list[dict[str, int]]:
        """Return legal actions for the current player and phase."""
        return self._get_legal_actions()

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def _new_state(self, seed: int | None) -> GameState:
        rng = np.random.default_rng(seed)
        return GameState(
            territory_owner=np.zeros(NUM_TERRITORIES, dtype=np.int32),
            armies=np.zeros(NUM_TERRITORIES, dtype=np.int32),
            cards=[[] for _ in range(self.n_players)],
            deck=list(range(44)),
            discard_pile=[],
            trade_count=0,
            current_player=0,
            phase=PHASE_REINFORCE,
            reinforcements_remaining=0,
            turn_capture=0,
            turns_elapsed=0,
            eliminated=np.zeros(self.n_players, dtype=np.int32),
            n_players=self.n_players,
            rng=rng,
        )

    def _setup_board(self) -> None:
        s = self.state
        s.rng.shuffle(s.deck)
        if self.curriculum is not None:
            self._setup_curriculum_board()
        else:
            order = np.arange(NUM_TERRITORIES)
            s.rng.shuffle(order)
            for idx, terr in enumerate(order):
                player = idx % self.n_players
                s.territory_owner[terr] = player
                s.armies[terr] = 1
            self._place_initial_armies()
        self._compute_reinforcements()
        s.phase = PHASE_TRADE if self._has_tradeable_cards(s.current_player) else PHASE_REINFORCE

    def _setup_curriculum_board(self) -> None:
        """Seed a near-won board so a weak policy can still reach a win.

        ``advantaged_player`` owns all but ``k`` territories per opponent
        (opponents start with 1 army each); its extra starting armies are piled
        onto its land so the lead is real, not just territorial.
        """
        s = self.state
        cur = self.curriculum or {}
        adv = int(cur.get("advantaged_player", 0))
        if "army_advantage" in cur:
            self._setup_army_advantage_board(adv, float(cur["army_advantage"]))
            return
        k = max(1, int(cur.get("opponent_territories", 1)))
        order = np.arange(NUM_TERRITORIES)
        s.rng.shuffle(order)
        idx = 0
        for player in range(self.n_players):
            if player == adv:
                continue
            for _ in range(min(k, NUM_TERRITORIES - idx - 1)):
                terr = int(order[idx])
                s.territory_owner[terr] = player
                s.armies[terr] = 1
                idx += 1
        for terr in order[idx:]:
            s.territory_owner[int(terr)] = adv
            s.armies[int(terr)] = 1
        # Pile the advantaged player's remaining starting armies on its land.
        owned = np.where(s.territory_owner == adv)[0]
        extra = STARTING_ARMIES[self.n_players] - len(owned)
        if extra > 0 and len(owned) > 0:
            for terr in s.rng.choice(owned, size=int(extra), replace=True):
                s.armies[int(terr)] += 1

    def _setup_army_advantage_board(self, adv: int, multiplier: float) -> None:
        """Even territory split, learner army-rich (multiplier x normal mass).

        Games stay full-size while the learner has a material edge to convert
        into a win; the trainer anneals the multiplier toward 1.0 (balanced).
        """
        s = self.state
        order = np.arange(NUM_TERRITORIES)
        s.rng.shuffle(order)
        for i, terr in enumerate(order):
            s.territory_owner[int(terr)] = i % self.n_players
            s.armies[int(terr)] = 1
        self._place_initial_armies()
        # Pile extra armies on the advantaged player to reach ``multiplier`` ×.
        owned = np.where(s.territory_owner == adv)[0]
        base = STARTING_ARMIES[self.n_players]
        extra = int(round((multiplier - 1.0) * base))
        if extra > 0 and len(owned) > 0:
            for terr in s.rng.choice(owned, size=extra, replace=True):
                s.armies[int(terr)] += 1

    def _place_initial_armies(self) -> None:
        s = self.state
        per_player = STARTING_ARMIES[self.n_players]
        placed = np.bincount(s.territory_owner, minlength=self.n_players)
        remaining = per_player - placed
        for player in range(self.n_players):
            owned = np.where(s.territory_owner == player)[0]
            if len(owned) == 0 or remaining[player] <= 0:
                continue
            additions = s.rng.choice(owned, size=int(remaining[player]), replace=True)
            for terr in additions:
                s.armies[terr] += 1

    # ------------------------------------------------------------------
    # Reinforcements
    # ------------------------------------------------------------------

    def _compute_reinforcements(self) -> None:
        s = self.state
        player = s.current_player
        n_territories = int(np.sum(s.territory_owner == player))
        from_territories = max(3, n_territories // 3)
        from_continents = sum(
            bonus
            for name, bonus in CONTINENT_BONUSES.items()
            if self._controls_continent(player, name)
        )
        s.reinforcements_remaining = from_territories + from_continents

    def _controls_continent(self, player: int, name: str) -> bool:
        s = self.state
        return all(s.territory_owner[t] == player for t in CONTINENTS[name])

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _dispatch_action(
        self,
        action: dict[str, int],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        s = self.state
        prev = self._snapshot()
        handler = {
            PHASE_TRADE: self._handle_trade,
            PHASE_REINFORCE: self._handle_reinforce,
            PHASE_ATTACK: self._handle_attack,
            PHASE_CAPTURE_MOVE: self._handle_capture_move,
            PHASE_FORTIFY: self._handle_fortify,
        }.get(s.phase)
        if handler is None:
            return self._invalid_step()
        if not handler(action):
            return self._invalid_step()
        terminated = self._check_win()
        if terminated:
            reward = self.reward_config.sparse_win
        else:
            reward = self._compute_reward(s.current_player, prev)
        return self._get_obs(), float(reward), bool(terminated), False, self._get_info()

    def _invalid_step(
        self,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        s = self.state
        _log.debug(
            "invalid action — player=%d phase=%d",
            s.current_player,
            s.phase,
        )
        return (
            self._get_obs(),
            float(self.reward_config.invalid_action_penalty),
            False,
            False,
            self._get_info(),
        )

    # ------------------------------------------------------------------
    # Trade
    # ------------------------------------------------------------------

    def _handle_trade(self, action: dict[str, int]) -> bool:
        s = self.state
        a, b, c = action["param_a"], action["param_b"], action["param_c"]
        player = s.current_player
        hand = s.cards[player]
        if action["action_type"] == 5:  # skip
            if self._has_tradeable_cards(player):
                return False
            s.phase = PHASE_REINFORCE
            return True
        if not self._is_valid_trade([a, b, c], player):
            return False
        bonus = get_trade_value(s.trade_count)
        s.trade_count += 1
        s.reinforcements_remaining += bonus
        for idx in sorted([a, b, c], reverse=True):
            if 0 <= idx < len(hand):
                s.discard_pile.append(hand.pop(idx))
        if self._has_tradeable_cards(player):
            return True
        s.phase = PHASE_REINFORCE if s.reinforcements_remaining > 0 else PHASE_ATTACK
        return True

    def _is_valid_trade(self, indices: list[int], player: int) -> bool:
        s = self.state
        hand = s.cards[player]
        if len(set(indices)) != 3 or any(i < 0 or i >= len(hand) for i in indices):
            return False
        symbols = [CARD_SYMBOLS[hand[i]] for i in indices]
        return self._forms_valid_set(symbols)

    def _forms_valid_set(self, symbols: list[int]) -> bool:
        if 3 in symbols:
            return True
        return len(set(symbols)) == 1 or len(set(symbols)) == 3

    def _has_tradeable_cards(self, player: int) -> bool:
        s = self.state
        hand = s.cards[player]
        if len(hand) < 3:
            return False
        from itertools import combinations

        for combo in combinations(range(len(hand)), 3):
            if self._is_valid_trade(list(combo), player):
                return True
        return False

    def _must_trade(self) -> bool:
        s = self.state
        return s.phase == PHASE_TRADE and self._has_tradeable_cards(s.current_player)

    # ------------------------------------------------------------------
    # Reinforce
    # ------------------------------------------------------------------

    def _handle_reinforce(self, action: dict[str, int]) -> bool:
        s = self.state
        terr = action["param_a"]
        count = action["param_b"]
        player = s.current_player
        if action["action_type"] == 5 and s.reinforcements_remaining == 0:  # end phase
            s.phase = PHASE_ATTACK
            return True
        if count <= 0 or count > s.reinforcements_remaining:
            return False
        if s.territory_owner[terr] != player:
            return False
        s.armies[terr] += count
        s.reinforcements_remaining -= count
        if s.reinforcements_remaining == 0:
            s.phase = PHASE_ATTACK
        return True

    # ------------------------------------------------------------------
    # Attack
    # ------------------------------------------------------------------

    def _handle_attack(self, action: dict[str, int]) -> bool:
        s = self.state
        if action["action_type"] == 5:  # end attack
            s.phase = PHASE_FORTIFY
            return True
        attacker = action["param_a"]
        defender = action["param_b"]
        dice = action["param_c"]
        player = s.current_player
        if s.territory_owner[attacker] != player:
            return False
        if s.territory_owner[defender] == player:
            return False
        if defender not in ADJACENCY[attacker]:
            return False
        if s.armies[attacker] < 2:
            return False
        if dice < 1 or dice > 3 or dice >= s.armies[attacker]:
            return False
        self._resolve_attack(attacker, defender, dice)
        return True

    def _resolve_attack(self, attacker: int, defender: int, dice: int) -> None:
        s = self.state
        attacker_dice = np.sort(s.rng.integers(1, 7, size=dice))[::-1]
        defender_dice_count = min(2, int(s.armies[defender]))
        defender_dice = np.sort(s.rng.integers(1, 7, size=defender_dice_count))[::-1]
        pairs = min(len(attacker_dice), len(defender_dice))
        for i in range(pairs):
            if attacker_dice[i] > defender_dice[i]:
                s.armies[defender] -= 1
            else:
                s.armies[attacker] -= 1
        if s.armies[defender] == 0:
            defender_player = int(s.territory_owner[defender])
            s.territory_owner[defender] = s.current_player
            s.armies[defender] = 0  # armies moved by _handle_capture_move
            s.turn_capture = 1
            s.last_capture_dice = dice
            s.last_attacker = attacker
            s.last_defender = defender
            _log.debug(
                "capture — player=%d took territory=%d from player=%d",
                s.current_player,
                defender,
                defender_player,
            )
            self._eliminate_player(defender_player)
            if s.armies[attacker] >= 2:
                s.phase = PHASE_CAPTURE_MOVE
            else:
                s.phase = PHASE_FORTIFY
        elif s.armies[attacker] < 2:
            s.phase = PHASE_FORTIFY

    def _eliminate_player(self, player: int) -> None:
        s = self.state
        remaining = int(np.sum(s.territory_owner == player))
        if remaining > 0:
            _log.debug(
                "eliminate guard — player=%d still holds %d territories, skipping",
                player,
                remaining,
            )
            return
        _log.info(
            "player=%d eliminated by player=%d | eliminated=%s",
            player,
            s.current_player,
            s.eliminated.tolist(),
        )
        s.eliminated[player] = 1
        eliminator = s.current_player
        transferred = s.cards[player].copy()
        s.cards[eliminator].extend(transferred)
        s.cards[player] = []
        while len(s.cards[eliminator]) > MAX_CARDS:
            self._forced_trade(eliminator)

    def _forced_trade(self, player: int) -> None:
        s = self.state
        hand = s.cards[player]
        from itertools import combinations

        for combo in combinations(range(len(hand)), 3):
            if self._is_valid_trade(list(combo), player):
                bonus = get_trade_value(s.trade_count)
                s.trade_count += 1
                for idx in sorted(combo, reverse=True):
                    s.discard_pile.append(hand.pop(idx))
                s.reinforcements_remaining += bonus
                return

    # ------------------------------------------------------------------
    # Capture move
    # ------------------------------------------------------------------

    def _handle_capture_move(self, action: dict[str, int]) -> bool:
        s = self.state
        move = action["param_a"]
        attacker = s.last_attacker
        defender = s.last_defender
        dice = s.last_capture_dice
        available = s.armies[attacker]
        if move < dice or move > available - 1:  # keep ≥1 army in attacker
            return False
        s.armies[defender] += move
        s.armies[attacker] -= move
        s.phase = PHASE_ATTACK if s.armies[attacker] >= 2 else PHASE_FORTIFY
        return True

    # ------------------------------------------------------------------
    # Fortify
    # ------------------------------------------------------------------

    def _handle_fortify(self, action: dict[str, int]) -> bool:
        s = self.state
        if action["action_type"] == 5:  # skip
            self._end_turn()
            return True
        source = action["param_a"]
        dest = action["param_b"]
        count = action["param_c"]
        player = s.current_player
        if s.territory_owner[source] != player or s.territory_owner[dest] != player:
            return False
        if source == dest:
            return False
        if count < 1 or count >= s.armies[source]:
            return False
        reachable = (
            dest in ADJACENCY[source]
            if self.fortify_adjacent_only
            else self._is_connected(source, dest, player)
        )
        if not reachable:
            return False
        s.armies[source] -= count
        s.armies[dest] += count
        self._end_turn()
        return True

    def _is_connected(self, source: int, target: int, player: int) -> bool:
        """Legacy connected-chain reachability (used only when fortify_adjacent_only=False).

        DFS over own territories via ADJACENCY.  Returns True when *target* is
        reachable from *source* through any chain of *player*-owned territories.
        """
        visited = set()
        stack = [source]
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in visited:
                continue
            visited.add(node)
            for neighbor in ADJACENCY[node]:
                if self.state.territory_owner[neighbor] == player and neighbor not in visited:
                    stack.append(neighbor)
        return False

    def _end_turn(self) -> None:
        s = self.state
        prev_player = s.current_player
        if s.turn_capture and len(s.cards[s.current_player]) < MAX_CARDS:
            self._draw_card(s.current_player)
        s.turns_elapsed += 1
        self._next_player()
        _log.debug(
            "turn end — player=%d → player=%d | territories=%s",
            prev_player,
            s.current_player,
            np.bincount(s.territory_owner, minlength=s.n_players).tolist(),
        )
        self._compute_reinforcements()
        s.turn_capture = 0
        s.last_attacker = -1
        s.last_defender = -1
        s.last_capture_dice = 0
        s.phase = PHASE_TRADE if self._has_tradeable_cards(s.current_player) else PHASE_REINFORCE

    def _draw_card(self, player: int) -> None:
        s = self.state
        if not s.deck:
            s.rng.shuffle(s.discard_pile)
            s.deck = s.discard_pile
            s.discard_pile = []
        if s.deck:
            s.cards[player].append(s.deck.pop())

    def _next_player(self) -> None:
        s = self.state
        for _ in range(s.n_players):
            s.current_player = (s.current_player + 1) % s.n_players
            if not s.eliminated[s.current_player]:
                break

    # ------------------------------------------------------------------
    # Win / reward
    # ------------------------------------------------------------------

    def _check_win(self) -> bool:
        s = self.state
        active = [
            p
            for p in range(s.n_players)
            if not s.eliminated[p] and np.sum(s.territory_owner == p) > 0
        ]
        if len(active) <= 1:
            _log.info("win detected — active=%s eliminated=%s", active, s.eliminated.tolist())
        elif len(active) <= 2:
            _log.debug("near-end — active=%s eliminated=%s", active, s.eliminated.tolist())
        return len(active) <= 1

    def territory_margin(self, player: int) -> float:
        """Return ``player``'s territory lead over the strongest opponent.

        Normalised to ``[-1, 1]`` by the board size. Used to give a graded
        terminal signal on games that hit the turn cap without an elimination
        (a draw): +1 means the player owns every territory, -1 means it owns
        none while a rival owns all. The dense per-step rewards never encode
        "who was ahead when the clock ran out", so without this the value head
        sees no win/loss target on unresolved games.
        """
        s = self.state
        if s.territory_owner.size == 0 or player >= s.n_players:
            return 0.0
        counts = np.bincount(s.territory_owner, minlength=s.n_players)
        own = int(counts[player])
        opponents = [int(counts[p]) for p in range(s.n_players) if p != player]
        best_opp = max(opponents) if opponents else 0
        return (own - best_opp) / NUM_TERRITORIES

    def _compute_reward(self, player: int, prev: dict[str, Any]) -> float:
        cfg = self.reward_config
        s = self.state
        reward = 0.0
        if s.eliminated[player]:
            return cfg.sparse_loss
        # Territory delta
        prev_count = int(np.sum(prev["territory_owner"] == player))
        curr_count = int(np.sum(s.territory_owner == player))
        reward += cfg.dense_territory_delta * (curr_count - prev_count)
        # Continent bonus delta
        prev_bonus = sum(
            bonus
            for name, bonus in CONTINENT_BONUSES.items()
            if all(prev["territory_owner"][t] == player for t in CONTINENTS[name])
        )
        curr_bonus = sum(
            bonus
            for name, bonus in CONTINENT_BONUSES.items()
            if self._controls_continent(player, name)
        )
        reward += cfg.dense_continent_bonus_delta * (curr_bonus - prev_bonus)
        # Army ratio
        total = int(s.armies.sum())
        if total > 0:
            player_armies = int(s.armies[s.territory_owner == player].sum())
            reward += cfg.dense_army_ratio * (player_armies / total)
        # Elimination bonus
        for p in range(s.n_players):
            if p == player:
                continue
            if prev["eliminated"][p] == 0 and s.eliminated[p] == 1:
                reward += cfg.dense_elimination_bonus
        # Potential-based progress shaping toward winning: reward increases in
        # territory margin (own lead over the strongest rival). Telescopes to
        # the win, densifying the gradient toward conquering/eliminating without
        # distorting the optimal policy.
        if cfg.dense_progress_weight != 0.0:
            prev_margin = self._margin_from_owner(prev["territory_owner"], player)
            curr_margin = self.territory_margin(player)
            reward += cfg.dense_progress_weight * (curr_margin - prev_margin)
        # Per-step living penalty: pressure to finish games rather than turtle.
        reward += cfg.step_penalty
        return reward

    def _margin_from_owner(self, owner: np.ndarray, player: int) -> float:
        """Territory margin (own lead over strongest rival, [-1,1]) for an owner array."""
        if owner.size == 0 or player >= self.state.n_players:
            return 0.0
        counts = np.bincount(owner, minlength=self.state.n_players)
        own = int(counts[player])
        opponents = [int(counts[p]) for p in range(self.state.n_players) if p != player]
        best_opp = max(opponents) if opponents else 0
        return (own - best_opp) / NUM_TERRITORIES

    def _snapshot(self) -> dict[str, Any]:
        s = self.state
        return {
            "territory_owner": s.territory_owner.copy(),
            "armies": s.armies.copy(),
            "eliminated": s.eliminated.copy(),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _is_valid_action(self, action: dict[str, int]) -> bool:
        if not isinstance(action, dict):
            return False
        for key in ("action_type", "param_a", "param_b", "param_c", "param_d"):
            if key not in action:
                return False
        return True

    def _get_legal_actions(self) -> list[dict[str, int]]:
        s = self.state
        actions: list[dict[str, int]] = []
        player = s.current_player

        def add(action_type: int, a: int, b: int, c: int, d: int) -> None:
            actions.append(
                {
                    "action_type": action_type,
                    "param_a": a,
                    "param_b": b,
                    "param_c": c,
                    "param_d": d,
                }
            )

        if s.phase == PHASE_TRADE:
            if not self._must_trade():
                add(5, 0, 0, 0, 0)
            hand = s.cards[player]
            from itertools import combinations

            for combo in combinations(range(len(hand)), 3):
                if self._is_valid_trade(list(combo), player):
                    a, b, c = combo
                    add(0, a, b, c, 0)
        elif s.phase == PHASE_REINFORCE:
            if s.reinforcements_remaining == 0:
                add(5, 0, 0, 0, 0)
            else:
                owned = np.where(s.territory_owner == player)[0]
                cap = min(s.reinforcements_remaining, MAX_ARMIES_PARAM)
                for terr in owned:
                    for count in range(1, cap + 1):
                        add(1, int(terr), count, 0, 0)
        elif s.phase == PHASE_ATTACK:
            add(5, 0, 0, 0, 0)
            owned = np.where((s.territory_owner == player) & (s.armies >= 2))[0]
            for attacker in owned:
                for target in ADJACENCY[attacker]:
                    if s.territory_owner[target] != player:
                        max_dice = min(3, int(s.armies[attacker]) - 1)
                        for dice in range(1, max_dice + 1):
                            add(2, int(attacker), int(target), dice, 0)
        elif s.phase == PHASE_CAPTURE_MOVE:
            attacker = s.last_attacker
            dice = s.last_capture_dice
            available = s.armies[attacker]
            cap = min(int(available) - 1, MAX_ARMIES_PARAM)  # keep ≥1 in attacker
            for move in range(dice, cap + 1):
                add(3, move, 0, 0, 0)
        elif s.phase == PHASE_FORTIFY:
            add(5, 0, 0, 0, 0)
            sources = np.where((s.territory_owner == player) & (s.armies >= 2))[0]
            if self.fortify_adjacent_only:
                for source in sources:
                    cap = min(int(s.armies[source]) - 1, MAX_ARMIES_PARAM)
                    for target in ADJACENCY[source]:
                        if s.territory_owner[target] == player:
                            for count in range(1, cap + 1):
                                add(4, int(source), int(target), count, 0)
            else:
                all_owned = {int(t) for t in np.where(s.territory_owner == player)[0]}
                for source in sources:
                    cap = min(int(s.armies[source]) - 1, MAX_ARMIES_PARAM)
                    for target in all_owned:
                        if target == int(source):
                            continue
                        if self._is_connected(int(source), target, player):
                            for count in range(1, cap + 1):
                                add(4, int(source), target, count, 0)
        return actions
