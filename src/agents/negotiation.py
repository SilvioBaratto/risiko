"""Negotiation orchestrator for the eval-only diplomacy layer (issue #101).

Orchestrates bounded negotiation rounds: validates LLM payloads, applies
mutual-consent alliance formation (order-independent), and records war
declarations as broken alliances + grudges. The RL learner (slot 0) is
always excluded from speakers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from src.agents.diplomacy import DiplomacyState
from src.agents.negotiation_prompt import render_negotiation_prompt
from src.agents.player_config import PlayerConfig
from src.agents.reputation import ReputationBook
from src.config import DiplomacyConfig
from src.utils.log import get_logger

_log = get_logger(__name__)

LEARNER_SLOT: int = 0


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _valid_ids(ids: Any, speaker: int, n_players: int) -> tuple[int, ...]:
    """Drop out-of-range, self-referential, and duplicate ids; return as tuple."""
    seen: set[int] = set()
    result: list[int] = []
    for pid in ids or []:
        if isinstance(pid, int) and 0 <= pid < n_players and pid != speaker and pid not in seen:
            seen.add(pid)
            result.append(pid)
    return tuple(result)


# ---------------------------------------------------------------------------
# NegotiationMessage — validated, frozen payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NegotiationMessage:
    """Validated, sanitized negotiation output from one LLM speaker."""

    speaker: int
    propose_alliance_with: tuple[int, ...]
    accept_alliance_with: tuple[int, ...]
    declare_war_on: tuple[int, ...]
    attack_priority: tuple[int, ...]

    @staticmethod
    def from_raw(speaker: int, raw: dict[str, Any], n_players: int) -> NegotiationMessage:
        """Build from raw LLM dict: drop invalid ids, dedupe each list."""
        clean = lambda f: _valid_ids(raw.get(f), speaker, n_players)  # noqa: E731
        return NegotiationMessage(
            speaker=speaker,
            propose_alliance_with=clean("propose_alliance_with"),
            accept_alliance_with=clean("accept_alliance_with"),
            declare_war_on=clean("declare_war_on"),
            attack_priority=clean("attack_priority"),
        )


# ---------------------------------------------------------------------------
# State-mutation helpers (pure; separated for testability)
# ---------------------------------------------------------------------------


def _form_mutual_alliances(messages: list[NegotiationMessage], state: DiplomacyState) -> None:
    """Form alliances where both sides propose/accept each other (order-independent).

    Collected after ALL messages in the round so the result is independent of
    speaker iteration order (sorted(speakers) ensures determinism).
    """
    expressed: dict[int, set[int]] = {}
    for msg in messages:
        expressed[msg.speaker] = set(msg.propose_alliance_with + msg.accept_alliance_with)
    speakers = sorted(expressed)
    for i, a in enumerate(speakers):
        for b in speakers[i + 1 :]:
            if b in expressed.get(a, set()) and a in expressed.get(b, set()):
                state.form_alliance(a, b)


def _apply_war_declarations(
    messages: list[NegotiationMessage], state: DiplomacyState, turn: int = 0
) -> None:
    """Break alliances and record grudges for war declarations (sorted for determinism)."""
    for msg in sorted(messages, key=lambda m: m.speaker):
        for target in msg.declare_war_on:
            state.declare_war_on(attacker=msg.speaker, target=target, turn=turn)


# ---------------------------------------------------------------------------
# Default call_fn (no-op)
# ---------------------------------------------------------------------------


def _noop_call_fn(player_id: int, prompt: str) -> None:  # noqa: ARG001
    """Sentinel: always returns None so all speakers are treated as timed-out."""
    return None


# ---------------------------------------------------------------------------
# NegotiationOrchestrator
# ---------------------------------------------------------------------------


class NegotiationOrchestrator:
    """Orchestrates bounded negotiation rounds before each game turn.

    ``call_fn`` receives ``(player_id: int, prompt: str) -> dict | None``.
    ``None`` is the fallback contract (timeout/parse error) — treated as a
    silent no-op for that speaker; the round continues.
    """

    def __init__(
        self,
        state: DiplomacyState,
        reputation: ReputationBook,
        cfg: DiplomacyConfig,
        profiles: Sequence[PlayerConfig],
        *,
        call_fn: Callable[[int, str], dict[str, Any] | None] | None = None,
        learner_slot: int = LEARNER_SLOT,
    ) -> None:
        """Initialise with shared state, config, player profiles, and injectable call_fn.

        ``call_fn`` defaults to ``None``, which means all speakers return None
        (silent no-op — no alliances form). Inject a real or mocked call_fn for
        production or test use.
        """
        self._state = state
        self._reputation = reputation
        self._cfg = cfg
        self._profiles: dict[int, PlayerConfig] = {p.player_id: p for p in profiles}
        self._call_fn: Callable[[int, str], dict[str, Any] | None] = (
            call_fn if call_fn is not None else _noop_call_fn
        )
        self._learner = learner_slot
        self.call_count: int = 0

    def run_round(self, obs: dict[str, Any], speakers: Sequence[int]) -> int:
        """Run one negotiation round for active (non-learner) speakers.

        Returns the number of LLM calls made (one per active speaker).
        Alliances and war declarations are applied after collecting all
        messages so the outcome is order-independent.
        """
        active = sorted(s for s in speakers if s != self._learner)
        messages = self._collect(active, obs)
        _form_mutual_alliances(messages, self._state)
        _apply_war_declarations(messages, self._state)
        self.call_count += len(active)
        _log.info("negotiation round: %d calls (total=%d)", len(active), self.call_count)
        return len(active)

    def _collect(self, speakers: list[int], obs: dict[str, Any]) -> list[NegotiationMessage]:
        """Call each speaker once; None responses are silent no-ops."""
        n = len(self._profiles)
        result: list[NegotiationMessage] = []
        for speaker in speakers:
            raw = self._call_fn(speaker, self._prompt_for(speaker, obs))
            if raw is not None:
                result.append(NegotiationMessage.from_raw(speaker, raw, n))
        return result

    def _prompt_for(self, speaker: int, obs: dict[str, Any]) -> str:
        """Build the negotiation prompt for this speaker.

        Nobody is ever told that the leader is themselves. The action prompt has always
        guarded this (``build_diplomacy_note``: ``leader if leader != player else None``);
        the negotiation prompt did not, so in 1 traced negotiation out of 6 a model read
        ``Leader (most territories): Player N`` with N being its own seat.

        That is not a cosmetic slip. The coalition strategy's whole instruction is
        "gang up on the leader" — pointed at itself, it is degenerate, and the logs show
        exactly that: the diplomat, told it was the leader, declared war on a random
        player and proposed no alliances at all. The aggressor, in the same spot,
        declared war on a player and offered it an alliance in the same reply.
        """
        allies = sorted(p for p in self._profiles if self._state.are_allied(speaker, p))
        owners = obs.get("territory_owner", [])
        leader = self._state.leader(owners) if len(owners) > 0 else None
        return render_negotiation_prompt(
            player_id=speaker,
            board=obs,
            allies=allies,
            leader=leader if leader != speaker else None,
            grudges={},
            max_chars=self._cfg.max_message_tokens * 5,
        )


__all__ = ["NegotiationMessage", "NegotiationOrchestrator", "LEARNER_SLOT"]
