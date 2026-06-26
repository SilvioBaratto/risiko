"""Tournament driver: frozen dataclasses, YAML loader, and ledger primitives.

Entry point for Cycle 8 — all later tournament tasks extend this module.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import yaml

import src.agents.ollama_client as _ollama_client
from src.agents.llm_opponent import LLMOpponent
from src.agents.ollama_client import list_ollama_models
from src.agents.player_config import PlayerConfig
from src.agents.reputation import ReputationBook
from src.agents.strategies import (
    CATALOG_BY_NAME,
    get_strategy,
    to_player_config,
)
from src.config import DiplomacyConfig
from src.env import RisikoEnv
from src.multi_agent import DiplomacyRunner
from src.utils.env import ensure_env_loaded
from src.utils.log import get_logger

_log = get_logger("tournament")

_N_PLAYERS = 6
_MAX_TURNS: int = 200
_NEGOTIATION_MAX_TOKENS: int = 512


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TournamentConfig:
    """Parsed and validated tournament configuration."""

    name: str
    seed: int
    n_games: int
    max_concurrency: int
    n_rounds: int
    strategies: tuple[str, ...]
    models: tuple[str, ...]
    temperature: float
    top_p: float


@dataclass(frozen=True)
class GamePlan:
    """Per-game execution plan derived from TournamentConfig."""

    game_index: int
    seed: int
    assignment: dict[str, str]


@dataclass(frozen=True)
class LedgerRecord:
    """JSON-serialisable projection of a single finished game."""

    game_index: int
    seed: int
    winner: int | None
    winner_strategy: str | None
    winner_model: str | None
    n_turns: int
    elimination_order: list[int]
    seat_strategies: dict[str, str]
    seat_models: dict[str, str]
    assignment: dict[str, str]
    card_trade_turns: list[int]
    llm_call_count: int

    def to_json_line(self) -> str:
        """Serialise to a single JSON line with no trailing newline."""
        return json.dumps(_to_serialisable(self.__dict__))

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> LedgerRecord:
        """Reconstruct a LedgerRecord from a parsed JSON dict."""
        return cls(**data)


@dataclass(frozen=True)
class TournamentSummary:
    """Aggregate statistics returned by run_tournament."""

    run_dir: Path
    target_games: int
    games_completed_this_run: int
    total_games_in_ledger: int
    total_llm_calls: int


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_tournament_config(path: Path) -> TournamentConfig:
    """Parse *path* via yaml.safe_load and return a validated TournamentConfig.

    Raises ValueError when models ≠ 6, strategies ≠ 6, or any strategy
    slug is not in CATALOG_BY_NAME.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return _build_config(raw)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_config(raw: dict[str, Any]) -> TournamentConfig:
    models = raw.get("models", [])
    strategies = raw.get("strategies", [])
    _validate_counts(strategies, models)
    _validate_slugs(strategies)
    return TournamentConfig(
        name=raw["name"],
        seed=int(raw["seed"]),
        n_games=int(raw["n_games"]),
        max_concurrency=int(raw["max_concurrency"]),
        n_rounds=int(raw["n_rounds"]),
        strategies=tuple(strategies),
        models=tuple(models),
        temperature=float(raw.get("temperature", 0.7)),
        top_p=float(raw.get("top_p", 0.9)),
    )


def _validate_counts(strategies: list, models: list) -> None:
    if len(models) != _N_PLAYERS:
        raise ValueError(f"Expected {_N_PLAYERS} models, got {len(models)}: {models}")
    if len(strategies) != _N_PLAYERS:
        raise ValueError(f"Expected {_N_PLAYERS} strategies, got {len(strategies)}: {strategies}")


def _validate_slugs(strategies: list[str]) -> None:
    unknown = [s for s in strategies if s not in CATALOG_BY_NAME]
    if unknown:
        raise ValueError(
            f"Unknown strategy slug(s) {unknown}. Valid names: {sorted(CATALOG_BY_NAME)}"
        )


def _to_serialisable(obj: Any) -> Any:
    """Recursively cast numpy scalars/arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Preflight model verification (issue #119)
# ---------------------------------------------------------------------------


def run_preflight(models: Sequence[str]) -> None:
    """Verify all *models* are available on the Ollama server before any game.

    Mirrors ``risiko_rl.cli._preflight_validate_model`` without the Typer
    dependency: raises ``RuntimeError`` with the missing names so the caller
    (CLI or tests) can handle it uniformly.  If ``list_ollama_models`` returns
    ``None`` (server unqueryable), logs a warning and proceeds rather than
    aborting a valid run on a flaky probe.
    """
    available = list_ollama_models()
    if available is None:
        _log.warning("Could not query Ollama model list; skipping preflight model check.")
        return
    missing = sorted(set(models) - set(available))
    if not missing:
        return
    raise RuntimeError(
        f"Model(s) not available on Ollama server: {', '.join(missing)}. "
        "Pull the model (`ollama pull <name>`) or check available models."
    )


# ---------------------------------------------------------------------------
# Seeded strategy↔model bijection (issue #118)
# ---------------------------------------------------------------------------


def assign_strategies_to_models(
    strategies: Sequence[str],
    models: Sequence[str],
    *,
    seed: int,
) -> dict[str, str]:
    """Return a bijection {strategy: model} permuted by *seed*.

    Uses np.random.default_rng so the same seed always yields the same
    permutation and different seeds yield varied permutations.
    The two sequences must have equal length.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(models))
    models_list = list(models)
    return {strategies[i]: models_list[perm[i]] for i in range(len(strategies))}


def build_game_plan(config: TournamentConfig, game_index: int) -> GamePlan:
    """Return a deterministic GamePlan for *game_index* derived from *config*.

    Per-game seed = config.seed + game_index — simple, collision-free for any
    tournament size. The assignment RNG is isolated from the env/gameplay seed
    stream to avoid correlation.
    """
    per_game_seed = config.seed + game_index
    assignment = assign_strategies_to_models(
        list(config.strategies), list(config.models), seed=per_game_seed
    )
    return GamePlan(game_index=game_index, seed=per_game_seed, assignment=assignment)


# ---------------------------------------------------------------------------
# Resumable-ledger primitives (implemented in full in issue #117)
# ---------------------------------------------------------------------------


def pending_game_indices(total_games: int, completed: frozenset[int]) -> set[int]:
    """Return indices not yet in *completed*."""
    return set(range(total_games)) - completed


def pending_games(total: int, completed_ids: set[int]) -> set[int]:
    """Return game indices in ``[0, total)`` not present in *completed_ids*.

    Thin alias over :func:`pending_game_indices`; out-of-range completed ids
    are naturally excluded since they fall outside ``range(total)``.
    """
    return pending_game_indices(total, frozenset(completed_ids))


def load_completed_indices(ledger: Path) -> set[int]:
    """Parse game_index from each JSONL line; tolerate missing file and bad lines."""
    if not ledger.exists():
        return set()
    indices: set[int] = set()
    with open(ledger) as fh:
        for line in fh:
            with contextlib.suppress(Exception):
                indices.add(json.loads(line)["game_index"])
    return indices


def load_ledger(path: str | Path) -> set[int]:
    """Return completed game indices from *path*; alias for load_completed_indices."""
    return load_completed_indices(Path(path))


def append_record(ledger: Path, record: LedgerRecord) -> None:
    """Append one JSON line to *ledger*, flushing and fsyncing for durability."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as fh:
        fh.write(record.to_json_line() + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# Concurrent scheduler (issue #121)
# ---------------------------------------------------------------------------


def _game_index_of(obj: Any) -> int:
    """Return game_index from a GamePlan, LedgerRecord, or plain dict."""
    return obj["game_index"] if isinstance(obj, dict) else obj.game_index


def _llm_calls_of(record: Any) -> int:
    """Return llm_call_count from a record; 0 when absent."""
    if isinstance(record, dict):
        return record.get("llm_call_count", 0)
    return getattr(record, "llm_call_count", 0)


def _filter_pending(plans: list, ledger_path: Path | None) -> list:
    """Return plans not yet recorded in *ledger_path*."""
    if ledger_path is None:
        return list(plans)
    done = load_completed_indices(ledger_path)
    return [p for p in plans if _game_index_of(p) not in done]


def _write_result(ledger_path: Path, record: Any) -> None:
    """Append *record* to *ledger_path* as a durable JSON line."""
    line = json.dumps(record) if isinstance(record, dict) else record.to_json_line()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _log_completion(record: Any, llm_count: int, cumulative: int, games_done: int) -> None:
    """Log per-game and cumulative LLM-call counts."""
    _log.info(
        "game %d done; llm_calls=%d cumulative_llm_calls=%d games_done=%d",
        _game_index_of(record),
        llm_count,
        cumulative,
        games_done,
    )


class TournamentRunner:
    """Run pending games concurrently up to *max_concurrency*."""

    def __init__(
        self,
        max_concurrency: int = _N_PLAYERS,
        max_retries: int = 3,
        base_backoff: float = 1.0,
        run_dir: str | Path | None = None,
        ledger_path: str | Path | None = None,
    ) -> None:
        """Initialise runner with concurrency, backoff, and optional ledger path."""
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None

    def run(
        self,
        plans: list[GamePlan],
        play_one: Callable[[GamePlan], LedgerRecord],
        on_result: Callable[[LedgerRecord], None],
    ) -> int:
        """Execute pending plans concurrently; call *on_result* on the main thread.

        Each game is guarded by ``_play_with_backoff`` so transient HTTP errors
        (e.g. 429 rate-limit) trigger exponential-backoff retries before failing.
        """
        pending = _filter_pending(plans, self.ledger_path)
        ctx: dict[str, int] = {"games": 0, "llm": 0}
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = {
                pool.submit(self._play_with_backoff, lambda p=p: play_one(p)): p for p in pending
            }
            for fut in as_completed(futures):
                self._on_game_done(fut.result(), on_result, ctx)
        return ctx["games"]

    def _on_game_done(
        self,
        record: Any,
        on_result: Callable[[Any], None],
        ctx: dict[str, int],
    ) -> None:
        """Process one finished game on the main thread (serialises ledger writes)."""
        on_result(record)
        llm_count = _llm_calls_of(record)
        ctx["llm"] += llm_count
        ctx["games"] += 1
        _log_completion(record, llm_count, ctx["llm"], ctx["games"])
        if self.ledger_path is not None:
            _write_result(self.ledger_path, record)

    def _play_with_backoff(
        self,
        fn: Callable[[], Any],
        max_retries: int | None = None,
    ) -> Any:
        """Call fn(); retry on httpx.HTTPError with exponential backoff.

        Note: LLMOpponent already catches most action-call errors internally and
        falls back to RandomAgent, so this wrapper is a safety net for errors that
        propagate to game level (e.g. negotiation calls, preflight).
        """
        retries = max_retries if max_retries is not None else self.max_retries
        for attempt in range(retries + 1):
            try:
                return fn()
            except httpx.HTTPError:
                if attempt == retries:
                    raise
                time.sleep(self.base_backoff * (2**attempt))


# ---------------------------------------------------------------------------
# Seat building (issue #120)
# ---------------------------------------------------------------------------


@dataclass
class _DiplomacyWiring:
    """Wiring parameters for a tournament diplomacy runner (not a runner itself)."""

    enabled: bool
    n_rounds: int
    learner_id: int
    reputation_book: Any

    def run_game(self, **_: Any) -> None:
        """Placeholder consumed by _play_one_game which builds the real DiplomacyRunner."""


def build_seats(
    assignment: dict[str, str],
    config: TournamentConfig,
) -> list[PlayerConfig]:
    """Return one PlayerConfig per slot, in strategy-list order.

    ``assignment`` is a bijection {strategy_slug: model}.  Slot *i* receives
    the model for ``config.strategies[i]``.  The ``strategy_hint`` and posture
    levers (trust_propensity, vengefulness) come from the strategy catalog via
    ``to_player_config``.
    """
    return [
        to_player_config(
            get_strategy(strategy),
            player_id=i,
            temperature=config.temperature,
            top_p=config.top_p,
            model=assignment[strategy],
        )
        for i, strategy in enumerate(config.strategies)
    ]


def build_diplomacy_runner(
    config: TournamentConfig,
    reputation_book: Any,
) -> _DiplomacyWiring:
    """Return a wiring record for a tournament diplomacy runner.

    The runner is built with ``enabled=True``, ``n_rounds=config.n_rounds``,
    and ``learner_id=-1`` (no RL learner in pure-LLM tournament mode).  The
    shared ``reputation_book`` is stored for later use by ``_play_one_game``.
    """
    return _DiplomacyWiring(
        enabled=True,
        n_rounds=config.n_rounds,
        learner_id=-1,
        reputation_book=reputation_book,
    )


def build_negotiation_call_fn(
    player_id: int,
    player_configs: Sequence[PlayerConfig],
    base_url: str,
) -> Callable | None:
    """Return a callable ``(player_id, prompt) -> dict | None`` for *player_id*.

    Routes negotiation prompts through ``call_ollama_for_negotiation`` (not the
    action-index path).  Strips a trailing ``/v1`` from *base_url* (Ollama cloud
    compatibility).  The returned callable returns ``None`` on any LLM/HTTP error.
    """
    cfg = player_configs[player_id]
    root = base_url.removesuffix("/v1")
    key = os.environ.get("OLLAMA_API_KEY", "ollama")

    def fn(*args: Any) -> dict | None:
        prompt: str = args[1] if len(args) > 1 else ""
        result = _ollama_client.call_ollama_for_negotiation(
            root=root,
            key=key,
            model=cfg.model,
            prompt=prompt,
            max_message_tokens=_NEGOTIATION_MAX_TOKENS,
            temperature=cfg.temperature,
        )
        return result if isinstance(result, dict) else None

    return fn


def run_game(
    plan: GamePlan,
    config: TournamentConfig,
    book: ReputationBook | None = None,
) -> LedgerRecord:
    """Module-level alias for _play_one_game; preserved for patching compatibility."""
    return _play_one_game(plan, config, book)


def _play_one_game(
    plan: GamePlan,
    config: TournamentConfig,
    book: ReputationBook | None = None,
) -> LedgerRecord:
    """Run one 6-player LLM game and return a serialisable LedgerRecord.

    All seats are ``LLMOpponent`` agents wired to their assigned models via
    ``build_seats``.  Diplomacy is enabled with ``learner_id=-1`` so every
    player participates in negotiations.  LLM calls are counted across both
    action picks and negotiation speaker turns.

    *book* is the session-shared ``ReputationBook``; a fresh one is created when
    ``None`` (direct call / test path without run_tournament).
    """
    if book is None:
        book = ReputationBook()
    ensure_env_loaded()
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    seats = build_seats(plan.assignment, config)
    agents: list[LLMOpponent] = [LLMOpponent(player_config=seat) for seat in seats]

    diplomacy_cfg = DiplomacyConfig(enabled=True, n_rounds=config.n_rounds)

    per_player_fns = [build_negotiation_call_fn(i, seats, base_url) for i in range(_N_PLAYERS)]

    def _routing_call_fn(player_id: int, prompt: str) -> dict | None:
        fn = per_player_fns[player_id]
        if fn is None:
            return None
        try:
            result = fn(player_id, prompt)
            return result if isinstance(result, dict) else None
        except Exception:
            return None

    env = RisikoEnv(n_players=_N_PLAYERS)
    runner = DiplomacyRunner(
        env=env,
        agents=agents,
        max_turns=_MAX_TURNS,
        diplomacy_cfg=diplomacy_cfg,
        reputation=book,
        learner_id=-1,
        profiles=seats,
        call_fn=_routing_call_fn,
    )

    result = runner.run_game(seed=plan.seed)

    action_calls = sum(a.llm_call_count for a in agents)
    total_llm_calls = action_calls + runner.call_count

    winner = result.winner
    winner_strategy: str | None = None
    winner_model: str | None = None
    if winner is not None:
        winner_strategy = list(config.strategies)[winner]
        winner_model = plan.assignment.get(winner_strategy)

    seat_strategies = {str(i): list(config.strategies)[i] for i in range(_N_PLAYERS)}
    seat_models = {str(i): seats[i].model for i in range(_N_PLAYERS)}

    return LedgerRecord(
        game_index=plan.game_index,
        seed=plan.seed,
        winner=winner,
        winner_strategy=winner_strategy,
        winner_model=winner_model,
        n_turns=result.n_turns,
        elimination_order=result.elimination_order,
        seat_strategies=seat_strategies,
        seat_models=seat_models,
        assignment=dict(plan.assignment),
        card_trade_turns=result.card_trade_turns,
        llm_call_count=total_llm_calls,
    )


# ---------------------------------------------------------------------------
# Orchestration helpers (issue #122)
# ---------------------------------------------------------------------------


def _ledger_totals(ledger: Path) -> tuple[int, int]:
    """Return (total_game_count, total_llm_calls) by reading the full ledger."""
    if not ledger.exists():
        return 0, 0
    count, total_llm = 0, 0
    with open(ledger) as fh:
        for line in fh:
            with contextlib.suppress(Exception):
                data = json.loads(line)
                count += 1
                total_llm += data.get("llm_call_count", 0)
    return count, total_llm


def _cap_pending(pending: set[int], max_games: int | None) -> set[int]:
    """Slice *pending* to at most *max_games* indices (sorted for determinism)."""
    if max_games is None:
        return pending
    return set(sorted(pending)[:max_games])


def _resolve_play_fn(
    config: TournamentConfig,
    play_fn: Callable | None,
    skip_preflight: bool,
    book: ReputationBook,
) -> Callable[[GamePlan], LedgerRecord]:
    """Return the effective play function; run preflight when using the real one."""
    if play_fn is not None:
        return play_fn
    if not skip_preflight:
        run_preflight(config.models)
    return lambda plan: _play_one_game(plan, config, book)


def _build_summary(
    config: TournamentConfig,
    ledger: Path,
    games_done: int,
) -> TournamentSummary:
    """Build TournamentSummary from the post-run ledger totals."""
    total_games, total_llm = _ledger_totals(ledger)
    return TournamentSummary(
        run_dir=ledger.parent,
        target_games=config.n_games,
        games_completed_this_run=games_done,
        total_games_in_ledger=total_games,
        total_llm_calls=total_llm,
    )


def run_tournament(
    config: TournamentConfig,
    ledger: Path,
    *,
    play_fn: Callable[[GamePlan], LedgerRecord] | None = None,
    max_games: int | None = None,
    skip_preflight: bool = False,
) -> TournamentSummary:
    """Orchestrate a resumable tournament run.

    Loads completed indices from *ledger*, schedules only pending games,
    applies *max_games* cap, runs concurrently, appends each result, and
    returns a TournamentSummary derived from the full (post-run) ledger.
    """
    book = ReputationBook()
    effective_play_fn = _resolve_play_fn(config, play_fn, skip_preflight, book)
    completed = load_completed_indices(ledger)
    pending = _cap_pending(pending_game_indices(config.n_games, frozenset(completed)), max_games)
    plans = [build_game_plan(config, idx) for idx in sorted(pending)]
    runner = TournamentRunner(max_concurrency=config.max_concurrency)
    games_done = runner.run(plans, effective_play_fn, lambda r: append_record(ledger, r))
    return _build_summary(config, ledger, games_done)


__all__ = [
    "TournamentConfig",
    "GamePlan",
    "LedgerRecord",
    "TournamentSummary",
    "TournamentRunner",
    "load_tournament_config",
    "assign_strategies_to_models",
    "build_game_plan",
    "pending_game_indices",
    "pending_games",
    "load_completed_indices",
    "load_ledger",
    "append_record",
    "run_preflight",
    "run_game",
    "build_seats",
    "build_diplomacy_runner",
    "build_negotiation_call_fn",
    "_play_one_game",
    "run_tournament",
]
