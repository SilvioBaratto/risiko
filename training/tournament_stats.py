"""Tournament statistics data model, tolerant ledger reader, and marginal aggregation.

Foundation module for Cycle 9 analysis: frozen dataclasses for the leaderboard,
a tolerant reader that converts games.jsonl into plain dicts, and per-strategy /
per-model win-rate (Wilson CI) + survival-placement aggregation.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from training.monte_carlo import wilson_ci  # noqa: F401 — re-exported for callers

__all__ = [
    "StrategyStat",
    "ModelStat",
    "MatrixCell",
    "Leaderboard",
    "read_ledger",
    "_diplomacy_of",
    "wilson_ci",
    "placements",
    "aggregate_strategies",
    "aggregate_models",
    "aggregate_diplomacy",
    "with_diplomacy",
]

_REQUIRED_KEYS: frozenset[str] = frozenset({"seat_strategies", "seat_models", "elimination_order"})


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyStat:
    """Aggregate win-rate and placement statistics for one strategy.

    Args:
        strategy: Strategy slug (e.g. ``australia_lock``).
        games: Total games in which this strategy participated.
        wins: Games won.
        win_rate: ``wins / games``.
        ci_low: Wilson 95% CI lower bound.
        ci_high: Wilson 95% CI upper bound.
        mean_placement: Average finishing place (1 = winner).
        betrayals: Total betrayal events (populated by diplomacy slice; default 0).
        alliances: Total alliance events (populated by diplomacy slice; default 0).
    """

    strategy: str
    games: int
    wins: int
    win_rate: float
    ci_low: float
    ci_high: float
    mean_placement: float
    betrayals: int = field(default=0)
    alliances: int = field(default=0)


@dataclass(frozen=True)
class ModelStat:
    """Aggregate win-rate and placement statistics for one model.

    Args:
        model: Model name string.
        games: Total games participated.
        wins: Games won.
        win_rate: ``wins / games``.
        ci_low: Wilson 95% CI lower bound.
        ci_high: Wilson 95% CI upper bound.
        mean_placement: Average finishing place (1 = winner).
    """

    model: str
    games: int
    wins: int
    win_rate: float
    ci_low: float
    ci_high: float
    mean_placement: float


@dataclass(frozen=True)
class MatrixCell:
    """Win statistics for one strategy × model pairing.

    Args:
        strategy: Strategy slug.
        model: Model name string.
        games: Games where this pairing appeared.
        wins: Games won by this pairing.
        win_rate: ``wins / games``.
    """

    strategy: str
    model: str
    games: int
    wins: int
    win_rate: float


@dataclass(frozen=True)
class Leaderboard:
    """Full tournament leaderboard for one run.

    Args:
        run: Run identifier string (directory name).
        n_games: Total well-formed games counted.
        n_malformed: Lines skipped due to parse errors or missing keys.
        n_players: Players per game (inferred from records; default 6).
        strategies: Ordered tuple of :class:`StrategyStat` entries.
        models: Ordered tuple of :class:`ModelStat` entries.
        matrix: Flat tuple of ``MatrixCell`` entries.
    """

    run: str
    n_games: int
    n_malformed: int
    n_players: int
    strategies: tuple  # tuple[StrategyStat, ...] at runtime
    models: tuple  # tuple[ModelStat, ...] at runtime
    matrix: tuple[MatrixCell, ...]


# ---------------------------------------------------------------------------
# Ledger reader
# ---------------------------------------------------------------------------


def read_ledger(path: Path) -> tuple[list[dict], int]:
    """Parse *path* (JSONL) into well-formed records and a malformed count.

    Args:
        path: Path to the ``games.jsonl`` ledger file.

    Returns:
        ``(well_formed_records, n_malformed)``.  Missing file returns
        ``([], 0)``.  Malformed JSON lines and records missing any of
        ``seat_strategies``, ``seat_models``, or ``elimination_order`` are
        counted in ``n_malformed`` and excluded from the record list.
    """
    if not path.is_file():
        return [], 0
    records: list[dict] = []
    n_malformed = 0
    with path.open() as fh:
        for line in fh:
            ok, rec = _parse_line(line)
            if ok and rec is not None:
                records.append(rec)
            else:
                n_malformed += 1
    return records, n_malformed


def _parse_line(line: str) -> tuple[bool, dict | None]:
    """Return ``(True, record)`` for a valid line, ``(False, None)`` otherwise."""
    with contextlib.suppress(Exception):
        data = json.loads(line)
        if isinstance(data, dict) and data.keys() >= _REQUIRED_KEYS:
            return True, data
        return False, None
    return False, None


# ---------------------------------------------------------------------------
# Optional diplomacy helper
# ---------------------------------------------------------------------------


def _diplomacy_of(rec: dict) -> dict:
    """Return the optional diplomacy fields from *rec*, defaulting to empty dicts.

    Args:
        rec: A well-formed ledger record dict.

    Returns:
        ``{"betrayals": <dict or {}>, "alliances": <dict or {}>}``.
        When either key is absent the corresponding value is an empty dict.
        The caller must never require these fields to exist.
    """
    return {
        "betrayals": rec.get("betrayals", {}),
        "alliances": rec.get("alliances", {}),
    }


# ---------------------------------------------------------------------------
# Per-strategy diplomacy aggregation (issue #127)
# ---------------------------------------------------------------------------


def _resolve_key(key: str, seat_strategies: dict[str, str]) -> str:
    """Seat indices (``"0"``–``"5"``) resolve via seat_strategies; others pass through."""
    return seat_strategies.get(key, key)


def _sum_counts(
    counts: dict[str, int],
    seat_strategies: dict[str, str],
    acc: dict[str, int],
) -> None:
    """Accumulate *counts* into *acc*, resolving seat vs. strategy keys first."""
    for key, n in counts.items():
        s = _resolve_key(key, seat_strategies)
        acc[s] = acc.get(s, 0) + n


def aggregate_diplomacy(records: Iterable[dict]) -> dict[str, tuple[int, int]]:
    """Return per-strategy ``(betrayals, alliances)`` totals summed across *records*.

    Both seat-keyed (``{"0": n}``) and strategy-keyed (``{"aggressive_blitz": n}``)
    forms are tolerated.  Records missing diplomacy fields contribute 0.
    """
    b_acc: dict[str, int] = {}
    a_acc: dict[str, int] = {}
    seen: set[str] = set()
    for rec in records:
        seat_strats = rec.get("seat_strategies", {})
        seen.update(seat_strats.values())
        dip = _diplomacy_of(rec)
        _sum_counts(dip["betrayals"], seat_strats, b_acc)
        _sum_counts(dip["alliances"], seat_strats, a_acc)
    return {s: (b_acc.get(s, 0), a_acc.get(s, 0)) for s in seen}


def with_diplomacy(
    strategy_stats: tuple[StrategyStat, ...],
    records: Iterable[dict],
) -> tuple[StrategyStat, ...]:
    """Return enriched ``StrategyStat`` tuple with betrayals/alliances from *records*.

    Uses :func:`dataclasses.replace` — all other fields on each stat are unchanged.
    """
    diplomacy = aggregate_diplomacy(list(records))
    return tuple(
        replace(
            stat,
            betrayals=diplomacy.get(stat.strategy, (0, 0))[0],
            alliances=diplomacy.get(stat.strategy, (0, 0))[1],
        )
        for stat in strategy_stats
    )


# ---------------------------------------------------------------------------
# Wilson CI guard
# ---------------------------------------------------------------------------


def _safe_wilson(wins: int, n: int) -> tuple[float, float]:
    """Return Wilson 95% CI, or ``(0.0, 0.0)`` when *n* == 0."""
    if n == 0:
        return 0.0, 0.0
    return wilson_ci(wins, n)


# ---------------------------------------------------------------------------
# Placement computation
# ---------------------------------------------------------------------------


def placements(rec: dict) -> dict[int, int]:
    """Map every seat in *rec* to its finishing placement (1 = best).

    Seats absent from ``elimination_order`` are survivors → placement 1.
    Eliminated seats rank from 2 (last out) upward by reverse elimination order.
    Tolerates a missing or empty ``elimination_order``.
    """
    elim: list[int] = rec.get("elimination_order") or []
    n_elim = len(elim)
    result = {int(k): 1 for k in rec["seat_strategies"]}
    for idx, seat in enumerate(elim):
        result[seat] = n_elim - idx + 1
    return result


# ---------------------------------------------------------------------------
# Per-strategy and per-model marginal aggregation
# ---------------------------------------------------------------------------

_Acc = dict  # {games: int, wins: int, placement_sum: int}


def _new_acc() -> _Acc:
    return {"games": 0, "wins": 0, "placement_sum": 0}


def _finalise_strategy(name: str, acc: _Acc) -> StrategyStat:
    games, wins = acc["games"], acc["wins"]
    win_rate = wins / games if games else 0.0
    ci_low, ci_high = _safe_wilson(wins, games)
    return StrategyStat(
        strategy=name,
        games=games,
        wins=wins,
        win_rate=win_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        mean_placement=acc["placement_sum"] / games if games else 0.0,
    )


def aggregate_strategies(records: Iterable[dict]) -> tuple[StrategyStat, ...]:
    """Return one :class:`StrategyStat` per strategy, sorted win_rate desc then ci_low desc."""
    accum: dict[str, _Acc] = {}
    for rec in records:
        seat_pl = placements(rec)
        winner_strategy = rec.get("winner_strategy")
        for seat_str, strategy in rec["seat_strategies"].items():
            acc = accum.setdefault(strategy, _new_acc())
            acc["games"] += 1
            acc["wins"] += 1 if strategy == winner_strategy else 0
            acc["placement_sum"] += seat_pl[int(seat_str)]
    stats = (_finalise_strategy(name, acc) for name, acc in accum.items())
    return tuple(sorted(stats, key=lambda s: (-s.win_rate, -s.ci_low, s.strategy)))


def _finalise_model(name: str, acc: _Acc) -> ModelStat:
    games, wins = acc["games"], acc["wins"]
    win_rate = wins / games if games else 0.0
    ci_low, ci_high = _safe_wilson(wins, games)
    return ModelStat(
        model=name,
        games=games,
        wins=wins,
        win_rate=win_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        mean_placement=acc["placement_sum"] / games if games else 0.0,
    )


def aggregate_models(records: Iterable[dict]) -> tuple[ModelStat, ...]:
    """Return one :class:`ModelStat` per model, sorted win_rate desc then ci_low desc."""
    accum: dict[str, _Acc] = {}
    for rec in records:
        seat_pl = placements(rec)
        winner_model = rec.get("winner_model")
        for seat_str, model in rec["seat_models"].items():
            acc = accum.setdefault(model, _new_acc())
            acc["games"] += 1
            acc["wins"] += 1 if model == winner_model else 0
            acc["placement_sum"] += seat_pl[int(seat_str)]
    stats = (_finalise_model(name, acc) for name, acc in accum.items())
    return tuple(sorted(stats, key=lambda m: (-m.win_rate, -m.ci_low, m.model)))
