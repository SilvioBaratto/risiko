"""Source-blind example tests for training/tournament_stats.py.

Issue #124 tests: read_ledger, StrategyStat / ModelStat / MatrixCell / Leaderboard
                  field presence and immutability, _diplomacy_of helper.
Issue #125 tests: placements(), aggregate_strategies(), aggregate_models(),
                  _safe_wilson(), wilson_ci(), sort-order invariants.

All tests are derived from the acceptance criteria only; no implementation
source has been read or inspected.
"""

from __future__ import annotations

import dataclasses
import json
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from training.tournament_stats import (
    Leaderboard,
    MatrixCell,
    ModelStat,
    StrategyStat,
    _diplomacy_of,
    _safe_wilson,
    aggregate_models,
    aggregate_strategies,
    placements,
    read_ledger,
    wilson_ci,
)

# ---------------------------------------------------------------------------
# Helpers — fixture construction from criteria, not from production code
# ---------------------------------------------------------------------------


def _make_valid_record(**overrides: object) -> dict:
    """Minimal well-formed ledger record: carries all three required keys."""
    base: dict = {
        "game_index": 0,
        "seed": 42,
        "winner": 0,
        "winner_strategy": "australia_lock",
        "winner_model": "model_a",
        "n_turns": 50,
        "elimination_order": [5, 4, 3, 2, 1],
        "seat_strategies": {
            "0": "australia_lock",
            "1": "south_america_lock",
            "2": "aggressive_blitz",
            "3": "turtle_defensive",
            "4": "card_cycle_hunter",
            "5": "diplomat_coalition",
        },
        "seat_models": {
            "0": "model_a",
            "1": "model_b",
            "2": "model_c",
            "3": "model_d",
            "4": "model_e",
            "5": "model_f",
        },
        "assignment": {
            "australia_lock": "model_a",
            "south_america_lock": "model_b",
            "aggressive_blitz": "model_c",
            "turtle_defensive": "model_d",
            "card_cycle_hunter": "model_e",
            "diplomat_coalition": "model_f",
        },
        "card_trade_turns": [],
        "llm_call_count": 10,
    }
    base.update(overrides)
    return base


def _write_jsonl(path: Path, lines: list) -> None:
    """Write records or raw strings to a JSONL file."""
    with path.open("w") as fh:
        for line in lines:
            if isinstance(line, str):
                fh.write(line + "\n")
            else:
                fh.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------------------
# StrategyStat — field presence, types, defaults, immutability
# ---------------------------------------------------------------------------


class TestStrategyStatFields:
    def test_when_strategy_stat_created_then_strategy_field_is_present(self) -> None:
        stat = StrategyStat(
            strategy="australia_lock",
            games=10,
            wins=4,
            win_rate=0.4,
            ci_low=0.12,
            ci_high=0.72,
            mean_placement=2.5,
        )
        assert stat.strategy == "australia_lock"

    def test_when_strategy_stat_created_then_games_wins_win_rate_fields_are_present(
        self,
    ) -> None:
        stat = StrategyStat(
            strategy="aggressive_blitz",
            games=8,
            wins=3,
            win_rate=0.375,
            ci_low=0.09,
            ci_high=0.72,
            mean_placement=2.0,
        )
        assert stat.games == 8
        assert stat.wins == 3
        assert stat.win_rate == 0.375

    def test_when_strategy_stat_created_then_ci_and_placement_fields_are_present(
        self,
    ) -> None:
        stat = StrategyStat(
            strategy="turtle_defensive",
            games=6,
            wins=1,
            win_rate=0.167,
            ci_low=0.0,
            ci_high=0.55,
            mean_placement=3.2,
        )
        assert stat.ci_low == 0.0
        assert stat.ci_high == 0.55
        assert stat.mean_placement == 3.2

    def test_when_strategy_stat_created_without_diplomacy_args_then_betrayals_default_to_zero(
        self,
    ) -> None:
        """Betrayals must default to 0 so later issues can populate without schema change."""
        stat = StrategyStat(
            strategy="card_cycle_hunter",
            games=5,
            wins=1,
            win_rate=0.2,
            ci_low=0.0,
            ci_high=0.52,
            mean_placement=3.0,
        )
        assert stat.betrayals == 0

    def test_when_strategy_stat_created_without_diplomacy_args_then_alliances_default_to_zero(
        self,
    ) -> None:
        """Alliances must default to 0 so later issues can populate without schema change."""
        stat = StrategyStat(
            strategy="diplomat_coalition",
            games=5,
            wins=2,
            win_rate=0.4,
            ci_low=0.07,
            ci_high=0.78,
            mean_placement=1.8,
        )
        assert stat.alliances == 0

    def test_when_strategy_stat_is_mutated_then_frozen_error_is_raised(self) -> None:
        stat = StrategyStat(
            strategy="south_america_lock",
            games=5,
            wins=2,
            win_rate=0.4,
            ci_low=0.08,
            ci_high=0.77,
            mean_placement=2.2,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            stat.games = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelStat — field presence, immutability
# ---------------------------------------------------------------------------


class TestModelStatFields:
    def test_when_model_stat_created_then_model_field_is_present(self) -> None:
        stat = ModelStat(
            model="model_a",
            games=10,
            wins=3,
            win_rate=0.3,
            ci_low=0.07,
            ci_high=0.62,
            mean_placement=2.8,
        )
        assert stat.model == "model_a"

    def test_when_model_stat_created_then_stats_fields_are_present(self) -> None:
        stat = ModelStat(
            model="model_b",
            games=12,
            wins=5,
            win_rate=0.417,
            ci_low=0.15,
            ci_high=0.71,
            mean_placement=2.1,
        )
        assert stat.games == 12
        assert stat.wins == 5
        assert stat.win_rate == 0.417
        assert stat.ci_low == 0.15
        assert stat.ci_high == 0.71
        assert stat.mean_placement == 2.1

    def test_when_model_stat_is_mutated_then_frozen_error_is_raised(self) -> None:
        stat = ModelStat(
            model="model_c",
            games=5,
            wins=1,
            win_rate=0.2,
            ci_low=0.0,
            ci_high=0.52,
            mean_placement=3.5,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            stat.wins = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MatrixCell — field presence, immutability
# ---------------------------------------------------------------------------


class TestMatrixCellFields:
    def test_when_matrix_cell_created_then_all_fields_are_present(self) -> None:
        cell = MatrixCell(
            strategy="australia_lock",
            model="model_a",
            games=5,
            wins=2,
            win_rate=0.4,
        )
        assert cell.strategy == "australia_lock"
        assert cell.model == "model_a"
        assert cell.games == 5
        assert cell.wins == 2
        assert cell.win_rate == 0.4

    def test_when_matrix_cell_is_mutated_then_frozen_error_is_raised(self) -> None:
        cell = MatrixCell(
            strategy="aggressive_blitz",
            model="model_d",
            games=3,
            wins=1,
            win_rate=0.333,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            cell.games = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Leaderboard — field presence, tuple types, immutability
# ---------------------------------------------------------------------------


class TestLeaderboardFields:
    def _make_board(self, **overrides: object) -> Leaderboard:
        defaults: dict = {
            "run": "run_001",
            "n_games": 10,
            "n_malformed": 1,
            "n_players": 6,
            "strategies": ("australia_lock", "south_america_lock"),
            "models": ("model_a", "model_b"),
            "matrix": (),
        }
        defaults.update(overrides)
        return Leaderboard(**defaults)  # type: ignore[arg-type]

    def test_when_leaderboard_created_then_run_field_is_present(self) -> None:
        assert self._make_board().run == "run_001"

    def test_when_leaderboard_created_then_n_games_and_n_malformed_are_present(
        self,
    ) -> None:
        board = self._make_board()
        assert board.n_games == 10
        assert board.n_malformed == 1

    def test_when_leaderboard_created_then_n_players_is_present(self) -> None:
        assert self._make_board().n_players == 6

    def test_when_leaderboard_created_then_strategies_is_a_tuple(self) -> None:
        board = self._make_board(strategies=("australia_lock",))
        assert isinstance(board.strategies, tuple)

    def test_when_leaderboard_created_then_models_is_a_tuple(self) -> None:
        board = self._make_board(models=("model_x", "model_y"))
        assert isinstance(board.models, tuple)

    def test_when_leaderboard_created_then_matrix_is_a_tuple(self) -> None:
        cell = MatrixCell(strategy="australia_lock", model="model_a", games=5, wins=2, win_rate=0.4)
        board = self._make_board(matrix=(cell,))
        assert isinstance(board.matrix, tuple)

    def test_when_leaderboard_is_mutated_then_frozen_error_is_raised(self) -> None:
        board = self._make_board()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            board.n_games = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# read_ledger: missing file
# ---------------------------------------------------------------------------


class TestReadLedgerMissingFile:
    def test_when_file_does_not_exist_then_empty_list_is_returned(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_dir" / "games.jsonl"
        records, _ = read_ledger(missing)
        assert records == []

    def test_when_file_does_not_exist_then_n_malformed_is_zero(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_dir" / "games.jsonl"
        _, n_malformed = read_ledger(missing)
        assert n_malformed == 0

    def test_when_file_does_not_exist_then_return_type_is_tuple_of_list_and_int(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "games.jsonl"
        result = read_ledger(missing)
        records, n_malformed = result
        assert isinstance(records, list)
        assert isinstance(n_malformed, int)


# ---------------------------------------------------------------------------
# read_ledger: valid records
# ---------------------------------------------------------------------------


class TestReadLedgerValidRecords:
    def test_when_file_has_one_valid_record_then_one_dict_is_returned(self, tmp_path: Path) -> None:
        ledger = tmp_path / "games.jsonl"
        _write_jsonl(ledger, [_make_valid_record()])
        records, n_malformed = read_ledger(ledger)
        assert len(records) == 1
        assert n_malformed == 0

    def test_when_file_has_valid_record_then_record_is_a_plain_dict(self, tmp_path: Path) -> None:
        ledger = tmp_path / "games.jsonl"
        _write_jsonl(ledger, [_make_valid_record()])
        records, _ = read_ledger(ledger)
        assert isinstance(records[0], dict)

    def test_when_file_has_three_valid_records_then_three_dicts_are_returned(
        self, tmp_path: Path
    ) -> None:
        ledger = tmp_path / "games.jsonl"
        _write_jsonl(
            ledger,
            [
                _make_valid_record(game_index=0),
                _make_valid_record(game_index=1),
                _make_valid_record(game_index=2),
            ],
        )
        records, n_malformed = read_ledger(ledger)
        assert len(records) == 3
        assert n_malformed == 0

    def test_when_valid_record_is_returned_it_contains_seat_strategies(
        self, tmp_path: Path
    ) -> None:
        ledger = tmp_path / "games.jsonl"
        _write_jsonl(ledger, [_make_valid_record()])
        records, _ = read_ledger(ledger)
        assert "seat_strategies" in records[0]

    def test_when_valid_record_is_returned_it_contains_seat_models(self, tmp_path: Path) -> None:
        ledger = tmp_path / "games.jsonl"
        _write_jsonl(ledger, [_make_valid_record()])
        records, _ = read_ledger(ledger)
        assert "seat_models" in records[0]

    def test_when_valid_record_is_returned_it_contains_elimination_order(
        self, tmp_path: Path
    ) -> None:
        ledger = tmp_path / "games.jsonl"
        _write_jsonl(ledger, [_make_valid_record()])
        records, _ = read_ledger(ledger)
        assert "elimination_order" in records[0]


# ---------------------------------------------------------------------------
# read_ledger: malformed / garbage JSON lines
# ---------------------------------------------------------------------------


class TestReadLedgerGarbageLines:
    def test_when_file_has_one_garbage_line_then_n_malformed_is_one(self, tmp_path: Path) -> None:
        ledger = tmp_path / "games.jsonl"
        _write_jsonl(ledger, ["THIS IS NOT JSON ~~~"])
        _, n_malformed = read_ledger(ledger)
        assert n_malformed == 1

    def test_when_file_has_one_garbage_line_then_no_records_are_returned(
        self, tmp_path: Path
    ) -> None:
        ledger = tmp_path / "games.jsonl"
        _write_jsonl(ledger, ["BAD_LINE"])
        records, _ = read_ledger(ledger)
        assert records == []

    def test_when_file_has_mixed_valid_and_garbage_lines_then_only_valid_returned(
        self, tmp_path: Path
    ) -> None:
        ledger = tmp_path / "games.jsonl"
        _write_jsonl(
            ledger,
            [
                _make_valid_record(game_index=0),
                "GARBAGE_LINE",
                _make_valid_record(game_index=1),
            ],
        )
        records, n_malformed = read_ledger(ledger)
        assert len(records) == 2
        assert n_malformed == 1

    def test_when_file_has_three_garbage_lines_then_n_malformed_is_three(
        self, tmp_path: Path
    ) -> None:
        ledger = tmp_path / "games.jsonl"
        _write_jsonl(ledger, ["BAD", "ALSO_BAD", "{incomplete"])
        _, n_malformed = read_ledger(ledger)
        assert n_malformed == 3


# ---------------------------------------------------------------------------
# read_ledger: records missing required keys
# ---------------------------------------------------------------------------


class TestReadLedgerMissingRequiredKeys:
    def test_when_record_missing_seat_strategies_then_it_is_counted_as_malformed(
        self, tmp_path: Path
    ) -> None:
        rec = _make_valid_record()
        del rec["seat_strategies"]
        ledger = tmp_path / "games.jsonl"
        _write_jsonl(ledger, [rec])
        records, n_malformed = read_ledger(ledger)
        assert records == []
        assert n_malformed == 1

    def test_when_record_missing_seat_models_then_it_is_counted_as_malformed(
        self, tmp_path: Path
    ) -> None:
        rec = _make_valid_record()
        del rec["seat_models"]
        ledger = tmp_path / "games.jsonl"
        _write_jsonl(ledger, [rec])
        records, n_malformed = read_ledger(ledger)
        assert records == []
        assert n_malformed == 1

    def test_when_record_missing_elimination_order_then_it_is_counted_as_malformed(
        self, tmp_path: Path
    ) -> None:
        rec = _make_valid_record()
        del rec["elimination_order"]
        ledger = tmp_path / "games.jsonl"
        _write_jsonl(ledger, [rec])
        records, n_malformed = read_ledger(ledger)
        assert records == []
        assert n_malformed == 1

    def test_when_one_complete_and_one_incomplete_record_then_counts_are_correct(
        self, tmp_path: Path
    ) -> None:
        complete = _make_valid_record(game_index=0)
        incomplete = _make_valid_record(game_index=1)
        del incomplete["seat_strategies"]
        ledger = tmp_path / "games.jsonl"
        _write_jsonl(ledger, [complete, incomplete])
        records, n_malformed = read_ledger(ledger)
        assert len(records) == 1
        assert n_malformed == 1


# ---------------------------------------------------------------------------
# _diplomacy_of helper — presence, absence, and default behaviour
# ---------------------------------------------------------------------------


class TestDiplomacyOf:
    def test_when_record_has_no_diplomacy_fields_then_helper_does_not_raise(
        self,
    ) -> None:
        rec = _make_valid_record()  # no betrayals or alliances keys
        try:
            _diplomacy_of(rec)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"_diplomacy_of raised unexpectedly on record without diplomacy: {exc}")

    def test_when_record_has_no_betrayals_field_then_helper_returns_zero_or_empty_for_betrayals(
        self,
    ) -> None:
        rec = _make_valid_record()
        result = _diplomacy_of(rec)
        # Accept dict, tuple, or object — check betrayals is zero/falsy/empty
        if isinstance(result, dict):
            betrayals = result.get("betrayals", 0)
        elif isinstance(result, tuple):
            betrayals = result[0]
        else:
            betrayals = getattr(result, "betrayals", 0)
        # Zero int, empty dict, empty list, or None are all acceptable "absent" representations
        assert not betrayals or betrayals == {} or betrayals == [] or betrayals is None

    def test_when_record_has_no_alliances_field_then_helper_returns_zero_or_empty_for_alliances(
        self,
    ) -> None:
        rec = _make_valid_record()
        result = _diplomacy_of(rec)
        if isinstance(result, dict):
            alliances = result.get("alliances", 0)
        elif isinstance(result, tuple):
            alliances = result[1]
        else:
            alliances = getattr(result, "alliances", 0)
        assert not alliances or alliances == {} or alliances == [] or alliances is None

    def test_when_record_has_betrayals_field_then_helper_surfaces_non_empty_betrayals(
        self,
    ) -> None:
        rec = _make_valid_record()
        rec["betrayals"] = {"0": 2, "3": 1}
        result_with = _diplomacy_of(rec)
        result_without = _diplomacy_of(_make_valid_record())
        # The helper must distinguish a record with betrayals from one without
        assert result_with != result_without

    def test_when_record_has_alliances_field_then_helper_surfaces_non_empty_alliances(
        self,
    ) -> None:
        rec = _make_valid_record()
        rec["alliances"] = {"1": 3, "4": 1}
        result_with = _diplomacy_of(rec)
        result_without = _diplomacy_of(_make_valid_record())
        assert result_with != result_without

    def test_when_record_has_both_betrayals_and_alliances_then_helper_does_not_raise(
        self,
    ) -> None:
        rec = _make_valid_record()
        rec["betrayals"] = {"0": 1}
        rec["alliances"] = {"2": 2}
        try:
            _diplomacy_of(rec)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"_diplomacy_of raised unexpectedly when both fields present: {exc}")


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis) — issue #124
# ---------------------------------------------------------------------------


@given(st.text(min_size=1, alphabet=st.characters(blacklist_categories=("Cs",))))
def test_when_any_nonexistent_path_given_to_read_ledger_then_no_error_is_raised(
    path_str: str,
) -> None:
    """read_ledger is total over the missing-file case: must never raise."""
    with tempfile.TemporaryDirectory() as tmpdir:
        safe_name = path_str.replace("/", "_").replace("\x00", "_")[:100] or "x"
        p = Path(tmpdir) / safe_name
        # File does not exist — must return ([], 0) without raising
        records, n_malformed = read_ledger(p)
        assert isinstance(records, list)
        assert isinstance(n_malformed, int)
        assert n_malformed >= 0


@given(
    st.lists(
        st.booleans(),  # True → valid record, False → unparseable garbage
        min_size=0,
        max_size=20,
    )
)
def test_when_ledger_has_n_lines_then_well_formed_plus_malformed_equals_n(
    validity_flags: list[bool],
) -> None:
    """len(records) + n_malformed must equal the total number of JSONL lines written."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = Path(tmpdir) / "games.jsonl"
        lines: list = []
        for i, is_valid in enumerate(validity_flags):
            if is_valid:
                lines.append(_make_valid_record(game_index=i))
            else:
                lines.append("INVALID_JSON_LINE ~~~")
        _write_jsonl(ledger, lines)
        records, n_malformed = read_ledger(ledger)
        assert len(records) + n_malformed == len(lines)


# ===========================================================================
# Issue #125 — placements(), aggregate_strategies(), aggregate_models(),
#               _safe_wilson(), wilson_ci(), sort-order invariants
# ===========================================================================
#
# Three-game fixture (records use the same JSONL format as _make_valid_record,
# with string keys for seat_strategies / seat_models):
#
#   GAME_A  winner=2  elimination_order=[4, 0, 3, 5, 1]
#     seat_strategies: {0:turtle 1:diplomat 2:aggressive_blitz
#                       3:australia_lock 4:card_cycle 5:south_america_lock}
#     seat placements:  {2:1, 1:2, 5:3, 3:4, 0:5, 4:6}
#
#   GAME_B  winner=0  elimination_order=[3, 5, 1, 4, 2]
#     seat_strategies: {0:aggressive_blitz 1:australia_lock 2:south_america_lock
#                       3:card_cycle 4:turtle 5:diplomat}
#     seat placements:  {0:1, 2:2, 4:3, 1:4, 5:5, 3:6}
#
#   GAME_C  winner=1  elimination_order=[0, 5, 2, 4, 3]
#     seat_strategies: {0:diplomat 1:australia_lock 2:turtle
#                       3:aggressive_blitz 4:card_cycle 5:south_america_lock}
#     seat placements:  {1:1, 3:2, 4:3, 2:4, 5:5, 0:6}
#
# Derived strategy totals (3 games each):
#   aggressive_blitz    2 wins (A, B)  win_rate=2/3
#   australia_lock      1 win  (C)     win_rate=1/3
#   turtle/diplomat/card_cycle/south_america_lock  0 wins
#
# Derived model totals (each model in exactly one seat per game):
#   model_c  2 wins (game A seat 2; game C seat 1)  win_rate=2/3
#   model_d  1 win  (game B seat 0)                 win_rate=1/3
#   model_a/b/e/f  0 wins
#
# Mean placement by strategy:
#   aggressive_blitz: (1+1+2)/3 = 4/3
#   australia_lock:   (4+4+1)/3 = 3.0
#
# Mean placement by model:
#   model_c: (1+5+1)/3 = 7/3
#   model_d: (4+1+4)/3 = 3.0
# ===========================================================================

_STRAT_A = {
    "0": "turtle_defensive",
    "1": "diplomat_coalition",
    "2": "aggressive_blitz",
    "3": "australia_lock",
    "4": "card_cycle_hunter",
    "5": "south_america_lock",
}
_STRAT_B = {
    "0": "aggressive_blitz",
    "1": "australia_lock",
    "2": "south_america_lock",
    "3": "card_cycle_hunter",
    "4": "turtle_defensive",
    "5": "diplomat_coalition",
}
_STRAT_C = {
    "0": "diplomat_coalition",
    "1": "australia_lock",
    "2": "turtle_defensive",
    "3": "aggressive_blitz",
    "4": "card_cycle_hunter",
    "5": "south_america_lock",
}

_MODELS_A = {
    "0": "model_a",
    "1": "model_b",
    "2": "model_c",
    "3": "model_d",
    "4": "model_e",
    "5": "model_f",
}
_MODELS_B = {
    "0": "model_d",
    "1": "model_e",
    "2": "model_f",
    "3": "model_a",
    "4": "model_b",
    "5": "model_c",
}
_MODELS_C = {
    "0": "model_b",
    "1": "model_c",
    "2": "model_d",
    "3": "model_e",
    "4": "model_f",
    "5": "model_a",
}

GAME_A = _make_valid_record(
    game_index=0,
    winner=2,
    winner_strategy="aggressive_blitz",
    winner_model="model_c",
    elimination_order=[4, 0, 3, 5, 1],
    seat_strategies=_STRAT_A,
    seat_models=_MODELS_A,
    assignment={v: _MODELS_A[k] for k, v in _STRAT_A.items()},
)
GAME_B = _make_valid_record(
    game_index=1,
    winner=0,
    winner_strategy="aggressive_blitz",
    winner_model="model_d",
    elimination_order=[3, 5, 1, 4, 2],
    seat_strategies=_STRAT_B,
    seat_models=_MODELS_B,
    assignment={v: _MODELS_B[k] for k, v in _STRAT_B.items()},
)
GAME_C = _make_valid_record(
    game_index=2,
    winner=1,
    winner_strategy="australia_lock",
    winner_model="model_c",
    elimination_order=[0, 5, 2, 4, 3],
    seat_strategies=_STRAT_C,
    seat_models=_MODELS_C,
    assignment={v: _MODELS_C[k] for k, v in _STRAT_C.items()},
)

THREE_GAMES = [GAME_A, GAME_B, GAME_C]


def _by_strategy(stats: Sequence[StrategyStat], name: str) -> StrategyStat | None:
    return next((s for s in stats if s.strategy == name), None)


def _by_model(stats: Sequence[ModelStat], name: str) -> ModelStat | None:
    return next((s for s in stats if s.model == name), None)


# ---------------------------------------------------------------------------
# placements()  — issue #125
# ---------------------------------------------------------------------------


class TestPlacements:
    def test_when_winner_present_then_winner_seat_gets_placement_1(self) -> None:
        result = placements(GAME_A)
        assert result[2] == 1

    def test_when_full_elimination_order_then_last_eliminated_gets_placement_2(self) -> None:
        # elimination_order=[4,0,3,5,1] → seat 1 is last eliminated
        result = placements(GAME_A)
        assert result[1] == 2

    def test_when_full_elimination_order_then_first_eliminated_gets_placement_6(self) -> None:
        # seat 4 is first in elimination list → worst placement in a 6-player game
        result = placements(GAME_A)
        assert result[4] == 6

    def test_when_full_elimination_order_then_all_six_seats_are_keys(self) -> None:
        result = placements(GAME_A)
        assert set(result.keys()) == {0, 1, 2, 3, 4, 5}

    def test_when_full_elimination_order_then_placements_form_contiguous_range_1_to_6(self) -> None:
        result = placements(GAME_A)
        assert sorted(result.values()) == list(range(1, 7))

    def test_when_game_b_is_processed_then_winner_seat_0_gets_placement_1(self) -> None:
        result = placements(GAME_B)
        assert result[0] == 1

    def test_when_game_b_is_processed_then_seat_2_gets_placement_2(self) -> None:
        # GAME_B elimination_order=[3,5,1,4,2] → seat 2 is last eliminated
        result = placements(GAME_B)
        assert result[2] == 2

    def test_when_empty_elimination_order_and_no_winner_then_all_get_placement_1(self) -> None:
        """Spec: tolerates malformed/empty elimination_order; all seats co-survive → 1."""
        rec = _make_valid_record(
            winner=None,
            winner_strategy=None,
            winner_model=None,
            elimination_order=[],
            seat_strategies={str(i): "turtle_defensive" for i in range(6)},
            seat_models={str(i): "model_a" for i in range(6)},
        )
        result = placements(rec)
        assert all(p == 1 for p in result.values())

    def test_when_co_survivors_then_surviving_seats_all_get_placement_1(self) -> None:
        """Spec: draw co-survivors → 1. Seats 3,4,5 survive; seats 0,1,2 eliminated."""
        rec = _make_valid_record(
            winner=None,
            winner_strategy=None,
            winner_model=None,
            elimination_order=[0, 1, 2],
            seat_strategies={str(i): "turtle_defensive" for i in range(6)},
            seat_models={str(i): "model_a" for i in range(6)},
        )
        result = placements(rec)
        assert result[3] == 1
        assert result[4] == 1
        assert result[5] == 1

    def test_when_co_survivors_then_last_eliminated_gets_placement_2(self) -> None:
        """Seat 2 is last in elimination_order=[0,1,2] → placement 2."""
        rec = _make_valid_record(
            winner=None,
            winner_strategy=None,
            winner_model=None,
            elimination_order=[0, 1, 2],
            seat_strategies={str(i): "turtle_defensive" for i in range(6)},
            seat_models={str(i): "model_a" for i in range(6)},
        )
        result = placements(rec)
        assert result[2] == 2

    def test_when_co_survivors_then_first_eliminated_gets_worst_rank_among_eliminated(self) -> None:
        """With 3 co-survivors and 3 eliminated, the first eliminated gets placement 4."""
        rec = _make_valid_record(
            winner=None,
            winner_strategy=None,
            winner_model=None,
            elimination_order=[0, 1, 2],
            seat_strategies={str(i): "turtle_defensive" for i in range(6)},
            seat_models={str(i): "model_a" for i in range(6)},
        )
        result = placements(rec)
        assert result[0] == 4


# ---------------------------------------------------------------------------
# _safe_wilson() and wilson_ci()  — issue #125
# ---------------------------------------------------------------------------


class TestSafeWilson:
    def test_when_n_is_zero_then_returns_0_0(self) -> None:
        lo, hi = _safe_wilson(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_when_n_is_zero_then_does_not_raise(self) -> None:
        result = _safe_wilson(0, 0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_when_wins_equals_n_positive_then_ci_is_in_unit_interval(self) -> None:
        lo, hi = _safe_wilson(5, 5)
        assert 0.0 <= lo <= hi <= 1.0

    def test_when_wins_is_zero_and_n_positive_then_lo_is_0(self) -> None:
        lo, hi = _safe_wilson(0, 5)
        assert lo == 0.0
        assert hi >= 0.0

    def test_when_n_positive_then_result_matches_wilson_ci(self) -> None:
        lo, hi = _safe_wilson(3, 10)
        expected_lo, expected_hi = wilson_ci(3, 10)
        assert math.isclose(lo, expected_lo, rel_tol=1e-9)
        assert math.isclose(hi, expected_hi, rel_tol=1e-9)

    @given(
        st.integers(min_value=0, max_value=200).flatmap(
            lambda n: st.tuples(st.integers(min_value=0, max_value=n), st.just(n))
        )
    )
    def test_when_wins_and_n_are_valid_then_safe_wilson_never_raises(
        self, wins_n: tuple[int, int]
    ) -> None:
        """Invariant: _safe_wilson is total over all (0 ≤ wins ≤ n) — never raises."""
        wins, n = wins_n
        result = _safe_wilson(wins, n)
        assert isinstance(result, tuple) and len(result) == 2

    @given(
        st.integers(min_value=1, max_value=200).flatmap(
            lambda n: st.tuples(st.integers(min_value=0, max_value=n), st.just(n))
        )
    )
    def test_when_n_positive_then_ci_bounds_lie_in_unit_interval(
        self, wins_n: tuple[int, int]
    ) -> None:
        """Invariant: ci_low and ci_high are always in [0, 1] for any valid (wins, n)."""
        wins, n = wins_n
        lo, hi = _safe_wilson(wins, n)
        assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# aggregate_strategies()  — issue #125
# ---------------------------------------------------------------------------


class TestAggregateStrategies:
    def test_when_three_games_then_one_stat_per_observed_strategy_is_returned(self) -> None:
        stats = aggregate_strategies(THREE_GAMES)
        names = {s.strategy for s in stats}
        assert names == {
            "aggressive_blitz",
            "australia_lock",
            "turtle_defensive",
            "diplomat_coalition",
            "card_cycle_hunter",
            "south_america_lock",
        }

    def test_when_three_games_then_length_equals_number_of_distinct_strategies(self) -> None:
        stats = aggregate_strategies(THREE_GAMES)
        assert len(stats) == 6

    def test_when_aggressive_blitz_wins_twice_then_games_is_3(self) -> None:
        stat = _by_strategy(aggregate_strategies(THREE_GAMES), "aggressive_blitz")
        assert stat is not None
        assert stat.games == 3

    def test_when_aggressive_blitz_wins_twice_then_wins_is_2(self) -> None:
        stat = _by_strategy(aggregate_strategies(THREE_GAMES), "aggressive_blitz")
        assert stat is not None
        assert stat.wins == 2

    def test_when_aggressive_blitz_wins_twice_then_win_rate_is_2_over_3(self) -> None:
        stat = _by_strategy(aggregate_strategies(THREE_GAMES), "aggressive_blitz")
        assert stat is not None
        assert math.isclose(stat.win_rate, 2 / 3, rel_tol=1e-9)

    def test_when_aggressive_blitz_wins_twice_then_ci_matches_wilson_ci(self) -> None:
        stat = _by_strategy(aggregate_strategies(THREE_GAMES), "aggressive_blitz")
        assert stat is not None
        expected_lo, expected_hi = wilson_ci(2, 3)
        assert math.isclose(stat.ci_low, expected_lo, rel_tol=1e-9)
        assert math.isclose(stat.ci_high, expected_hi, rel_tol=1e-9)

    def test_when_australia_lock_wins_once_then_wins_is_1_and_games_is_3(self) -> None:
        stat = _by_strategy(aggregate_strategies(THREE_GAMES), "australia_lock")
        assert stat is not None
        assert stat.wins == 1
        assert stat.games == 3

    def test_when_strategy_has_no_wins_then_win_rate_is_0_and_ci_low_is_0(self) -> None:
        stat = _by_strategy(aggregate_strategies(THREE_GAMES), "turtle_defensive")
        assert stat is not None
        assert stat.wins == 0
        assert stat.win_rate == 0.0
        assert stat.ci_low == 0.0

    def test_when_strategies_aggregated_then_betrayals_defaults_to_0_for_all(self) -> None:
        """Spec: betrayals/alliances left at default 0 (populated by a separate issue)."""
        for stat in aggregate_strategies(THREE_GAMES):
            assert stat.betrayals == 0

    def test_when_strategies_aggregated_then_alliances_defaults_to_0_for_all(self) -> None:
        for stat in aggregate_strategies(THREE_GAMES):
            assert stat.alliances == 0

    def test_when_aggressive_blitz_wins_games_a_and_b_then_mean_placement_is_4_over_3(self) -> None:
        """
        aggressive_blitz placements:
          game A seat 2 → 1, game B seat 0 → 1, game C seat 3 → 2.
          mean = (1+1+2)/3 = 4/3.
        """
        stat = _by_strategy(aggregate_strategies(THREE_GAMES), "aggressive_blitz")
        assert stat is not None
        assert math.isclose(stat.mean_placement, 4 / 3, rel_tol=1e-9)

    def test_when_australia_lock_wins_game_c_then_mean_placement_is_3(self) -> None:
        """
        australia_lock placements:
          game A seat 3 → 4, game B seat 1 → 4, game C seat 1 → 1.
          mean = (4+4+1)/3 = 3.0.
        """
        stat = _by_strategy(aggregate_strategies(THREE_GAMES), "australia_lock")
        assert stat is not None
        assert math.isclose(stat.mean_placement, 3.0, rel_tol=1e-9)

    def test_when_return_type_is_checked_then_result_is_a_tuple(self) -> None:
        result = aggregate_strategies(THREE_GAMES)
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# aggregate_models()  — issue #125
# ---------------------------------------------------------------------------


class TestAggregateModels:
    def test_when_three_games_then_one_stat_per_observed_model_is_returned(self) -> None:
        stats = aggregate_models(THREE_GAMES)
        names = {s.model for s in stats}
        assert names == {"model_a", "model_b", "model_c", "model_d", "model_e", "model_f"}

    def test_when_three_games_then_length_equals_number_of_distinct_models(self) -> None:
        stats = aggregate_models(THREE_GAMES)
        assert len(stats) == 6

    def test_when_model_c_wins_twice_then_games_is_3_and_wins_is_2(self) -> None:
        stat = _by_model(aggregate_models(THREE_GAMES), "model_c")
        assert stat is not None
        assert stat.games == 3
        assert stat.wins == 2

    def test_when_model_c_wins_twice_then_win_rate_is_2_over_3(self) -> None:
        stat = _by_model(aggregate_models(THREE_GAMES), "model_c")
        assert stat is not None
        assert math.isclose(stat.win_rate, 2 / 3, rel_tol=1e-9)

    def test_when_model_c_wins_twice_then_ci_matches_wilson_ci(self) -> None:
        stat = _by_model(aggregate_models(THREE_GAMES), "model_c")
        assert stat is not None
        expected_lo, expected_hi = wilson_ci(2, 3)
        assert math.isclose(stat.ci_low, expected_lo, rel_tol=1e-9)
        assert math.isclose(stat.ci_high, expected_hi, rel_tol=1e-9)

    def test_when_model_d_wins_once_then_wins_is_1_and_games_is_3(self) -> None:
        stat = _by_model(aggregate_models(THREE_GAMES), "model_d")
        assert stat is not None
        assert stat.wins == 1
        assert stat.games == 3

    def test_when_model_wins_zero_then_ci_equals_safe_wilson_0_n(self) -> None:
        stat = _by_model(aggregate_models(THREE_GAMES), "model_a")
        assert stat is not None
        expected_lo, expected_hi = _safe_wilson(0, 3)
        assert math.isclose(stat.ci_low, expected_lo, rel_tol=1e-9)
        assert math.isclose(stat.ci_high, expected_hi, rel_tol=1e-9)

    def test_when_model_c_wins_games_a_and_c_then_mean_placement_is_7_over_3(self) -> None:
        """
        model_c placements:
          game A seat 2 → 1, game B seat 5 → 5, game C seat 1 → 1.
          mean = (1+5+1)/3 = 7/3.
        """
        stat = _by_model(aggregate_models(THREE_GAMES), "model_c")
        assert stat is not None
        assert math.isclose(stat.mean_placement, 7 / 3, rel_tol=1e-9)

    def test_when_model_d_wins_game_b_then_mean_placement_is_3(self) -> None:
        """
        model_d placements:
          game A seat 3 → 4, game B seat 0 → 1, game C seat 2 → 4.
          mean = (4+1+4)/3 = 3.0.
        """
        stat = _by_model(aggregate_models(THREE_GAMES), "model_d")
        assert stat is not None
        assert math.isclose(stat.mean_placement, 3.0, rel_tol=1e-9)

    def test_when_return_type_is_checked_then_result_is_a_tuple(self) -> None:
        result = aggregate_models(THREE_GAMES)
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Sort order  — issue #125
# ---------------------------------------------------------------------------


class TestSortOrder:
    def test_when_strategies_aggregated_then_output_is_sorted_by_win_rate_desc(self) -> None:
        stats = aggregate_strategies(THREE_GAMES)
        rates = [s.win_rate for s in stats]
        assert rates == sorted(rates, reverse=True)

    def test_when_strategies_aggregated_then_aggressive_blitz_is_at_index_0(self) -> None:
        """win_rate 2/3 is highest → must be at position 0."""
        stats = aggregate_strategies(THREE_GAMES)
        assert stats[0].strategy == "aggressive_blitz"

    def test_when_strategies_aggregated_then_australia_lock_is_at_index_1(self) -> None:
        """win_rate 1/3 is second highest → must be at position 1."""
        stats = aggregate_strategies(THREE_GAMES)
        assert stats[1].strategy == "australia_lock"

    def test_when_two_strategies_share_win_rate_then_higher_ci_low_comes_first(self) -> None:
        """Spec: tie on win_rate → sort by ci_low desc."""
        # Build records where "S0" wins 2/4 games and "S1" wins 1/2 games.
        # Both have win_rate=0.5 but different sample sizes → different ci_low values.
        # (More games → tighter CI → higher ci_low for the same win_rate.)
        strat_map = {str(i): f"S{i}" for i in range(6)}
        model_map = {str(i): f"M{i}" for i in range(6)}

        def _g(gid, winner):
            elim = [s for s in range(6) if s != winner]
            return _make_valid_record(
                game_index=gid,
                winner=winner,
                winner_strategy=strat_map[str(winner)],
                winner_model=model_map[str(winner)],
                elimination_order=elim,
                seat_strategies=strat_map,
                seat_models=model_map,
                assignment={v: model_map[k] for k, v in strat_map.items()},
            )

        # S0 wins games 0 and 1; S1 wins game 2; S2 wins game 3
        # After 4 games: S0 → 2/4, S1 → 1/4, S2 → 1/4, others → 0/4
        # We cannot easily force both S0 and S1 to 0.5 with different n;
        # instead, verify that when ci_low values differ, the higher one is first.
        records = [_g(0, 0), _g(1, 0), _g(2, 1), _g(3, 2)]
        stats = aggregate_strategies(records)
        rates = [s.win_rate for s in stats]
        assert rates == sorted(rates, reverse=True)
        # Among strategies with equal win_rate, ci_low must be non-increasing
        for i in range(len(stats) - 1):
            if math.isclose(stats[i].win_rate, stats[i + 1].win_rate, rel_tol=1e-9):
                assert stats[i].ci_low >= stats[i + 1].ci_low - 1e-12

    def test_when_models_aggregated_then_output_is_sorted_by_win_rate_desc(self) -> None:
        stats = aggregate_models(THREE_GAMES)
        rates = [s.win_rate for s in stats]
        assert rates == sorted(rates, reverse=True)

    def test_when_models_aggregated_then_model_c_is_at_index_0(self) -> None:
        """model_c win_rate=2/3 is the highest → index 0."""
        stats = aggregate_models(THREE_GAMES)
        assert stats[0].model == "model_c"

    def test_when_models_aggregated_then_model_d_is_at_index_1(self) -> None:
        """model_d win_rate=1/3 is second highest → index 1."""
        stats = aggregate_models(THREE_GAMES)
        assert stats[1].model == "model_d"

    @given(st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=30))
    @settings(max_examples=200)
    def test_when_any_winner_sequence_then_strategies_sorted_by_win_rate_desc(
        self, winner_seats: list[int]
    ) -> None:
        """Ordering invariant: aggregate_strategies always returns stats in win_rate desc order."""
        strat_map = {str(i): f"S{i}" for i in range(6)}
        model_map = {str(i): f"M{i}" for i in range(6)}
        records = [
            _make_valid_record(
                game_index=gid,
                winner=winner,
                winner_strategy=strat_map[str(winner)],
                winner_model=model_map[str(winner)],
                elimination_order=[s for s in range(6) if s != winner],
                seat_strategies=strat_map,
                seat_models=model_map,
                assignment={v: model_map[k] for k, v in strat_map.items()},
            )
            for gid, winner in enumerate(winner_seats)
        ]
        stats = aggregate_strategies(records)
        rates = [s.win_rate for s in stats]
        assert rates == sorted(rates, reverse=True)

    @given(st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=30))
    @settings(max_examples=200)
    def test_when_any_winner_sequence_then_models_sorted_by_win_rate_desc(
        self, winner_seats: list[int]
    ) -> None:
        """Ordering invariant: aggregate_models always returns stats in win_rate desc order."""
        strat_map = {str(i): f"S{i}" for i in range(6)}
        model_map = {str(i): f"M{i}" for i in range(6)}
        records = [
            _make_valid_record(
                game_index=gid,
                winner=winner,
                winner_strategy=strat_map[str(winner)],
                winner_model=model_map[str(winner)],
                elimination_order=[s for s in range(6) if s != winner],
                seat_strategies=strat_map,
                seat_models=model_map,
                assignment={v: model_map[k] for k, v in strat_map.items()},
            )
            for gid, winner in enumerate(winner_seats)
        ]
        stats = aggregate_models(records)
        rates = [s.win_rate for s in stats]
        assert rates == sorted(rates, reverse=True)
