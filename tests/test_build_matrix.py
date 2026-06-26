"""
Example tests for build_matrix() and tournament resumability.
Issue #126 — feat: strategy×model win matrix.

Derived from acceptance criteria only; no implementation source was read.
All fixtures are inline dicts (JSONL record shape); no live LLM calls.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from training.tournament import pending_games
from training.tournament_analysis import build_matrix

# ── record fixture helpers ───────────────────────────────────────────────────


def _rec(
    seat_strategies: list[str],
    seat_models: list[str],
    winner_strategy: str | None,
    winner_model: str | None,
) -> dict:
    """Build a game record using seat-list fields (assignment absent)."""
    return {
        "seat_strategies": seat_strategies,
        "seat_models": seat_models,
        "winner_strategy": winner_strategy,
        "winner_model": winner_model,
    }


def _rec_with_assignment(
    seat_strategies: list[str],
    seat_models: list[str],
    winner_strategy: str | None,
    winner_model: str | None,
) -> dict:
    """Build a game record with an explicit assignment list."""
    return {
        "assignment": [
            {"strategy": s, "model": m} for s, m in zip(seat_strategies, seat_models, strict=False)
        ],
        "winner_strategy": winner_strategy,
        "winner_model": winner_model,
    }


# ── return-type contracts ────────────────────────────────────────────────────


def test_when_records_given_then_build_matrix_returns_tuple():
    records = [_rec(["Australia Lock"], ["glm-5.2:cloud"], "Australia Lock", "glm-5.2:cloud")]
    assert isinstance(build_matrix(records), tuple)


def test_when_no_records_given_then_build_matrix_returns_empty_collection():
    result = build_matrix([])
    assert len(result) == 0


# ── one cell per observed (strategy, model) pairing ─────────────────────────


def test_when_one_game_one_seat_then_exactly_one_cell_is_returned():
    records = [_rec(["Australia Lock"], ["glm-5.2:cloud"], "Australia Lock", "glm-5.2:cloud")]
    result = build_matrix(records)
    assert len(result) == 1
    assert result[0].strategy == "Australia Lock"
    assert result[0].model == "glm-5.2:cloud"


def test_when_two_seats_per_game_then_two_cells_are_returned():
    records = [
        _rec(
            ["Australia Lock", "Turtle"],
            ["glm-5.2:cloud", "minimax-m3:cloud"],
            "Australia Lock",
            "glm-5.2:cloud",
        )
    ]
    assert len(build_matrix(records)) == 2


def test_when_same_pairing_in_two_games_then_one_aggregated_cell_is_returned():
    records = [
        _rec(["Aggressive Blitz"], ["minimax-m3:cloud"], "Aggressive Blitz", "minimax-m3:cloud"),
        _rec(["Aggressive Blitz"], ["minimax-m3:cloud"], "Aggressive Blitz", "minimax-m3:cloud"),
    ]
    result = build_matrix(records)
    assert len(result) == 1


def test_when_same_pairing_in_two_games_then_games_count_is_two():
    records = [
        _rec(["Aggressive Blitz"], ["minimax-m3:cloud"], "Aggressive Blitz", "minimax-m3:cloud"),
        _rec(["Aggressive Blitz"], ["minimax-m3:cloud"], None, None),
    ]
    assert build_matrix(records)[0].games == 2


# ── games / wins / win_rate correctness ─────────────────────────────────────


def test_when_one_win_out_of_three_games_then_win_rate_is_one_third():
    records = [
        _rec(["Diplomat"], ["qwen3.5:cloud"], "Diplomat", "qwen3.5:cloud"),
        _rec(["Diplomat"], ["qwen3.5:cloud"], None, None),
        _rec(["Diplomat"], ["qwen3.5:cloud"], None, None),
    ]
    cell = build_matrix(records)[0]
    assert cell.games == 3
    assert cell.wins == 1
    assert abs(cell.win_rate - 1 / 3) < 1e-9


def test_when_zero_wins_then_win_rate_is_zero():
    records = [
        _rec(["Turtle"], ["glm-5.1:cloud"], None, None),
        _rec(["Turtle"], ["glm-5.1:cloud"], None, None),
    ]
    cell = build_matrix(records)[0]
    assert cell.wins == 0
    assert cell.win_rate == 0.0


# ── pairing from assignment field ────────────────────────────────────────────


def test_when_assignment_field_present_then_pairings_come_from_it():
    record = _rec_with_assignment(
        ["Card-Cycle", "South America Lock"],
        ["kimi-k2.6:cloud", "deepseek-v4-flash:cloud"],
        "Card-Cycle",
        "kimi-k2.6:cloud",
    )
    pairs = {(c.strategy, c.model) for c in build_matrix([record])}
    assert ("Card-Cycle", "kimi-k2.6:cloud") in pairs
    assert ("South America Lock", "deepseek-v4-flash:cloud") in pairs


def test_when_assignment_field_present_then_cross_pairings_are_not_emitted():
    record = _rec_with_assignment(
        ["Card-Cycle", "South America Lock"],
        ["kimi-k2.6:cloud", "deepseek-v4-flash:cloud"],
        "Card-Cycle",
        "kimi-k2.6:cloud",
    )
    pairs = {(c.strategy, c.model) for c in build_matrix([record])}
    assert ("Card-Cycle", "deepseek-v4-flash:cloud") not in pairs
    assert ("South America Lock", "kimi-k2.6:cloud") not in pairs


# ── fallback: reconstruct pairing from seat_strategies / seat_models ─────────


def test_when_assignment_missing_then_seat_position_determines_pairing():
    records = [
        _rec(
            ["Turtle", "Aggressive Blitz"],
            ["glm-5.1:cloud", "glm-5.2:cloud"],
            "Turtle",
            "glm-5.1:cloud",
        )
    ]
    pairs = {(c.strategy, c.model) for c in build_matrix(records)}
    assert ("Turtle", "glm-5.1:cloud") in pairs  # seat 0 → seat 0
    assert ("Aggressive Blitz", "glm-5.2:cloud") in pairs  # seat 1 → seat 1


def test_when_assignment_missing_then_cross_seat_pairings_are_not_emitted():
    records = [
        _rec(
            ["Turtle", "Aggressive Blitz"],
            ["glm-5.1:cloud", "glm-5.2:cloud"],
            "Turtle",
            "glm-5.1:cloud",
        )
    ]
    pairs = {(c.strategy, c.model) for c in build_matrix(records)}
    assert ("Turtle", "glm-5.2:cloud") not in pairs
    assert ("Aggressive Blitz", "glm-5.1:cloud") not in pairs


# ── only cells with games > 0 are emitted ───────────────────────────────────


def test_when_cells_emitted_then_every_cell_has_games_greater_than_zero():
    records = [
        _rec(
            ["Diplomat", "Turtle"],
            ["qwen3.5:cloud", "kimi-k2.6:cloud"],
            "Diplomat",
            "qwen3.5:cloud",
        )
    ]
    for cell in build_matrix(records):
        assert cell.games > 0


# ── sort order: strategy then model ──────────────────────────────────────────


def test_when_multiple_pairings_then_cells_sorted_by_strategy_then_model():
    records = [
        _rec(
            ["Turtle", "Australia Lock", "Diplomat"],
            ["glm-5.2:cloud", "minimax-m3:cloud", "kimi-k2.6:cloud"],
            "Turtle",
            "glm-5.2:cloud",
        )
    ]
    result = build_matrix(records)
    keys = [(c.strategy, c.model) for c in result]
    assert keys == sorted(keys)


def test_when_same_strategy_appears_with_two_models_then_sorted_by_model_name():
    records = [
        _rec(
            ["Aggressive Blitz", "Aggressive Blitz"],
            ["qwen3.5:cloud", "glm-5.2:cloud"],
            "Aggressive Blitz",
            "qwen3.5:cloud",
        )
    ]
    same_strategy = [c for c in build_matrix(records) if c.strategy == "Aggressive Blitz"]
    models = [c.model for c in same_strategy]
    assert models == sorted(models)


# ── winner attribution ───────────────────────────────────────────────────────


def test_when_winner_is_named_then_winning_cell_gets_one_win():
    records = [
        _rec(
            ["Australia Lock", "Aggressive Blitz"],
            ["glm-5.2:cloud", "minimax-m3:cloud"],
            "Australia Lock",
            "glm-5.2:cloud",
        )
    ]
    result = build_matrix(records)
    winner = next(
        c for c in result if c.strategy == "Australia Lock" and c.model == "glm-5.2:cloud"
    )
    assert winner.wins == 1


def test_when_winner_is_named_then_losing_cell_gets_zero_wins():
    records = [
        _rec(
            ["Australia Lock", "Aggressive Blitz"],
            ["glm-5.2:cloud", "minimax-m3:cloud"],
            "Australia Lock",
            "glm-5.2:cloud",
        )
    ]
    result = build_matrix(records)
    loser = next(
        c for c in result if c.strategy == "Aggressive Blitz" and c.model == "minimax-m3:cloud"
    )
    assert loser.wins == 0


def test_when_winner_is_none_then_every_cell_gets_a_game_but_no_win():
    records = [
        _rec(["Australia Lock", "Turtle"], ["glm-5.2:cloud", "minimax-m3:cloud"], None, None)
    ]
    for cell in build_matrix(records):
        assert cell.games == 1
        assert cell.wins == 0


def test_when_multiple_draws_then_wins_stay_zero_and_games_accumulate():
    records = [
        _rec(["Diplomat"], ["qwen3.5:cloud"], None, None),
        _rec(["Diplomat"], ["qwen3.5:cloud"], None, None),
        _rec(["Diplomat"], ["qwen3.5:cloud"], None, None),
    ]
    cell = build_matrix(records)[0]
    assert cell.games == 3
    assert cell.wins == 0


# ── matrix covers all observed pairings ─────────────────────────────────────


def test_when_two_games_with_disjoint_pairings_then_all_four_pairings_appear():
    records = [
        _rec(
            ["Australia Lock", "Diplomat"],
            ["glm-5.2:cloud", "kimi-k2.6:cloud"],
            "Australia Lock",
            "glm-5.2:cloud",
        ),
        _rec(
            ["Turtle", "Card-Cycle"],
            ["minimax-m3:cloud", "deepseek-v4-flash:cloud"],
            "Turtle",
            "minimax-m3:cloud",
        ),
    ]
    result = build_matrix(records)
    pairs = {(c.strategy, c.model) for c in result}
    assert ("Australia Lock", "glm-5.2:cloud") in pairs
    assert ("Diplomat", "kimi-k2.6:cloud") in pairs
    assert ("Turtle", "minimax-m3:cloud") in pairs
    assert ("Card-Cycle", "deepseek-v4-flash:cloud") in pairs
    assert len(result) == 4


# ── resumability: pending_games ──────────────────────────────────────────────


def test_when_some_games_completed_then_pending_games_excludes_them():
    result = set(pending_games(total=5, completed_ids={0, 1, 2}))
    assert result == {3, 4}


def test_when_no_games_completed_then_all_indices_are_pending():
    result = set(pending_games(total=3, completed_ids=set()))
    assert result == {0, 1, 2}


def test_when_all_games_completed_then_no_pending_games_returned():
    result = list(pending_games(total=4, completed_ids={0, 1, 2, 3}))
    assert result == []


def test_when_total_is_zero_then_no_pending_games_returned():
    result = list(pending_games(total=0, completed_ids=set()))
    assert result == []


def test_when_completed_ids_exceed_total_then_no_pending_games_returned():
    # completed set contains ids beyond the target; should still be safe
    result = list(pending_games(total=3, completed_ids={0, 1, 2, 5, 99}))
    assert set(result).issubset({0, 1, 2})
    assert 0 not in result and 1 not in result and 2 not in result


# ── RL train path unchanged ──────────────────────────────────────────────────


def test_when_rl_modules_imported_then_no_import_error_is_raised():
    """Tournament addition must not break existing RL training imports."""
    from src.models.actor_critic import ActorCritic  # noqa: F401
    from src.models.ppo import PPOTrainer  # noqa: F401
    from training.self_play import SelfPlayTrainer  # noqa: F401


# ── property-based tests ─────────────────────────────────────────────────────

_STRATEGIES = [
    "Australia Lock",
    "South America Lock",
    "Aggressive Blitz",
    "Turtle",
    "Card-Cycle",
    "Diplomat",
]
_MODELS = [
    "glm-5.2:cloud",
    "minimax-m3:cloud",
    "glm-5.1:cloud",
    "kimi-k2.6:cloud",
    "deepseek-v4-flash:cloud",
    "qwen3.5:cloud",
]

_strategy_st = st.sampled_from(_STRATEGIES)
_model_st = st.sampled_from(_MODELS)

_record_st = st.fixed_dictionaries(
    {
        "seat_strategies": st.lists(_strategy_st, min_size=1, max_size=6),
        "seat_models": st.lists(_model_st, min_size=1, max_size=6),
        "winner_strategy": st.one_of(st.none(), _strategy_st),
        "winner_model": st.one_of(st.none(), _model_st),
    }
)


def _same_length(records: list[dict]) -> list[dict]:
    return [r for r in records if len(r["seat_strategies"]) == len(r["seat_models"])]


@given(records=st.lists(_record_st, min_size=0, max_size=20))
def test_when_any_valid_records_given_then_cells_are_sorted_by_strategy_then_model(records):
    result = build_matrix(_same_length(records))
    keys = [(c.strategy, c.model) for c in result]
    assert keys == sorted(keys)


@given(records=st.lists(_record_st, min_size=1, max_size=20))
def test_when_any_valid_records_given_then_every_emitted_cell_has_positive_games(records):
    for cell in build_matrix(_same_length(records)):
        assert cell.games > 0


@given(records=st.lists(_record_st, min_size=1, max_size=20))
def test_when_any_valid_records_given_then_win_rate_equals_wins_divided_by_games(records):
    for cell in build_matrix(_same_length(records)):
        assert cell.games > 0
        assert abs(cell.win_rate - cell.wins / cell.games) < 1e-9


@given(
    total=st.integers(min_value=0, max_value=50),
    completed=st.lists(st.integers(min_value=0, max_value=49), max_size=50).map(set),
)
def test_when_pending_games_called_then_no_completed_id_appears_in_result(total, completed):
    result_set = set(pending_games(total=total, completed_ids=completed))
    assert result_set.isdisjoint(completed)


@given(
    total=st.integers(min_value=0, max_value=50),
    completed=st.lists(st.integers(min_value=0, max_value=49), max_size=50).map(set),
)
def test_when_pending_games_called_then_all_returned_ids_are_valid_game_indices(total, completed):
    for gid in pending_games(total=total, completed_ids=completed):
        assert 0 <= gid < total


@given(
    total=st.integers(min_value=0, max_value=50),
    completed=st.lists(st.integers(min_value=0, max_value=49), max_size=50).map(set),
)
def test_when_pending_games_called_then_pending_plus_relevant_completed_equals_total(
    total, completed
):
    # IDs in [0, total) that were already completed should not reappear
    in_range_completed = {gid for gid in completed if gid < total}
    pending = set(pending_games(total=total, completed_ids=completed))
    assert pending | in_range_completed == set(range(total))
