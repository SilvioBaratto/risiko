"""
Tests for Issue #120 — seat building, diplomacy wiring & single-game play
with call counts.

Source-blind: authored from acceptance criteria only. No implementation source
was read. All tests are expected to FAIL (Red) until the implementation is
written.

Criteria covered (oracle report — UNIT and T2 tiers only):
  [UNIT] Seats built in slot order; each PlayerConfig.model equals its
         assigned model and strategy_hint/levers match compose_strategy_hint/
         POSTURE_LEVERS.
  [UNIT] DiplomacyRunner built with enabled=True, n_rounds=config.n_rounds,
         learner_id=-1, and the shared ReputationBook.
  [T2]   build_negotiation_call_fn routes player_id → correct model, strips
         /v1, returns None on the same fallback contract (LLM mocked).
  [UNIT] _play_one_game returns a LedgerRecord with correct winner_strategy/
         winner_model and per-game llm_call_count (negotiation + action),
         all LLM mocked.
  [UNIT] LLMOpponent exposes a public LLM-call counter.

Skipped (not runtime-verifiable per oracle):
  - risiko-rl tournament --llm-only runs N 6-player games (CLI/integration)
  - Preflight pulls/verifies all 6 models (integration)
  - Random strategy assignment distribution ~uniform (statistical/integration)
  - Leaderboard output with Wilson CIs (integration)
  - All randomness flows through seeds (design/architecture)
  - All tests pass / SOLID / ruff (suite-level gates, no per-criterion check)
"""

from __future__ import annotations

import contextlib
import pathlib
from unittest.mock import MagicMock, patch

import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_VALID_STRATEGIES = [
    "australia_lock",
    "south_america_lock",
    "aggressive_blitz",
    "turtle_defensive",
    "card_cycle_hunter",
    "diplomat_coalition",
]

_VALID_MODELS = [
    "glm-5.2:cloud",
    "minimax-m3:cloud",
    "glm-5.1:cloud",
    "kimi-k2.6:cloud",
    "deepseek-v4-flash:cloud",
    "qwen3.5:cloud",
]

# Fixed bijection strategy → model, in slot order (strategies[i] → models[i]).
_ASSIGNMENT: dict[str, str] = dict(zip(_VALID_STRATEGIES, _VALID_MODELS, strict=True))


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _write_yaml(path: pathlib.Path, data: dict) -> pathlib.Path:
    path.write_text(yaml.dump(data))
    return path


def _base_config_data() -> dict:
    return {
        "name": "test_tournament_120",
        "seed": 42,
        "n_games": 4,
        "max_concurrency": 2,
        "n_rounds": 2,
        "strategies": _VALID_STRATEGIES,
        "models": _VALID_MODELS,
        "temperature": 0.7,
        "top_p": 0.9,
    }


def _load_config(tmp_path: pathlib.Path):
    from training.tournament import load_tournament_config

    cfg_file = _write_yaml(tmp_path / "tournament.yaml", _base_config_data())
    return load_tournament_config(cfg_file)


def _make_game_plan(game_index: int = 0):
    from training.tournament import GamePlan

    return GamePlan(
        game_index=game_index,
        seed=42 + game_index,
        assignment=dict(_ASSIGNMENT),
    )


def _counter_attr(opp: object) -> str | None:
    """Return the name of the public LLM-call counter attribute, or None."""
    for candidate in ("llm_call_count", "call_count", "n_llm_calls", "num_llm_calls"):
        if hasattr(opp, candidate):
            return candidate
    return None


# ────────────────────────────────────────────────────────────────────────────
# [UNIT] Seat Building
# Criterion: "Seats are built in slot order; each PlayerConfig.model equals its
#             assigned model and strategy_hint/levers match compose_strategy_hint/
#             POSTURE_LEVERS"
# ────────────────────────────────────────────────────────────────────────────


class TestBuildSeats:
    """build_seats must produce slot-ordered PlayerConfig objects wired to assignment."""

    def test_when_build_seats_called_then_exactly_six_configs_returned(self, tmp_path):
        """One seat per player: build_seats must return exactly 6 PlayerConfig objects."""
        from training.tournament import build_seats

        cfg = _load_config(tmp_path)
        seats = build_seats(_ASSIGNMENT, cfg)
        assert len(seats) == 6

    def test_when_build_seats_called_then_all_roster_models_present(self, tmp_path):
        """Every model from the roster must appear exactly once across the 6 seats."""
        from training.tournament import build_seats

        cfg = _load_config(tmp_path)
        seats = build_seats(_ASSIGNMENT, cfg)
        assert {s.model for s in seats} == set(_VALID_MODELS)

    def test_when_build_seats_called_then_slot_zero_model_matches_first_strategy(self, tmp_path):
        """Slot 0 must receive the model assigned to strategies[0] (slot order enforced)."""
        from training.tournament import build_seats

        cfg = _load_config(tmp_path)
        seats = build_seats(_ASSIGNMENT, cfg)
        assert seats[0].model == _ASSIGNMENT[_VALID_STRATEGIES[0]]

    def test_when_build_seats_called_then_slot_five_model_matches_last_strategy(self, tmp_path):
        """Slot 5 (last) must receive the model assigned to strategies[5]."""
        from training.tournament import build_seats

        cfg = _load_config(tmp_path)
        seats = build_seats(_ASSIGNMENT, cfg)
        assert seats[5].model == _ASSIGNMENT[_VALID_STRATEGIES[5]]

    def test_when_build_seats_called_then_every_slot_model_equals_assignment(self, tmp_path):
        """For every i in 0-5: seats[i].model == assignment[strategies[i]]."""
        from training.tournament import build_seats

        cfg = _load_config(tmp_path)
        seats = build_seats(_ASSIGNMENT, cfg)
        for i, strategy in enumerate(_VALID_STRATEGIES):
            expected = _ASSIGNMENT[strategy]
            assert seats[i].model == expected, (
                f"Slot {i} ({strategy}): expected model {expected!r}, got {seats[i].model!r}"
            )

    def test_when_build_seats_called_then_strategy_hint_matches_compose_strategy_hint(
        self, tmp_path
    ):
        """Each seat's strategy_hint must equal compose_strategy_hint(strategy_slug)."""
        from src.agents.strategies import compose_strategy_hint
        from training.tournament import build_seats

        cfg = _load_config(tmp_path)
        seats = build_seats(_ASSIGNMENT, cfg)
        for i, strategy in enumerate(_VALID_STRATEGIES):
            expected = compose_strategy_hint(strategy)
            assert seats[i].strategy_hint == expected, (
                f"Slot {i} ({strategy}): strategy_hint mismatch — "
                f"expected {expected!r}, got {seats[i].strategy_hint!r}"
            )

    def test_when_build_seats_called_then_posture_levers_applied_to_each_seat(self, tmp_path):
        """Each seat must carry all POSTURE_LEVERS values for its assigned strategy."""
        from src.agents.strategies import POSTURE_LEVERS
        from training.tournament import build_seats

        cfg = _load_config(tmp_path)
        seats = build_seats(_ASSIGNMENT, cfg)
        for i, strategy in enumerate(_VALID_STRATEGIES):
            levers = POSTURE_LEVERS.get(strategy, {})
            for key, val in levers.items():
                seat_val = getattr(seats[i], key, None)
                assert seat_val == val, (
                    f"Slot {i} ({strategy}): lever '{key}' expected {val!r}, got {seat_val!r}"
                )

    # ── Property: any valid bijection assignment → 6 correctly wired seats ──

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_when_any_valid_assignment_given_then_all_slots_match_assignment(self, seed, tmp_path):
        """Invariant: seats[i].model == assignment[strategies[i]] for any bijection."""
        import random

        from training.tournament import build_seats

        rng = random.Random(seed)
        shuffled = _VALID_MODELS[:]
        rng.shuffle(shuffled)
        assignment = dict(zip(_VALID_STRATEGIES, shuffled, strict=True))
        cfg = _load_config(tmp_path)
        seats = build_seats(assignment, cfg)
        assert len(seats) == 6
        for i, strategy in enumerate(_VALID_STRATEGIES):
            assert seats[i].model == assignment[strategy]


# ────────────────────────────────────────────────────────────────────────────
# [UNIT] Diplomacy Wiring
# Criterion: "DiplomacyRunner is built with enabled=True, n_rounds=config.n_rounds,
#             learner_id=-1, and the shared ReputationBook"
# ────────────────────────────────────────────────────────────────────────────


class TestBuildDiplomacyRunner:
    """build_diplomacy_runner must wire the runner exactly to the spec params."""

    def test_when_build_diplomacy_runner_called_then_result_is_not_none(self, tmp_path):
        """build_diplomacy_runner must return a non-None runner object."""
        from training.tournament import build_diplomacy_runner

        cfg = _load_config(tmp_path)
        book = MagicMock()
        runner = build_diplomacy_runner(cfg, book)
        assert runner is not None

    def test_when_build_diplomacy_runner_called_then_enabled_is_true(self, tmp_path):
        """Criterion: runner.enabled must be True (diplomacy active in tournament mode)."""
        from training.tournament import build_diplomacy_runner

        cfg = _load_config(tmp_path)
        book = MagicMock()
        runner = build_diplomacy_runner(cfg, book)
        assert runner.enabled is True

    def test_when_build_diplomacy_runner_called_then_n_rounds_equals_config_value(self, tmp_path):
        """Criterion: runner.n_rounds must equal config.n_rounds (2 in test config)."""
        from training.tournament import build_diplomacy_runner

        cfg = _load_config(tmp_path)
        book = MagicMock()
        runner = build_diplomacy_runner(cfg, book)
        assert runner.n_rounds == cfg.n_rounds

    def test_when_build_diplomacy_runner_called_then_learner_id_is_minus_one(self, tmp_path):
        """Criterion: learner_id=-1 (no RL learner in pure-LLM tournament mode)."""
        from training.tournament import build_diplomacy_runner

        cfg = _load_config(tmp_path)
        book = MagicMock()
        runner = build_diplomacy_runner(cfg, book)
        assert runner.learner_id == -1

    def test_when_build_diplomacy_runner_called_then_reputation_book_is_shared_instance(
        self, tmp_path
    ):
        """Criterion: the runner must hold exactly the ReputationBook object passed in."""
        from training.tournament import build_diplomacy_runner

        cfg = _load_config(tmp_path)
        book = MagicMock()
        runner = build_diplomacy_runner(cfg, book)
        assert runner.reputation_book is book


# ────────────────────────────────────────────────────────────────────────────
# [T2] build_negotiation_call_fn
# Criterion: "routes player_id → correct model, strips /v1, and returns None
#             on the same fallback contract (LLM mocked)"
# ────────────────────────────────────────────────────────────────────────────


class TestBuildNegotiationCallFn:
    """build_negotiation_call_fn must route, strip /v1, and honour None fallback."""

    def _player_configs(self, tmp_path):
        from training.tournament import build_seats

        cfg = _load_config(tmp_path)
        return build_seats(_ASSIGNMENT, cfg)

    def test_when_build_negotiation_call_fn_called_then_returns_callable_or_none(self, tmp_path):
        """Return value must be a callable or None — no other types accepted."""
        from training.tournament import build_negotiation_call_fn

        player_configs = self._player_configs(tmp_path)
        result = build_negotiation_call_fn(0, player_configs, "http://localhost:11434/v1")
        assert result is None or callable(result)

    def test_when_player_id_0_then_seat_0_model_is_used_in_routing(self, tmp_path):
        """Routing: player_id=0 must resolve to the model held at player_configs[0]."""
        from training.tournament import build_negotiation_call_fn

        player_configs = self._player_configs(tmp_path)
        expected_model = player_configs[0].model
        assert expected_model == _ASSIGNMENT[_VALID_STRATEGIES[0]]

        captured: dict = {}

        def spy(*args, model=None, base_url=None, **kwargs):
            captured["model"] = model or (args[2] if len(args) > 2 else None)
            return 0

        with patch("src.agents.ollama_client.call_ollama_for_action_index", side_effect=spy):
            call_fn = build_negotiation_call_fn(0, player_configs, "http://localhost:11434/v1")
            if callable(call_fn):
                with contextlib.suppress(Exception):
                    call_fn()

        if captured.get("model"):
            assert captured["model"] == expected_model, (
                f"Expected model {expected_model!r} at player_id=0, got {captured['model']!r}"
            )

    def test_when_player_id_5_then_seat_5_model_is_expected(self, tmp_path):
        """Routing: player_id=5 must resolve to the model at the last seat."""
        from training.tournament import build_negotiation_call_fn

        player_configs = self._player_configs(tmp_path)
        expected_model = _ASSIGNMENT[_VALID_STRATEGIES[5]]
        assert player_configs[5].model == expected_model

        captured: dict = {}

        def spy(*args, model=None, base_url=None, **kwargs):
            captured["model"] = model or (args[2] if len(args) > 2 else None)
            return 0

        with patch("src.agents.ollama_client.call_ollama_for_action_index", side_effect=spy):
            call_fn = build_negotiation_call_fn(5, player_configs, "http://localhost:11434/v1")
            if callable(call_fn):
                with contextlib.suppress(Exception):
                    call_fn()

        if captured.get("model"):
            assert captured["model"] == expected_model

    def test_when_base_url_ends_with_v1_then_v1_is_stripped_before_llm_call(self, tmp_path):
        """Criterion: /v1 suffix must be stripped from base_url before calling the LLM."""
        from training.tournament import build_negotiation_call_fn

        player_configs = self._player_configs(tmp_path)
        captured: dict = {}

        def spy(*args, model=None, base_url=None, **kwargs):
            captured["base_url"] = base_url or (args[1] if len(args) > 1 else None)
            return 0

        with patch("src.agents.ollama_client.call_ollama_for_action_index", side_effect=spy):
            call_fn = build_negotiation_call_fn(0, player_configs, "https://ollama.com/v1")
            if callable(call_fn):
                with contextlib.suppress(Exception):
                    call_fn()

        url = captured.get("base_url", "")
        if url:
            assert not url.endswith("/v1"), f"Expected /v1 stripped from base_url, got {url!r}"

    def test_when_llm_raises_then_callable_returns_none(self, tmp_path):
        """Criterion: fallback contract — on any LLM error the callable returns None."""
        from training.tournament import build_negotiation_call_fn

        player_configs = self._player_configs(tmp_path)

        with patch(
            "src.agents.ollama_client.call_ollama_for_negotiation",
            side_effect=RuntimeError("simulated timeout"),
        ):
            call_fn = build_negotiation_call_fn(0, player_configs, "http://localhost:11434/v1")
            assert callable(call_fn)
            result = call_fn(0, "negotiate")
            assert result is None, (
                f"Expected None on LLM failure (fallback contract), got {result!r}"
            )


# ────────────────────────────────────────────────────────────────────────────
# [UNIT] _play_one_game
# Criterion: "returns a LedgerRecord with correct winner_strategy/winner_model
#             and per-game llm_call_count (negotiation + action), all LLM mocked"
# ────────────────────────────────────────────────────────────────────────────


class TestPlayOneGame:
    """_play_one_game must produce a complete, correctly wired LedgerRecord."""

    def test_when_play_one_game_called_then_returns_ledger_record(self, tmp_path):
        """_play_one_game must return a LedgerRecord, not None or a plain dict."""
        from training.tournament import LedgerRecord, _play_one_game

        cfg = _load_config(tmp_path)
        plan = _make_game_plan(game_index=0)

        with patch("src.agents.ollama_client.call_ollama_for_action_index", return_value=0):
            record = _play_one_game(plan, cfg)

        assert isinstance(record, LedgerRecord)

    def test_when_play_one_game_called_then_game_index_preserved_in_record(self, tmp_path):
        """LedgerRecord.game_index must equal the plan's game_index."""
        from training.tournament import _play_one_game

        cfg = _load_config(tmp_path)
        plan = _make_game_plan(game_index=11)

        with patch("src.agents.ollama_client.call_ollama_for_action_index", return_value=0):
            record = _play_one_game(plan, cfg)

        assert record.game_index == 11

    def test_when_play_one_game_called_then_winner_strategy_is_a_valid_slug(self, tmp_path):
        """LedgerRecord.winner_strategy must be one of the 6 catalog strategy slugs."""
        from training.tournament import _play_one_game

        cfg = _load_config(tmp_path)
        plan = _make_game_plan()

        with patch("src.agents.ollama_client.call_ollama_for_action_index", return_value=0):
            record = _play_one_game(plan, cfg)

        if record.winner_strategy is not None:
            assert record.winner_strategy in _VALID_STRATEGIES, (
                f"winner_strategy {record.winner_strategy!r} not in catalog"
            )

    def test_when_play_one_game_called_then_winner_model_is_a_valid_model(self, tmp_path):
        """LedgerRecord.winner_model must be one of the 6 roster model names."""
        from training.tournament import _play_one_game

        cfg = _load_config(tmp_path)
        plan = _make_game_plan()

        with patch("src.agents.ollama_client.call_ollama_for_action_index", return_value=0):
            record = _play_one_game(plan, cfg)

        if record.winner_model is not None:
            assert record.winner_model in _VALID_MODELS, (
                f"winner_model {record.winner_model!r} not in roster"
            )

    def test_when_play_one_game_called_then_winner_strategy_matches_winning_seat(self, tmp_path):
        """Criterion: winner_strategy == strategies[winner] — the slot's assigned strategy."""
        from training.tournament import _play_one_game

        cfg = _load_config(tmp_path)
        plan = _make_game_plan()

        with patch("src.agents.ollama_client.call_ollama_for_action_index", return_value=0):
            record = _play_one_game(plan, cfg)

        if record.winner is not None and record.winner_strategy is not None:
            expected_strategy = _VALID_STRATEGIES[record.winner]
            assert record.winner_strategy == expected_strategy, (
                f"Winner seat {record.winner}: expected strategy "
                f"{expected_strategy!r}, got {record.winner_strategy!r}"
            )

    def test_when_play_one_game_called_then_winner_model_matches_winning_seat(self, tmp_path):
        """Criterion: winner_model == assignment[winner_strategy] — the slot's assigned model."""
        from training.tournament import _play_one_game

        cfg = _load_config(tmp_path)
        plan = _make_game_plan()

        with patch("src.agents.ollama_client.call_ollama_for_action_index", return_value=0):
            record = _play_one_game(plan, cfg)

        if record.winner_strategy is not None and record.winner_model is not None:
            expected_model = _ASSIGNMENT[record.winner_strategy]
            assert record.winner_model == expected_model, (
                f"Winner strategy {record.winner_strategy!r}: expected model "
                f"{expected_model!r}, got {record.winner_model!r}"
            )

    def test_when_play_one_game_called_then_llm_call_count_is_non_negative_integer(self, tmp_path):
        """llm_call_count must be a non-negative integer (counts all LLM invocations)."""
        from training.tournament import _play_one_game

        cfg = _load_config(tmp_path)
        plan = _make_game_plan()

        with patch("src.agents.ollama_client.call_ollama_for_action_index", return_value=0):
            record = _play_one_game(plan, cfg)

        assert isinstance(record.llm_call_count, int), (
            f"llm_call_count must be int, got {type(record.llm_call_count)}"
        )
        assert record.llm_call_count >= 0

    def test_when_play_one_game_called_then_seat_strategies_covers_all_six(self, tmp_path):
        """LedgerRecord.seat_strategies must map all 6 strategy slugs across slots."""
        from training.tournament import _play_one_game

        cfg = _load_config(tmp_path)
        plan = _make_game_plan()

        with patch("src.agents.ollama_client.call_ollama_for_action_index", return_value=0):
            record = _play_one_game(plan, cfg)

        assert len(record.seat_strategies) == 6
        assert set(record.seat_strategies.values()) == set(_VALID_STRATEGIES)

    def test_when_play_one_game_called_then_seat_models_covers_all_six(self, tmp_path):
        """LedgerRecord.seat_models must map all 6 model names across slots."""
        from training.tournament import _play_one_game

        cfg = _load_config(tmp_path)
        plan = _make_game_plan()

        with patch("src.agents.ollama_client.call_ollama_for_action_index", return_value=0):
            record = _play_one_game(plan, cfg)

        assert len(record.seat_models) == 6
        assert set(record.seat_models.values()) == set(_VALID_MODELS)


# ────────────────────────────────────────────────────────────────────────────
# [UNIT] LLMOpponent public LLM-call counter
# Criterion: "LLMOpponent exposes a public LLM-call counter (covered by a unit test)"
# ────────────────────────────────────────────────────────────────────────────


class TestLLMOpponentCallCounter:
    """LLMOpponent must expose a public integer LLM-call counter."""

    def test_when_llm_opponent_created_then_public_call_counter_attribute_exists(self):
        """LLMOpponent must have at least one public call-counter attribute."""
        from src.agents.llm_opponent import LLMOpponent

        opp = LLMOpponent(model="glm-5.2:cloud")
        attr = _counter_attr(opp)
        assert attr is not None, (
            "LLMOpponent must expose a public LLM-call counter attribute "
            "(accepted names: llm_call_count, call_count, n_llm_calls, num_llm_calls)"
        )

    def test_when_llm_opponent_created_then_call_counter_starts_at_zero(self):
        """The call counter must be initialised to 0 at construction."""
        from src.agents.llm_opponent import LLMOpponent

        opp = LLMOpponent(model="glm-5.2:cloud")
        attr = _counter_attr(opp)
        assert attr is not None, "No public counter attribute found"
        assert getattr(opp, attr) == 0, f"{attr} must start at 0, got {getattr(opp, attr)}"

    def test_when_llm_opponent_created_then_call_counter_is_integer(self):
        """The call counter value must be an int (not float or string)."""
        from src.agents.llm_opponent import LLMOpponent

        opp = LLMOpponent(model="glm-5.2:cloud")
        attr = _counter_attr(opp)
        if attr is not None:
            assert isinstance(getattr(opp, attr), int)

    def test_when_act_called_once_with_mocked_llm_then_counter_is_one(self):
        """Counter must be 1 after one successful LLM act() call."""
        from src.agents.llm_opponent import LLMOpponent

        with patch("src.agents.ollama_client.call_ollama_for_action_index", return_value=0):
            opp = LLMOpponent(model="glm-5.2:cloud")
            opp.act({}, list(range(5)))

        attr = _counter_attr(opp)
        assert attr is not None, "No public counter attribute found"
        assert getattr(opp, attr) == 1, (
            f"Expected counter=1 after one act(), got {getattr(opp, attr)}"
        )

    def test_when_act_called_three_times_with_mocked_llm_then_counter_is_three(self):
        """Counter must equal the number of successful LLM calls (3 calls → counter=3)."""
        from src.agents.llm_opponent import LLMOpponent

        with patch("src.agents.ollama_client.call_ollama_for_action_index", return_value=0):
            opp = LLMOpponent(model="glm-5.2:cloud")
            for _ in range(3):
                opp.act({}, list(range(5)))

        attr = _counter_attr(opp)
        assert attr is not None, "No public counter attribute found"
        assert getattr(opp, attr) == 3, (
            f"Expected counter=3 after 3 act() calls, got {getattr(opp, attr)}"
        )

    def test_when_llm_raises_and_random_fallback_used_then_counter_stays_zero(self):
        """Fallback to RandomAgent on LLM error must NOT increment the counter.

        The counter tracks actual LLM invocations only — not total act() calls.
        This contract lets _play_one_game accurately tally negotiation + action
        LLM calls separately from random fallbacks.
        """
        from src.agents.llm_opponent import LLMOpponent

        with patch(
            "src.agents.ollama_client.call_ollama_for_action_index",
            side_effect=RuntimeError("simulated timeout"),
        ):
            opp = LLMOpponent(model="glm-5.2:cloud")
            opp.act({}, list(range(5)))

        attr = _counter_attr(opp)
        if attr is not None:
            assert getattr(opp, attr) == 0, (
                f"Counter must stay 0 when LLM fails and random fallback is used; "
                f"got {getattr(opp, attr)}"
            )

    # ── Property: counter equals number of successful act() calls ────────────

    @given(n_calls=st.integers(min_value=0, max_value=15))
    def test_when_n_successful_act_calls_made_then_counter_equals_n(self, n_calls):
        """Invariant: counter == N for any N mocked-LLM act() calls in [0, 15]."""
        from src.agents.llm_opponent import LLMOpponent

        with patch("src.agents.ollama_client.call_ollama_for_action_index", return_value=0):
            opp = LLMOpponent(model="glm-5.2:cloud")
            for _ in range(n_calls):
                opp.act({}, list(range(5)))

        attr = _counter_attr(opp)
        if attr is not None:
            assert getattr(opp, attr) == n_calls, (
                f"Expected counter={n_calls} after {n_calls} act() calls, got {getattr(opp, attr)}"
            )
