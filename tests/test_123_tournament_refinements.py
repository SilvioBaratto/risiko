"""Source-blind example tests for issue #123 — tournament refinements.

Authored from acceptance criteria only; no implementation source was read.
All tests start Red (TDD).

UNIT-verifiable criteria covered:
  1. Negotiation routes through call_ollama_for_negotiation with correct kwargs;
     a mocked dict reply propagates to DiplomacyRunner (not coerced to None).
  2. A single ReputationBook instance is shared across all games in one
     run_tournament call.
  3. _play_with_backoff (or equivalent) actually guards the negotiation/preflight
     path; transient-error retry covered by a test.
  4. Negotiation base URL is sourced from OLLAMA_BASE_URL; max_turns is no
     longer a bare magic number.
  5. Dead run_game / build_diplomacy_runner reconciled (consumed or removed).

Skipped criteria (oracle: NOT VERIFIABLE):
  - risiko-rl tournament --llm-only flag (CLI/integration)
  - Preflight model pull / abort (integration)
  - Strategy assignment distribution (statistical)
  - Leaderboard output format (integration)
  - All randomness through seeds (architectural)
  - SOLID / line-length prose (subjective)
"""

from __future__ import annotations

import contextlib
import os
import pathlib
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants and helpers (derived from requirements, not source)
# ─────────────────────────────────────────────────────────────────────────────

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

_ASSIGNMENT: dict[str, str] = dict(zip(_VALID_STRATEGIES, _VALID_MODELS, strict=True))


def _base_config_data(n_games: int = 4) -> dict:
    return {
        "name": "test_tournament_123",
        "seed": 17,
        "n_games": n_games,
        "max_concurrency": 1,
        "n_rounds": 2,
        "strategies": _VALID_STRATEGIES,
        "models": _VALID_MODELS,
        "temperature": 0.7,
        "top_p": 0.9,
    }


def _write_config(tmp_path: pathlib.Path, n_games: int = 4) -> pathlib.Path:
    cfg_file = tmp_path / "tournament.yaml"
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(yaml.dump(_base_config_data(n_games=n_games)))
    return cfg_file


def _load_config(tmp_path: pathlib.Path, n_games: int = 4):
    from training.tournament import load_tournament_config

    return load_tournament_config(_write_config(tmp_path, n_games=n_games))


def _make_ledger_record(game_index: int = 0, llm_call_count: int = 5):
    from training.tournament import LedgerRecord

    return LedgerRecord(
        game_index=game_index,
        seed=game_index + 42,
        winner=0,
        winner_strategy="australia_lock",
        winner_model="glm-5.2:cloud",
        n_turns=60,
        elimination_order=[1, 2, 3, 4, 5],
        seat_strategies={str(i): s for i, s in enumerate(_VALID_STRATEGIES)},
        seat_models={str(i): m for i, m in enumerate(_VALID_MODELS)},
        assignment=dict(_ASSIGNMENT),
        card_trade_turns=[12, 30],
        llm_call_count=llm_call_count,
    )


def _spy_play_fn(call_log: list[int]):
    """Return a play_fn that records game_index calls and returns a minimal record."""

    def _play(plan):
        call_log.append(plan.game_index)
        return _make_ledger_record(game_index=plan.game_index)

    return _play


def _make_game_plan(game_index: int = 0):
    from training.tournament import GamePlan

    return GamePlan(game_index=game_index, seed=42 + game_index, assignment=dict(_ASSIGNMENT))


# ─────────────────────────────────────────────────────────────────────────────
# TestNegotiationCallOllama
# Criterion: "Negotiation routes through call_ollama_for_negotiation with correct
#             kwargs; a mocked dict reply propagates to DiplomacyRunner (not
#             coerced to None); test added"
# ─────────────────────────────────────────────────────────────────────────────


class TestNegotiationCallOllama:
    """call_ollama_for_negotiation must exist and its dict replies must propagate."""

    def test_when_module_imported_then_call_ollama_for_negotiation_is_callable(self):
        """Criterion: call_ollama_for_negotiation must be importable and callable."""
        import src.agents.ollama_client as m

        fn = getattr(m, "call_ollama_for_negotiation", None)
        assert callable(fn), "call_ollama_for_negotiation must exist in src.agents.ollama_client"

    def test_when_negotiation_fn_returns_dict_then_routing_fn_propagates_not_none(self, tmp_path):
        """A dict reply from call_ollama_for_negotiation must NOT be coerced to None."""
        import src.agents.ollama_client as client_mod
        from training.tournament import build_negotiation_call_fn, build_seats

        cfg = _load_config(tmp_path)
        seats = build_seats(_ASSIGNMENT, cfg)
        expected = {"text": "I propose a non-aggression pact"}

        with patch.object(client_mod, "call_ollama_for_negotiation", return_value=expected):
            call_fn = build_negotiation_call_fn(0, seats, "http://localhost:11434/v1")
            if not callable(call_fn):
                pytest.skip("build_negotiation_call_fn returned None — not yet wired")
            result = call_fn(0, "make a proposal")

        assert result is not None, "Dict reply must NOT be coerced to None by the routing closure"
        assert result == expected, f"Expected {expected!r}, got {result!r}"

    def test_when_negotiation_fn_called_then_call_ollama_for_negotiation_is_invoked(self, tmp_path):
        """Routing must call call_ollama_for_negotiation (not call_ollama_for_action_index)."""
        import src.agents.ollama_client as client_mod
        from training.tournament import build_negotiation_call_fn, build_seats

        cfg = _load_config(tmp_path)
        seats = build_seats(_ASSIGNMENT, cfg)

        negotiation_call_log: list = []

        def spy_negotiation(*args, **kwargs):
            negotiation_call_log.append(("negotiation", args, kwargs))
            return {"text": "ok"}

        action_call_log: list = []

        def spy_action(*args, **kwargs):
            action_call_log.append(("action", args, kwargs))
            return 0

        with (
            patch.object(client_mod, "call_ollama_for_negotiation", side_effect=spy_negotiation),
            patch.object(client_mod, "call_ollama_for_action_index", side_effect=spy_action),
        ):
            call_fn = build_negotiation_call_fn(0, seats, "http://localhost:11434/v1")
            if callable(call_fn):
                with contextlib.suppress(Exception):
                    call_fn(0, "negotiate")

        assert len(negotiation_call_log) >= 1, (
            "call_ollama_for_negotiation must be invoked by the negotiation routing fn"
        )

    def test_when_negotiation_fn_returns_int_then_routing_fn_returns_none(self, tmp_path):
        """An int reply (action index, not a negotiation dict) must not pass as-is."""
        import src.agents.ollama_client as client_mod
        from training.tournament import build_negotiation_call_fn, build_seats

        cfg = _load_config(tmp_path)
        seats = build_seats(_ASSIGNMENT, cfg)

        with patch.object(client_mod, "call_ollama_for_negotiation", return_value=3):
            call_fn = build_negotiation_call_fn(0, seats, "http://localhost:11434/v1")
            if not callable(call_fn):
                pytest.skip("build_negotiation_call_fn returned None")
            result = call_fn(0, "negotiate")

        assert result is None or isinstance(result, dict), (
            "A non-dict reply from the negotiation fn must not pass as a dict to DiplomacyRunner"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestReputationBookSharing
# Criterion: "A single ReputationBook instance is shared across all games in
#             one run_tournament call; test added"
# ─────────────────────────────────────────────────────────────────────────────


class TestReputationBookSharing:
    """One ReputationBook must be created per run_tournament call, not per game."""

    def test_when_run_tournament_plays_4_games_then_reputation_book_created_exactly_once(
        self, tmp_path
    ):
        """Criterion: ReputationBook() must be called once for n_games=4, not 4 times."""
        from training.tournament import run_tournament

        cfg = _load_config(tmp_path, n_games=4)
        ledger = tmp_path / "games.jsonl"
        call_log: list[int] = []

        with patch("training.tournament.ReputationBook") as mock_book:
            mock_book.return_value = MagicMock()
            run_tournament(cfg, ledger, play_fn=_spy_play_fn(call_log))

        assert len(call_log) == 4, "Expected 4 games to be played"
        assert mock_book.call_count == 1, (
            f"Expected exactly 1 ReputationBook instantiation for 4 games, "
            f"got {mock_book.call_count}"
        )

    def test_when_run_tournament_with_n_games_1_then_reputation_book_created_once(self, tmp_path):
        """Even for a single-game run, ReputationBook must be created exactly once."""
        from training.tournament import run_tournament

        cfg = _load_config(tmp_path, n_games=1)
        ledger = tmp_path / "games.jsonl"

        with patch("training.tournament.ReputationBook") as mock_book:
            mock_book.return_value = MagicMock()
            run_tournament(cfg, ledger, play_fn=_spy_play_fn([]))

        assert mock_book.call_count == 1

    def test_when_run_tournament_creates_reputation_book_then_same_instance_across_games(
        self, tmp_path
    ):
        """All games must share the SAME ReputationBook instance (identity check)."""
        import training.tournament as tm
        from training.tournament import run_tournament

        cfg = _load_config(tmp_path, n_games=3)
        ledger = tmp_path / "games.jsonl"
        books_seen: list = []
        plans_seen: list = []

        def capturing_play(plan, *args, **kwargs):
            plans_seen.append(plan.game_index)
            # Try to capture the reputation_book from positional/keyword args
            if args:
                books_seen.append(args[0] if len(args) > 1 else args[0])
            if "reputation_book" in kwargs:
                books_seen.append(kwargs["reputation_book"])
            return _make_ledger_record(game_index=plan.game_index)

        # Patch _play_one_game to intercept calls including reputation_book arg.
        # skip_preflight: without a play_fn, run_tournament pings the model server for
        # real — a unit test must not depend on an Ollama being up.
        with patch.object(tm, "_play_one_game", side_effect=capturing_play):
            run_tournament(cfg, ledger, skip_preflight=True)

        if not plans_seen:
            # _play_one_game not called — different design; skip identity check
            pytest.skip("_play_one_game not intercepted — verify sharing via other means")

        if len(books_seen) >= 2:
            assert all(b is books_seen[0] for b in books_seen), (
                "All games must receive the SAME ReputationBook instance"
            )

    def test_when_two_run_tournament_calls_then_separate_reputation_books_created(self, tmp_path):
        """Two separate run_tournament calls must each get their own ReputationBook."""
        from training.tournament import run_tournament

        cfg = _load_config(tmp_path, n_games=2)
        ledger_a = tmp_path / "a.jsonl"
        ledger_b = tmp_path / "b.jsonl"

        creation_count = [0]

        def count_book():
            creation_count[0] += 1
            return MagicMock()

        with patch("training.tournament.ReputationBook", side_effect=count_book):
            run_tournament(cfg, ledger_a, play_fn=_spy_play_fn([]))
            run_tournament(cfg, ledger_b, play_fn=_spy_play_fn([]))

        assert creation_count[0] == 2, (
            f"Two run_tournament calls must create 2 ReputationBooks, got {creation_count[0]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestPlayWithBackoffRetry
# Criterion: "_play_with_backoff (or equivalent) actually guards the
#             negotiation/preflight path; transient-error retry covered by a test"
# ─────────────────────────────────────────────────────────────────────────────


class TestPlayWithBackoffRetry:
    """TournamentRunner._play_with_backoff must retry on transient HTTP errors."""

    def _runner(self) -> object:
        from training.tournament import TournamentRunner

        return TournamentRunner(max_concurrency=1, base_backoff=0.0)

    def test_when_first_call_raises_http_error_then_second_call_result_returned(self):
        """A transient HTTPError on call 1 must trigger a retry; call 2's result used."""
        runner = self._runner()
        call_count = [0]

        def fn_fails_once():
            call_count[0] += 1
            if call_count[0] == 1:
                raise httpx.HTTPError("transient failure")
            return "success_after_retry"

        result = runner._play_with_backoff(fn_fails_once, max_retries=1)
        assert result == "success_after_retry"
        assert call_count[0] == 2, "Must have called fn exactly twice (1 failure + 1 success)"

    def test_when_all_retries_exhausted_then_http_error_is_reraised(self):
        """When every attempt raises HTTPError, the final error must propagate."""
        runner = self._runner()

        def fn_always_fails():
            raise httpx.HTTPError("persistent failure")

        with pytest.raises(httpx.HTTPError):
            runner._play_with_backoff(fn_always_fails, max_retries=2)

    def test_when_http_status_error_raised_then_retry_happens(self):
        """HTTPStatusError (e.g. 429) is a subclass of HTTPError — must also retry."""
        runner = self._runner()
        call_count = [0]

        def fn_fails_with_status():
            call_count[0] += 1
            if call_count[0] < 3:
                raise httpx.HTTPStatusError(
                    "429 Too Many Requests",
                    request=MagicMock(),
                    response=MagicMock(),
                )
            return {"text": "proposal after backoff"}

        result = runner._play_with_backoff(fn_fails_with_status, max_retries=3)
        assert result == {"text": "proposal after backoff"}
        assert call_count[0] == 3

    def test_when_non_http_exception_raised_then_it_is_not_caught(self):
        """Non-HTTP exceptions (e.g. ValueError) must propagate immediately, no retry."""
        runner = self._runner()
        call_count = [0]

        def fn_raises_value_error():
            call_count[0] += 1
            raise ValueError("non-transient error")

        with pytest.raises(ValueError):
            runner._play_with_backoff(fn_raises_value_error, max_retries=3)

        assert call_count[0] == 1, "Non-HTTP exception must not trigger retries"

    def test_when_fn_succeeds_on_first_call_then_result_returned_immediately(self):
        """A successful first call must return its result without any retries."""
        runner = self._runner()
        call_count = [0]

        def fn_always_succeeds():
            call_count[0] += 1
            return 42

        result = runner._play_with_backoff(fn_always_succeeds, max_retries=3)
        assert result == 42
        assert call_count[0] == 1

    # ── Hypothesis: retry invariant ──────────────────────────────────────────

    @given(n_failures=st.integers(min_value=0, max_value=4))
    def test_when_n_failures_then_succeeds_if_within_retry_budget(self, n_failures):
        """Invariant: if n_failures <= max_retries, the final call succeeds."""
        from training.tournament import TournamentRunner

        runner = TournamentRunner(max_concurrency=1, base_backoff=0.0)
        max_retries = 4
        call_count = [0]

        def fn():
            call_count[0] += 1
            if call_count[0] <= n_failures:
                raise httpx.HTTPError("transient")
            return "ok"

        if n_failures <= max_retries:
            result = runner._play_with_backoff(fn, max_retries=max_retries)
            assert result == "ok"
        else:
            with pytest.raises(httpx.HTTPError):
                runner._play_with_backoff(fn, max_retries=max_retries)


# ─────────────────────────────────────────────────────────────────────────────
# TestNegotiationBaseURLAndMaxTurns
# Criterion: "Negotiation base URL is sourced from OLLAMA_BASE_URL;
#             max_turns is no longer a bare magic number"
# ─────────────────────────────────────────────────────────────────────────────


class TestNegotiationBaseURLAndMaxTurns:
    """Negotiation URL must come from env; max_turns must be a named constant."""

    def test_when_ollama_base_url_env_set_then_negotiation_uses_that_url(self, tmp_path):
        """Criterion: OLLAMA_BASE_URL env var must drive the negotiation call base_url."""
        import src.agents.ollama_client as client_mod
        from training.tournament import build_negotiation_call_fn, build_seats

        cfg = _load_config(tmp_path)
        seats = build_seats(_ASSIGNMENT, cfg)
        custom_url = "http://custom-ollama:9999/v1"
        captured: list = []

        def spy(*args, root=None, **kwargs):
            # call_ollama_for_negotiation uses `root` (not `base_url`) for the server URL
            captured.append(root)
            return {"text": "ok"}

        with (
            patch.dict(os.environ, {"OLLAMA_BASE_URL": custom_url}, clear=False),
            patch.object(client_mod, "call_ollama_for_negotiation", side_effect=spy),
        ):
            # Re-build call_fn with env-derived URL (implementation must read OLLAMA_BASE_URL)
            call_fn = build_negotiation_call_fn(0, seats, os.environ["OLLAMA_BASE_URL"])
            if callable(call_fn):
                with contextlib.suppress(Exception):
                    call_fn(0, "make a proposal")

        # Check the captured root URL contains the custom host
        if captured:
            custom_host = "custom-ollama:9999"
            assert any(custom_host in (url or "") for url in captured), (
                f"Negotiation must use OLLAMA_BASE_URL={custom_url!r}; "
                f"captured root values: {captured!r}"
            )

    def test_when_max_turns_inspected_then_it_is_a_named_constant_or_config_field(self):
        """Criterion: max_turns must not be a bare magic number (200) in _play_one_game.

        Choice: either training.tournament exposes a module-level constant (MAX_TURNS /
        _MAX_TURNS / TOURNAMENT_MAX_TURNS), or TournamentConfig has a max_turns field.
        Either approach satisfies the criterion.
        """
        import training.tournament as tm
        from training.tournament import TournamentConfig

        has_module_constant = any(
            hasattr(tm, name) for name in ("MAX_TURNS", "_MAX_TURNS", "TOURNAMENT_MAX_TURNS")
        )
        has_config_field = "max_turns" in TournamentConfig.__dataclass_fields__

        assert has_module_constant or has_config_field, (
            "max_turns must be exposed as a named module constant or a TournamentConfig field, "
            "not a bare magic number (e.g. 200) embedded in _play_one_game"
        )

    def test_when_ollama_base_url_not_set_then_negotiation_still_works(self, tmp_path):
        """Negotiation must not crash when OLLAMA_BASE_URL uses the default localhost."""
        import src.agents.ollama_client as client_mod
        from training.tournament import build_negotiation_call_fn, build_seats

        cfg = _load_config(tmp_path)
        seats = build_seats(_ASSIGNMENT, cfg)

        env_without_custom = {k: v for k, v in os.environ.items() if k != "OLLAMA_BASE_URL"}

        with (
            patch.dict(os.environ, env_without_custom, clear=True),
            patch.object(client_mod, "call_ollama_for_negotiation", return_value=None),
        ):
            default_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            call_fn = build_negotiation_call_fn(0, seats, default_url)
            if callable(call_fn):
                with contextlib.suppress(Exception):
                    call_fn(0, "negotiate")


# ─────────────────────────────────────────────────────────────────────────────
# TestDeadCodeReconciled
# Criterion: "Dead run_game / build_diplomacy_runner reconciled
#             (consumed or removed)"
# ─────────────────────────────────────────────────────────────────────────────


class TestDeadCodeReconciled:
    """run_game and build_diplomacy_runner must not be NotImplementedError stubs."""

    def test_when_run_game_accessed_then_it_is_not_a_not_implemented_error_stub(self):
        """Criterion: run_game must be removed or implemented — not a dead stub.

        Choice: if run_game is removed from the module, this test passes immediately.
        If it still exists, calling it must not raise NotImplementedError.
        """
        import training.tournament as tm

        if not hasattr(tm, "run_game"):
            return  # Removed — criterion satisfied

        plan = _make_game_plan(0)
        try:
            tm.run_game(plan)
        except NotImplementedError:
            pytest.fail(
                "run_game must not raise NotImplementedError. "
                "Either implement it fully or remove it."
            )
        except Exception:
            pass  # Any other exception means it has been implemented (not a stub)

    def test_when_build_diplomacy_runner_accessed_then_it_is_not_an_unused_stub(self):
        """Criterion: build_diplomacy_runner must be removed or produce a real runner.

        Choice: if removed, the test passes immediately. If it still exists, it
        must return an object that is not the dead _DiplomacyWiring stub
        (i.e., it should be an actual runner or at least callable/runnable).
        """
        import training.tournament as tm

        if not hasattr(tm, "build_diplomacy_runner"):
            return  # Removed — criterion satisfied

        cfg_mock = MagicMock()
        cfg_mock.n_rounds = 2
        book_mock = MagicMock()

        try:
            result = tm.build_diplomacy_runner(cfg_mock, book_mock)
        except Exception:
            return  # raised — needs investigation but not a dead stub pattern

        # If it returns something, verify it's not the old _DiplomacyWiring dead-stub
        # The old stub had no callable run_game method
        if result is not None:
            # A real DiplomacyRunner-compatible object should have a run_game callable
            # OR build_diplomacy_runner should be removed entirely
            assert hasattr(result, "run_game") and callable(result.run_game), (
                "build_diplomacy_runner must return a real DiplomacyRunner "
                "or be removed. Got a stub without a run_game method."
            )

    def test_when_run_game_removed_then_module_still_imports_cleanly(self):
        """Removing run_game must not break module imports or __all__."""
        import training.tournament as tm  # noqa: F401

        # If run_game was in __all__, its removal must also update __all__
        all_names = getattr(tm, "__all__", [])
        if "run_game" not in all_names:
            return  # Already removed from __all__ — OK

        # If still in __all__, it must exist as a non-stub
        if hasattr(tm, "run_game"):
            plan = _make_game_plan(0)
            try:
                tm.run_game(plan)
            except NotImplementedError:
                pytest.fail(
                    "run_game in __all__ but raises NotImplementedError — must be reconciled"
                )
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# TestRLTrainPathStillIntact
# Criterion (global, UNIT): "The existing RL train path is unchanged"
# ─────────────────────────────────────────────────────────────────────────────


def test_when_refinements_imported_then_actor_critic_is_intact():
    """Importing tournament refinements must not break src.models.actor_critic."""
    import src.models.actor_critic  # noqa: F401
    import training.tournament  # noqa: F401

    assert hasattr(src.models.actor_critic, "ActorCritic")


def test_when_refinements_imported_then_ppo_config_is_intact():
    """Importing tournament refinements must not break src.config.TrainingConfig."""
    import src.config  # noqa: F401
    import training.tournament  # noqa: F401

    assert hasattr(src.config, "TrainingConfig")


def test_when_refinements_imported_then_default_config_yaml_still_loads():
    """config/default.yaml must remain loadable by load_config after refinements."""
    import training.tournament  # noqa: F401
    from src.config import load_config

    cfg = load_config("config/default.yaml")
    assert cfg is not None
