"""
Issue #107 — fix(evaluation): social-eval uses plain MultiAgentRunner and
fabricates total_negotiation_calls.

RED tests authored against the issue body acceptance criteria only.
All tests in this file must FAIL before the fix lands.

Acceptance criteria:
- DiplomacyRunner is built when cfg.enabled=True; MultiAgentRunner otherwise.
- total_negotiation_calls = summed observed runner.call_count (not n_games * n_rounds * n_speakers).
- A parse-failure / disabled run reports 0, never the formula.
- Cross-game reputation carry: a betrayal in game i raises betrayal_count visible in game i+1.
- Determinism: mocked LLM + fixed seed => identical aggregate result (diplomacy path exercised).
- run_social_eval accepts profiles and cfg: DiplomacyConfig keyword args.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

from src.agents.reputation import ReputationBook
from src.config import DiplomacyConfig

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_game_result(winner=None, n_turns=10):
    r = MagicMock()
    r.winner = winner
    r.n_turns = n_turns
    return r


def _dummy_learner():
    m = MagicMock()
    m.act.return_value = {}
    m.act_with_meta.return_value = ({}, math.nan, math.nan)
    return m


def _dummy_pool(n=5):
    agents = []
    for _ in range(n):
        a = MagicMock()
        a.act.return_value = {}
        a.act_with_meta.return_value = ({}, math.nan, math.nan)
        agents.append(a)
    return agents


def _mock_diplomacy_runner_cls(winner_sequence, call_count_per_game=2):
    """Return a MagicMock that looks like DiplomacyRunner."""
    cls = MagicMock()
    instance = cls.return_value
    instance.run.side_effect = [_make_game_result(w) for w in winner_sequence]
    instance.call_count = call_count_per_game
    return cls


# ---------------------------------------------------------------------------
# Criterion: run_social_eval accepts profiles and cfg kwargs (signature fix)
# ---------------------------------------------------------------------------


class TestSignatureAcceptsProfilesAndCfg:
    """The fixed signature must accept profiles and cfg without TypeError."""

    def test_when_called_with_cfg_kwarg_then_no_type_error(self):
        """run_social_eval(cfg=DiplomacyConfig()) must not raise TypeError."""
        from training.social_eval import run_social_eval  # noqa: PLC0415

        cfg = DiplomacyConfig(enabled=False)
        with patch("training.social_eval.MultiAgentRunner") as mock_cls:
            mock_cls.return_value.run.return_value = _make_game_result(0)
            # Must not raise TypeError for unexpected keyword argument 'cfg'
            result = run_social_eval(
                learner=_dummy_learner(),
                llm_pool=_dummy_pool(5),
                n_games=1,
                cfg=cfg,
            )
        assert hasattr(result, "total_negotiation_calls")

    def test_when_called_with_profiles_kwarg_then_no_type_error(self):
        """run_social_eval(profiles=[...]) must not raise TypeError."""
        from training.social_eval import run_social_eval  # noqa: PLC0415

        with patch("training.social_eval.MultiAgentRunner") as mock_cls:
            mock_cls.return_value.run.return_value = _make_game_result(0)
            result = run_social_eval(
                learner=_dummy_learner(),
                llm_pool=_dummy_pool(5),
                n_games=1,
                profiles=None,
            )
        assert hasattr(result, "total_negotiation_calls")


# ---------------------------------------------------------------------------
# Criterion: total_negotiation_calls = 0 when diplomacy is disabled
# ---------------------------------------------------------------------------


class TestTotalCallsIsZeroWhenDiplomacyDisabled:
    """
    When cfg.enabled is False (the default), no negotiation runs.
    total_negotiation_calls must be 0, NOT the formula n_games*n_rounds*n_speakers.
    """

    def _run(self, n_games, n_rounds, n_speakers=5):
        from training.social_eval import run_social_eval  # noqa: PLC0415

        with patch("training.social_eval.MultiAgentRunner") as mock_cls:
            mock_cls.return_value.run.return_value = _make_game_result(0)
            return run_social_eval(
                learner=_dummy_learner(),
                llm_pool=_dummy_pool(n_speakers),
                n_games=n_games,
                n_rounds=n_rounds,
                cfg=DiplomacyConfig(enabled=False),
            )

    def test_when_diplomacy_disabled_then_total_calls_is_zero_not_formula(self):
        """n_games=3, n_rounds=2, n_speakers=5 => formula=30; fix must report 0."""
        result = self._run(n_games=3, n_rounds=2, n_speakers=5)
        assert result.total_negotiation_calls == 0, (
            f"Expected 0 when diplomacy disabled, got {result.total_negotiation_calls} "
            "(formula n_games*n_rounds*n_speakers would be 30)"
        )

    def test_when_diplomacy_disabled_and_n_games_one_then_total_calls_is_zero(self):
        result = self._run(n_games=1, n_rounds=1, n_speakers=5)
        assert result.total_negotiation_calls == 0


# ---------------------------------------------------------------------------
# Criterion: total_negotiation_calls = summed observed DiplomacyRunner.call_count
# ---------------------------------------------------------------------------


class TestTotalCallsEqualsObservedCallCount:
    """
    total_negotiation_calls must come from runner.call_count (observed) not a formula.
    """

    def test_when_diplomacy_enabled_total_calls_equals_runner_call_count_sum(self):
        """If each game's runner reports call_count=3 and n_games=2, total must be 6."""
        from training.social_eval import run_social_eval  # noqa: PLC0415

        cfg = DiplomacyConfig(enabled=True, n_rounds=1)
        per_game_calls = 3
        n_games = 2

        with patch("training.social_eval.DiplomacyRunner") as mock_cls:
            instance = mock_cls.return_value
            instance.run.return_value = _make_game_result(0)
            instance.call_count = per_game_calls

            result = run_social_eval(
                learner=_dummy_learner(),
                llm_pool=_dummy_pool(5),
                n_games=n_games,
                cfg=cfg,
            )

        expected = per_game_calls * n_games
        assert result.total_negotiation_calls == expected, (
            f"Expected observed total={expected}, got {result.total_negotiation_calls}. "
            "total_negotiation_calls must derive from runner.call_count, not a formula."
        )

    def test_when_diplomacy_enabled_and_zero_calls_then_total_is_zero(self):
        """DiplomacyRunner with call_count=0 must yield total=0 (noop/parse-failure path)."""
        from training.social_eval import run_social_eval  # noqa: PLC0415

        cfg = DiplomacyConfig(enabled=True, n_rounds=1)

        with patch("training.social_eval.DiplomacyRunner") as mock_cls:
            instance = mock_cls.return_value
            instance.run.return_value = _make_game_result(0)
            instance.call_count = 0

            result = run_social_eval(
                learner=_dummy_learner(),
                llm_pool=_dummy_pool(5),
                n_games=3,
                cfg=cfg,
            )

        assert result.total_negotiation_calls == 0, (
            "A noop call_fn / no negotiation must report 0, never the formula."
        )

    def test_when_diplomacy_enabled_total_calls_bounded_by_n_games_times_upper_per_game(self):
        """Observed count must not exceed the theoretical upper bound per the spec."""
        from training.social_eval import run_social_eval  # noqa: PLC0415

        n_games = 4
        n_players = 6
        n_rounds = 2
        n_speakers = n_players - 1  # learner excluded
        per_game_upper = n_rounds * n_speakers
        cfg = DiplomacyConfig(enabled=True, n_rounds=n_rounds)

        with patch("training.social_eval.DiplomacyRunner") as mock_cls:
            instance = mock_cls.return_value
            instance.run.return_value = _make_game_result(0)
            # simulate a run that hit the exact cap
            instance.call_count = per_game_upper

            result = run_social_eval(
                learner=_dummy_learner(),
                llm_pool=_dummy_pool(n_speakers),
                n_games=n_games,
                cfg=cfg,
            )

        assert result.total_negotiation_calls <= n_games * per_game_upper


# ---------------------------------------------------------------------------
# Criterion: DiplomacyRunner used when enabled; MultiAgentRunner when disabled
# ---------------------------------------------------------------------------


class TestRunnerSelectionByConfig:
    """The runner class selected must match cfg.enabled."""

    def test_when_cfg_enabled_true_then_diplomacy_runner_is_instantiated(self):
        from training.social_eval import run_social_eval  # noqa: PLC0415

        cfg = DiplomacyConfig(enabled=True)
        with (
            patch("training.social_eval.DiplomacyRunner") as mock_dr,
            patch("training.social_eval.MultiAgentRunner") as mock_mar,
        ):
            mock_dr.return_value.run.return_value = _make_game_result(0)
            mock_dr.return_value.call_count = 0

            run_social_eval(
                learner=_dummy_learner(),
                llm_pool=_dummy_pool(5),
                n_games=1,
                cfg=cfg,
            )

        assert mock_dr.called, "DiplomacyRunner must be used when cfg.enabled=True"
        assert not mock_mar.called, "MultiAgentRunner must NOT be used when cfg.enabled=True"

    def test_when_cfg_enabled_false_then_multi_agent_runner_is_used(self):
        from training.social_eval import run_social_eval  # noqa: PLC0415

        cfg = DiplomacyConfig(enabled=False)
        with (
            patch("training.social_eval.DiplomacyRunner") as mock_dr,
            patch("training.social_eval.MultiAgentRunner") as mock_mar,
        ):
            mock_mar.return_value.run.return_value = _make_game_result(0)

            run_social_eval(
                learner=_dummy_learner(),
                llm_pool=_dummy_pool(5),
                n_games=1,
                cfg=cfg,
            )

        assert mock_mar.called, "MultiAgentRunner must be used when cfg.enabled=False"
        assert not mock_dr.called, "DiplomacyRunner must NOT be used when cfg.enabled=False"


# ---------------------------------------------------------------------------
# Criterion: cross-game reputation carry via shared ReputationBook
# ---------------------------------------------------------------------------


class TestCrossGameReputationCarry:
    """
    A betrayal recorded in game i must be visible (via the shared book) in game i+1.
    The fix requires that the *same* ReputationBook is passed to each DiplomacyRunner.
    We verify this by spying on the book instance that reaches the runner constructor.
    """

    def test_when_diplomacy_enabled_same_reputation_book_is_passed_to_every_game_runner(self):
        """The shared book must be passed to each DiplomacyRunner instantiation."""
        from training.social_eval import run_social_eval  # noqa: PLC0415

        cfg = DiplomacyConfig(enabled=True)
        book_instances_passed: list[object] = []

        def spy_runner(*args, **kwargs):
            rep = kwargs.get("reputation")
            book_instances_passed.append(id(rep) if rep is not None else None)
            m = MagicMock()
            m.run.return_value = _make_game_result(0)
            m.call_count = 0
            return m

        with patch("training.social_eval.DiplomacyRunner", side_effect=spy_runner):
            run_social_eval(
                learner=_dummy_learner(),
                llm_pool=_dummy_pool(5),
                n_games=3,
                cfg=cfg,
            )

        assert len(book_instances_passed) == 3, "Expected 3 DiplomacyRunner instantiations"
        assert len(set(book_instances_passed)) == 1, (
            "All games must share the SAME ReputationBook instance for cross-game carry. "
            f"Got {len(set(book_instances_passed))} distinct ids."
        )

    def test_when_betrayal_recorded_via_shared_book_then_betrayal_count_persists(self):
        """After a betrayal in game 1, betrayal_count(1) > 0 across the whole session."""
        from training.social_eval import run_social_eval  # noqa: PLC0415

        cfg = DiplomacyConfig(enabled=True)
        captured_books: list[ReputationBook] = []

        def spy_runner(*args, **kwargs):
            book = kwargs.get("reputation")
            captured_books.append(book)
            m = MagicMock()
            m.run.return_value = _make_game_result(0)
            m.call_count = 0
            if book is not None and len(captured_books) == 1:
                # Simulate a betrayal in the first game
                book.record_betrayal(1)
            return m

        with patch("training.social_eval.DiplomacyRunner", side_effect=spy_runner):
            run_social_eval(
                learner=_dummy_learner(),
                llm_pool=_dummy_pool(5),
                n_games=2,
                cfg=cfg,
            )

        assert len(captured_books) == 2
        # The same book instance — betrayal from game 1 must be visible after game 2
        assert captured_books[0] is captured_books[1], "Book must be shared across games"
        assert captured_books[1].betrayal_count(1) == 1, (
            "Betrayal from game 1 must be visible in the shared book after game 2"
        )


# ---------------------------------------------------------------------------
# Criterion: determinism — mocked LLM + fixed seed => identical aggregate result
# ---------------------------------------------------------------------------


class TestDeterminismWithMockedLLMAndFixedSeed:
    """
    Running run_social_eval twice with the same seed and same mock outputs
    must produce identical SocialEvalResult values.
    The diplomacy path (DiplomacyRunner) must be exercised — not a no-op MAR.
    """

    def _run_once(self, seed, cfg, per_game_calls, winners):
        from training.social_eval import run_social_eval  # noqa: PLC0415

        with patch("training.social_eval.DiplomacyRunner") as mock_cls:
            instance = mock_cls.return_value
            instance.run.side_effect = [_make_game_result(w, n_turns=10) for w in winners]
            instance.call_count = per_game_calls
            return run_social_eval(
                learner=_dummy_learner(),
                llm_pool=_dummy_pool(5),
                n_games=len(winners),
                seed=seed,
                cfg=cfg,
            )

    def test_when_mocked_llm_and_same_seed_then_results_are_identical(self):
        cfg = DiplomacyConfig(enabled=True, n_rounds=1)
        winners = [0, 1, 0, None, 0]
        per_game_calls = 5

        result_a = self._run_once(seed=42, cfg=cfg, per_game_calls=per_game_calls, winners=winners)
        result_b = self._run_once(seed=42, cfg=cfg, per_game_calls=per_game_calls, winners=winners)

        assert result_a.learner_win_rate == result_b.learner_win_rate
        assert result_a.total_negotiation_calls == result_b.total_negotiation_calls
        assert result_a.draw_rate == result_b.draw_rate
