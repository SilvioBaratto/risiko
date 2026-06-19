"""
Source-blind example tests for issue #104:
  feat(evaluation): social-eval harness + CLI command (training/social_eval.py)

Every test is derived from the acceptance-criteria text only.
No implementation source was read during authoring.

Verifiable criteria covered
────────────────────────────
[UNIT] get_legal_actions() is never empty in PHASE_TRADE under a forced trade.
[UNIT] run_social_eval reports learner_win_rate with a Wilson CI (reusing wilson_ci).
[UNIT] total_negotiation_calls is bounded by n_games * n_rounds * n_speakers and is logged.
[UNIT] One ReputationBook is shared across the n_games loop and reset per run_social_eval
       call; cross-game reputation visibly carries between games.
[UNIT] A social-eval CLI command parses, loads the pool, prints win-rate + call cost,
       and appears in --help.

Skipped (oracle: NOT VERIFIABLE or boilerplate gate)
─────────────────────────────────────────────────────
- Forced-trade / skip-action illegality (no standalone unit signal in the oracle)
- Mocked LLM + fixed seed => identical aggregate result
- obs_dim stays 137
- All tests pass / SOLID / ruff / coverage gates
"""

import logging
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers (shared across test classes)
# ──────────────────────────────────────────────────────────────────────────────


def _make_game_result(winner):
    """Minimal GameResult stub; winner=None means the game timed out."""
    r = MagicMock()
    r.winner = winner
    r.n_turns = 10
    return r


def _mock_runner_cls(winner_sequence):
    """
    Return a MagicMock that looks like MultiAgentRunner and yields one
    GameResult per run() call, driven by winner_sequence.
    """
    cls = MagicMock()
    cls.return_value.run.side_effect = [_make_game_result(w) for w in winner_sequence]
    return cls


def _dummy_learner():
    m = MagicMock()
    m.act.return_value = 0
    return m


def _dummy_pool(n=5):
    agents = []
    for _ in range(n):
        a = MagicMock()
        a.act.return_value = 0
        agents.append(a)
    return agents


# ──────────────────────────────────────────────────────────────────────────────
# Criterion: get_legal_actions() is never empty in PHASE_TRADE under a forced trade
# ──────────────────────────────────────────────────────────────────────────────


class TestPhaseTradeLegalActionsNonEmpty:
    """
    Design choice (from spec): a player holding any valid card set is in a
    forced-trade situation.  The simplest valid set is three cards of the same
    type (3-of-a-kind).  PHASE_TRADE is a module-level integer constant in
    src/env.py; state.hands[player] is the list of card-type ints for that player.
    """

    def test_when_player_holds_three_of_a_kind_in_phase_trade_then_legal_actions_not_empty(self):
        from src.env import PHASE_TRADE, RisikoEnv  # noqa: PLC0415

        env = RisikoEnv(n_players=2)
        env.reset(seed=0)
        state = env.state
        # Give current player three infantry card IDs (IDs 0,1,2 all map to symbol 0).
        state.cards[state.current_player] = [0, 1, 2]
        state.phase = PHASE_TRADE

        actions = env.get_legal_actions()

        assert len(actions) > 0, (
            "get_legal_actions() must not be empty when player holds a valid set "
            "and env is in PHASE_TRADE (forced trade)."
        )

    def test_when_player_holds_one_of_each_in_phase_trade_then_legal_actions_not_empty(self):
        """1-of-each (infantry/cavalry/cannon): card IDs 0, 14, 28 → symbols 0, 1, 2."""
        from src.env import PHASE_TRADE, RisikoEnv  # noqa: PLC0415

        env = RisikoEnv(n_players=2)
        env.reset(seed=1)
        state = env.state
        state.cards[state.current_player] = [0, 14, 28]  # infantry, cavalry, artillery
        state.phase = PHASE_TRADE

        assert len(env.get_legal_actions()) > 0

    @given(base_id=st.integers(min_value=0, max_value=13))
    @settings(max_examples=30)
    def test_when_any_three_infantry_cards_are_held_then_legal_actions_is_not_empty(self, base_id):
        """
        Invariant: any 3 infantry-type card IDs (0–13) form a valid 3-of-a-kind
        and must produce at least one legal action in PHASE_TRADE.
        IDs are distinct so the hand is valid (base_id, base_id+14, base_id+28 use
        the same card: for uniqueness we use consecutive IDs within the infantry range).
        """
        from src.env import PHASE_TRADE, RisikoEnv  # noqa: PLC0415

        # Three distinct IDs all in infantry range (0-13), cycling to stay in range.
        ids = [base_id % 14, (base_id + 1) % 14, (base_id + 2) % 14]
        # If any two coincide (e.g. base_id=12 gives [12,13,0]), they are still valid
        # as unique positions in the hand list; _is_valid_trade checks index uniqueness.
        env = RisikoEnv(n_players=2)
        env.reset(seed=42)
        state = env.state
        state.cards[state.current_player] = ids
        state.phase = PHASE_TRADE

        assert len(env.get_legal_actions()) > 0, f"Expected non-empty legal actions for ids={ids}"


# ──────────────────────────────────────────────────────────────────────────────
# Criterion: run_social_eval reports learner_win_rate with a Wilson CI
# ──────────────────────────────────────────────────────────────────────────────


class TestSocialEvalWinRateAndCI:
    """
    run_social_eval must return a result object with:
      - learner_win_rate  float in [0, 1]
      - ci_low            float in [0, 1]
      - ci_high           float in [0, 1]
    such that  ci_low <= learner_win_rate <= ci_high.

    The criterion says Wilson CI is *reused* from an existing wilson_ci helper,
    so we test the observable output contract, not the internal call.

    Design choice: the learner is the first participant (index 0).  Games where
    winner == 0 count as learner wins; winner == None (timeout) counts as no win.
    """

    def _run(self, winners, n_rounds=1, n_speakers=5):
        from training.social_eval import run_social_eval  # noqa: PLC0415

        learner = _dummy_learner()
        pool = _dummy_pool(n_speakers)
        with patch(
            "training.social_eval.MultiAgentRunner",
            _mock_runner_cls(winners),
        ):
            return run_social_eval(
                learner=learner,
                llm_pool=pool,
                n_games=len(winners),
                n_rounds=n_rounds,
            )

    def test_when_social_eval_completes_then_result_has_learner_win_rate_field(self):
        result = self._run([0, 1, 0])
        assert hasattr(result, "learner_win_rate"), (
            "run_social_eval result must expose a learner_win_rate attribute."
        )

    def test_when_social_eval_completes_then_learner_win_rate_is_in_unit_interval(self):
        result = self._run([0, 1, None])
        assert 0.0 <= result.learner_win_rate <= 1.0

    def test_when_learner_wins_all_games_then_win_rate_is_one(self):
        result = self._run([0, 0, 0])
        assert result.learner_win_rate == pytest.approx(1.0)

    def test_when_learner_wins_no_games_then_win_rate_is_zero(self):
        result = self._run([1, 2, 3])
        assert result.learner_win_rate == pytest.approx(0.0)

    def test_when_social_eval_completes_then_result_has_ci_low_and_ci_high(self):
        result = self._run([0, 0, 1])
        assert hasattr(result, "ci_low"), "Result must expose ci_low (Wilson CI lower bound)."
        assert hasattr(result, "ci_high"), "Result must expose ci_high (Wilson CI upper bound)."

    def test_when_social_eval_completes_then_ci_bounds_are_in_unit_interval(self):
        result = self._run([0, 1, 0, 1, 0])
        assert 0.0 <= result.ci_low <= 1.0
        assert 0.0 <= result.ci_high <= 1.0

    def test_when_social_eval_completes_then_ci_is_ordered(self):
        result = self._run([0, 1, 0, 1, 0])
        assert result.ci_low <= result.ci_high

    def test_when_social_eval_completes_then_win_rate_lies_within_ci(self):
        result = self._run([0, 0, 1, None, 0])  # 3 wins out of 5 games
        assert result.ci_low <= result.learner_win_rate <= result.ci_high, (
            "Wilson CI must bracket the observed win-rate."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Criterion: total_negotiation_calls bounded by n_games * n_rounds * n_speakers
#            and is logged
# ──────────────────────────────────────────────────────────────────────────────


class TestNegotiationCallsBounded:
    """
    total_negotiation_calls is the count of LLM negotiation invocations across
    the full eval run.  The spec gives an explicit upper bound:
        total_negotiation_calls <= n_games * n_rounds * n_speakers
    where n_speakers = len(llm_pool).

    The value must also appear in log output (logging.INFO level or below).
    """

    def _run(self, n_games, n_rounds, n_speakers=5):
        from training.social_eval import run_social_eval  # noqa: PLC0415

        learner = _dummy_learner()
        pool = _dummy_pool(n_speakers)
        with patch(
            "training.social_eval.MultiAgentRunner",
            _mock_runner_cls([0] * n_games),
        ):
            return run_social_eval(
                learner=learner,
                llm_pool=pool,
                n_games=n_games,
                n_rounds=n_rounds,
            )

    def test_when_social_eval_runs_then_result_has_total_negotiation_calls_field(self):
        result = self._run(n_games=2, n_rounds=1)
        assert hasattr(result, "total_negotiation_calls"), (
            "run_social_eval result must expose total_negotiation_calls."
        )

    def test_when_social_eval_runs_then_total_negotiation_calls_is_non_negative(self):
        result = self._run(n_games=2, n_rounds=1)
        assert result.total_negotiation_calls >= 0

    def test_when_social_eval_runs_then_total_calls_does_not_exceed_upper_bound(self):
        n_games, n_rounds, n_speakers = 3, 2, 5
        result = self._run(n_games=n_games, n_rounds=n_rounds, n_speakers=n_speakers)
        upper = n_games * n_rounds * n_speakers
        bound_desc = f"n_games({n_games})*n_rounds({n_rounds})*n_speakers({n_speakers})={upper}"
        assert result.total_negotiation_calls <= upper, (
            f"total_negotiation_calls={result.total_negotiation_calls} exceeds {bound_desc}"
        )

    def test_when_n_games_is_zero_then_total_negotiation_calls_is_zero(self):
        result = self._run(n_games=0, n_rounds=1)
        assert result.total_negotiation_calls == 0

    def test_when_social_eval_runs_then_total_negotiation_calls_is_logged(self, caplog):
        with caplog.at_level(logging.INFO):
            result = self._run(n_games=2, n_rounds=1)
        log_text = caplog.text
        # The criterion says the value "is logged" — either the word negotiation
        # or the numeric count itself must appear.
        appears = (
            "negotiation" in log_text.lower() or str(result.total_negotiation_calls) in log_text
        )
        assert appears, (
            f"total_negotiation_calls must be written to the log.  Got log: {log_text!r}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Criterion: One ReputationBook shared across n_games, reset per run_social_eval call
# ──────────────────────────────────────────────────────────────────────────────


class TestReputationBookLifecycle:
    """
    The spec states:
      "One ReputationBook is shared across the n_games loop and reset per
       run_social_eval call; cross-game reputation visibly carries between games."

    Observable contract
    ────────────────────
    1. Exactly one ReputationBook is instantiated per run_social_eval call.
    2. A second call instantiates a different (new) ReputationBook.
    3. Within a single call the same book instance is updated after each game.

    We spy via a local SpyBook class injected as a patch target so that the real
    ReputationBook constructor is never invoked; the patch target is the name as
    it appears inside training/social_eval.py.
    """

    def _run_with_spy_book(self, n_games, spy_cls):
        from training.social_eval import run_social_eval  # noqa: PLC0415

        learner = _dummy_learner()
        pool = _dummy_pool(5)
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "training.social_eval.MultiAgentRunner",
                    _mock_runner_cls([0] * n_games),
                )
            )
            stack.enter_context(patch("training.social_eval.ReputationBook", spy_cls))
            return run_social_eval(
                learner=learner,
                llm_pool=pool,
                n_games=n_games,
                n_rounds=1,
            )

    def test_when_social_eval_runs_then_exactly_one_reputation_book_is_created(self):
        instances = []

        class SpyBook:
            def __init__(self, *args, **kwargs):
                instances.append(self)

            def update(self, *args, **kwargs):
                pass

            def reset(self, *args, **kwargs):
                pass

        self._run_with_spy_book(n_games=3, spy_cls=SpyBook)

        assert len(instances) == 1, (
            f"Expected exactly 1 ReputationBook per run_social_eval call, "
            f"but {len(instances)} were instantiated."
        )

    def test_when_social_eval_called_twice_then_each_call_gets_fresh_reputation_book(self):
        instances = []

        class SpyBook:
            def __init__(self, *args, **kwargs):
                instances.append(self)

            def update(self, *args, **kwargs):
                pass

            def reset(self, *args, **kwargs):
                pass

        self._run_with_spy_book(n_games=2, spy_cls=SpyBook)
        self._run_with_spy_book(n_games=2, spy_cls=SpyBook)

        assert len(instances) == 2, (
            "Each run_social_eval call must create its own fresh ReputationBook; "
            f"got {len(instances)} total instantiations across 2 calls."
        )
        assert instances[0] is not instances[1], (
            "The two calls must not share the same ReputationBook instance."
        )

    def test_when_games_run_then_same_book_instance_is_used_across_all_games(self):
        """Cross-game reputation carries: the book is updated after every game."""
        update_ids: list[int] = []

        class SpyBook:
            def __init__(self, *args, **kwargs):
                pass

            def update(self_inner, *args, **kwargs):  # noqa: N805
                update_ids.append(id(self_inner))

            def reset(self, *args, **kwargs):
                pass

        self._run_with_spy_book(n_games=3, spy_cls=SpyBook)

        assert len(update_ids) >= 3, (
            "ReputationBook.update() must be called at least once per game "
            f"(n_games=3).  Called {len(update_ids)} time(s)."
        )
        unique = set(update_ids)
        assert len(unique) == 1, (
            f"All ReputationBook.update() calls must originate from the same instance "
            f"(cross-game carry).  Got {len(unique)} distinct id(s)."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Criterion: social-eval CLI command parses, prints win-rate + call cost, --help
# ──────────────────────────────────────────────────────────────────────────────


class TestSocialEvalCLI:
    """
    The spec requires a `social-eval` sub-command in the Typer CLI
    (risiko_rl/cli.py).  It must:
      - appear in the top-level --help output
      - have its own --help that exits cleanly
      - print win-rate and call cost when invoked

    We patch run_social_eval so the test never touches the network or the env.
    """

    @staticmethod
    def _mock_result(win_rate=0.5, ci_low=0.3, ci_high=0.7, calls=6):
        r = MagicMock()
        r.learner_win_rate = win_rate
        r.ci_low = ci_low
        r.ci_high = ci_high
        r.total_negotiation_calls = calls
        return r

    def test_when_help_is_invoked_then_social_eval_command_appears_in_output(self):
        from typer.testing import CliRunner  # noqa: PLC0415

        from risiko_rl.cli import app as main  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0, f"--help failed:\n{result.output}"
        assert "social-eval" in result.output, (
            "social-eval must appear in the top-level --help listing."
        )

    def test_when_social_eval_help_is_invoked_then_it_exits_with_zero(self):
        from typer.testing import CliRunner  # noqa: PLC0415

        from risiko_rl.cli import app as main  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(main, ["social-eval", "--help"])

        assert result.exit_code == 0, f"social-eval --help must exit 0.  Got:\n{result.output}"

    def test_when_social_eval_runs_via_cli_then_output_contains_win_rate(self):
        from typer.testing import CliRunner  # noqa: PLC0415

        from risiko_rl.cli import app as main  # noqa: PLC0415

        runner = CliRunner()
        with patch(
            "training.social_eval.run_social_eval",
            return_value=self._mock_result(win_rate=0.4),
        ):
            result = runner.invoke(main, ["social-eval", "--n-games", "2"])

        out_lower = result.output.lower()
        assert "win" in out_lower or "rate" in out_lower, (
            f"CLI output must mention win-rate.  Got:\n{result.output}"
        )

    def test_when_social_eval_runs_via_cli_then_output_contains_call_cost(self):
        from typer.testing import CliRunner  # noqa: PLC0415

        from risiko_rl.cli import app as main  # noqa: PLC0415

        runner = CliRunner()
        with patch(
            "training.social_eval.run_social_eval",
            return_value=self._mock_result(calls=12),
        ):
            result = runner.invoke(main, ["social-eval", "--n-games", "2"])

        out_lower = result.output.lower()
        assert "call" in out_lower or "negotiat" in out_lower or "12" in result.output, (
            f"CLI output must print call cost (total_negotiation_calls).  Got:\n{result.output}"
        )

    def test_when_social_eval_runs_via_cli_then_exit_code_is_zero(self):
        from typer.testing import CliRunner  # noqa: PLC0415

        from risiko_rl.cli import app as main  # noqa: PLC0415

        runner = CliRunner()
        with patch(
            "training.social_eval.run_social_eval",
            return_value=self._mock_result(),
        ):
            result = runner.invoke(main, ["social-eval", "--n-games", "2"])

        assert result.exit_code == 0, f"social-eval must exit 0 on success.  Got:\n{result.output}"
