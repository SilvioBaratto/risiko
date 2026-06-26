"""Example tests for seeded strategy↔model bijection and per-game plan — Issue #118.

Authored source-blind from acceptance criteria only; no implementation source was read.
All tests are expected to FAIL (Red) until the implementation is written.

Criteria covered (from oracle report — UNIT-verifiable only):
  - assign_strategies_to_models(strategies, models, *, seed) returns a bijection
    (every strategy mapped; every model used exactly once)
  - Identical for the same (seed, game_index); differs across game indices
    (not all assignments identical)
  - build_game_plan(config, game_index) returns a GamePlan with derived
    per-game seed and assignment
"""

from __future__ import annotations

import pathlib

import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

CONFIG_PATH = pathlib.Path("config/tournament.yaml")

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: pathlib.Path, data: dict) -> pathlib.Path:
    path.write_text(yaml.dump(data))
    return path


def _base_config_data() -> dict:
    return {
        "name": "test_tournament_118",
        "seed": 7,
        "n_games": 12,
        "max_concurrency": 2,
        "n_rounds": 2,
        "strategies": _VALID_STRATEGIES,
        "models": _VALID_MODELS,
        "temperature": 0.7,
        "top_p": 0.9,
    }


def _load_config(tmp_path: pathlib.Path):
    """Write a minimal valid tournament.yaml and parse it into a TournamentConfig."""
    from training.tournament import load_tournament_config

    cfg_file = _write_yaml(tmp_path / "tournament.yaml", _base_config_data())
    return load_tournament_config(cfg_file)


# ---------------------------------------------------------------------------
# TestAssignStrategiesToModels
# Criterion: assign_strategies_to_models(strategies, models, *, seed) returns a
#            bijection — every strategy mapped; every model used exactly once.
# ---------------------------------------------------------------------------


class TestAssignStrategiesToModels:
    """assign_strategies_to_models must always produce a strict bijection."""

    def test_when_assigned_then_all_strategies_appear_as_keys(self):
        """Every strategy slug must be a key in the returned mapping."""
        from training.tournament import assign_strategies_to_models

        result = assign_strategies_to_models(_VALID_STRATEGIES, _VALID_MODELS, seed=0)
        assert set(result.keys()) == set(_VALID_STRATEGIES)

    def test_when_assigned_then_all_models_appear_as_values(self):
        """Every model name must appear as a value in the returned mapping (surjective)."""
        from training.tournament import assign_strategies_to_models

        result = assign_strategies_to_models(_VALID_STRATEGIES, _VALID_MODELS, seed=0)
        assert set(result.values()) == set(_VALID_MODELS)

    def test_when_assigned_then_no_model_used_more_than_once(self):
        """Values are unique — every model is used at most once (injective)."""
        from training.tournament import assign_strategies_to_models

        result = assign_strategies_to_models(_VALID_STRATEGIES, _VALID_MODELS, seed=0)
        values = list(result.values())
        assert len(values) == len(set(values))

    def test_when_assigned_then_mapping_length_equals_strategy_count(self):
        """Length of the returned mapping equals the number of input strategies."""
        from training.tournament import assign_strategies_to_models

        result = assign_strategies_to_models(_VALID_STRATEGIES, _VALID_MODELS, seed=42)
        assert len(result) == len(_VALID_STRATEGIES)

    def test_when_same_seed_used_twice_then_results_are_identical(self):
        """Same seed ⟹ identical assignment (deterministic)."""
        from training.tournament import assign_strategies_to_models

        first = assign_strategies_to_models(_VALID_STRATEGIES, _VALID_MODELS, seed=99)
        second = assign_strategies_to_models(_VALID_STRATEGIES, _VALID_MODELS, seed=99)
        assert first == second

    def test_when_different_seeds_used_then_at_least_one_assignment_differs(self):
        """Different seeds must not all collapse to an identical assignment (variety exists)."""
        from training.tournament import assign_strategies_to_models

        assignments = [
            frozenset(assign_strategies_to_models(_VALID_STRATEGIES, _VALID_MODELS, seed=s).items())
            for s in range(20)
        ]
        assert len(set(assignments)) > 1

    def test_when_assigned_then_each_value_is_one_of_the_input_models(self):
        """No invented models appear: every value in the result ∈ the provided models list."""
        from training.tournament import assign_strategies_to_models

        result = assign_strategies_to_models(_VALID_STRATEGIES, _VALID_MODELS, seed=5)
        for model in result.values():
            assert model in _VALID_MODELS

    def test_when_assigned_then_each_key_is_one_of_the_input_strategies(self):
        """No invented strategies appear: every key in the result ∈ the provided strategies list."""
        from training.tournament import assign_strategies_to_models

        result = assign_strategies_to_models(_VALID_STRATEGIES, _VALID_MODELS, seed=5)
        for strategy in result:
            assert strategy in _VALID_STRATEGIES

    # ── Hypothesis: bijection invariant for all valid seeds ──────────────────

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_when_any_seed_used_then_result_is_always_a_bijection(self, seed):
        """Invariant: assign_strategies_to_models always returns a bijection for any seed."""
        from training.tournament import assign_strategies_to_models

        result = assign_strategies_to_models(_VALID_STRATEGIES, _VALID_MODELS, seed=seed)
        assert set(result.keys()) == set(_VALID_STRATEGIES)
        assert set(result.values()) == set(_VALID_MODELS)
        values = list(result.values())
        assert len(values) == len(set(values))

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_when_same_seed_applied_twice_then_results_are_equal(self, seed):
        """Invariant: determinism holds for any seed in the valid domain."""
        from training.tournament import assign_strategies_to_models

        first = assign_strategies_to_models(_VALID_STRATEGIES, _VALID_MODELS, seed=seed)
        second = assign_strategies_to_models(_VALID_STRATEGIES, _VALID_MODELS, seed=seed)
        assert first == second


# ---------------------------------------------------------------------------
# TestBuildGamePlan
# Criterion: build_game_plan(config, game_index) returns a GamePlan with
#            derived per-game seed and assignment.
# Sub-criterion: Identical for same (seed, game_index); differs across indices.
# ---------------------------------------------------------------------------


class TestBuildGamePlan:
    """build_game_plan must return a well-formed GamePlan with seed and assignment."""

    def test_when_build_game_plan_called_then_returns_game_plan_instance(self, tmp_path):
        """Return type is GamePlan (not None, not a plain dict)."""
        from training.tournament import GamePlan, build_game_plan

        cfg = _load_config(tmp_path)
        plan = build_game_plan(cfg, game_index=0)
        assert isinstance(plan, GamePlan)

    def test_when_build_game_plan_called_then_plan_has_game_index(self, tmp_path):
        """The returned GamePlan stores the game_index that was passed in."""
        from training.tournament import build_game_plan

        cfg = _load_config(tmp_path)
        plan = build_game_plan(cfg, game_index=7)
        assert plan.game_index == 7

    def test_when_build_game_plan_called_with_zero_then_game_index_is_zero(self, tmp_path):
        """game_index=0 is stored exactly as 0 in the plan."""
        from training.tournament import build_game_plan

        cfg = _load_config(tmp_path)
        plan = build_game_plan(cfg, game_index=0)
        assert plan.game_index == 0

    def test_when_build_game_plan_called_then_plan_has_integer_seed(self, tmp_path):
        """GamePlan exposes a per-game seed that is an integer derived from the config."""
        from training.tournament import build_game_plan

        cfg = _load_config(tmp_path)
        plan = build_game_plan(cfg, game_index=0)
        assert isinstance(plan.seed, int)

    def test_when_build_game_plan_called_then_plan_has_assignment_with_six_entries(self, tmp_path):
        """GamePlan exposes a strategy→model assignment covering all 6 strategies."""
        from training.tournament import build_game_plan

        cfg = _load_config(tmp_path)
        plan = build_game_plan(cfg, game_index=0)
        assert len(plan.assignment) == 6

    def test_when_build_game_plan_called_then_assignment_strategies_match_config(self, tmp_path):
        """Assignment keys are exactly the config's strategy slugs (no extras or missing)."""
        from training.tournament import build_game_plan

        cfg = _load_config(tmp_path)
        plan = build_game_plan(cfg, game_index=3)
        assert set(plan.assignment.keys()) == set(_VALID_STRATEGIES)

    def test_when_build_game_plan_called_then_assignment_models_match_config(self, tmp_path):
        """Assignment values are exactly the config's models (no extras or missing)."""
        from training.tournament import build_game_plan

        cfg = _load_config(tmp_path)
        plan = build_game_plan(cfg, game_index=3)
        assert set(plan.assignment.values()) == set(_VALID_MODELS)

    def test_when_build_game_plan_called_then_assignment_has_no_duplicate_models(self, tmp_path):
        """Each model appears at most once in the assignment (injective)."""
        from training.tournament import build_game_plan

        cfg = _load_config(tmp_path)
        plan = build_game_plan(cfg, game_index=1)
        values = list(plan.assignment.values())
        assert len(values) == len(set(values))

    def test_when_same_config_and_index_used_then_identical_seed_returned(self, tmp_path):
        """build_game_plan is deterministic: same (config, game_index) ⟹ same per-game seed."""
        from training.tournament import build_game_plan

        cfg = _load_config(tmp_path)
        plan_a = build_game_plan(cfg, game_index=5)
        plan_b = build_game_plan(cfg, game_index=5)
        assert plan_a.seed == plan_b.seed

    def test_when_same_config_and_index_used_then_identical_assignment_returned(self, tmp_path):
        """build_game_plan is deterministic: same (config, game_index) ⟹ same assignment."""
        from training.tournament import build_game_plan

        cfg = _load_config(tmp_path)
        plan_a = build_game_plan(cfg, game_index=5)
        plan_b = build_game_plan(cfg, game_index=5)
        assert plan_a.assignment == plan_b.assignment

    def test_when_different_game_indices_used_then_seeds_are_distinct(self, tmp_path):
        """Different game_index values must produce distinct per-game seeds."""
        from training.tournament import build_game_plan

        cfg = _load_config(tmp_path)
        seeds = {build_game_plan(cfg, game_index=i).seed for i in range(10)}
        assert len(seeds) == 10

    def test_when_different_game_indices_used_then_not_all_assignments_identical(self, tmp_path):
        """Over many game indices, assignments must not all be the same permutation."""
        from training.tournament import build_game_plan

        cfg = _load_config(tmp_path)
        assignments = [
            frozenset(build_game_plan(cfg, game_index=i).assignment.items()) for i in range(20)
        ]
        assert len(set(assignments)) > 1

    # ── Hypothesis: bijection + determinism invariants for any game_index ────

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(game_index=st.integers(min_value=0, max_value=9999))
    def test_when_any_game_index_used_then_plan_assignment_is_always_a_bijection(
        self, game_index, tmp_path
    ):
        """Invariant: build_game_plan always returns a full bijection in .assignment."""
        from training.tournament import build_game_plan

        cfg = _load_config(tmp_path)
        plan = build_game_plan(cfg, game_index=game_index)
        assert set(plan.assignment.keys()) == set(_VALID_STRATEGIES)
        assert set(plan.assignment.values()) == set(_VALID_MODELS)
        values = list(plan.assignment.values())
        assert len(values) == len(set(values))

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(game_index=st.integers(min_value=0, max_value=9999))
    def test_when_same_game_index_used_twice_then_plan_seed_and_assignment_are_identical(
        self, game_index, tmp_path
    ):
        """Invariant: determinism holds for any game_index in [0, 9999]."""
        from training.tournament import build_game_plan

        cfg = _load_config(tmp_path)
        plan_a = build_game_plan(cfg, game_index=game_index)
        plan_b = build_game_plan(cfg, game_index=game_index)
        assert plan_a.seed == plan_b.seed
        assert plan_a.assignment == plan_b.assignment


# ---------------------------------------------------------------------------
# TestUniformDistribution
# Criterion: each strategy's assigned-model distribution is ~uniform over many
#            seeded games (chi-square tolerance test, no scipy required).
# ---------------------------------------------------------------------------


class TestUniformDistribution:
    """Over many seeded games every model should appear ~equally often per strategy."""

    def test_when_many_seeds_used_then_per_strategy_model_distribution_is_uniform(self):
        """Chi-square statistic must be within critical value at alpha=0.05 (df=5)."""
        import numpy as np

        from training.tournament import assign_strategies_to_models

        n_games = 360
        k = len(_VALID_STRATEGIES)  # 6
        expected = n_games / k  # 60 per (strategy, model) cell

        counts = {s: dict.fromkeys(_VALID_MODELS, 0) for s in _VALID_STRATEGIES}
        for seed in range(n_games):
            for strategy, model in assign_strategies_to_models(
                _VALID_STRATEGIES, _VALID_MODELS, seed=seed
            ).items():
                counts[strategy][model] += 1

        # Chi-square critical value at alpha=0.05, df=k-1=5 is ~11.07
        chi2_critical = 11.071
        for strategy in _VALID_STRATEGIES:
            observed = np.array([counts[strategy][m] for m in _VALID_MODELS], dtype=float)
            chi2 = float(np.sum((observed - expected) ** 2 / expected))
            counts_by_model = dict(zip(_VALID_MODELS, observed.astype(int), strict=True))
            assert chi2 < chi2_critical, (
                f"Strategy '{strategy}': chi2={chi2:.3f} > critical {chi2_critical}; "
                f"counts={counts_by_model}"
            )
