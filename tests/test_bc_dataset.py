"""
Source-blind example tests for training.bc_dataset.generate_bc_dataset (issue #80).

All tests derived exclusively from the acceptance criteria for issue #80:
"feat: generate_bc_dataset orchestrator — HeuristicAgent self-play → sharded dataset".
No implementation source has been read.  Tests are written in the Red phase and
are expected to fail until the implementation satisfies each criterion.

Skipped (per oracle — NOT VERIFIABLE):
  - Byte-identical shards on same seed (runtime non-determinism across runs)
  - Progress/stats logging (no concrete unit check; stats dataclass is recommended
    but not asserted here since the oracle marks the criterion non-verifiable)
  - pytest -m "not integration" gate (suite-level, not per-criterion)
  - SOLID / code-quality prose
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

OBS_DIM = 137


# ── Minimal config stubs (derived from acceptance-criteria field names only) ─


@dataclasses.dataclass(frozen=True)
class _BCStub:
    n_games: int
    dataset_dir: str
    shard_size: int = 50
    explore_eps: float = 0.1
    seed: int = 0
    n_players: int = 6
    max_turns: int = 200


@dataclasses.dataclass(frozen=True)
class _PPOStub:
    gamma: float = 0.99


@dataclasses.dataclass(frozen=True)
class _CfgStub:
    bc: _BCStub
    ppo: _PPOStub = dataclasses.field(default_factory=_PPOStub)


def _cfg(
    dataset_dir,
    *,
    n_games: int = 2,
    shard_size: int = 50,
    explore_eps: float = 0.1,
    seed: int = 0,
    gamma: float = 0.99,
) -> _CfgStub:
    return _CfgStub(
        bc=_BCStub(
            n_games=n_games,
            dataset_dir=str(dataset_dir),
            shard_size=shard_size,
            explore_eps=explore_eps,
            seed=seed,
        ),
        ppo=_PPOStub(gamma=gamma),
    )


# ── Module accessibility ─────────────────────────────────────────────────────


def test_when_module_imported_then_generate_bc_dataset_is_accessible():
    """Criterion: training/bc_dataset.py exposes a generate_bc_dataset(cfg) entry point."""
    from training.bc_dataset import generate_bc_dataset  # noqa: F401


# ── No LLM / network calls ───────────────────────────────────────────────────


def test_when_module_imported_then_no_llm_imports_are_present():
    """
    Criterion: generate_bc_dataset makes no LLM / network calls.
    Proxy: the module source contains no references to LLMOpponent, ollama,
    or openai — sufficient to detect accidental LLM wiring at import time.
    """
    import training.bc_dataset as _mod

    src = inspect.getsource(_mod)
    assert "LLMOpponent" not in src, "bc_dataset must not reference LLMOpponent"
    assert "ollama" not in src.lower(), "bc_dataset must not reference ollama"
    assert "openai" not in src.lower(), "bc_dataset must not reference openai"


# ── Output files are created ─────────────────────────────────────────────────


def test_when_dataset_generated_then_at_least_one_npz_shard_is_created(tmp_path):
    """Criterion: output is sharded .npz files under cfg.bc.dataset_dir."""
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2))
    shards = list(tmp_path.glob("shard_*.npz"))
    assert len(shards) >= 1, "Expected ≥1 .npz shard under dataset_dir"


def test_when_dataset_generated_then_manifest_json_is_created(tmp_path):
    """Criterion: manifest.json written alongside shards."""
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2))
    assert (tmp_path / "manifest.json").exists(), "manifest.json must exist after generation"


# ── manifest.json fields ─────────────────────────────────────────────────────


def test_when_manifest_read_then_all_required_fields_are_present(tmp_path):
    """Criterion: manifest records counts, obs_dim, seed, and config hash."""
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2, seed=7))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for field in ("total_rows", "obs_dim", "seed", "config_hash"):
        assert field in manifest, f"manifest missing required field '{field}'"


def test_when_manifest_read_then_obs_dim_is_137(tmp_path):
    """Criterion: obs columns == 137."""
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["obs_dim"] == OBS_DIM


def test_when_manifest_read_then_seed_matches_config_seed(tmp_path):
    """Criterion: every run logs its seed."""
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2, seed=42))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["seed"] == 42


# ── Shard shape: obs == (n_rows, 137) float32 ────────────────────────────────


def test_when_shard_loaded_then_obs_has_137_columns(tmp_path):
    """Criterion: every decision records the 137-dim flatten_obs."""
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2))
    shard = next(tmp_path.glob("shard_*.npz"))
    assert np.load(shard)["obs"].shape[1] == OBS_DIM


def test_when_shard_loaded_then_obs_dtype_is_float32(tmp_path):
    """Criterion: observations stored as float32 (matching flatten_obs output)."""
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2))
    shard = next(tmp_path.glob("shard_*.npz"))
    assert np.load(shard)["obs"].dtype == np.float32


# ── Shard row count ≤ shard_size ─────────────────────────────────────────────


def test_when_shards_loaded_then_each_has_at_most_shard_size_rows(tmp_path):
    """Criterion: sharded ≤ shard_size rows each, trailing partial shard allowed."""
    from training.bc_dataset import generate_bc_dataset

    shard_size = 10
    generate_bc_dataset(_cfg(tmp_path, n_games=6, shard_size=shard_size))
    for f in tmp_path.glob("shard_*.npz"):
        n = np.load(f)["obs"].shape[0]
        assert n <= shard_size, f"{f.name}: {n} rows > shard_size={shard_size}"


# ── Total rows == sum of shard rows ─────────────────────────────────────────


def test_when_manifest_read_then_total_rows_equals_sum_of_all_shard_rows(tmp_path):
    """
    Criterion: total row count equals the sum over games of len(result.trajectories);
    manifest total_rows must equal the actual row count across shards.
    """
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=3, shard_size=20))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    shard_total = sum(np.load(f)["obs"].shape[0] for f in tmp_path.glob("shard_*.npz"))
    assert manifest["total_rows"] == shard_total


def test_when_any_n_games_run_then_total_rows_is_positive(tmp_path):
    """
    Every game produces at least one decision; total_rows must be > 0 when n_games ≥ 1.
    """
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=1))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["total_rows"] > 0, "Expected at least one recorded decision for n_games=1"


# ── Label arrays present and aligned ─────────────────────────────────────────


def test_when_shard_loaded_then_label_index_acting_player_mc_return_are_present(tmp_path):
    """
    Criterion: every decision records the chosen action's index into legal_actions,
    the acting player, and the discounted MC return from bc_returns.
    """
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2))
    shard = next(tmp_path.glob("shard_*.npz"))
    data = np.load(shard)
    for key in ("label_index", "acting_player", "mc_return"):
        assert key in data, f"shard missing required array '{key}'"


def test_when_shard_loaded_then_all_arrays_have_same_row_count(tmp_path):
    """Criterion: obs columns aligned with label arrays — all arrays same length."""
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2))
    shard = next(tmp_path.glob("shard_*.npz"))
    data = np.load(shard)
    n_rows = data["obs"].shape[0]
    for key in (
        "label_index",
        "acting_player",
        "mc_return",
        "action_type",
        "param_a",
        "param_b",
        "param_c",
        "param_d",
    ):
        assert data[key].shape[0] == n_rows, (
            f"'{key}' length {data[key].shape[0]} != obs rows {n_rows}"
        )


# ── All randomness flows through set_global_seeds ────────────────────────────


def test_when_generate_bc_dataset_called_then_set_global_seeds_is_invoked(tmp_path, monkeypatch):
    """
    Criterion: all randomness flows through set_global_seeds; the seed from
    cfg.bc.seed must be passed to set_global_seeds.
    """
    import training.bc_dataset as _mod

    calls: list[int] = []
    monkeypatch.setattr(_mod, "set_global_seeds", lambda s: calls.append(s))
    _mod.generate_bc_dataset(_cfg(tmp_path, n_games=2, seed=55))
    assert 55 in calls, f"set_global_seeds(55) was not called; calls: {calls}"


# ── BCConfig is a frozen dataclass (no magic numbers) ────────────────────────


def test_when_bc_config_inspected_then_it_is_a_frozen_dataclass():
    """
    Criterion: every threshold/weight introduced lives in BCConfig or NetworkConfig
    (no magic numbers in source); BCConfig follows the frozen-config convention.
    """
    from src.config import BCConfig

    assert dataclasses.is_dataclass(BCConfig)
    instance = BCConfig()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        instance.n_games = 9999  # type: ignore[misc]


def test_when_bc_config_inspected_then_explore_eps_field_is_present():
    """Criterion: explore_eps configurable via BCConfig (not hardcoded)."""
    from src.config import BCConfig

    names = {f.name for f in dataclasses.fields(BCConfig)}
    assert "explore_eps" in names, "BCConfig must declare 'explore_eps' field"


# ── n_games controls game count ──────────────────────────────────────────────


def test_when_n_games_is_larger_then_total_rows_is_nondecreasing(tmp_path):
    """
    Criterion: runs cfg.bc.n_games HeuristicAgent self-play games.
    5 games must yield at least as many rows as 1 game (monotonicity).
    """
    from training.bc_dataset import generate_bc_dataset

    dir1 = tmp_path / "g1"
    dir5 = tmp_path / "g5"
    dir1.mkdir()
    dir5.mkdir()
    generate_bc_dataset(_cfg(dir1, n_games=1, seed=0))
    generate_bc_dataset(_cfg(dir5, n_games=5, seed=0))
    rows_1 = json.loads((dir1 / "manifest.json").read_text())["total_rows"]
    rows_5 = json.loads((dir5 / "manifest.json").read_text())["total_rows"]
    assert rows_5 >= rows_1, (
        f"5 games ({rows_5} rows) produced fewer rows than 1 game ({rows_1} rows)"
    )


# ── ε-greedy: explore_eps=0 and explore_eps=1 both produce valid datasets ───


def test_when_explore_eps_is_zero_then_dataset_generation_succeeds(tmp_path):
    """
    Criterion: ε-greedy state coverage — explore_eps=0.0 is a valid config value.
    The dataset must still be created correctly.
    """
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2, explore_eps=0.0))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["total_rows"] > 0


def test_when_explore_eps_is_one_then_dataset_generation_succeeds(tmp_path):
    """
    Criterion: ε-greedy — with explore_eps=1.0 every executed action is random,
    but the label is always the heuristic's choice; dataset must still be valid.
    """
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2, explore_eps=1.0))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["total_rows"] > 0
    shard = next(tmp_path.glob("shard_*.npz"))
    assert np.load(shard)["obs"].shape[1] == OBS_DIM


# ── .gitignore excludes data/bc/ ─────────────────────────────────────────────


def test_when_gitignore_read_then_data_bc_is_excluded():
    """Criterion: data/bc/ added to .gitignore and excluded from git."""
    gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
    assert gitignore.exists(), ".gitignore not found at project root"
    content = gitignore.read_text()
    assert "data/bc" in content, "Expected 'data/bc' pattern in .gitignore; not found"


# ── Performance smoke test ────────────────────────────────────────────────────


@pytest.mark.slow
def test_when_200_games_generated_then_wall_time_is_under_60_seconds(tmp_path):
    """
    Criterion: ≥10k games generate in minutes on CPU.
    Proxy: 200 games must complete in <60 s (= 200 games/min minimum rate,
    which extrapolates to 10k games in ≤50 min; 'minutes not hours').
    Marked slow — excluded from the default suite via -m 'not slow'.
    """
    from training.bc_dataset import generate_bc_dataset

    start = time.perf_counter()
    generate_bc_dataset(_cfg(tmp_path, n_games=200, shard_size=2000, seed=0))
    elapsed = time.perf_counter() - start
    assert elapsed < 60.0, f"200 games took {elapsed:.1f}s; need ≤60s to meet ≥200 games/min target"


# ── Round-trip: pretrain → load into PPO → one update step ──────────────────


@pytest.mark.integration
def test_when_pretrained_pt_loaded_into_ppo_trainer_then_update_step_succeeds(tmp_path):
    """
    Global criterion: pretrain → load models/pretrained.pt into the PPO trainer →
    one update step passes with no architecture changes to SelfPlayTrainer / PPO loop.
    Marked integration — excluded from the default suite via -m 'not integration'.
    """
    from training.bc_trainer import pretrain

    from training.bc_dataset import generate_bc_dataset
    from training.self_play import SelfPlayTrainer

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_path = tmp_path / "pretrained.pt"

    generate_bc_dataset(_cfg(data_dir, n_games=5, shard_size=50, seed=0))
    pretrain(_cfg(data_dir, n_games=5, seed=0), output_path=str(output_path))
    assert output_path.exists(), "pretrain() must write a checkpoint to output_path"

    trainer = SelfPlayTrainer.from_checkpoint(str(output_path))
    trainer.update_step()


# ── Property-based tests (invariants derived from criteria) ──────────────────


@given(st.integers(min_value=1, max_value=8))
@settings(max_examples=5, deadline=None)
def test_pbt_manifest_total_rows_always_equals_shard_sum(n_games):
    """
    Invariant: manifest['total_rows'] == sum of all shard row counts,
    for every valid n_games in [1, 8].
    Derived from: 'total row count equals the sum over games of len(result.trajectories)'.
    """
    from training.bc_dataset import generate_bc_dataset

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        generate_bc_dataset(_cfg(p, n_games=n_games, shard_size=30))
        manifest = json.loads((p / "manifest.json").read_text())
        shard_total = sum(np.load(f)["obs"].shape[0] for f in p.glob("shard_*.npz"))
        assert manifest["total_rows"] == shard_total, (
            f"n_games={n_games}: manifest={manifest['total_rows']} shard_sum={shard_total}"
        )


@given(st.integers(min_value=1, max_value=8))
@settings(max_examples=5, deadline=None)
def test_pbt_obs_dim_is_always_137_for_any_n_games(n_games):
    """
    Invariant: obs columns are always exactly 137 for all valid n_games.
    Derived from: 'obs columns == 137' and fixed flatten_obs dimension.
    """
    from training.bc_dataset import generate_bc_dataset

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        generate_bc_dataset(_cfg(p, n_games=n_games, shard_size=30))
        for f in p.glob("shard_*.npz"):
            assert np.load(f)["obs"].shape[1] == OBS_DIM, (
                f"n_games={n_games}, {f.name}: obs.shape[1]={np.load(f)['obs'].shape[1]}"
            )


@given(st.integers(min_value=1, max_value=8), st.integers(min_value=1, max_value=12))
@settings(max_examples=5, deadline=None)
def test_pbt_every_shard_has_at_most_shard_size_rows(n_games, shard_size):
    """
    Invariant: every shard has ≤ shard_size rows, for all (n_games, shard_size) pairs.
    Derived from: 'sharded ≤ shard_size rows each'.
    """
    from training.bc_dataset import generate_bc_dataset

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        generate_bc_dataset(_cfg(p, n_games=n_games, shard_size=shard_size))
        for f in p.glob("shard_*.npz"):
            rows = np.load(f)["obs"].shape[0]
            assert rows <= shard_size, (
                f"n_games={n_games} shard_size={shard_size}: {f.name} has {rows} rows"
            )


@given(st.integers(min_value=1, max_value=8))
@settings(max_examples=5, deadline=None)
def test_pbt_label_arrays_always_aligned_with_obs_rows(n_games):
    """
    Invariant: for every shard, each label array has the same length as obs rows,
    for all valid n_games.
    Derived from: 'Every decision records … aligned arrays'.
    """
    from training.bc_dataset import generate_bc_dataset

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        generate_bc_dataset(_cfg(p, n_games=n_games, shard_size=30))
        for f in p.glob("shard_*.npz"):
            data = np.load(f)
            n_rows = data["obs"].shape[0]
            for key in (
                "label_index",
                "acting_player",
                "mc_return",
                "action_type",
                "param_a",
                "param_b",
                "param_c",
                "param_d",
            ):
                assert data[key].shape[0] == n_rows, (
                    f"n_games={n_games} {f.name}: '{key}' length {data[key].shape[0]} != {n_rows}"
                )


# ── Action-component labels (issue #81 fix) ──────────────────────────────────


_ACTION_COMPONENT_KEYS = ("action_type", "param_a", "param_b", "param_c", "param_d")


def test_when_shard_loaded_then_all_action_component_arrays_are_present(tmp_path):
    """
    Fix #81: _row_labels must include action_type, param_a..d from rec.label_action.
    The policy network has independent categorical heads over those five fields;
    behavioural cloning requires them as supervision targets.
    """
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2))
    shard = next(tmp_path.glob("shard_*.npz"))
    data = np.load(shard)
    for key in _ACTION_COMPONENT_KEYS:
        assert key in data, f"shard missing required action-component array '{key}'"


def test_when_shard_loaded_then_action_component_arrays_are_aligned_with_obs(tmp_path):
    """
    Fix #81: action_type, param_a..d must be row-aligned with the obs array.
    """
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2))
    shard = next(tmp_path.glob("shard_*.npz"))
    data = np.load(shard)
    n_rows = data["obs"].shape[0]
    for key in _ACTION_COMPONENT_KEYS:
        assert data[key].shape[0] == n_rows, (
            f"'{key}' length {data[key].shape[0]} != obs rows {n_rows}"
        )


def test_when_shard_loaded_then_action_type_values_are_non_negative_integers(tmp_path):
    """
    Fix #81: action_type stores categorical indices — must be non-negative integers.
    """
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2))
    for f in tmp_path.glob("shard_*.npz"):
        data = np.load(f)
        assert np.issubdtype(data["action_type"].dtype, np.integer), (
            "action_type must have integer dtype"
        )
        assert (data["action_type"] >= 0).all(), "action_type values must be non-negative"


def test_when_shard_loaded_then_param_arrays_are_non_negative_integers(tmp_path):
    """
    Fix #81: param_a..d store action parameters — must be non-negative integers.
    """
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2))
    for f in tmp_path.glob("shard_*.npz"):
        data = np.load(f)
        for key in ("param_a", "param_b", "param_c", "param_d"):
            assert np.issubdtype(data[key].dtype, np.integer), f"'{key}' must have integer dtype"
            assert (data[key] >= 0).all(), f"'{key}' values must be non-negative"


@given(st.integers(min_value=1, max_value=6))
@settings(max_examples=5, deadline=None)
def test_pbt_action_component_arrays_always_present_for_any_n_games(n_games):
    """
    Invariant: action_type, param_a..d are present in every shard for all valid n_games.
    """
    from training.bc_dataset import generate_bc_dataset

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        generate_bc_dataset(_cfg(p, n_games=n_games, shard_size=30))
        for f in p.glob("shard_*.npz"):
            data = np.load(f)
            for key in _ACTION_COMPONENT_KEYS:
                assert key in data, f"n_games={n_games} {f.name}: missing '{key}'"


# ── Determinism: same seed + config → array-equal shards (issue #81 fix) ─────


def test_when_same_seed_run_twice_then_shards_are_array_equal(tmp_path):
    """
    Fix #81: two generate_bc_dataset runs with identical seed+config must produce
    shards whose arrays compare equal under np.array_equal field-by-field.
    The determinism guarantee was previously skipped/not implemented.
    """
    from training.bc_dataset import generate_bc_dataset

    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    dir_a.mkdir()
    dir_b.mkdir()

    generate_bc_dataset(_cfg(dir_a, n_games=3, shard_size=50, seed=42))
    generate_bc_dataset(_cfg(dir_b, n_games=3, shard_size=50, seed=42))

    shards_a = sorted(dir_a.glob("shard_*.npz"))
    shards_b = sorted(dir_b.glob("shard_*.npz"))
    assert len(shards_a) > 0, "no shards produced"
    assert len(shards_a) == len(shards_b), (
        f"shard count differs: {len(shards_a)} vs {len(shards_b)}"
    )
    for pa, pb in zip(shards_a, shards_b, strict=True):
        da, db = np.load(pa), np.load(pb)
        assert set(da.files) == set(db.files), f"array keys differ in {pa.name}"
        for key in da.files:
            assert np.array_equal(da[key], db[key]), (
                f"{pa.name} key='{key}': arrays differ between identical-seed runs"
            )


def test_when_same_seed_run_twice_then_manifest_total_rows_and_hash_match(tmp_path):
    """
    Fix #81: two identical-seed runs must produce manifests with equal total_rows
    and config_hash.
    """
    from training.bc_dataset import generate_bc_dataset

    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    dir_a.mkdir()
    dir_b.mkdir()

    generate_bc_dataset(_cfg(dir_a, n_games=3, seed=7))
    generate_bc_dataset(_cfg(dir_b, n_games=3, seed=7))

    ma = json.loads((dir_a / "manifest.json").read_text())
    mb = json.loads((dir_b / "manifest.json").read_text())
    assert ma["total_rows"] == mb["total_rows"], (
        f"total_rows differ: {ma['total_rows']} vs {mb['total_rows']}"
    )
    assert ma["config_hash"] == mb["config_hash"], "config_hash differs between identical-seed runs"


# ── reward_config is threaded through (issue #81 fix) ─────────────────────────


@dataclasses.dataclass(frozen=True)
class _RewardStub:
    """Minimal reward-config stub with default values."""

    sparse_win: float = 1.0
    sparse_loss: float = -1.0
    dense_territory_delta: float = 0.01
    dense_continent_bonus_delta: float = 0.05
    dense_army_ratio: float = 0.005
    dense_elimination_bonus: float = 0.1
    invalid_action_penalty: float = -0.01
    terminal_margin_weight: float = 0.5


@dataclasses.dataclass(frozen=True)
class _CfgWithReward:
    bc: _BCStub
    ppo: _PPOStub = dataclasses.field(default_factory=_PPOStub)
    reward: _RewardStub = dataclasses.field(default_factory=_RewardStub)


def test_when_cfg_has_reward_field_then_dataset_generation_succeeds(tmp_path):
    """
    Fix #81: _build_runner must accept cfg.reward and pass it to RisikoEnv.
    A config with a .reward attribute must not crash dataset generation.
    """
    from training.bc_dataset import generate_bc_dataset

    cfg = _CfgWithReward(
        bc=_BCStub(n_games=2, dataset_dir=str(tmp_path)),
        reward=_RewardStub(),
    )
    generate_bc_dataset(cfg)
    assert (tmp_path / "manifest.json").exists()


def test_when_cfg_lacks_reward_field_then_dataset_generation_still_succeeds(tmp_path):
    """
    Fix #81: _build_runner must fall back gracefully when cfg has no .reward attribute.
    The duck-typed stubs used in the rest of the test suite lack .reward; must not crash.
    """
    from training.bc_dataset import generate_bc_dataset

    generate_bc_dataset(_cfg(tmp_path, n_games=2))
    assert (tmp_path / "manifest.json").exists()
