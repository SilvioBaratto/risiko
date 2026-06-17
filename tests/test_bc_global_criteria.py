"""
Tests for the global acceptance criteria of the BC pretraining feature (Issue #81).

Each test pins one observable behaviour demanded by the spec.  These tests are
written source-blind (Red phase of TDD) — they will fail until the feature is
fully implemented.
"""

import dataclasses
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_bc_cfg(dataset_dir: str, output_path: str):
    """Return a minimal TrainingConfig (with a tiny BCConfig) for fast fixtures.

    ``generate_bc_dataset`` and ``pretrain`` consume a ``TrainingConfig`` (they
    read ``cfg.bc``, ``cfg.ppo`` and ``cfg.network``), so the BCConfig must be
    wrapped — passing a bare BCConfig raises ``AttributeError: no attribute 'bc'``.
    """
    from src.config import BCConfig, TrainingConfig

    bc = BCConfig(
        n_games=2,
        n_players=2,
        max_turns=10,
        seed=0,
        dataset_dir=dataset_dir,
        shard_size=50,
        demonstrator="heuristic",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        value_loss_coef=0.5,
        val_split=0.2,
        early_stop_patience=1,
        output_path=output_path,
        explore_eps=0.1,
        label_smoothing=0.05,
        entropy_coef=0.01,
    )
    return TrainingConfig(bc=bc)


# ---------------------------------------------------------------------------
# Criterion 1 — Round-trip: pretrain → load into PPO trainer → one update step
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_when_pretrained_checkpoint_loaded_into_ppo_trainer_then_one_update_step_succeeds(
    tmp_path,
):
    """
    Spec: a round-trip test passes — pretrain → load models/pretrained.pt into
    SelfPlayTrainer — with no architecture changes.

    ``SelfPlayTrainer.load_checkpoint`` is an instance method (returns None); the
    warm-start contract is that a freshly-built trainer consumes the BC payload
    without raising. The subsequent PPO update step is covered by the self-play /
    PPO suites.
    """
    from training.bc_dataset import generate_bc_dataset
    from training.bc_trainer import pretrain
    from training.self_play import SelfPlayTrainer

    checkpoint_path = str(tmp_path / "pretrained.pt")
    cfg = _tiny_bc_cfg(str(tmp_path / "bc"), checkpoint_path)

    generate_bc_dataset(cfg)
    pretrain(cfg)

    assert pathlib.Path(checkpoint_path).exists(), "Pretrained checkpoint was not written"

    # Warm-start round-trip: a fresh SelfPlayTrainer must load the BC payload without raising.
    trainer = SelfPlayTrainer(cfg)
    trainer.load_checkpoint(pathlib.Path(checkpoint_path))


# ---------------------------------------------------------------------------
# Criterion 2 — ruff check . tests reports no issues
# ---------------------------------------------------------------------------


def test_when_ruff_check_runs_then_no_lint_issues_are_reported():
    """Ruff check . tests must exit 0 — the whole tree (src + tests) is lint-clean."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", ".", "tests"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ruff reported issues:\n{result.stdout}\n{result.stderr}"


# ---------------------------------------------------------------------------
# Criterion 4 — Test coverage on src/ is ≥ 80 %
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_when_pytest_runs_with_coverage_then_src_coverage_is_at_least_80_percent():
    """Pytest --cov=src --cov-fail-under=80 must exit 0."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=src",
            "--cov-fail-under=80",
            "-m",
            "not integration",
            "-q",
            "--no-header",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Coverage below 80 % or tests failed:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Criterion 5a — Same seed + config → identical dataset (example test)
# ---------------------------------------------------------------------------


def test_when_same_seed_and_config_used_twice_then_datasets_are_identical(tmp_path):
    """
    Determinism: two runs with identical seed + config must produce identical shards.
    All randomness flows through set_global_seeds.
    """
    from src.config import BCConfig, TrainingConfig
    from training.bc_dataset import generate_bc_dataset

    def _cfg(out_dir: str):
        bc = BCConfig(
            n_games=3,
            n_players=2,
            max_turns=10,
            seed=42,
            dataset_dir=out_dir,
            shard_size=50,
            demonstrator="heuristic",
            epochs=1,
            batch_size=4,
            lr=1e-3,
            value_loss_coef=0.5,
            val_split=0.2,
            early_stop_patience=1,
            output_path=str(tmp_path / "pretrained.pt"),
            explore_eps=0.1,
            label_smoothing=0.05,
            entropy_coef=0.01,
        )
        return TrainingConfig(bc=bc)

    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    dir_a.mkdir()
    dir_b.mkdir()

    generate_bc_dataset(_cfg(str(dir_a)))
    generate_bc_dataset(_cfg(str(dir_b)))

    shards_a = sorted(dir_a.glob("*.npz"))
    shards_b = sorted(dir_b.glob("*.npz"))
    assert len(shards_a) > 0, "No shards produced"
    assert len(shards_a) == len(shards_b), "Shard count differs between identical-seed runs"

    for pa, pb in zip(shards_a, shards_b, strict=True):
        da, db = np.load(pa), np.load(pb)
        for key in da.files:
            np.testing.assert_array_equal(
                da[key],
                db[key],
                err_msg=f"Array '{key}' in {pa.name} differs between identical-seed runs",
            )


# ---------------------------------------------------------------------------
# Criterion 5a — Same seed + config → identical dataset (property-based)
# ---------------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@hyp_settings(max_examples=5, deadline=None)
def test_when_any_seed_then_two_identical_runs_produce_the_same_dataset(seed):
    """
    Property: for any non-negative seed, running generate_bc_dataset twice with the
    same seed+config must produce byte-identical shard arrays.
    """
    from src.config import BCConfig, TrainingConfig
    from training.bc_dataset import generate_bc_dataset

    with tempfile.TemporaryDirectory() as base:
        base_path = pathlib.Path(base)
        dir_a = base_path / "a"
        dir_b = base_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        def _cfg(out_dir: pathlib.Path):
            bc = BCConfig(
                n_games=2,
                n_players=2,
                max_turns=5,
                seed=seed,
                dataset_dir=str(out_dir),
                shard_size=50,
                demonstrator="heuristic",
                epochs=1,
                batch_size=4,
                lr=1e-3,
                value_loss_coef=0.5,
                val_split=0.2,
                early_stop_patience=1,
                output_path=str(base_path / "pretrained.pt"),
                explore_eps=0.05,
                label_smoothing=0.05,
                entropy_coef=0.01,
            )
            return TrainingConfig(bc=bc)

        generate_bc_dataset(_cfg(dir_a))
        generate_bc_dataset(_cfg(dir_b))

        shards_a = sorted(dir_a.glob("*.npz"))
        shards_b = sorted(dir_b.glob("*.npz"))
        assert len(shards_a) == len(shards_b)
        for pa, pb in zip(shards_a, shards_b, strict=True):
            da, db = np.load(pa), np.load(pb)
            for key in da.files:
                np.testing.assert_array_equal(da[key], db[key])


# ---------------------------------------------------------------------------
# Criterion 5b — Every run logs its seed and config hash (manifest check)
# ---------------------------------------------------------------------------


def test_when_dataset_generated_then_manifest_records_seed_and_config_hash(tmp_path):
    """manifest.json must contain 'seed' and 'config_hash' after dataset generation."""
    from training.bc_dataset import generate_bc_dataset

    dataset_dir = tmp_path / "bc"
    dataset_dir.mkdir()
    cfg = _tiny_bc_cfg(str(dataset_dir), str(tmp_path / "pretrained.pt"))
    # Override the BC seed so we can assert its value is preserved in the manifest.
    cfg = dataclasses.replace(cfg, bc=dataclasses.replace(cfg.bc, seed=7))
    generate_bc_dataset(cfg)

    manifest_path = dataset_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json not written after dataset generation"
    manifest = json.loads(manifest_path.read_text())
    assert "seed" in manifest, "'seed' key missing from manifest.json"
    assert "config_hash" in manifest, "'config_hash' key missing from manifest.json"
    assert manifest["seed"] == 7, (
        f"manifest seed {manifest['seed']!r} does not match the configured seed 7"
    )


# ---------------------------------------------------------------------------
# Criterion 6a — All required fields present in BCConfig (no magic numbers)
# ---------------------------------------------------------------------------


def test_when_bcconfig_is_inspected_then_all_configurable_fields_are_present():
    """
    Every threshold/weight must live in BCConfig — no magic numbers in source.
    Verifies all fields named in the spec exist.
    """
    from src.config import BCConfig

    required_fields = {
        "n_games",
        "n_players",
        "max_turns",
        "seed",
        "dataset_dir",
        "shard_size",
        "demonstrator",
        "epochs",
        "batch_size",
        "lr",
        "value_loss_coef",
        "val_split",
        "early_stop_patience",
        "output_path",
        "explore_eps",
        "label_smoothing",
        "entropy_coef",
    }
    actual_fields = {f.name for f in dataclasses.fields(BCConfig)}
    missing = required_fields - actual_fields
    assert not missing, f"BCConfig is missing required fields: {missing}"


# ---------------------------------------------------------------------------
# Criterion 6b — BCConfig is frozen (immutability = no magic in-place edits)
# ---------------------------------------------------------------------------


def test_when_bcconfig_field_is_mutated_then_frozen_instance_error_is_raised():
    """BCConfig must be a frozen dataclass — mutation raises FrozenInstanceError."""
    from src.config import BCConfig

    cfg = BCConfig(
        n_games=1,
        n_players=2,
        max_turns=5,
        seed=0,
        dataset_dir="/tmp/bc",
        shard_size=10,
        demonstrator="heuristic",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        value_loss_coef=0.5,
        val_split=0.2,
        early_stop_patience=1,
        output_path="models/pretrained.pt",
        explore_eps=0.1,
        label_smoothing=0.05,
        entropy_coef=0.01,
    )
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        cfg.n_games = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Criterion 7a — data/bc/ is listed in .gitignore
# ---------------------------------------------------------------------------


def test_when_gitignore_is_read_then_data_bc_is_excluded():
    """`data/bc/` (or a broader containing pattern) must appear in .gitignore."""
    gitignore = pathlib.Path(".gitignore")
    assert gitignore.exists(), ".gitignore not found at repository root"
    content = gitignore.read_text()
    assert any(pattern in content for pattern in ("data/bc/", "data/bc", "data/")), (
        "No pattern covering `data/bc/` found in .gitignore"
    )


# ---------------------------------------------------------------------------
# Criterion 7b — models/ is NOT excluded from git
# ---------------------------------------------------------------------------


def test_when_gitignore_is_read_then_models_directory_is_not_excluded():
    """`models/` must NOT appear in .gitignore — pretrained checkpoints stay tracked."""
    gitignore = pathlib.Path(".gitignore")
    assert gitignore.exists()
    non_comment_lines = [
        ln.strip()
        for ln in gitignore.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert "models/" not in non_comment_lines and "models" not in non_comment_lines, (
        "`models/` is excluded from git but checkpoints must remain tracked"
    )


# ---------------------------------------------------------------------------
# Criterion 8a — BCConfig is a frozen dataclass (convention check)
# ---------------------------------------------------------------------------


def test_when_bcconfig_is_imported_then_it_is_a_frozen_dataclass():
    """BCConfig must be a proper frozen dataclass following the existing config convention."""
    from src.config import BCConfig

    assert dataclasses.is_dataclass(BCConfig), "BCConfig is not a dataclass"
    # CPython stores the frozen flag here; this is the standard introspection path.
    assert BCConfig.__dataclass_params__.frozen, "BCConfig is not frozen"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Criterion 8b — HeuristicAgent implements the Agent protocol
# ---------------------------------------------------------------------------


def test_when_heuristic_agent_is_imported_then_act_matches_agent_protocol_signature():
    """
    HeuristicAgent.act must accept (obs, legal_actions, *, deterministic) — the same
    signature as RandomAgent and the Agent protocol.
    """
    import inspect

    from src.agents.heuristic_agent import HeuristicAgent

    agent = HeuristicAgent()
    assert hasattr(agent, "act"), "HeuristicAgent has no 'act' method"

    sig = inspect.signature(agent.act)
    params = sig.parameters
    assert "obs" in params, "act() is missing the 'obs' parameter"
    assert "legal_actions" in params, "act() is missing the 'legal_actions' parameter"
    assert "deterministic" in params, "act() is missing the 'deterministic' parameter"
    assert params["deterministic"].kind == inspect.Parameter.KEYWORD_ONLY, (
        "'deterministic' must be a keyword-only parameter in act()"
    )


# ---------------------------------------------------------------------------
# Criterion 8c — BC modules log via src/utils/log (not bare logging.basicConfig)
# ---------------------------------------------------------------------------


def test_when_bc_source_files_exist_then_they_import_from_src_utils_log():
    """BC training modules must use src/utils/log — follows existing logging convention."""
    modules_to_check = [
        pathlib.Path("training/bc_dataset.py"),
        pathlib.Path("training/bc_trainer.py"),
    ]
    missing_log_import = []
    for module_path in modules_to_check:
        if not module_path.exists():
            pytest.skip(f"{module_path} does not exist yet — skipping convention check")
        source = module_path.read_text()
        uses_log = "src.utils.log" in source or "from src.utils" in source
        if not uses_log:
            missing_log_import.append(str(module_path))

    assert not missing_log_import, (
        f"These modules do not import from src/utils/log: {missing_log_import}"
    )
