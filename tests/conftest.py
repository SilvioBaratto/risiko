"""Session-wide pytest setup.

Typer/Rich colorize and wrap ``--help`` output based on the terminal and on
``FORCE_COLOR``/``COLUMNS`` env vars. Under CI those defaults inject ANSI color
codes *between* characters (so ``--override`` is split by SGR escapes around the
two dashes) and wrap at 80 columns, which breaks plain substring assertions that
pass on a local wide, color-free terminal.

Pin a plain, wide, deterministic rendering before any test (in-process
``CliRunner`` or ``subprocess``, which inherits ``os.environ``) runs, so CLI
help assertions behave identically everywhere.
"""

import os

# Force color OFF for both Rich (NO_COLOR) and Click (NO_COLOR / CLICOLOR).
os.environ["NO_COLOR"] = "1"
os.environ["CLICOLOR"] = "0"
os.environ["TERM"] = "dumb"
# Defeat any CI-injected color forcing.
os.environ.pop("FORCE_COLOR", None)
os.environ.pop("CLICOLOR_FORCE", None)
# Wide enough that long option names/descriptions never wrap mid-token.
os.environ["COLUMNS"] = "200"


# ---------------------------------------------------------------------------
# Keep the default CI gate (``pytest -m "not integration"``) fast.
#
# A handful of behavioral-cloning dataset-generation tests each run full
# heuristic self-play (seconds-to-minutes apiece; the hypothesis property
# tests regenerate a dataset per example). After games became player-turn
# capped they pushed the default suite to ~30 min. Auto-mark them
# ``integration`` so the fast gate skips them — they still run under an
# explicit ``pytest -m integration`` invocation.
# ---------------------------------------------------------------------------
_HEAVY_BC_SUBSTRINGS = (
    "test_pbt_",  # hypothesis property tests over generate_bc_dataset
    "test_when_pretrained_checkpoint_loaded_into_ppo_trainer_then_one_update_step_succeeds",
    "test_when_shards_loaded_then_each_has_at_most_shard_size_rows",
    "test_when_same_seed_run_twice_then_",
    "test_when_n_games_is_larger_then_total_rows_is_nondecreasing",
    "test_when_generate_bc_dataset_called_twice_with_same_seed_then_manifest_matches",
    "test_when_same_seed_and_config_then_dataset_manifests_are_identical",
)


def pytest_collection_modifyitems(config, items):
    """Tag known-heavy BC dataset-generation tests as ``integration``."""
    import pytest

    for item in items:
        if any(s in item.nodeid for s in _HEAVY_BC_SUBSTRINGS):
            item.add_marker(pytest.mark.integration)
