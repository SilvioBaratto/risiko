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

import pytest

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
    for item in items:
        if any(s in item.nodeid for s in _HEAVY_BC_SUBSTRINGS):
            item.add_marker(pytest.mark.integration)


# ---------------------------------------------------------------------------
# No unit test may reach a real Ollama server.
#
# Several tests patch ``call_ollama_for_action_index`` but exercise a code path
# that also negotiates, so the negotiation call escaped to whatever Ollama is
# listening on localhost. With rate-limit waiting enabled that call blocks for
# as long as the server keeps the socket open — the suite hung for 15+ minutes
# with a live connection to :11434 and no output. A developer with Ollama
# running got a slow, network-dependent suite; CI got a fast one. Same tests.
#
# Every unit test now fails loudly instead of dialling out. The guard sits at the
# transport (``httpx.post``/``httpx.get``), which is the lowest layer above the
# socket: tests that legitimately exercise the client by mocking the transport
# themselves simply patch over it, and tests that mean to reach a server carry the
# ``integration`` marker and are exempt.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_real_ollama(request, monkeypatch):
    """Make any un-mocked HTTP request raise instead of blocking on the network."""
    if request.node.get_closest_marker("integration"):
        return

    def _blocked(url, *args, **kwargs):
        raise AssertionError(
            f"a unit test tried to make a real HTTP request to {url!r} — mock the call "
            "(call_ollama_for_action_index / call_ollama_for_negotiation / httpx.post), "
            "or mark the test as integration"
        )

    monkeypatch.setattr("src.agents.ollama_client.httpx.post", _blocked, raising=True)
    monkeypatch.setattr("src.agents.ollama_client.httpx.get", _blocked, raising=True)
