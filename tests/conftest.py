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
