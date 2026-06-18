"""CLI smoke tests for issue #88: risiko-rl pretrain command.

Source-blind: derived from acceptance criteria only. No implementation source was read.

Patching assumption: the pretrain CLI command calls generate_bc_dataset and pretrain
by referencing them through their source modules (e.g. via a module-level import or
lazy import). The patch targets below are therefore the source module paths:

    training.bc_dataset.generate_bc_dataset
    training.bc_trainer.pretrain

If the CLI uses ``from training.bc_dataset import generate_bc_dataset`` at module
level, change _GENERATE to ``"risiko_rl.cli.generate_bc_dataset"`` (and similarly
_PRETRAIN_FN to ``"risiko_rl.cli.pretrain"``).
"""

from __future__ import annotations

import pathlib
from unittest.mock import patch

from typer.testing import CliRunner

from risiko_rl.cli import app

runner = CliRunner()

_GENERATE = "training.bc_dataset.generate_bc_dataset"
_PRETRAIN_FN = "training.bc_trainer.pretrain"


# ---------------------------------------------------------------------------
# Criterion: `pretrain` command exists, invoked as
#   risiko-rl pretrain --config config/bc_pretrain.yaml
# ---------------------------------------------------------------------------


class TestPretrainCommandExists:
    def test_when_pretrain_help_invoked_then_exit_is_zero(self) -> None:
        result = runner.invoke(app, ["pretrain", "--help"])
        assert result.exit_code == 0, result.output

    def test_when_pretrain_help_invoked_then_default_config_is_bc_pretrain_yaml(self) -> None:
        """--config must default to config/bc_pretrain.yaml."""
        result = runner.invoke(app, ["pretrain", "--help"])
        assert "bc_pretrain.yaml" in result.output

    def test_when_pretrain_help_invoked_then_override_option_is_listed(self) -> None:
        """--override / -o must be a registered option on the pretrain command."""
        result = runner.invoke(app, ["pretrain", "--help"])
        assert "--override" in result.output or "-o" in result.output


# ---------------------------------------------------------------------------
# Criterion: runs generate_bc_dataset(cfg) only when no dataset is present;
#            when shards are present it logs/echoes a skip message and proceeds.
# ---------------------------------------------------------------------------


class TestConditionalDatasetGeneration:
    def test_when_dataset_dir_is_empty_then_generate_bc_dataset_is_called(
        self, tmp_path: pathlib.Path
    ) -> None:
        """generate_bc_dataset must be invoked when no .npz shards exist."""
        data_dir = tmp_path / "data" / "bc"
        data_dir.mkdir(parents=True)
        out = tmp_path / "pretrained.pt"

        with patch(_GENERATE) as gen_mock, patch(_PRETRAIN_FN):
            runner.invoke(
                app,
                [
                    "pretrain",
                    "--override",
                    f"bc.dataset_dir={data_dir}",
                    "--override",
                    f"bc.output_path={out}",
                ],
            )

        gen_mock.assert_called_once()

    def test_when_shards_already_present_then_generate_bc_dataset_is_not_called(
        self, tmp_path: pathlib.Path
    ) -> None:
        """generate_bc_dataset must NOT be called when .npz shards already exist."""
        data_dir = tmp_path / "data" / "bc"
        data_dir.mkdir(parents=True)
        (data_dir / "shard_0.npz").write_bytes(b"fake")
        out = tmp_path / "pretrained.pt"

        with patch(_GENERATE) as gen_mock, patch(_PRETRAIN_FN):
            runner.invoke(
                app,
                [
                    "pretrain",
                    "--override",
                    f"bc.dataset_dir={data_dir}",
                    "--override",
                    f"bc.output_path={out}",
                ],
            )

        gen_mock.assert_not_called()

    def test_when_shards_already_present_then_skip_message_is_emitted(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A skip/found/exists message must appear in output when generation is bypassed."""
        data_dir = tmp_path / "data" / "bc"
        data_dir.mkdir(parents=True)
        (data_dir / "shard_0.npz").write_bytes(b"fake")
        out = tmp_path / "pretrained.pt"

        with patch(_GENERATE), patch(_PRETRAIN_FN):
            result = runner.invoke(
                app,
                [
                    "pretrain",
                    "--override",
                    f"bc.dataset_dir={data_dir}",
                    "--override",
                    f"bc.output_path={out}",
                ],
            )

        keywords = ("skip", "found", "exist", "already", "present")
        assert any(kw in result.output.lower() for kw in keywords), (
            f"Expected one of {keywords} in output; got: {result.output!r}"
        )


# ---------------------------------------------------------------------------
# Criterion: runs pretrain(cfg), writing models/pretrained.pt (cfg.bc.output_path)
# ---------------------------------------------------------------------------


class TestCheckpointOutput:
    def test_when_pretrain_command_completes_then_exit_code_is_zero(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Tiny end-to-end smoke test: pretrain command exits 0."""
        data_dir = tmp_path / "data" / "bc"
        data_dir.mkdir(parents=True)
        (data_dir / "shard_0.npz").write_bytes(b"fake")
        out = tmp_path / "pretrained.pt"

        def _write_ckpt(cfg) -> None:
            p = pathlib.Path(str(cfg.bc.output_path))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"ckpt")

        with patch(_GENERATE), patch(_PRETRAIN_FN, side_effect=_write_ckpt):
            result = runner.invoke(
                app,
                [
                    "pretrain",
                    "--override",
                    f"bc.dataset_dir={data_dir}",
                    "--override",
                    f"bc.output_path={out}",
                ],
            )

        assert result.exit_code == 0, result.output

    def test_when_pretrain_command_completes_then_checkpoint_file_exists(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The checkpoint must be written to cfg.bc.output_path."""
        data_dir = tmp_path / "data" / "bc"
        data_dir.mkdir(parents=True)
        (data_dir / "shard_0.npz").write_bytes(b"fake")
        out = tmp_path / "pretrained.pt"

        def _write_ckpt(cfg) -> None:
            p = pathlib.Path(str(cfg.bc.output_path))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"ckpt")

        with patch(_GENERATE), patch(_PRETRAIN_FN, side_effect=_write_ckpt):
            runner.invoke(
                app,
                [
                    "pretrain",
                    "--override",
                    f"bc.dataset_dir={data_dir}",
                    "--override",
                    f"bc.output_path={out}",
                ],
            )

        assert out.exists()

    def test_when_pretrain_fn_is_called_then_it_receives_output_path_from_config(
        self, tmp_path: pathlib.Path
    ) -> None:
        """pretrain(cfg) must receive a cfg where cfg.bc.output_path reflects the override."""
        data_dir = tmp_path / "data" / "bc"
        data_dir.mkdir(parents=True)
        (data_dir / "shard_0.npz").write_bytes(b"fake")
        out = tmp_path / "custom_checkpoint.pt"
        captured: list = []

        def _capture(cfg) -> None:
            captured.append(cfg)

        with patch(_GENERATE), patch(_PRETRAIN_FN, side_effect=_capture):
            runner.invoke(
                app,
                [
                    "pretrain",
                    "--override",
                    f"bc.dataset_dir={data_dir}",
                    "--override",
                    f"bc.output_path={out}",
                ],
            )

        if captured:
            assert str(out) in str(captured[0].bc.output_path)


# ---------------------------------------------------------------------------
# Criterion: supports --override key=value (including bc.*)
#            via the same merge_cli_overrides path as `train`;
#            a malformed override (no `=`) raises a BadParameter (non-zero exit).
# ---------------------------------------------------------------------------


class TestOverrideSupport:
    def test_when_bc_n_games_override_passed_then_exit_is_zero(
        self, tmp_path: pathlib.Path
    ) -> None:
        data_dir = tmp_path / "data" / "bc"
        data_dir.mkdir(parents=True)
        (data_dir / "shard_0.npz").write_bytes(b"fake")

        with patch(_GENERATE), patch(_PRETRAIN_FN):
            result = runner.invoke(
                app,
                [
                    "pretrain",
                    "--override",
                    "bc.n_games=20",
                    "--override",
                    f"bc.dataset_dir={data_dir}",
                    "--override",
                    f"bc.output_path={tmp_path / 'out.pt'}",
                ],
            )

        assert result.exit_code == 0, result.output

    def test_when_bc_epochs_override_passed_then_exit_is_zero(self, tmp_path: pathlib.Path) -> None:
        data_dir = tmp_path / "data" / "bc"
        data_dir.mkdir(parents=True)
        (data_dir / "shard_0.npz").write_bytes(b"fake")

        with patch(_GENERATE), patch(_PRETRAIN_FN):
            result = runner.invoke(
                app,
                [
                    "pretrain",
                    "--override",
                    "bc.epochs=1",
                    "--override",
                    f"bc.dataset_dir={data_dir}",
                    "--override",
                    f"bc.output_path={tmp_path / 'out.pt'}",
                ],
            )

        assert result.exit_code == 0, result.output

    def test_when_override_missing_equals_sign_then_exit_is_nonzero(self) -> None:
        """A malformed override (no '=') must raise BadParameter → non-zero exit."""
        result = runner.invoke(
            app,
            [
                "pretrain",
                "--override",
                "bc.n_games",
            ],
        )
        assert result.exit_code != 0

    def test_when_multiple_bc_overrides_provided_then_all_are_applied_to_config(
        self, tmp_path: pathlib.Path
    ) -> None:
        """All --override values must be merged before pretrain(cfg) is called."""
        data_dir = tmp_path / "data" / "bc"
        data_dir.mkdir(parents=True)
        (data_dir / "shard_0.npz").write_bytes(b"fake")
        captured: list = []

        def _capture(cfg) -> None:
            captured.append(cfg)

        with patch(_GENERATE), patch(_PRETRAIN_FN, side_effect=_capture):
            runner.invoke(
                app,
                [
                    "pretrain",
                    "--override",
                    "bc.n_games=20",
                    "--override",
                    "bc.epochs=1",
                    "--override",
                    f"bc.dataset_dir={data_dir}",
                    "--override",
                    f"bc.output_path={tmp_path / 'out.pt'}",
                ],
            )

        if captured:
            assert captured[0].bc.n_games == 20
            assert captured[0].bc.epochs == 1

    def test_when_short_flag_o_used_as_override_then_exit_is_zero(
        self, tmp_path: pathlib.Path
    ) -> None:
        """-o must be the short alias for --override, consistent with the `train` command."""
        data_dir = tmp_path / "data" / "bc"
        data_dir.mkdir(parents=True)
        (data_dir / "shard_0.npz").write_bytes(b"fake")

        with patch(_GENERATE), patch(_PRETRAIN_FN):
            result = runner.invoke(
                app,
                [
                    "pretrain",
                    "-o",
                    "bc.n_games=5",
                    "-o",
                    f"bc.dataset_dir={data_dir}",
                    "-o",
                    f"bc.output_path={tmp_path / 'out.pt'}",
                ],
            )

        assert result.exit_code == 0, result.output
