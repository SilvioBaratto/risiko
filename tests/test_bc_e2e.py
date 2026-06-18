"""End-to-end warm-start round-trip test for issue #90.

Covers (issue #90, criterion):
  "An end-to-end test (@pytest.mark.integration/slow) runs generate →
  pretrain → loads models/pretrained.pt into a SelfPlayTrainer via the
  existing checkpoint path → completes one update step without error."

"One update step" is operationalised as ``SelfPlayTrainer.update_step()``,
the method designed for this exact integration-test use case.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _e2e_config(dataset_dir: str, output_path: str):
    """Minimal TrainingConfig for the full generate → pretrain pipeline."""
    from src.config import BCConfig, PPOConfig, TrainingConfig  # noqa: PLC0415

    bc = BCConfig(
        n_games=3,
        n_players=2,
        max_turns=20,
        seed=0,
        dataset_dir=dataset_dir,
        shard_size=500,
        demonstrator="heuristic",
        epochs=2,
        batch_size=8,
        lr=1e-3,
        value_loss_coef=0.5,
        val_split=0.25,
        early_stop_patience=5,
        output_path=output_path,
        explore_eps=0.0,
        label_smoothing=0.05,
        entropy_coef=0.01,
    )
    return TrainingConfig(total_timesteps=100, ppo=PPOConfig(lr=1e-3), bc=bc)


# ---------------------------------------------------------------------------
# Criterion: generate → pretrain → load_checkpoint → one update step
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_when_full_pretrain_pipeline_runs_then_one_ppo_update_step_succeeds(
    tmp_path,
):
    """
    E2E round-trip:
      1. generate_bc_dataset(cfg)  — real HeuristicAgent self-play
      2. pretrain(cfg)             — BC training → writes models/pretrained.pt
      3. SelfPlayTrainer(cfg)      — no architecture changes needed
      4. trainer.load_checkpoint(ckpt_path)
      5. trainer.update_step()     — must not raise

    Passes only when the pretrained checkpoint is fully compatible with the
    existing SelfPlayTrainer / PPO loop architecture.
    """
    from training.bc_dataset import generate_bc_dataset  # noqa: PLC0415
    from training.bc_trainer import pretrain  # noqa: PLC0415
    from training.self_play import SelfPlayTrainer  # noqa: PLC0415

    dataset_dir = str(tmp_path / "data" / "bc")
    output_path = str(tmp_path / "models" / "pretrained.pt")

    cfg = _e2e_config(dataset_dir, output_path)

    # Steps 1 & 2: real data generation + BC training
    generate_bc_dataset(cfg)
    pretrain(cfg)
    ckpt_path = tmp_path / "models" / "pretrained.pt"
    assert ckpt_path.exists(), "pretrain() must write a checkpoint to cfg.bc.output_path"

    # Steps 3 & 4: load into SelfPlayTrainer (no architecture changes needed)
    trainer = SelfPlayTrainer(cfg)
    trainer.load_checkpoint(str(ckpt_path))

    # Step 5: one PPO update step — must not raise
    trainer.update_step()


@pytest.mark.integration
def test_when_full_pretrain_pipeline_runs_then_all_model_parameters_require_grad(tmp_path):
    """
    After loading the BC checkpoint into SelfPlayTrainer, all model parameters
    must have requires_grad=True — no accidental freezing from an arch mismatch.
    """
    from training.bc_dataset import generate_bc_dataset  # noqa: PLC0415
    from training.bc_trainer import pretrain  # noqa: PLC0415
    from training.self_play import SelfPlayTrainer  # noqa: PLC0415

    dataset_dir = str(tmp_path / "data" / "bc")
    output_path = str(tmp_path / "models" / "pretrained.pt")
    cfg = _e2e_config(dataset_dir, output_path)

    generate_bc_dataset(cfg)
    pretrain(cfg)

    trainer = SelfPlayTrainer(cfg)
    trainer.load_checkpoint(output_path)

    params = list(trainer._agent._net.parameters())
    assert params, "SelfPlayTrainer._agent._net must have parameters after loading BC checkpoint"
    frozen = [p for p in params if not p.requires_grad]
    assert not frozen, (
        f"{len(frozen)} parameter(s) have requires_grad=False after loading the BC checkpoint; "
        "the PPO loop needs all parameters trainable"
    )


@pytest.mark.integration
def test_when_pretrain_checkpoint_loaded_then_update_step_produces_no_error(tmp_path):
    """
    After generate → pretrain → load_checkpoint, update_step() must not raise
    — proving the obs_dim=137 architectural constant and action_dims are
    preserved throughout the full pipeline.
    """
    from training.bc_dataset import generate_bc_dataset  # noqa: PLC0415
    from training.bc_trainer import pretrain  # noqa: PLC0415
    from training.self_play import SelfPlayTrainer  # noqa: PLC0415

    dataset_dir = str(tmp_path / "data" / "bc")
    output_path = str(tmp_path / "models" / "pretrained.pt")
    cfg = _e2e_config(dataset_dir, output_path)

    generate_bc_dataset(cfg)
    pretrain(cfg)

    trainer = SelfPlayTrainer(cfg)
    trainer.load_checkpoint(output_path)

    # update_step runs one real game episode + one PPO gradient update;
    # any obs_dim or action_dim mismatch surfaces here as a RuntimeError
    trainer.update_step()
