"""Source-blind tests for training/bc_loss.py.

Tests authored from the issue #82 acceptance criteria alone — no implementation
source was read. They define the contract the implementation must satisfy.

Naming convention: ``test_when_<precondition>_then_<expected_result>``.
"""

from __future__ import annotations

import math

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from training.bc_loss import (
    LossStats,
    combined_loss,
    policy_cross_entropy,
    policy_entropy,
    top1_accuracy,
    value_mse,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers derived from the acceptance criteria text.
#
# "ACTION_DIMS = {"action_type":6, "param_a":201, "param_b":201, "param_c":201,
# "param_d":1} — 5 independent Categorical heads"
# ---------------------------------------------------------------------------

ACTION_DIMS: dict[str, int] = {
    "action_type": 6,
    "param_a": 201,
    "param_b": 201,
    "param_c": 201,
    "param_d": 1,
}

_B = 4  # default batch size


def _uniform_logits(batch: int = _B) -> dict[str, torch.Tensor]:
    """Zero logits — uniform distribution over each head's vocabulary."""
    return {name: torch.zeros(batch, dim) for name, dim in ACTION_DIMS.items()}


def _peaked_logits(batch: int = _B, target_idx: int = 0) -> dict[str, torch.Tensor]:
    """Logits sharply peaked on ``target_idx`` for every head."""
    out: dict[str, torch.Tensor] = {}
    for name, dim in ACTION_DIMS.items():
        t = torch.full((batch, dim), -1e9)
        clamped = min(target_idx, dim - 1)
        t[:, clamped] = 1e9
        out[name] = t
    return out


def _targets(batch: int = _B, index: int = 0) -> dict[str, torch.Tensor]:
    """Target label ``index`` (clamped to valid range) for every head."""
    return {
        name: torch.full((batch,), min(index, dim - 1), dtype=torch.long)
        for name, dim in ACTION_DIMS.items()
    }


# ---------------------------------------------------------------------------
# policy_cross_entropy
# ---------------------------------------------------------------------------


class TestPolicyCrossEntropy:
    """Criterion: sums F.cross_entropy across 5 heads.

    peaked→≈0, uniform→~log(dim), label_smoothing>0 raises the floor.
    """

    def test_when_logits_peaked_on_target_then_cross_entropy_is_near_zero(self) -> None:
        result = policy_cross_entropy(
            _peaked_logits(target_idx=0), _targets(index=0), label_smoothing=0.0
        )
        assert result.item() < 1e-2

    def test_when_logits_are_uniform_then_cross_entropy_approximates_sum_of_log_dims(self) -> None:
        """Uniform logits → CE per head ≈ log(dim); summed across all 5 heads."""
        result = policy_cross_entropy(_uniform_logits(), _targets(), label_smoothing=0.0)
        # log(6)+log(201)+log(201)+log(201)+log(1) ≈ 17.60; param_d contributes log(1)=0
        expected = sum(math.log(max(dim, 1)) for dim in ACTION_DIMS.values())
        assert abs(result.item() - expected) < 2.0

    def test_when_label_smoothing_positive_then_floor_raised_above_hard_label_ce(self) -> None:
        logits = _peaked_logits(target_idx=0)
        targets = _targets(index=0)
        no_smooth = policy_cross_entropy(logits, targets, label_smoothing=0.0)
        with_smooth = policy_cross_entropy(logits, targets, label_smoothing=0.1)
        assert with_smooth.item() > no_smooth.item()

    def test_when_called_then_result_is_scalar_tensor(self) -> None:
        result = policy_cross_entropy(_uniform_logits(), _targets(), label_smoothing=0.0)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0

    def test_when_label_smoothing_is_zero_then_cross_entropy_is_non_negative(self) -> None:
        result = policy_cross_entropy(_uniform_logits(), _targets(), label_smoothing=0.0)
        assert result.item() >= 0.0

    @given(
        st.floats(min_value=0.0, max_value=0.45, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=0.45, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_when_label_smoothing_increases_then_ce_on_peaked_logits_is_non_decreasing(
        self, smooth_lo: float, smooth_hi: float
    ) -> None:
        """Monotonicity invariant: higher smoothing never lowers CE on peaked logits."""
        logits = _peaked_logits(target_idx=0)
        targets = _targets(index=0)
        ce_lo = policy_cross_entropy(logits, targets, label_smoothing=smooth_lo).item()
        ce_hi = policy_cross_entropy(logits, targets, label_smoothing=smooth_hi).item()
        if smooth_hi >= smooth_lo:
            assert ce_hi >= ce_lo - 1e-4


# ---------------------------------------------------------------------------
# value_mse
# ---------------------------------------------------------------------------


class TestValueMse:
    """Criterion: mirrors nnf.mse_loss(value.squeeze(-1), mc_return); handles (B,1) shape."""

    def test_when_value_matches_mc_return_exactly_then_mse_is_zero(self) -> None:
        mc = torch.tensor([1.0, -1.0, 0.5, 2.0])
        value = mc.unsqueeze(-1)  # (B, 1)
        assert value_mse(value, mc).item() < 1e-6

    def test_when_value_differs_from_mc_return_then_mse_is_positive(self) -> None:
        mc = torch.zeros(4)
        value = torch.ones(4, 1)
        assert value_mse(value, mc).item() > 0.0

    def test_when_value_has_b1_shape_then_result_is_scalar_tensor(self) -> None:
        mc = torch.zeros(6)
        value = torch.zeros(6, 1)
        result = value_mse(value, mc)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0

    def test_when_error_magnitude_doubles_then_mse_quadruples(self) -> None:
        """MSE is mean-squared, so doubling the error quadruples the loss."""
        mc = torch.zeros(4)
        mse_1 = value_mse(torch.ones(4, 1), mc).item()
        mse_2 = value_mse(2.0 * torch.ones(4, 1), mc).item()
        assert abs(mse_2 / mse_1 - 4.0) < 1e-4

    @given(
        st.lists(
            st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=12,
        )
    )
    @settings(max_examples=30)
    def test_when_value_exactly_matches_mc_return_then_mse_is_always_near_zero(
        self, vals: list[float]
    ) -> None:
        """Round-trip invariant: value == mc_return → mse ≈ 0 for all valid inputs."""
        mc = torch.tensor(vals)
        value = mc.unsqueeze(-1)
        assert value_mse(value, mc).item() < 1e-5


# ---------------------------------------------------------------------------
# policy_entropy
# ---------------------------------------------------------------------------


class TestPolicyEntropy:
    """Criterion: sums Categorical(logits=...).entropy().mean() across heads."""

    def test_when_logits_are_uniform_then_entropy_is_strictly_positive(self) -> None:
        result = policy_entropy(_uniform_logits())
        assert result.item() > 0.0

    def test_when_logits_are_peaked_then_entropy_is_near_zero(self) -> None:
        result = policy_entropy(_peaked_logits(target_idx=0))
        assert result.item() < 1e-2

    def test_when_uniform_vs_peaked_then_uniform_entropy_exceeds_peaked(self) -> None:
        assert policy_entropy(_uniform_logits()).item() > policy_entropy(_peaked_logits()).item()

    def test_when_called_then_result_is_non_negative_scalar_tensor(self) -> None:
        result = policy_entropy(_uniform_logits())
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
        assert result.item() >= 0.0

    @given(st.integers(min_value=1, max_value=12))
    @settings(max_examples=20)
    def test_when_batch_size_varies_then_entropy_is_always_non_negative(self, batch: int) -> None:
        """Invariant: entropy ≥ 0 for any valid batch size."""
        assert policy_entropy(_uniform_logits(batch=batch)).item() >= -1e-6


# ---------------------------------------------------------------------------
# combined_loss
# ---------------------------------------------------------------------------


class TestCombinedLoss:
    """Criterion: (Tensor, LossStats) = policy_ce + vcoef*value_mse − ecoef*entropy."""

    def test_when_called_then_returns_scalar_tensor_and_lossstats(self) -> None:
        loss, stats = combined_loss(
            _uniform_logits(),
            _targets(),
            torch.zeros(_B, 1),
            torch.zeros(_B),
            value_loss_coef=0.5,
            entropy_coef=0.01,
            label_smoothing=0.0,
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert isinstance(stats, LossStats)

    def test_when_entropy_coef_positive_then_higher_entropy_lowers_total_loss(self) -> None:
        """Entropy is a bonus (subtracted): entropy_coef=0 vs entropy_coef=1 on same logits."""
        logits = _uniform_logits()
        targets = _targets()
        mc = torch.zeros(_B)
        value = torch.zeros(_B, 1)

        loss_no_bonus, _ = combined_loss(
            logits,
            targets,
            value,
            mc,
            value_loss_coef=0.0,
            entropy_coef=0.0,
            label_smoothing=0.0,
        )
        loss_with_bonus, _ = combined_loss(
            logits,
            targets,
            value,
            mc,
            value_loss_coef=0.0,
            entropy_coef=1.0,
            label_smoothing=0.0,
        )
        # uniform logits have positive entropy; bonus subtracts it so loss is lower
        assert loss_with_bonus.item() < loss_no_bonus.item()

    def test_when_value_error_increases_then_combined_loss_increases(self) -> None:
        """↑ value-error raises total loss."""
        logits = _uniform_logits()
        targets = _targets()
        mc = torch.zeros(_B)

        loss_correct, _ = combined_loss(
            logits,
            targets,
            torch.zeros(_B, 1),
            mc,
            value_loss_coef=0.5,
            entropy_coef=0.0,
            label_smoothing=0.0,
        )
        loss_wrong, _ = combined_loss(
            logits,
            targets,
            torch.ones(_B, 1) * 10.0,
            mc,
            value_loss_coef=0.5,
            entropy_coef=0.0,
            label_smoothing=0.0,
        )
        assert loss_wrong.item() > loss_correct.item()

    def test_when_value_loss_coef_is_zero_then_value_error_does_not_affect_loss(self) -> None:
        logits = _uniform_logits()
        targets = _targets()
        mc = torch.zeros(_B)

        loss_zero_err, _ = combined_loss(
            logits,
            targets,
            torch.zeros(_B, 1),
            mc,
            value_loss_coef=0.0,
            entropy_coef=0.0,
            label_smoothing=0.0,
        )
        loss_huge_err, _ = combined_loss(
            logits,
            targets,
            999.0 * torch.ones(_B, 1),
            mc,
            value_loss_coef=0.0,
            entropy_coef=0.0,
            label_smoothing=0.0,
        )
        assert abs(loss_zero_err.item() - loss_huge_err.item()) < 1e-4

    def test_when_formula_applied_using_stats_then_result_matches_returned_loss(self) -> None:
        """Closed-form check: loss == policy_ce + vcoef*value_mse − ecoef*entropy."""
        vcoef, ecoef = 0.5, 0.01
        loss, stats = combined_loss(
            _uniform_logits(),
            _targets(),
            torch.zeros(_B, 1),
            torch.zeros(_B),
            value_loss_coef=vcoef,
            entropy_coef=ecoef,
            label_smoothing=0.0,
        )
        expected = stats.policy_ce + vcoef * stats.value_mse - ecoef * stats.entropy
        assert abs(loss.item() - expected) < 1e-4

    def test_when_lossstats_returned_then_all_four_fields_are_plain_floats(self) -> None:
        _, stats = combined_loss(
            _uniform_logits(),
            _targets(),
            torch.zeros(_B, 1),
            torch.zeros(_B),
            value_loss_coef=0.5,
            entropy_coef=0.01,
            label_smoothing=0.0,
        )
        for field in ("policy_ce", "value_mse", "entropy", "top1_acc"):
            val = getattr(stats, field)
            assert isinstance(val, float), f"LossStats.{field} must be float, got {type(val)}"


# ---------------------------------------------------------------------------
# top1_accuracy
# ---------------------------------------------------------------------------


class TestTop1Accuracy:
    """Criterion: returns per-head + aggregate top-1; param_d (dim=1) is always correct."""

    def test_when_logits_peaked_on_correct_target_then_all_per_head_accuracies_are_one(
        self,
    ) -> None:
        logits = _peaked_logits(target_idx=0)
        targets = _targets(index=0)
        result = top1_accuracy(logits, targets)
        for name in ACTION_DIMS:
            assert abs(result[name] - 1.0) < 1e-6, f"Expected acc=1 for head '{name}'"

    def test_when_logits_peaked_on_wrong_class_then_multi_class_head_accuracies_are_zero(
        self,
    ) -> None:
        """Peaked on index 0, target on index 1 → acc=0 for heads with dim≥2."""
        logits: dict[str, torch.Tensor] = {}
        targets: dict[str, torch.Tensor] = {}
        for name, dim in ACTION_DIMS.items():
            if dim >= 2:
                t = torch.full((_B, dim), -1e9)
                t[:, 0] = 1e9  # predict 0
                logits[name] = t
                targets[name] = torch.ones(_B, dtype=torch.long)  # true label 1
            else:
                logits[name] = torch.zeros(_B, dim)
                targets[name] = torch.zeros(_B, dtype=torch.long)

        result = top1_accuracy(logits, targets)
        for name, dim in ACTION_DIMS.items():
            if dim >= 2:
                assert result[name] < 1e-6, f"Expected acc=0 for head '{name}'"

    def test_when_param_d_has_dim_1_then_accuracy_is_always_one(self) -> None:
        """param_d has dim=1 — only index 0 exists, so top-1 prediction is always correct."""
        result = top1_accuracy(_uniform_logits(), _targets())
        assert abs(result["param_d"] - 1.0) < 1e-6

    def test_when_called_then_result_contains_per_head_and_aggregate_keys(self) -> None:
        result = top1_accuracy(_uniform_logits(), _targets())
        for name in ACTION_DIMS:
            assert name in result, f"Missing per-head key: '{name}'"
        assert "aggregate" in result, "Missing 'aggregate' key in top1_accuracy result"

    def test_when_accuracies_returned_then_all_values_are_in_zero_one_range(self) -> None:
        result = top1_accuracy(_uniform_logits(), _targets())
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"Accuracy out of [0, 1] for key='{key}': {val}"

    @given(st.integers(min_value=1, max_value=16))
    @settings(max_examples=20)
    def test_when_batch_size_varies_then_accuracies_are_always_in_zero_one(
        self, batch: int
    ) -> None:
        """Invariant: all accuracy values ∈ [0, 1] for any valid batch size."""
        logits = {name: torch.zeros(batch, dim) for name, dim in ACTION_DIMS.items()}
        targets = {name: torch.zeros(batch, dtype=torch.long) for name in ACTION_DIMS}
        result = top1_accuracy(logits, targets)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"acc out of [0,1]: '{key}'={val}"


# ---------------------------------------------------------------------------
# LossStats dataclass
# ---------------------------------------------------------------------------


class TestLossStats:
    """Criterion: LossStats dataclass with float fields policy_ce, value_mse, entropy, top1_acc."""

    def test_when_created_with_all_fields_then_values_are_accessible(self) -> None:
        stats = LossStats(policy_ce=0.5, value_mse=0.3, entropy=1.2, top1_acc=0.8)
        assert stats.policy_ce == 0.5
        assert stats.value_mse == 0.3
        assert stats.entropy == 1.2
        assert stats.top1_acc == 0.8

    def test_when_fields_inspected_then_all_four_required_fields_exist(self) -> None:
        stats = LossStats(policy_ce=0.0, value_mse=0.0, entropy=0.0, top1_acc=0.0)
        for field in ("policy_ce", "value_mse", "entropy", "top1_acc"):
            assert hasattr(stats, field), f"LossStats missing field '{field}'"
