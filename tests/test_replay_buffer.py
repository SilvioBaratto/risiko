"""Tests for the Rollout Buffer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.replay_buffer import RolloutBuffer


def _make_buffer(capacity: int = 10) -> RolloutBuffer:
    """Create a RolloutBuffer with a small capacity for testing."""
    return RolloutBuffer(capacity=capacity, device="cpu")


def _sample_obs(batch_size: int = 1) -> dict[str, np.ndarray]:
    """Create a synthetic observation dict."""
    return {
        "territory_owner": np.random.randint(0, 6, 42, dtype=np.int32),
        "armies": np.random.randint(1, 10, 42, dtype=np.int32),
        "phase": np.array(1, dtype=np.int32),
        "current_player": np.array(2, dtype=np.int32),
        "cards": np.random.randint(0, 2, (5, 4), dtype=np.int32),
        "continent_control": np.random.randint(0, 2, 6, dtype=np.int32),
        "trade_count": np.array(4, dtype=np.int32),
        "reinforcements_remaining": np.array(5, dtype=np.int32),
        "turn_capture": np.array(1, dtype=np.int32),
        "n_players": np.array(3, dtype=np.int32),
        "eliminated": np.zeros(6, dtype=np.int32),
    }


def _sample_action() -> dict[str, torch.Tensor]:
    """Create a synthetic action dict."""
    return {
        "action_type": torch.tensor(0),
        "param_a": torch.tensor(1),
        "param_b": torch.tensor(2),
        "param_c": torch.tensor(3),
        "param_d": torch.tensor(4),
    }


class TestInit:
    """Construction contracts."""

    def test_is_empty_at_start(self):
        """Fresh buffer has length 0."""
        buf = _make_buffer()
        assert len(buf) == 0

    def test_capacity_is_set(self):
        """Buffer stores the passed capacity."""
        buf = _make_buffer(capacity=50)
        assert buf.capacity == 50

    def test_device_is_set(self):
        """Buffer stores the passed device."""
        buf = _make_buffer()
        assert buf.device == "cpu"


class TestAdd:
    """Adding transitions."""

    def test_length_increases_after_add(self):
        """Adding a transition increments length."""
        buf = _make_buffer()
        buf.add(
            state=_sample_obs(),
            action=_sample_action(),
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=-1.0,
        )
        assert len(buf) == 1

    def test_can_add_multiple(self):
        """Adding multiple transitions grows length."""
        buf = _make_buffer()
        for _ in range(5):
            buf.add(
                state=_sample_obs(),
                action=_sample_action(),
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
            )
        assert len(buf) == 5

    def test_dict_actions_preserved(self):
        """Action dict keys are preserved in the buffer."""
        buf = _make_buffer()
        action = _sample_action()
        buf.add(
            state=_sample_obs(),
            action=action,
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=-1.0,
        )
        for key in action:
            assert key in buf.actions[0]

    def test_cannot_exceed_capacity(self):
        """Adding beyond capacity raises IndexError."""
        buf = _make_buffer(capacity=3)
        for _ in range(3):
            buf.add(
                state=_sample_obs(),
                action=_sample_action(),
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
            )
        with pytest.raises(IndexError):
            buf.add(
                state=_sample_obs(),
                action=_sample_action(),
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
            )


class TestComputeAdvantages:
    """Advantage computation contracts."""

    def test_computes_advantages(self):
        """compute_advantages sets advantages and returns."""
        buf = _make_buffer(capacity=5)
        for _ in range(5):
            buf.add(
                state=_sample_obs(),
                action=_sample_action(),
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
            )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
            gamma=0.99,
            gae_lambda=0.95,
        )
        assert buf.advantages is not None
        assert buf.returns is not None

    def test_raises_if_buffer_empty(self):
        """compute_advantages on empty buffer raises ValueError."""
        buf = _make_buffer()
        with pytest.raises(ValueError):
            buf.compute_advantages(
                next_value=torch.tensor([0.0]),
                next_done=torch.tensor([0.0]),
            )

    def test_advantage_shape_matches(self):
        """Advantages shape matches buffer length."""
        buf = _make_buffer(capacity=10)
        for _ in range(7):
            buf.add(
                state=_sample_obs(),
                action=_sample_action(),
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
            )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        assert buf.advantages is not None
        assert buf.advantages.shape == (7,)

    def test_returns_shape_matches(self):
        """Returns shape matches buffer length."""
        buf = _make_buffer(capacity=10)
        for _ in range(7):
            buf.add(
                state=_sample_obs(),
                action=_sample_action(),
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
            )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        assert buf.returns is not None
        assert buf.returns.shape == (7,)

    def test_done_resets(self):
        """A done in the middle resets GAE accumulation."""
        buf = _make_buffer(capacity=3)
        for _ in range(2):
            buf.add(
                state=_sample_obs(),
                action=_sample_action(),
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
            )
        buf.add(
            state=_sample_obs(),
            action=_sample_action(),
            reward=1.0,
            done=True,
            value=0.5,
            log_prob=-1.0,
        )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
            gamma=0.99,
            gae_lambda=0.95,
        )
        assert buf.advantages is not None
        # Step 1 (before done at step 2) should not bootstrap
        assert buf.advantages[1].item() == pytest.approx(0.5, abs=1e-5)
        # Step 2 still bootstraps from next_value since next_done is 0
        assert buf.advantages[2].item() == pytest.approx(0.995, abs=1e-5)


class TestGet:
    """Mini-batch generation."""

    def test_raises_before_compute_advantages(self):
        """get() raises ValueError if advantages not computed."""
        buf = _make_buffer(capacity=5)
        for _ in range(5):
            buf.add(
                state=_sample_obs(),
                action=_sample_action(),
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
            )
        with pytest.raises(ValueError):
            next(buf.get(batch_size=2))

    def test_mini_batch_size_matches(self):
        """Each mini-batch has exactly batch_size samples."""
        buf = _make_buffer(capacity=10)
        for _ in range(10):
            buf.add(
                state=_sample_obs(),
                action=_sample_action(),
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
            )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        batch = next(buf.get(batch_size=4))
        assert len(batch["obs"]["phase"]) == 4

    def test_mini_batches_cover_all(self):
        """All samples are yielded across mini-batches."""
        buf = _make_buffer(capacity=8)
        for _ in range(8):
            buf.add(
                state=_sample_obs(),
                action=_sample_action(),
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
            )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        batches = list(buf.get(batch_size=3))
        total = sum(len(b["obs"]["phase"]) for b in batches)
        assert total == 8

    def test_shuffling_changes_order(self):
        """get() shuffles samples across calls."""
        buf = _make_buffer(capacity=5)
        for i in range(5):
            buf.add(
                state=_sample_obs(),
                action=_sample_action(),
                reward=float(i),
                done=False,
                value=0.5,
                log_prob=-1.0,
            )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        for _ in range(20):
            b1 = next(buf.get(batch_size=5))
            b2 = next(buf.get(batch_size=5))
            if not torch.equal(b1["rewards"], b2["rewards"]):
                return
        pytest.fail("Shuffling never changed order in 20 tries")


class TestClear:
    """Buffer reset."""

    def test_clear_resets_length(self):
        """clear() resets length to 0."""
        buf = _make_buffer()
        for _ in range(5):
            buf.add(
                state=_sample_obs(),
                action=_sample_action(),
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
            )
        buf.clear()
        assert len(buf) == 0

    def test_clear_removes_advantages(self):
        """clear() resets advantages and returns."""
        buf = _make_buffer()
        for _ in range(5):
            buf.add(
                state=_sample_obs(),
                action=_sample_action(),
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
            )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        buf.clear()
        assert buf.advantages is None
        assert buf.returns is None


class TestDictActions:
    """Action dict preservation."""

    def test_action_dict_keys_match(self):
        """Buffered actions have the same keys as the input action."""
        buf = _make_buffer()
        action = _sample_action()
        buf.add(
            state=_sample_obs(),
            action=action,
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=-1.0,
        )
        assert set(buf.actions[0].keys()) == set(action.keys())

    def test_action_values_are_tensors(self):
        """Buffered action values are torch tensors."""
        buf = _make_buffer()
        buf.add(
            state=_sample_obs(),
            action=_sample_action(),
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=-1.0,
        )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        batch = next(buf.get(batch_size=1))
        for key in batch["actions"]:
            assert isinstance(batch["actions"][key], torch.Tensor)


class TestMiniBatchContents:
    """Contents of yielded mini-batches."""

    def test_batch_contains_obs(self):
        """Mini-batch contains observations."""
        buf = _make_buffer()
        buf.add(
            state=_sample_obs(),
            action=_sample_action(),
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=-1.0,
        )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        batch = next(buf.get(batch_size=1))
        assert "obs" in batch

    def test_batch_contains_actions(self):
        """Mini-batch contains actions."""
        buf = _make_buffer()
        buf.add(
            state=_sample_obs(),
            action=_sample_action(),
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=-1.0,
        )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        batch = next(buf.get(batch_size=1))
        assert "actions" in batch

    def test_batch_contains_advantages(self):
        """Mini-batch contains advantages."""
        buf = _make_buffer()
        buf.add(
            state=_sample_obs(),
            action=_sample_action(),
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=-1.0,
        )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        batch = next(buf.get(batch_size=1))
        assert "advantages" in batch

    def test_batch_contains_returns(self):
        """Mini-batch contains returns."""
        buf = _make_buffer()
        buf.add(
            state=_sample_obs(),
            action=_sample_action(),
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=-1.0,
        )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        batch = next(buf.get(batch_size=1))
        assert "returns" in batch

    def test_batch_contains_log_probs(self):
        """Mini-batch contains old log probs."""
        buf = _make_buffer()
        buf.add(
            state=_sample_obs(),
            action=_sample_action(),
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=-1.0,
        )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        batch = next(buf.get(batch_size=1))
        assert "log_probs" in batch

    def test_batch_contains_values(self):
        """Mini-batch contains old values."""
        buf = _make_buffer()
        buf.add(
            state=_sample_obs(),
            action=_sample_action(),
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=-1.0,
        )
        buf.compute_advantages(
            next_value=torch.tensor([0.5]),
            next_done=torch.tensor([0.0]),
        )
        batch = next(buf.get(batch_size=1))
        assert "values" in batch
