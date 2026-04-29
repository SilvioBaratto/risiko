"""Tests for observation flattening and stacking utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.utils import flatten_obs, get_obs_dim, stack_obs


def _to_tensor_batched(v) -> torch.Tensor:
    """Convert a numpy value (scalar or array) to a single-batch tensor."""
    return torch.as_tensor(v).unsqueeze(0)


def _single_obs(n_players: int = 3) -> dict[str, np.ndarray]:
    """Create a synthetic single observation matching RisikoEnv shapes."""
    return {
        "territory_owner": np.random.randint(0, n_players, 42, dtype=np.int32),
        "armies": np.random.randint(1, 10, 42, dtype=np.int32),
        "phase": np.array(1, dtype=np.int32),
        "current_player": np.array(2, dtype=np.int32),
        "cards": np.random.randint(0, 2, (5, 4), dtype=np.int32),
        "continent_control": np.random.randint(0, 2, 6, dtype=np.int32),
        "trade_count": np.array(4, dtype=np.int32),
        "reinforcements_remaining": np.array(5, dtype=np.int32),
        "turn_capture": np.array(1, dtype=np.int32),
        "n_players": np.array(n_players, dtype=np.int32),
        "eliminated": np.zeros(6, dtype=np.int32),
    }


def _batch(batch_size: int, n_players: int = 3) -> dict[str, torch.Tensor]:
    """Create a batched observation dict from synthetic data."""
    obs = _single_obs(n_players)
    return {k: torch.as_tensor(np.stack([obs[k] for _ in range(batch_size)])) for k in obs}


class TestGetObsDim:
    """Observation dimension computation."""

    def test_returns_137(self):
        """Flattened observation dimension is 137."""
        assert get_obs_dim() == 137


class TestFlattenObs:
    """flatten_obs correctness."""

    def test_single_obs_shape(self):
        """Single observation produces shape (1, 137)."""
        obs = _single_obs()
        batched = {k: _to_tensor_batched(v) for k, v in obs.items()}
        flat = flatten_obs(batched)
        assert flat.shape == (1, 137)
        assert flat.dtype == torch.float32

    def test_batch_obs_shape(self):
        """Batch of 8 observations produces shape (8, 137)."""
        batched = _batch(8)
        flat = flatten_obs(batched)
        assert flat.shape == (8, 137)

    def test_output_is_float32(self):
        """Flattened output is always float32."""
        batched = _batch(1)
        flat = flatten_obs(batched)
        assert flat.dtype == torch.float32

    def test_territory_owner_normalized(self):
        """Territory owner values are divided by 5.0."""
        obs = _single_obs()
        obs["territory_owner"] = np.full(42, 5, dtype=np.int32)
        batched = {k: _to_tensor_batched(v) for k, v in obs.items()}
        flat = flatten_obs(batched)
        assert flat[0, 0] == pytest.approx(1.0)
        assert flat[0, 41] == pytest.approx(1.0)

    def test_armies_normalized(self):
        """Army counts are divided by 999.0."""
        obs = _single_obs()
        obs["armies"] = np.full(42, 999, dtype=np.int32)
        batched = {k: _to_tensor_batched(v) for k, v in obs.items()}
        flat = flatten_obs(batched)
        assert flat[0, 42] == pytest.approx(1.0)
        assert flat[0, 83] == pytest.approx(1.0)

    def test_phase_one_hot(self):
        """Phase is one-hot encoded with 5 slots."""
        obs = _single_obs()
        obs["phase"] = np.array(3, dtype=np.int32)
        batched = {k: _to_tensor_batched(v) for k, v in obs.items()}
        flat = flatten_obs(batched)
        phase_slice = flat[0, 84:89].numpy()
        expected = np.zeros(5, dtype=np.float32)
        expected[3] = 1.0
        assert np.allclose(phase_slice, expected)

    def test_current_player_one_hot(self):
        """Current player is one-hot encoded with 6 slots."""
        obs = _single_obs()
        obs["current_player"] = np.array(4, dtype=np.int32)
        batched = {k: _to_tensor_batched(v) for k, v in obs.items()}
        flat = flatten_obs(batched)
        player_slice = flat[0, 89:95].numpy()
        expected = np.zeros(6, dtype=np.float32)
        expected[4] = 1.0
        assert np.allclose(player_slice, expected)

    def test_cards_preserved(self):
        """Cards matrix is flattened unchanged."""
        obs = _single_obs()
        expected = obs["cards"].flatten().astype(np.float32)
        batched = {k: _to_tensor_batched(v) for k, v in obs.items()}
        flat = flatten_obs(batched)
        card_slice = flat[0, 95:115].numpy()
        assert np.allclose(card_slice, expected)

    def test_continent_control_preserved(self):
        """Continent control vector is flattened unchanged."""
        obs = _single_obs()
        expected = obs["continent_control"].astype(np.float32)
        batched = {k: _to_tensor_batched(v) for k, v in obs.items()}
        flat = flatten_obs(batched)
        cont_slice = flat[0, 115:121].numpy()
        assert np.allclose(cont_slice, expected)

    def test_trade_count_normalized(self):
        """Trade count is divided by 999.0."""
        obs = _single_obs()
        obs["trade_count"] = np.array(999, dtype=np.int32)
        batched = {k: _to_tensor_batched(v) for k, v in obs.items()}
        flat = flatten_obs(batched)
        assert flat[0, 121].item() == pytest.approx(1.0)

    def test_reinforcements_remaining_normalized(self):
        """Reinforcements remaining is divided by 999.0."""
        obs = _single_obs()
        obs["reinforcements_remaining"] = np.array(500, dtype=np.int32)
        batched = {k: _to_tensor_batched(v) for k, v in obs.items()}
        flat = flatten_obs(batched)
        assert flat[0, 122].item() == pytest.approx(500 / 999.0)

    def test_turn_capture_preserved(self):
        """Turn capture is kept as binary scalar."""
        obs = _single_obs()
        obs["turn_capture"] = np.array(1, dtype=np.int32)
        batched = {k: _to_tensor_batched(v) for k, v in obs.items()}
        flat = flatten_obs(batched)
        assert flat[0, 123].item() == pytest.approx(1.0)

    def test_n_players_one_hot(self):
        """n_players is one-hot encoded with 7 slots."""
        obs = _single_obs(n_players=5)
        batched = {k: _to_tensor_batched(v) for k, v in obs.items()}
        flat = flatten_obs(batched)
        n_slice = flat[0, 124:131].numpy()
        expected = np.zeros(7, dtype=np.float32)
        expected[5] = 1.0
        assert np.allclose(n_slice, expected)

    def test_eliminated_preserved(self):
        """Eliminated vector is flattened unchanged."""
        obs = _single_obs()
        obs["eliminated"] = np.array([1, 0, 0, 1, 0, 0], dtype=np.int32)
        expected = obs["eliminated"].astype(np.float32)
        batched = {k: _to_tensor_batched(v) for k, v in obs.items()}
        flat = flatten_obs(batched)
        elim_slice = flat[0, 131:137].numpy()
        assert np.allclose(elim_slice, expected)

    def test_completeness_no_nan(self):
        """Flattened output contains no NaN values."""
        batched = _batch(4)
        flat = flatten_obs(batched)
        assert not torch.isnan(flat).any()


class TestStackObs:
    """stack_obs correctness."""

    def test_returns_dict(self):
        """Output is a dict with the same keys as input observations."""
        obs_list = [_single_obs() for _ in range(3)]
        stacked = stack_obs(obs_list, "cpu")
        assert isinstance(stacked, dict)
        assert set(stacked.keys()) == set(obs_list[0].keys())

    def test_batched_shapes(self):
        """Each value has an added batch dimension."""
        obs_list = [_single_obs() for _ in range(4)]
        stacked = stack_obs(obs_list, "cpu")
        for _k, v in stacked.items():
            assert v.shape[0] == 4

    def test_tensor_type(self):
        """Stacked values are torch.Tensor instances."""
        obs_list = [_single_obs() for _ in range(2)]
        stacked = stack_obs(obs_list, "cpu")
        for v in stacked.values():
            assert isinstance(v, torch.Tensor)

    def test_device_cpu(self):
        """Tensors are placed on cpu when requested."""
        obs_list = [_single_obs() for _ in range(2)]
        stacked = stack_obs(obs_list, "cpu")
        for v in stacked.values():
            assert v.device.type == "cpu"

    def test_scalar_keys_have_batch_dim(self):
        """Scalar observations like phase gain a batch dimension."""
        obs_list = [_single_obs() for _ in range(3)]
        stacked = stack_obs(obs_list, "cpu")
        assert stacked["phase"].shape == (3,)
        assert stacked["current_player"].shape == (3,)
        assert stacked["turn_capture"].shape == (3,)
        assert stacked["n_players"].shape == (3,)

    def test_list_of_one(self):
        """Single observation list still produces batch dim of 1."""
        obs_list = [_single_obs()]
        stacked = stack_obs(obs_list, "cpu")
        assert stacked["armies"].shape[0] == 1

    def test_values_preserved(self):
        """Stacked values match the original numpy arrays."""
        obs_list = [_single_obs() for _ in range(3)]
        stacked = stack_obs(obs_list, "cpu")
        for key in obs_list[0]:
            for i, obs in enumerate(obs_list):
                np.testing.assert_array_equal(
                    stacked[key][i].numpy(),
                    obs[key],
                )
