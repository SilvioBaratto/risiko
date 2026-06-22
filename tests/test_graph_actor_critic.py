"""Tests for GraphSAGEActorCritic — interface-compatibility with ActorCritic."""

from __future__ import annotations

import torch

from src.models.graph_actor_critic import GraphSAGEActorCritic
from src.models.utils import get_obs_dim
from src.utils.constants import ACTION_DIMS

OBS_DIM = get_obs_dim()


def _net() -> GraphSAGEActorCritic:
    return GraphSAGEActorCritic(OBS_DIM, hidden_size=64, num_layers=2, action_dims=ACTION_DIMS)


def test_forward_returns_per_head_logits_and_value():
    net = _net()
    logits, value = net(torch.rand(4, OBS_DIM))
    for name, dim in ACTION_DIMS.items():
        assert logits[name].shape == (4, dim)
    assert value.shape == (4, 1)


def test_single_obs_is_batched():
    net = _net()
    logits, value = net(torch.rand(OBS_DIM))
    assert logits["action_type"].shape == (1, ACTION_DIMS["action_type"])
    assert value.shape == (1, 1)


def test_get_action_and_value_samples_all_heads():
    net = _net()
    action, log_prob, entropy, value = net.get_action_and_value(torch.rand(5, OBS_DIM))
    assert set(action) == set(ACTION_DIMS)
    assert log_prob.shape == (5,)
    assert entropy.shape == (5,)
    assert value.shape == (5, 1)


def test_get_action_and_value_evaluates_given_action():
    net = _net()
    obs = torch.rand(3, OBS_DIM)
    action, _, _, _ = net.get_action_and_value(obs)
    _, log_prob, _, _ = net.get_action_and_value(obs, action=action)
    assert log_prob.shape == (3,)
    assert torch.isfinite(log_prob).all()


def test_action_mask_excludes_masked_choices():
    net = _net()
    obs = torch.rand(8, OBS_DIM)
    n = ACTION_DIMS["action_type"]
    mask = torch.zeros(8, n, dtype=torch.bool)
    mask[:, 0] = True  # only action_type 0 is legal
    masks = {"action_type": mask}
    action, _, _, _ = net.get_action_and_value(obs, action_masks=masks)
    assert torch.all(action["action_type"] == 0)
