"""GraphSAGE actor-critic with per-node action heads over the territory graph.

Drop-in replacement for ``ActorCritic`` — identical constructor signature and
``forward`` / ``get_action_and_value`` interface (takes the flat 137-dim obs,
returns the same per-head logits dict + value), so PPOTrainer, the rollout
buffer, action masking, and checkpointing are unchanged.

Why a GNN: the flat MLP can't see that Risiko is a graph. This trunk un-flattens
the obs into per-territory node features, runs GraphSAGE message passing over the
fixed ``ADJACENCY`` graph, and — crucially — selects territories with **per-node
action heads**: the territory-valued logits of ``param_a`` (attack source /
reinforce / fortify src / capture-move) and ``param_b`` (defender / fortify dst)
come directly from each territory's own node embedding, not from a pooled global
vector. (Pooling-then-flat-head, the earlier version, discarded exactly the
spatial signal those heads need — it couldn't even win the easiest curriculum
stage.) Count/dice logits (indices 42..200) and the non-spatial heads
(action_type, param_c, param_d, value) still come from the pooled board+global
embedding.

Flat-obs layout (see src/models/utils.py ``_CONCAT_ORDER``):
    [0:42]   territory_owner / 5.0
    [42:84]  armies / 999.0
    [84:89]  phase one-hot(5)
    [89:95]  current_player one-hot(6)
    [95:137] cards/continent_control/... (remaining globals)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.actor_critic import _build_distributions
from src.utils.constants import ADJACENCY, CONTINENTS, NUM_TERRITORIES

# Flat-obs slice boundaries (tied to utils._CONCAT_ORDER).
_OWNER_SLICE = (0, NUM_TERRITORIES)
_ARMIES_SLICE = (NUM_TERRITORIES, 2 * NUM_TERRITORIES)
_GLOBAL_START = 2 * NUM_TERRITORIES  # everything after armies is global scalars
_CURPLAYER_SLICE = (2 * NUM_TERRITORIES + 5, 2 * NUM_TERRITORIES + 11)  # one-hot(6)
_MAX_PLAYERS = 6
_NODE_FEAT_DIM = _MAX_PLAYERS + 1 + 1 + len(CONTINENTS)  # owner + armies + is_mine + continent
# Heads whose first NUM_TERRITORIES logits select a territory → score per-node.
_NODE_HEADS = ("param_a", "param_b")


def _row_normalized_adjacency() -> torch.Tensor:
    """Mean-aggregator adjacency (row-normalized) for the territory graph."""
    a = torch.zeros(NUM_TERRITORIES, NUM_TERRITORIES)
    for src, neighbors in ADJACENCY.items():
        for dst in neighbors:
            a[src, dst] = 1.0
    deg = a.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return a / deg


def _continent_one_hot() -> torch.Tensor:
    """Static (NUM_TERRITORIES, n_continents) one-hot of each territory's continent."""
    n = len(CONTINENTS)
    oh = torch.zeros(NUM_TERRITORIES, n)
    for ci, ids in enumerate(CONTINENTS.values()):
        for t in ids:
            oh[t, ci] = 1.0
    return oh


class _SAGELayer(nn.Module):
    """One GraphSAGE layer: h' = ReLU(W_self h + W_neigh (A_norm h))."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)  # stabilizes GNN optimization
        self.act = nn.ReLU()

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        neigh = torch.einsum("ij,bjf->bif", adj_norm, h)  # (B, N, F) neighbor mean
        return self.act(self.norm(self.self_lin(h) + self.neigh_lin(neigh)))


class _NodeAwareHead(nn.Module):
    """Logits where the first ``n_nodes`` come per-node, the rest from global.

    Used for territory-or-count params: a node's territory logit is scored from
    its own embedding (spatial); count/dice logits come from the board+global
    embedding.
    """

    def __init__(self, hidden: int, n_nodes: int, total_dim: int):
        super().__init__()
        self.node_score = nn.Linear(hidden, 1)
        self.rest = nn.Linear(hidden, total_dim - n_nodes)

    def forward(self, node_emb: torch.Tensor, global_emb: torch.Tensor) -> torch.Tensor:
        node_logits = self.node_score(node_emb).squeeze(-1)  # (B, N)
        rest_logits = self.rest(global_emb)  # (B, total - N)
        return torch.cat([node_logits, rest_logits], dim=-1)


class GraphSAGEActorCritic(nn.Module):
    """GraphSAGE trunk + per-node action heads; interface-compatible with ActorCritic."""

    def __init__(
        self,
        obs_dim: int,
        hidden_size: int,
        num_layers: int,
        action_dims: dict[str, int],
    ):
        """Args mirror ``ActorCritic``; ``num_layers`` = number of SAGE layers."""
        super().__init__()
        self.action_dims = action_dims
        self.obs_dim = obs_dim

        self.register_buffer("_adj_norm", _row_normalized_adjacency())
        self.register_buffer("_continent", _continent_one_hot())

        self.node_encoder = nn.Sequential(
            nn.Linear(_NODE_FEAT_DIM, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU()
        )
        self.sage_layers = nn.ModuleList(
            _SAGELayer(hidden_size, hidden_size) for _ in range(max(1, num_layers))
        )
        n_global = obs_dim - _GLOBAL_START
        self.global_encoder = nn.Sequential(
            nn.Linear(n_global, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU()
        )
        # pooled board (mean+max = 2*hidden) + globals (hidden) → joint hidden
        self.joint = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU()
        )

        self.global_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_size, dim)
                for name, dim in action_dims.items()
                if name not in _NODE_HEADS
            }
        )
        self.node_heads = nn.ModuleDict(
            {
                name: _NodeAwareHead(hidden_size, NUM_TERRITORIES, action_dims[name])
                for name in _NODE_HEADS
                if name in action_dims
            }
        )
        self.value = nn.Linear(hidden_size, 1)

    def _node_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Un-flatten the obs into per-territory node features (B, N, F)."""
        owner_norm = obs[:, _OWNER_SLICE[0] : _OWNER_SLICE[1]]  # (B, N)
        armies_norm = obs[:, _ARMIES_SLICE[0] : _ARMIES_SLICE[1]]  # (B, N)
        owner = (owner_norm * 5.0).round().long().clamp_(0, _MAX_PLAYERS - 1)  # (B, N)
        owner_oh = torch.nn.functional.one_hot(owner, _MAX_PLAYERS).float()  # (B, N, 6)
        cur_player = obs[:, _CURPLAYER_SLICE[0] : _CURPLAYER_SLICE[1]].argmax(dim=-1)  # (B,)
        is_mine = (owner == cur_player.unsqueeze(-1)).float().unsqueeze(-1)  # (B, N, 1)
        continent = self._continent.unsqueeze(0).expand(obs.shape[0], -1, -1)  # (B, N, 6)
        return torch.cat([owner_oh, armies_norm.unsqueeze(-1), is_mine, continent], dim=-1)

    def forward(self, obs: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass returning policy logits and state value."""
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        h = self.node_encoder(self._node_features(obs))  # (B, N, H)
        for layer in self.sage_layers:
            h = layer(h, self._adj_norm)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)  # (B, 2H)
        g = self.global_encoder(obs[:, _GLOBAL_START:])  # (B, H)
        joint = self.joint(torch.cat([pooled, g], dim=-1))  # (B, H)

        logits = {name: head(joint) for name, head in self.global_heads.items()}
        for name, head in self.node_heads.items():
            logits[name] = head(h, joint)
        return logits, self.value(joint)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: dict[str, torch.Tensor] | None = None,
        action_masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate an action; returns ``(action, log_prob, entropy, value)``."""
        logits, value = self.forward(obs)
        distributions = _build_distributions(logits, action_masks)
        if action is None:
            action = {name: dist.sample() for name, dist in distributions.items()}
        log_probs = torch.stack([distributions[name].log_prob(action[name]) for name in logits])
        entropies = torch.stack([distributions[name].entropy() for name in logits])
        return action, log_probs.sum(dim=0), entropies.sum(dim=0), value
