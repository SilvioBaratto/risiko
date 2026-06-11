"""End-to-end PPO log-prob round-trip check.

Runs ONE episode through the production pipeline (MultiAgentRunner →
SelfPlayTrainer-style buffer → first PPO update batch) and verifies that
``old_log_probs`` stored in the buffer match ``new_log_probs`` recomputed
at update time *before any gradient step*.

Healthy training requires |old - new| ≈ 0 on epoch 0. Before the fix this
diverged by ~17.7 (singleton vs no-mask).

Output: human-readable to stdout AND to logs/ppo_diagnostic.log.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.config import load_config
from src.env import RisikoEnv
from src.models.actor_critic import ActorCritic
from src.models.replay_buffer import RolloutBuffer
from src.models.utils import flatten_obs, get_obs_dim
from src.multi_agent import MultiAgentRunner
from src.utils.constants import ACTION_DIMS
from src.utils.log import setup_logging
from src.utils.seed import set_global_seeds

_LEARNER_ID = 0


def main() -> None:
    """Run one episode and verify stored vs recomputed log_probs match."""
    log_path = Path("logs/ppo_diagnostic.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(level="DEBUG", log_file=log_path)

    cfg = load_config(Path("config/random_6p_pretrain.yaml"))
    set_global_seeds(cfg.seed)

    env = RisikoEnv(n_players=cfg.self_play.n_players, reward_config=cfg.reward)
    net = ActorCritic(
        obs_dim=get_obs_dim(),
        hidden_size=cfg.network.hidden_sizes[0],
        num_layers=len(cfg.network.hidden_sizes),
        action_dims=ACTION_DIMS,
    )
    learner = PPOAgent(net, device="cpu")

    agents = [learner] + [RandomAgent() for _ in range(cfg.self_play.n_players - 1)]
    runner = MultiAgentRunner(env, agents, max_turns=300)
    result = runner.run_game(seed=cfg.seed)

    print("=" * 72)
    print(" PPO round-trip diagnostic")
    print("=" * 72)
    print(f"episode turns         : {result.n_turns}")
    learner_transitions = [
        t for t in result.trajectories if int(t.obs["current_player"]) == _LEARNER_ID
    ]
    print(f"learner transitions   : {len(learner_transitions)}")
    print()

    if not learner_transitions:
        print("RESULT: no learner transitions captured — episode ended before player 0 acted.")
        return

    import math as _math

    buffer = RolloutBuffer(capacity=len(learner_transitions), device="cpu")
    skipped_nan = 0
    for t in learner_transitions:
        if _math.isnan(t.log_prob) or _math.isnan(t.value):
            skipped_nan += 1
            continue
        action = {k: torch.tensor(v) for k, v in t.action.items()}
        buffer.add(t.obs, action, t.reward, False, t.value, t.log_prob, t.legal_actions)
    print(f"forced-skip transitions filtered (NaN): {skipped_nan}")
    print(f"trainable transitions in buffer       : {len(buffer)}")
    if len(buffer) == 0:
        print("RESULT: buffer empty after filtering. No learnable signal.")
        return

    buffer.compute_advantages(
        next_value=torch.tensor([0.0]),
        next_done=torch.tensor([0.0]),
        gamma=0.99,
        gae_lambda=0.95,
    )

    batches = list(buffer.get(batch_size=len(buffer), action_dims=ACTION_DIMS))
    batch = batches[0]
    obs_flat = flatten_obs(batch["obs"])
    with torch.no_grad():
        _, new_lp, _, _ = net.get_action_and_value(
            obs_flat,
            action=batch["actions"],
            action_masks=batch["action_masks"],
        )

    old_lp = batch["log_probs"]
    diff = (old_lp - new_lp).abs()

    print("Per-transition |old_log_prob - new_log_prob|:")
    for i, d in enumerate(diff.tolist()[:10]):
        print(
            f"  i={i:2d}  old={old_lp[i].item():+.4f}  new={new_lp[i].item():+.4f}  diff={d:+.4f}"
        )
    if len(diff) > 10:
        print(f"  ... ({len(diff) - 10} more transitions)")
    print()

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"max  |old - new| : {max_diff:.6f}")
    print(f"mean |old - new| : {mean_diff:.6f}")
    print()

    # PPO is healthy when approx_kl < 0.05. mean_diff is the closest analogue
    # to the per-update approx_kl reported in TensorBoard. Pre-fix this was
    # ~17.7; post-fix it should be ≪ 0.05.
    if mean_diff < 0.05 and max_diff < 0.5:
        print(
            f"RESULT: PASS. mean diff {mean_diff:.4f}, max diff {max_diff:.4f}. "
            "PPO will train correctly. (Tiny per-sample diffs are float32 noise.)"
        )
    elif mean_diff < 1.0:
        print(
            f"RESULT: SUSPICIOUS. mean diff {mean_diff:.4f} — within tolerance "
            "but check the failing transitions above."
        )
    else:
        print(f"RESULT: FAIL. mean diff {mean_diff:.4f} ≫ 0. Bug still present.")


if __name__ == "__main__":
    main()
