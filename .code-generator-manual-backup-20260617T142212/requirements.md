# Risiko RL — Optimal Strategy Discovery via PPO Self-Play vs LLM Opponents

## Description

Research project to answer the question **"What is the best strategy to win at Risiko (Risk)?"**
A PPO agent learns purely through self-play and repeated matches against an Azure OpenAI **GPT-4.1** model. Strategy is not hand-coded — it emerges from reinforcement learning signals (win/loss + shaped rewards). The research question is answered by analysing the converged policy: which territories to prioritise, when to attack vs consolidate, and how to manage cards and continents.

## Tech Stack

- **Language**: Python 3.11+
- **RL Framework**: PPO from scratch (PyTorch) — `src/models/`
- **LLM Model**: Azure OpenAI **GPT-4.1** (deployment `gpt-4.1`)
- **LLM Interface**: native `httpx` client (`src/agents/azure_openai.py`) calling `/chat/completions` with `response_format` json_schema (`strict: true`)
- **LLM Runtime**: Azure OpenAI endpoint (`AZURE_OPENAI_BASE_URL`); credentials in a git-ignored `.env` (`python-dotenv`)
- **Game Environment**: Gymnasium-compatible `RisikoEnv` (`src/env.py`)
- **CLI**: Typer / Click via `risiko_rl/cli.py`
- **Visualization**: Custom renderer in `visualization/render_game.py`
- **Logging / Metrics**: TensorBoard 2.20+
- **Tests**: pytest 9.0+
- **Package manager**: uv
- **Deploy**: local (no cloud)

## Goals

> **Research Question: What is the best strategy to win at Risiko?**

The RL agent discovers strategy from trial and error, not from human rules. Once trained, the learned policy is analysed to extract emergent heuristics:

- Which continents to prioritise and in what order
- When to attack aggressively vs fortify and consolidate
- How early to target card collection and when to trade
- Attack depth vs risk exposure trade-off
- How army distribution patterns evolve across game phases

## Scope

### 1. Risiko game environment

Fully rule-compliant Gymnasium env (`src/env.py`) covering all three phases, dice combat, card system, continent bonuses, and elimination.

- **Board**: 42 territories, 6 continents with correct adjacency graph
- **State space**: territory owner (one-hot per player), army counts, current phase, card hand, continent control flags; flattened dim = **137**
- **Action space**: hierarchical per-phase
  - *Reinforcement*: territory to reinforce + army count
  - *Attack*: attacker territory + defender territory + dice count (1–3)
  - *Fortification*: source territory + destination + army count (connected path required)
  - *Card trade*: which set of 3 cards to trade in (optional, before attack)
- **Dice resolution**: attacker rolls ≤3 dice, defender rolls ≤2 dice; compare descending pairs; ties go to defender
- **Card system**: draw 1 card on capturing ≥1 territory per turn; max 5 cards in hand; escalating trade-in values (4, 6, 8, 10, 12, 15, +5 each thereafter)
- **Win condition**: last player standing
- **Multi-player**: 2–6 players; environment rotates turns and tracks eliminations

Acceptance criteria:
- [ ] A territory always has ≥ 1 army; it can never be left empty
- [ ] Attacker needs ≥ 2 armies in the source territory to attack (1 must stay behind)
- [ ] Attacking territory and target must be adjacent on the board
- [ ] On capture, attacker moves ≥ (dice used in final roll) armies into the new territory
- [ ] Fortification is one move, over a connected chain of own territories, leaving ≥ 1 behind
- [ ] Cards: draw exactly 1 per turn if ≥ 1 territory captured; trade exactly 3 cards; max 5 in hand; trade only at turn start before attacking
- [ ] Continent bonuses: North America +5, South America +2, Europe +5, Africa +3, Asia +7, Australia +2
- [ ] Environment step is deterministic given a fixed random seed
- [ ] Code coverage ≥ 80% on `src/env.py`

### 2. PPO agent

`src/models/` implements actor-critic, PPO trainer (clipped surrogate + entropy + value loss), GAE, and rollout buffer.

```python
# src/models/actor_critic.py
class ActorCritic(nn.Module):
    shared_trunk: MLP          # obs → latent (dim 137 → hidden)
    policy_head: Linear        # latent → logits over legal actions
    value_head: Linear         # latent → scalar V(s)
```

- **Observation**: flattened `Dict` space — territory ownership (one-hot per player), army counts, current phase, card hand, continent control flags; fixed dim = **137**
- **Action masking**: only legal actions receive non-zero logit mass each step
- **PPO updates**: clipped surrogate loss + entropy bonus + value loss; gradient clipped at 0.5

Acceptance criteria:
- [ ] Actor-critic implements shared trunk → policy head + value head
- [ ] Only legal actions receive non-zero logit mass each step
- [ ] PPO update applies clipped surrogate loss + entropy bonus + value loss with gradient clipped at 0.5

### 3. Self-play training loop

`training/self_play.py` trains the learner against a frozen copy of itself; promotes the frozen opponent when the learner achieves > 55% win rate.

1. Learner agent plays N-game episodes against a frozen opponent copy (also PPO or random)
2. Rollout buffer accumulates transitions; GAE computed on `flush()`
3. After each batch, learner is updated via `PPOTrainer`
4. Every `eval_games` episodes, learner faces the frozen opponent: if win rate > `promote_threshold` (default 0.55), the frozen copy is replaced with the current learner weights
5. Periodically swap in the LLM opponent (`LLMOpponent`) to measure transfer performance

Reward shaping (`src/utils/reward_config.py`):

| Signal | Value | Notes |
|--------|-------|-------|
| Win | +1.0 | sparse, end of game |
| Loss | -1.0 | sparse, on elimination |
| Territory delta | +0.01 per net gain | dense, each turn |
| Continent bonus delta | +0.05 per new continent | dense |
| Army ratio delta | +0.005 per relative gain | dense |

All dense coefficients are ablatable via `RewardConfig`; sparse-only training is a supported mode.

Acceptance criteria:
- [ ] Frozen opponent is promoted to current learner weights when learner win rate > `promote_threshold` (default 0.55) over `eval_games` episodes
- [ ] Rollout buffer accumulates transitions and computes GAE on `flush()`
- [ ] All dense reward coefficients are ablatable via `RewardConfig`; sparse-only training is supported

### 4. LLM opponent

`src/agents/llm_opponent.py` Azure-OpenAI-backed player; the **primary baseline** the RL agent must learn to beat, used as an external baseline and mid-training evaluation target.

- Backed by `src/agents/azure_openai.py:call_azure_for_action_index()` — Azure OpenAI GPT-4.1 via `/chat/completions`
- The LLM picks an **INDEX** into the `legal_actions` list (output constrained to `{"action_index": <int>}` by `response_format` json_schema `strict`), so the chosen action is always legal by construction
- One `PlayerConfig` per player slot (temperature, top_p, strategy_hint, model)
- Per-call temperature / top_p passed directly in the request body
- ≤ 30 s timeout per move (hard timeout via `ThreadPoolExecutor`); falls back to `RandomAgent` on any HTTP/timeout/parse error

Default six-player LLM profile set (used in mixed evaluations):

| Player | Temperature | top_p | Strategy Hint |
|--------|-------------|-------|---------------|
| 0      | 0.1         | 0.9   | "Play greedily: maximise territory gain each turn." |
| 1      | 0.4         | 0.9   | "Focus on securing full continents before expanding." |
| 2      | 0.7         | 0.85  | "Eliminate the weakest player early; control cards." |
| 3      | 0.3         | 0.95  | "Fortify borders and expand only when safe." |
| 4      | 0.9         | 0.8   | "Play unpredictably; avoid predictable attack patterns." |
| 5      | 0.5         | 0.9   | "Balance attack and defence; trade cards conservatively." |

These profiles are overridable via YAML or CLI flags. The six-profile pool serves as a diverse opponent curriculum for the RL agent.

Acceptance criteria:
- [ ] Each player slot uses one `PlayerConfig` (temperature, top_p, strategy_hint, model)
- [ ] Per-call temperature and top_p injected into the Azure request body at call time
- [ ] `strategy_hint` is appended to the prompt as a one-line directive before the legal actions list
- [ ] ≤ 30 s timeout per move; falls back to `RandomAgent` on any HTTP/timeout/parse error
- [ ] LLM output is an index into `legal_actions`, enforced by json_schema `strict` — the chosen action is always legal

### 5. Multi-agent runner

`src/multi_agent.py` runs any mix of `Agent`-protocol objects (PPO, LLM, random) against one `RisikoEnv`.

Acceptance criteria:
- [ ] Runs any mix of `Agent`-protocol objects (PPO, LLM, random) against a single `RisikoEnv`

### 6. Monte Carlo baseline

`training/monte_carlo.py` runs random-vs-random and random-vs-LLM baselines to establish floor win rates.

Acceptance criteria:
- [ ] Runs random-vs-random and random-vs-LLM baselines to establish floor win rates

### 7. Evaluation harness

`training/evaluate.py` runs head-to-head matches between any two agents.

- Reports win rate with Wilson score confidence intervals, game length, territory curves, elimination order

Acceptance criteria:
- [ ] Reports win rate with Wilson score confidence intervals, game length, territory curves, and elimination order

### 8. Strategy analysis

`training/strategy_analysis.py` analyses `GameResult` sequences to extract emergent heuristics.

- Continent priority, attack aggressiveness (BFS distance distribution), card trade timing, early/mid/late-game phase tagging

Acceptance criteria:
- [ ] Extracts continent priority, attack aggressiveness (BFS distance distribution), card trade timing, and early/mid/late-game phase tagging from `GameResult` sequences

### 9. Visualization

`visualization/render_game.py` renders board state per turn (ASCII or matplotlib) and produces game replay GIFs via PIL animated GIF export.

Acceptance criteria:
- [ ] Renders board state per turn in ASCII or matplotlib
- [ ] Produces game replay GIFs via PIL animated GIF export

### 10. CLI

`risiko_rl/cli.py` exposes `train`, `evaluate`, `watch`, and `benchmark` commands.

- `train` accepts `--override key=value` for any nested config field (dot notation)

Acceptance criteria:
- [ ] Exposes `train`, `evaluate`, `watch`, and `benchmark` commands
- [ ] `train` accepts `--override key=value` for any nested config field (dot notation)

### 11. TensorBoard logging

Per-episode metrics logged to TensorBoard.

- Win rate, average game length, territory curve, continent capture rate, card trade frequency, policy entropy, value loss

Acceptance criteria:
- [ ] Logs per-episode win rate, average game length, territory curve, continent capture rate, card trade frequency, policy entropy, and value loss

## Non-goals

- Cloud deployment — runs local only.

## Constraints

- LLM calls require network access to the Azure OpenAI endpoint (no local model; the laptop cannot run a local LLM)
- LLM calls must be non-blocking and time-bounded (≤ 30 s timeout per move, fallback to random legal move)
- **Credentials**: `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` live in a git-ignored `.env` at project root (`.env.example` is the committed template), loaded once per process by `src/utils/env.py:ensure_env_loaded()` via `python-dotenv`. Auth uses the `api-key` header. The base URL embeds the `gpt-4.1` deployment; the client appends `/chat/completions?api-version=...`.
- Environment step must be deterministic given a fixed random seed (`src/utils/seed.py` fixes torch, numpy, and env seeds together)
- All hyperparameters (PPO learning rate, gamma, lambda, clip epsilon, entropy coefficient, network dims, per-player LLM temperature/top_p, number of games, number of players) configurable via YAML or CLI flags — no magic numbers in source
- Results saved as CSV + TensorBoard logs under `results/`; every run tagged with its seed and config file path for reproducibility
- Checkpoints saved every `save_freq` episodes to `models/`; training fully resumable from any `.pt` file
- Code coverage ≥ 80% on `src/env.py`
- **Index-based action selection**: the LLM picks an index into `legal_actions`; output is constrained to `{"action_index": <int>}` via `response_format` json_schema (`strict: true`) — never let the model invent a raw action
- **Per-call parameter override**: per-player `temperature` and `top_p` are injected into the Azure request body at call time
- **Strategy hint injection**: `strategy_hint` is appended to the prompt as a one-line directive before the legal actions list
- **Obs dim = 137**: fixed; changing the state representation requires updating `flatten_obs()` and the network input dim together
- **Secrets hygiene**: never commit `.env`; never hardcode the API key in source

### Project Structure

```
risiko/
├── .env                        # Azure OpenAI credentials (git-ignored)
├── .env.example                # Committed template for .env
├── src/
│   ├── env.py                  # Gymnasium Risiko environment
│   ├── multi_agent.py          # Multi-player runner (any Agent mix)
│   ├── config.py               # TrainingConfig, PPOConfig, NetworkConfig, SelfPlayConfig, RewardConfig
│   ├── checkpoint.py           # save/load checkpoint (config + model + optimizer)
│   ├── tb_logger.py            # TensorBoardLogger wrapping SummaryWriter
│   ├── agents/
│   │   ├── base.py             # Agent protocol
│   │   ├── llm_opponent.py     # Azure-OpenAI-backed LLM player (accepts PlayerConfig)
│   │   ├── azure_openai.py     # call_azure_for_action_index(): GPT-4.1 /chat/completions client
│   │   ├── action_prompt.py    # render_action_prompt(): provider-agnostic board/legal-action prompt
│   │   ├── player_config.py    # PlayerConfig (temperature, top_p, strategy_hint, model)
│   │   ├── ppo_agent.py        # Wraps trained ActorCritic for inference
│   │   └── random_agent.py     # Uniform random over legal actions
│   ├── models/
│   │   ├── actor_critic.py     # Shared trunk → policy head + value head
│   │   ├── ppo.py              # PPOTrainer: clipped loss, entropy, value loss
│   │   ├── gae.py              # compute_gae(rewards, values, dones, gamma, lambda)
│   │   └── replay_buffer.py    # RolloutBuffer: accumulates transitions, calls compute_gae on flush()
│   └── utils/
│       ├── seed.py             # set_global_seeds(seed)
│       ├── env.py              # ensure_env_loaded(): load .env once via python-dotenv
│       ├── obs_utils.py        # flatten_obs() / stack_obs(); fixed obs dim = 137
│       └── reward_config.py    # RewardConfig dataclass (all dense shaping coefficients)
├── training/
│   ├── self_play.py            # SelfPlayTrainer: learner vs frozen opponent; promotion logic
│   ├── monte_carlo.py          # Win-rate baselines: random vs random, random vs LLM
│   ├── strategy_analysis.py    # Post-game heuristic extraction from GameResult sequences
│   └── evaluate.py             # Head-to-head harness with Wilson score CI
├── visualization/
│   └── render_game.py          # ASCII + matplotlib renderer; PIL animated GIF export
├── risiko_rl/
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI entry point
│   └── agent_loader.py         # Load PlayerConfig from YAML / CLI
├── config/
│   ├── default.yaml            # Training config (PPO, network, self-play, reward shaping)
│   └── default_6p.yaml         # Default six-player LLM sampling profiles
├── models/                     # Saved checkpoints (.pt)
├── results/                    # Tournament CSVs, TensorBoard logs, heatmaps
└── requirements.txt
```

## Global acceptance criteria

- [ ] LLM opponent calls Azure OpenAI GPT-4.1 via `/chat/completions` with json_schema `strict` output
- [ ] LLM calls are non-blocking and time-bounded (≤ 30 s timeout per move, fallback to random legal move)
- [ ] Azure credentials load from a git-ignored `.env` (`AZURE_OPENAI_BASE_URL` / `_API_KEY` / `_API_VERSION`); `.env.example` is committed
- [ ] LLM output is an index into `legal_actions` — the chosen action is always legal by construction
- [ ] Environment step is deterministic given a fixed random seed (torch, numpy, env seeds fixed together)
- [ ] All hyperparameters configurable via YAML or CLI flags — no magic numbers in source
- [ ] Results saved as CSV + TensorBoard logs under `results/`; every run tagged with its seed and config file path
- [ ] Checkpoints saved every `save_freq` episodes to `models/`; training fully resumable from any `.pt` file
- [ ] Code coverage ≥ 80% on `src/env.py`
- [ ] `render_action_prompt()` is the single source of truth for the LLM prompt; the API key is never hardcoded in source
- [ ] Baseline win rates (6-player): Random agent ≈ 16%; GPT-4.1 opponent (flat temperature=0.5) ≈ 25–35%; Trained PPO (target) > 35% vs mixed LLM pool