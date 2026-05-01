# risiko-rl

**Research question:** What is the best strategy to win at Risiko (Risk)?

This project trains a PPO agent to discover optimal Risiko strategy purely through self-play and games against a locally-running Qwen LLM opponent. All training and inference run fully offline.

---

## Install

```bash
uv pip install -r requirements.txt
```

Python 3.12+ required. PyTorch, BAML, and all dependencies are pinned in `requirements.txt`.

---

## Ollama setup

The LLM opponent uses a custom Ollama model built from Qwen 3.5.

```bash
# Pull the base model
ollama pull qwen3.5:latest

# Build the risiko model with the project Modelfile
ollama create risiko -f Modelfile
```

The `Modelfile` sets a JSON-only system prompt, 8192-token context, and a repeat penalty tuned for rule-following output.

---

## Required environment variables

| Variable | Value | Purpose |
|---|---|---|
| `OLLAMA_KEEP_ALIVE` | `0` | Evict model from GPU RAM after every call |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Override Ollama endpoint (optional) |

### Why `OLLAMA_KEEP_ALIVE=0`

The BAML client accesses Ollama through the OpenAI-compatible `/v1/chat/completions` endpoint. Ollama **silently ignores** the per-call `keep_alive` parameter on this endpoint — it only works on the native `/api/generate` and `/api/chat` endpoints. Setting `OLLAMA_KEEP_ALIVE=0` at the server level ensures the model is evicted from GPU RAM after each call regardless of which endpoint is used.

The code also implements a second eviction layer: after every BAML response, `src/agents/ollama_eviction.py` fires a non-blocking POST to `http://localhost:11434/api/generate` with `{"keep_alive": 0}`. Both layers are required.

Start Ollama with the variable set:

```bash
OLLAMA_KEEP_ALIVE=0 ollama serve
```

Or export it in your shell profile before running any training command.

---

## Configuration

| File | Purpose |
|---|---|
| `config/default.yaml` | PPO hyperparameters, network shape, self-play settings, reward coefficients |
| `configs/default_6p.yaml` | Per-player LLM sampling profiles for 6-player games |

Key defaults in `config/default.yaml`:

```yaml
ppo:
  lr: 0.0003
  gamma: 0.99
  n_steps: 2048
  n_epochs: 10

self_play:
  n_players: 2
  promote_threshold: 0.55

reward:
  sparse_win: 1.0
  sparse_loss: -1.0
  dense_territory_delta: 0.01
  dense_continent_bonus_delta: 0.05
  dense_army_ratio: 0.005
```

---

## CLI

The CLI is installed as `risiko-rl` or run via `python -m risiko_rl.cli`.

| Command | Description |
|---|---|
| `train` | Run self-play PPO training |
| `evaluate` | Head-to-head evaluation between two agents |
| `watch` | Watch a single game rendered frame-by-frame |
| `benchmark` | Monte Carlo baselines (random vs random, random vs LLM) |

### `train`

```bash
# Train with default config
risiko-rl train

# Override any config field with dot notation
risiko-rl train --override ppo.lr=1e-4 --override ppo.gamma=0.97

# Resume from checkpoint, save to custom dir
risiko-rl train --config config/default.yaml --checkpoint-dir models/ --seed 0
```

### `evaluate`

```bash
# Random agent vs LLM, 100 games
risiko-rl evaluate --agent-a random --agent-b llm --n-games 100

# PPO checkpoint vs LLM, export CSV
risiko-rl evaluate --agent-a models/best.pt --agent-b llm --n-games 200 --output results.csv
```

### `watch`

```bash
# Watch two random agents in ASCII mode
risiko-rl watch --agent1 random --agent2 random --mode ascii

# Save PIL animated GIF replay
risiko-rl watch --agent1 models/best.pt --agent2 random --output replay.gif
```

### `benchmark`

```bash
# Run baseline tournament (50 games per matchup)
risiko-rl benchmark --games 50

# Include RL checkpoint and export CSV
risiko-rl benchmark --games 100 --rl-checkpoint models/best.pt --output baselines.csv
```

---

## TensorBoard

```bash
tensorboard --logdir runs/
```

Logged per episode: `episode/reward`, `episode/win`, `episode/win_rate_100`, `episode/length`, `episode/card_trade_frequency`, `episode/mean_territory`, `continent/*`, `train/policy_loss`, `train/value_loss`, `train/entropy_loss`.

---

## Baselines

Win rates at 6 players (random baseline ≈ 1/6 ≈ 16.7%):

| Agent | Win rate | Notes |
|---|---|---|
| Random | ≈ 16% | Uniform action sampling |
| Qwen LLM (`risiko` model) | ≈ 25–35% | Zero-shot tactical reasoning |
| PPO (target) | > 35% | Self-play + LLM curriculum |

---

## Six-player LLM profiles

Defined in `configs/default_6p.yaml`. Each slot uses the shared `risiko` Ollama model with different sampling parameters and a strategy hint injected into the prompt.

| Player | Temp | Top-p | Top-k | Repeat penalty | Strategy hint |
|---|---|---|---|---|---|
| 0 | 0.1 | 0.90 | 40 | 1.10 | Play greedily: maximise territory gain each turn. |
| 1 | 0.4 | 0.90 | 40 | 1.10 | Focus on securing full continents before expanding. |
| 2 | 0.7 | 0.85 | 50 | 1.15 | Eliminate the weakest player early; control cards. |
| 3 | 0.3 | 0.95 | 30 | 1.20 | Fortify borders and expand only when safe. |
| 4 | 0.9 | 0.80 | 60 | 1.05 | Play unpredictably; avoid predictable attack patterns. |
| 5 | 0.5 | 0.90 | 40 | 1.10 | Balance attack and defence; trade cards conservatively. |

---

## BAML

LLM function definitions live in `baml_src/`. After editing any `.baml` file, regenerate the client:

```bash
baml-cli generate
```

Never edit `baml_client/` directly — it is auto-generated.

---

## References

- [RULES.md](RULES.md) — full Risiko rule set used by the environment
- [CLAUDE.md](CLAUDE.md) — architecture overview and coding conventions
