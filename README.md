# risiko-rl

**Research question:** What is the best strategy to win at Risiko (Risk)?

This project trains a PPO agent to discover optimal Risiko strategy purely through self-play and games against an Ollama-served LLM opponent — run locally for free or against a hosted big model (e.g. `gpt-oss:120b`) on Ollama Cloud.

---

## Install

```bash
uv pip install -r requirements.txt
```

Python 3.12+ required. PyTorch and all dependencies are pinned in `requirements.txt`.

---

## Ollama setup

The LLM opponent calls **Ollama** via the OpenAI-compatible `/v1/chat/completions` endpoint with a `response_format` JSON schema (`strict: true`) that constrains the reply to `{"action_index": <int>}` — an index into the legal-action list, so the chosen move is legal by construction and the response is tiny.

One client serves both modes; they differ only in `base_url`, key, and model:

**Local (free, no key):** install [Ollama](https://ollama.com), pull a model, and you're done — no `.env` needed (the client defaults to `http://localhost:11434/v1`).

```bash
ollama pull gpt-oss:20b
```

**Cloud (Ollama Turbo, hosted big models like `gpt-oss:120b`):** copy the template and set your key:

```bash
cp .env.example .env
# then set OLLAMA_BASE_URL=https://ollama.com/v1 and OLLAMA_API_KEY=<your key>
```

`src/utils/env.py:ensure_env_loaded()` loads `.env` once per process (called by the CLI; lazily by the client). Auth uses the standard `Authorization: Bearer <key>` header. Local serving ignores the key, so any placeholder works.

---

## Environment variables (optional — only for cloud)

| Variable | Example | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `https://ollama.com/v1` (cloud) / `http://localhost:11434/v1` (local, default) | OpenAI-compatible base URL; the client appends `/chat/completions` |
| `OLLAMA_API_KEY` | `your-ollama-api-key` | Sent as `Authorization: Bearer <key>`; ignored by local serving |

`LLMOpponent` enforces a hard 30 s timeout per move via a `ThreadPoolExecutor` and falls back to a `RandomAgent` on any timeout, HTTP error, parse error, or out-of-range index — the returned action is always legal.

---

## Configuration

| File | Purpose |
|---|---|
| `config/default.yaml` | PPO hyperparameters, network shape, self-play settings, reward coefficients |
| `config/default_6p.yaml` | Per-player LLM sampling profiles for 6-player games |
| `config/llm_6p.yaml` | 6-player training: PPO learner vs 5 LLM opponents |

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
tensorboard --logdir results/runs/
```

Logged per episode: `episode/reward`, `episode/win`, `episode/win_rate_100`, `episode/length`, `episode/card_trade_frequency`, `episode/mean_territory`, `continent/*`, `train/policy_loss`, `train/value_loss`, `train/entropy_loss`.

---

## Baselines

Win rates at 6 players (random baseline ≈ 1/6 ≈ 16.7%):

| Agent | Win rate | Notes |
|---|---|---|
| Random | ≈ 16% | Uniform action sampling |
| LLM (gpt-oss) | ≈ 25–35% | Zero-shot tactical reasoning |
| PPO (target) | > 35% | Self-play + LLM curriculum |

---

## Six-player LLM profiles

Defined in `config/default_6p.yaml`. Each slot uses `gpt-oss:120b` (override per slot via the `model` field) with a distinct `temperature`/`top_p` and a strategy hint injected into the prompt.

| Player | Temp | Top-p | Strategy hint |
|---|---|---|---|
| 0 | 0.1 | 0.90 | Play greedily: maximise territory gain each turn. |
| 1 | 0.4 | 0.90 | Focus on securing full continents before expanding. |
| 2 | 0.7 | 0.85 | Eliminate the weakest player early; control cards. |
| 3 | 0.3 | 0.95 | Fortify borders and expand only when safe. |
| 4 | 0.9 | 0.80 | Play unpredictably; avoid predictable attack patterns. |
| 5 | 0.5 | 0.90 | Balance attack and defence; trade cards conservatively. |

---

## References

- [RULES.md](RULES.md) — full Risiko rule set used by the environment
- [CLAUDE.md](CLAUDE.md) — architecture overview and coding conventions
