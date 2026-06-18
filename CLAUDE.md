# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Research project: train a PPO agent to discover optimal Risiko (Risk) strategy purely through self-play and games against an Ollama-served LLM opponent (local models for free, or hosted models like `gpt-oss:120b` via Ollama Cloud for training). The question driving everything is *"What is the best strategy to win at Risiko?"*

## Commands

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run all tests
pytest

# Run a single test file
pytest tests/test_env_core.py -v

# Run tests with coverage (must hit ≥ 80% on src/)
pytest --cov=src --cov-report=term-missing

# Skip slow/integration tests
pytest -m "not slow and not integration"

# Lint / format
ruff check .
ruff format .

# CLI (installed as `risiko-rl` or via python -m)
risiko-rl pretrain --config config/bc_pretrain.yaml          # BC warm-start: dataset gen + supervised clone → models/pretrained.pt
risiko-rl pretrain --config config/bc_pretrain.yaml --override bc.n_games=5000
cp models/pretrained.pt models/latest.pt                     # warm-start PPO: place BC weights where train auto-resumes
risiko-rl train --config config/default.yaml
risiko-rl train --lr 1e-4 --seed 0 --override ppo.gamma=0.97
risiko-rl evaluate --checkpoint models/best.pt
risiko-rl watch --checkpoint models/best.pt
risiko-rl benchmark

# TensorBoard
tensorboard --logdir runs/
```

## Architecture

### Core layers

**`src/env.py`** — Gymnasium-compatible Risiko environment. Most critical component (≥ 80% test coverage required). Internal turn rotation: `step()` advances `state.current_player` automatically. The observation is a `Dict` space; call `get_legal_actions()` before every `step()`.

**`src/multi_agent.py`** — `MultiAgentRunner` coordinates any mix of `Agent`-protocol objects against one `RisikoEnv`. Returns `GameResult` (winner, turn count, per-player territory/army curves, elimination order).

**`src/agents/`** — All agents implement the `Agent` protocol (`base.py`):
- `random_agent.py` — uniform sample over legal actions
- `llm_opponent.py` — `LLMOpponent`: wraps `ThreadPoolExecutor` for hard timeout, falls back to `RandomAgent` on any HTTP/timeout/parse error; model, Ollama base_url/api_key configurable via constructor
- `ollama_client.py` — `call_ollama_for_action_index()`: HTTP client for Ollama's OpenAI-compatible `/v1/chat/completions` with `response_format` json_schema (`strict`) enforced output; the LLM picks an INDEX into the legal_actions list (guaranteed legal). One client serves local (`http://localhost:11434/v1`) and cloud (`https://ollama.com/v1`) — defaults to local with no key required
- `action_prompt.py` — `render_action_prompt()`: provider-agnostic board/legal-action prompt rendering shared by the LLM client
- `ppo_agent.py` — wraps a trained `ActorCritic` for inference

**`src/models/`** — PPO from scratch:
- `actor_critic.py` — shared trunk → policy head + value head
- `ppo.py` — `PPOTrainer`: clipped surrogate loss, entropy bonus, value loss; imports `ActorCritic`, `RolloutBuffer`, `compute_gae`
- `gae.py` — `compute_gae(rewards, values, dones, gamma, lambda)`
- `replay_buffer.py` — `RolloutBuffer`: accumulates transitions, calls `compute_gae` on `flush()`
- `utils.py` — `flatten_obs()` / `stack_obs()` (normalizes Box fields, one-hot encodes discrete fields); fixed obs dim = **137**

**`src/config.py`** — Frozen dataclasses (`TrainingConfig`, `PPOConfig`, `NetworkConfig`, `SelfPlayConfig`, `RewardConfig`). Load with `load_config(path)`, apply CLI overrides with `merge_cli_overrides(cfg, {"ppo.lr": "1e-4"})`.

**`src/checkpoint.py`** — `save_checkpoint()` / `load_checkpoint()` — serializes config + model + optimizer state; training is resumable from any saved `.pt` file.

**`src/tb_logger.py`** — `TensorBoardLogger`: wraps `SummaryWriter`; call `log_training_step()` and `log_game_result()` with a `GameResult`.

**`training/`**:
- `self_play.py` — `SelfPlayTrainer`: pits learner against frozen opponent copy; promotes opponent when learner win-rate > `self_play.promote_threshold` (default 0.55) over `self_play.eval_games` games
- `monte_carlo.py` — win-rate baselines: random vs random, random vs LLM, RL vs LLM
- `evaluate.py` — head-to-head harness; reports win rate, game length, territory curves, elimination order
- `strategy_analysis.py` — post-hoc analysis of `GameResult` sequences: BFS distances, continent control, early/mid/late-game phase tagging
- `bc_dataset.py` — offline self-play → sharded `.npz` dataset (`generate_bc_dataset`); `bc_trainer.py` — supervised BC clone into `ActorCritic`; `bc_eval.py` — pretrained win-rate gate with Wilson CIs (`evaluate_pretrained`). Warm-start convention: `cp models/pretrained.pt models/latest.pt` then `risiko-rl train` (auto-resumes `latest.pt` — no `--resume` flag).

**`visualization/render_game.py`** — ASCII + matplotlib renderer; PIL animated GIF export.

**`risiko_rl/cli.py`** — Typer CLI. `train` accepts `--override key=value` for any nested config field (dot notation). Entry point: `risiko_rl = "risiko_rl.cli:main"`.

### LLM / Ollama

- LLM opponents call **Ollama** via `src/agents/ollama_client.py` — `POST {OLLAMA_BASE_URL}/chat/completions` (OpenAI-compatible) with `response_format` json_schema (`strict: true`) enforcing `{"action_index": <int>}`. The LLM picks an index into the `legal_actions` list rather than inventing an action, so output is guaranteed legal and small. One client serves both local and cloud — they differ only in `base_url`, key, and model name.
- Config lives in a **git-ignored `.env`** at the project root (`.env.example` is the template): `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`; set to `https://ollama.com/v1` for cloud) and `OLLAMA_API_KEY` (only needed for cloud — local serving ignores it). `src/utils/env.py:ensure_env_loaded()` loads it once per process (called by the CLI; lazily by the client). Auth uses the standard `Authorization: Bearer <key>` header. Local Ollama needs no `.env` at all.
- Per-player LLM sampling profiles (temperature, top_p, strategy_hint, model) live in `config/default_6p.yaml`; loaded by `src/agents/player_config.py`.

### Key invariants the environment must enforce

- Territory always has ≥ 1 army; attacker needs ≥ 2 in source
- Attack only to adjacent enemy territory; on capture move ≥ (dice used) armies in
- Cards: draw 1 per turn if ≥ 1 territory captured; trade exactly 3; max 5 in hand; trade only before attacking
- Continent bonuses: NA +5, SA +2, EU +5, AF +3, AS +7, AU +2
- Fortification: single move over a connected chain of own territories, leave ≥ 1 behind

### Configuration & reproducibility

- All hyperparameters live in `config/default.yaml`; no magic numbers in source.
- `src/utils/seed.py` — `set_global_seeds(seed)` fixes `torch`, `numpy`, and env seeds together; log the seed for every run.
- Checkpoints saved every `save_freq` episodes to `models/`; resumable from any `.pt`.
- Reward shaping: sparse (win/loss) + dense (territory delta, continent bonus delta, army ratio). Dense rewards are ablatable via `RewardConfig` in `src/utils/reward_config.py`.

### Baselines to beat

Random agent ≈ 16% win rate (6-player); gpt-oss LLM ≈ 25–35%.
