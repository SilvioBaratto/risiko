#!/usr/bin/env bash
# Two-phase Risiko RL training.
#
# Phase 1 — fast pretraining vs random opponents (self-play + random fillers).
#          Each episode runs in seconds; PPO sees diverse states quickly.
#          Goal: a baseline policy that beats random before any LLM exposure.
#          Uses a coarse save_freq (checkpoints periodically — games are cheap).
#
# Phase 2 — LLM fine-tuning vs Ollama-served opponents (default glm-5.1:cloud).
#          Resumes from Phase 1's checkpoint (models/latest.pt). The learner
#          faces 5 LLM personalities. Each episode takes ~hours (LLM latency-
#          bound). save_freq=1 + rollout-buffer persistence make every finished
#          game durable and the run crash-resumable mid-buffer.
#
# Usage:
#   ./scripts/train_two_phase.sh                 # both phases, resume if checkpoint exists
#   ./scripts/train_two_phase.sh --fresh         # delete checkpoints, start Phase 1 from scratch
#   ./scripts/train_two_phase.sh --skip-phase1   # skip pretraining (assumes checkpoint exists)
#   ./scripts/train_two_phase.sh --skip-phase2   # only run pretraining
#
# Override defaults via env vars:
#   PHASE1_EPISODES=50000 PHASE2_EPISODES=500 MODEL=glm-5.1:cloud ./scripts/train_two_phase.sh

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

PHASE1_EPISODES="${PHASE1_EPISODES:-100000}"
PHASE2_EPISODES="${PHASE2_EPISODES:-1500}"
MODEL="${MODEL:-glm-5.1:cloud}"        # Ollama model for every Phase 2 LLM opponent
PHASE1_CONFIG="config/random_6p_pretrain.yaml"
PHASE2_CONFIG="config/llm_6p.yaml"
CHECKPOINT_DIR="models"
LATEST_CHECKPOINT="$CHECKPOINT_DIR/latest.pt"

# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

FRESH=0
SKIP_PHASE1=0
SKIP_PHASE2=0

for arg in "$@"; do
  case "$arg" in
    --fresh)        FRESH=1 ;;
    --skip-phase1)  SKIP_PHASE1=1 ;;
    --skip-phase2)  SKIP_PHASE2=1 ;;
    -h|--help)
      sed -n '2,22p' "$0" | sed 's/^# //; s/^#//'
      exit 0 ;;
    *)
      echo "Unknown flag: $arg (use --help)" >&2
      exit 2 ;;
  esac
done

# Move to repo root regardless of where the script was invoked from
cd "$(dirname "$0")/.."

# ------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------

if ! command -v risiko-rl >/dev/null 2>&1; then
  echo "error: 'risiko-rl' not found on PATH. Activate the conda env first:" >&2
  echo "  conda activate risiko" >&2
  exit 1
fi

for cfg in "$PHASE1_CONFIG" "$PHASE2_CONFIG"; do
  if [ ! -f "$cfg" ]; then
    echo "error: config not found: $cfg" >&2
    exit 1
  fi
done

# ------------------------------------------------------------------
# Optional: clean previous checkpoints
# ------------------------------------------------------------------

if [ "$FRESH" -eq 1 ]; then
  echo ">>> --fresh: removing existing checkpoints from $CHECKPOINT_DIR/"
  rm -f "$CHECKPOINT_DIR"/latest.pt "$CHECKPOINT_DIR"/checkpoint_*.pt
fi

# ------------------------------------------------------------------
# Phase 1 — pretraining vs random
# ------------------------------------------------------------------

if [ "$SKIP_PHASE1" -eq 0 ]; then
  echo
  echo "============================================================"
  echo "  Phase 1: pretraining vs random opponents"
  echo "  Target episodes: $PHASE1_EPISODES"
  echo "  Config:          $PHASE1_CONFIG"
  echo "  Expected wall time: ~hours (CPU)"
  echo "============================================================"

  RISIKO_CONSOLE_LEVEL=WARNING RISIKO_LOG_LEVEL=DEBUG \
    risiko-rl train \
      --config "$PHASE1_CONFIG" \
      --override "total_timesteps=$PHASE1_EPISODES"

  echo ">>> Phase 1 complete. Checkpoint: $LATEST_CHECKPOINT"
fi

# ------------------------------------------------------------------
# Phase 2 — LLM fine-tuning
# ------------------------------------------------------------------

if [ "$SKIP_PHASE2" -eq 0 ]; then
  if [ ! -f "$LATEST_CHECKPOINT" ]; then
    echo "error: $LATEST_CHECKPOINT does not exist; cannot start Phase 2." >&2
    echo "Run without --skip-phase1 first, or copy a checkpoint into place." >&2
    exit 1
  fi

  PHASE2_TARGET=$((PHASE1_EPISODES + PHASE2_EPISODES))

  echo
  echo "============================================================"
  echo "  Phase 2: LLM fine-tuning"
  echo "  Resuming from: $LATEST_CHECKPOINT"
  echo "  Model:           $MODEL"
  echo "  Target episodes: $PHASE2_TARGET (Phase 1 + $PHASE2_EPISODES new)"
  echo "  Config:          $PHASE2_CONFIG (save_freq=1 → every game durable)"
  echo "  Expected wall time: ~days (~hours/episode, LLM latency-bound)"
  echo "============================================================"
  echo
  echo "Tip: ensure .env has OLLAMA_BASE_URL set (and OLLAMA_API_KEY for cloud,"
  echo "  e.g. https://ollama.com/v1); local Ollama needs no key. On any"
  echo "  timeout/HTTP/parse error the opponent falls back to a random move."
  echo

  RISIKO_CONSOLE_LEVEL=WARNING RISIKO_LOG_LEVEL=DEBUG \
    risiko-rl train \
      --config "$PHASE2_CONFIG" \
      --model "$MODEL" \
      --override "total_timesteps=$PHASE2_TARGET"

  echo ">>> Phase 2 complete. Final checkpoint: $LATEST_CHECKPOINT"
fi

echo
echo "All done. Inspect TensorBoard:"
echo "  tensorboard --logdir results/runs/"
