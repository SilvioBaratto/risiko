#!/usr/bin/env bash
# Direct RL training against cloud LLM opponents — no pretraining phase.
#
# Self-detaching: by default the training runs in the background via nohup and
# survives the terminal closing (no tmux needed). A PID file tracks the run so
# it can be stopped/inspected later.
#
# The learner (slot 0) plays a 6-player game against 5 Ollama-served LLM
# opponents (default gemma4:31b-cloud). Asymmetric opponents → uneven boards →
# the terminal margin reward + done-flag give the value head real targets, so
# learning works without a self-play bootstrap (a 2p self-play pretrain is
# symmetric → 21/21 stalemate → zero terminal signal, so it is skipped).
# save_freq=1 + rollout-buffer persistence make every game durable and the run
# crash-resumable. Throughput is LLM-latency-bound; gemma4:31b-cloud answers
# in well under a second, so games run far faster than with glm.
#
# Usage:
#   ./scripts/train.sh                 # start detached (resume latest.pt if present)
#   ./scripts/train.sh --fresh         # wipe checkpoints, start from scratch
#   ./scripts/train.sh --foreground    # run in this terminal (no detach)
#   ./scripts/train.sh --status        # show whether a run is alive + tail log
#   ./scripts/train.sh --stop          # stop the running training
#
# Override defaults via env vars:
#   MODEL=gemma4:31b-cloud MAX_TURNS=150 N_STEPS=512 EPISODES=100000 \
#   N_PLAYERS=2 CONFIG=config/llm_6p.yaml ./scripts/train.sh
#
# Tip: for faster iteration use N_PLAYERS=2 (learner vs 1 LLM) — games resolve
# (real win/loss) and there is 1 LLM call/turn instead of 5.

set -euo pipefail

# ------------------------------------------------------------------
# Configuration (override via env vars)
# ------------------------------------------------------------------

MODEL="${MODEL:-gemma4:31b-cloud}"       # Ollama model for every LLM opponent
CONFIG="${CONFIG:-config/llm_6p.yaml}"   # LLM-mode config (llm_profiles_path set)
MAX_TURNS="${MAX_TURNS:-150}"            # player-turn cap (turns, not env steps)
N_STEPS="${N_STEPS:-512}"               # PPO rollout size — smaller = more frequent updates
EPISODES="${EPISODES:-}"                # total_timesteps override; empty → config value
N_PLAYERS="${N_PLAYERS:-}"             # player count override; empty → config value
CHECKPOINT_DIR="${CHECKPOINT_DIR:-models}"
LOG_FILE="${LOG_FILE:-logs/train.out}"
PID_FILE="${PID_FILE:-logs/train.pid}"

# Move to repo root regardless of where the script was invoked from
cd "$(dirname "$0")/.."

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

run_pid() { [ -f "$PID_FILE" ] && cat "$PID_FILE" 2>/dev/null || true; }
is_running() { local p; p="$(run_pid)"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

do_status() {
  if is_running; then
    echo "training RUNNING (pid $(run_pid))"
    echo "--- tail $LOG_FILE ---"; tail -n 8 "$LOG_FILE" 2>/dev/null | tr '\r' '\n' | tail -n 4
  else
    echo "training NOT running"
  fi
  exit 0
}

do_stop() {
  if is_running; then
    local p; p="$(run_pid)"
    echo ">>> stopping training (pid $p)"; kill "$p" 2>/dev/null || true
    sleep 2; kill -9 "$p" 2>/dev/null || true
    rm -f "$PID_FILE"; echo "stopped."
  else
    echo "no running training to stop."; rm -f "$PID_FILE" 2>/dev/null || true
  fi
  exit 0
}

# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

FRESH=0
FOREGROUND=0
for arg in "$@"; do
  case "$arg" in
    --fresh)      FRESH=1 ;;
    --foreground) FOREGROUND=1 ;;
    --status)     do_status ;;
    --stop)       do_stop ;;
    -h|--help)    sed -n '2,30p' "$0" | sed 's/^# //; s/^#//'; exit 0 ;;
    *)            echo "Unknown flag: $arg (use --help)" >&2; exit 2 ;;
  esac
done

# ------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------

if ! command -v risiko-rl >/dev/null 2>&1; then
  echo "error: 'risiko-rl' not found on PATH. Activate the conda env first:" >&2
  echo "  conda activate risiko" >&2
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "error: config not found: $CONFIG" >&2; exit 1
fi
if is_running; then
  echo "error: training already running (pid $(run_pid)). Use --stop first or --status." >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")" "$CHECKPOINT_DIR"

if [ "$FRESH" -eq 1 ]; then
  echo ">>> --fresh: removing existing checkpoints from $CHECKPOINT_DIR/"
  rm -f "$CHECKPOINT_DIR"/latest.pt "$CHECKPOINT_DIR"/checkpoint_*.pt
fi

# ------------------------------------------------------------------
# Build overrides
# ------------------------------------------------------------------

OVERRIDES=(--override "max_turns=$MAX_TURNS" --override "ppo.n_steps=$N_STEPS")
[ -n "$EPISODES" ]  && OVERRIDES+=(--override "total_timesteps=$EPISODES")
[ -n "$N_PLAYERS" ] && OVERRIDES+=(--override "self_play.n_players=$N_PLAYERS")

echo "============================================================"
echo "  Direct LLM training"
echo "  Model:        $MODEL"
echo "  Config:       $CONFIG"
echo "  max_turns:    $MAX_TURNS (player-turns)   ppo.n_steps: $N_STEPS"
echo "  players:      ${N_PLAYERS:-<config default>}   episodes: ${EPISODES:-<config default>}"
echo "  Expected wall time: LLM-latency-bound (fast with gemma4:31b-cloud)"
echo "============================================================"

# ------------------------------------------------------------------
# Launch
# ------------------------------------------------------------------

if [ "$FOREGROUND" -eq 1 ]; then
  exec env RISIKO_CONSOLE_LEVEL=WARNING RISIKO_LOG_LEVEL=DEBUG \
    risiko-rl train --config "$CONFIG" --model "$MODEL" "${OVERRIDES[@]}"
fi

# Detached: nohup + background → survives terminal hangup, no tmux needed.
nohup env RISIKO_CONSOLE_LEVEL=WARNING RISIKO_LOG_LEVEL=DEBUG \
  risiko-rl train --config "$CONFIG" --model "$MODEL" "${OVERRIDES[@]}" \
  > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
disown "$TRAIN_PID" 2>/dev/null || true
echo "$TRAIN_PID" > "$PID_FILE"

echo
echo ">>> training detached (pid $TRAIN_PID) — survives this terminal."
echo "    log:    tail -f $LOG_FILE"
echo "    detail: tail -f logs/risiko_debug.log"
echo "    status: ./scripts/train.sh --status"
echo "    stop:   ./scripts/train.sh --stop"
echo "    board:  tensorboard --logdir results/runs/"
