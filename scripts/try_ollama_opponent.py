"""One-shot manual check: real Ollama LLM opponent picks a legal action.

Local model: a single inference only (laptop RAM-friendly).
Cloud model: routed through the Ollama daemon to ollama.com (no local RAM).
"""

from __future__ import annotations

import logging
import sys
import time

from src.agents.llm_opponent import LLMOpponent
from src.agents.ollama_client import call_ollama_for_action_index
from src.env import RisikoEnv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    """Run one real Ollama action selection and print the chosen move."""
    env = RisikoEnv(n_players=3)
    obs, info = env.reset(seed=42)
    legal = info["legal_actions"]
    print(f"\nlegal_actions: {len(legal)} options; first 3: {legal[:3]}\n")

    label = sys.argv[1] if len(sys.argv) > 1 else "local"
    model = sys.argv[2] if len(sys.argv) > 2 else "gemma4:12b-mlx"
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 8192

    print(f"=== {label.upper()} — model={model} max_tokens={max_tokens} ===")
    t0 = time.time()
    idx = call_ollama_for_action_index(
        obs, legal, model=model, timeout=180.0, temperature=0.1, max_tokens=max_tokens
    )
    dt = time.time() - t0
    if idx is None:
        print(f"RESULT: no usable index ({dt:.1f}s) -> opponent falls back to random")
        return 1
    print(f"RESULT: real LLM chose idx={idx} -> {legal[idx]}  ({dt:.1f}s)")

    # Local: stop here (one inference only, RAM-friendly).
    # Cloud: also exercise the full agent wrapper (no local RAM cost).
    if label == "cloud":
        agent = LLMOpponent(model=model, timeout=120.0)
        action = agent.act(obs, legal)
        print(f"LLMOpponent.act() -> {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
