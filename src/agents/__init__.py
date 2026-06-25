"""Agent implementations for Risiko RL."""

from src.agents.base import Agent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.llm_opponent import LLMOpponent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategies import STRATEGY_CATALOG, Strategy

__all__ = [
    "Agent",
    "HeuristicAgent",
    "LLMOpponent",
    "PPOAgent",
    "RandomAgent",
    "Strategy",
    "STRATEGY_CATALOG",
]
