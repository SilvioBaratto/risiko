"""Post-training strategy analysis from GameResult data."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from typing import Any

from src.multi_agent import GameResult
from src.utils.constants import ADJACENCY, territory_to_continent


def _bfs_distance(start: int, end: int) -> int:
    """Shortest path distance between two territories."""
    if start == end:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        for neighbor in ADJACENCY.get(node, []):
            if neighbor == end:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return 0


def _phase_for_turn(turn: int, total_turns: int) -> str:
    """Classify a turn into early, mid, or late game."""
    if total_turns == 0:
        return "early"
    ratio = turn / total_turns
    if ratio < 1 / 3:
        return "early"
    if ratio < 2 / 3:
        return "mid"
    return "late"


def _counter_to_bucket(counter: Counter[str]) -> dict[str, int | float]:
    """Convert a distance Counter to {dist_str: count, 'mean': float, 'total': int}."""
    total = sum(counter.values())
    if total == 0:
        return {"mean": 0.0, "total": 0}
    mean = sum(int(k) * v for k, v in counter.items()) / total
    result: dict[str, int | float] = dict(counter)
    result["mean"] = mean
    result["total"] = total
    return result


class StrategyAnalyzer:
    """Extract learned policy heuristics from a batch of GameResults."""

    def __init__(
        self,
        game_results: Sequence[GameResult],
        learner_id: int = 0,
    ) -> None:
        """Initialise with game results and optional learner ID."""
        self._game_results = list(game_results)
        self._learner_id = learner_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def continent_priority(self) -> dict[str, dict[str, int | float]]:
        """Return per-continent action counts and percentages."""
        reinforce_counts: dict[str, int] = {}
        attack_counts: dict[str, int] = {}
        total_actions = 0

        for game in self._game_results:
            for action in game.action_log:
                if action.get("player") != self._learner_id:
                    continue
                action_type = action.get("action_type")
                if action_type == 1:  # reinforce
                    continent = territory_to_continent(action.get("param_a", -1))
                    if continent:
                        reinforce_counts[continent] = reinforce_counts.get(continent, 0) + 1
                        total_actions += 1
                elif action_type == 2:  # attack
                    continent = territory_to_continent(action.get("param_b", -1))
                    if continent:
                        attack_counts[continent] = attack_counts.get(continent, 0) + 1
                        total_actions += 1

        all_continents = [
            "North America",
            "South America",
            "Europe",
            "Africa",
            "Asia",
            "Australia",
        ]
        result: dict[str, dict[str, int | float]] = {}
        for name in all_continents:
            r = reinforce_counts.get(name, 0)
            a = attack_counts.get(name, 0)
            result[name] = {
                "reinforce": r,
                "attack": a,
                "reinforce_pct": r / total_actions if total_actions else 0.0,
                "attack_pct": a / total_actions if total_actions else 0.0,
            }
        return result

    def attack_aggressiveness(self) -> dict[str, float]:
        """Return average attacks per turn and capture success rate."""
        total_turns = sum(g.n_turns for g in self._game_results)
        attack_count = 0
        capture_count = 0

        for game in self._game_results:
            for action in game.action_log:
                if action.get("player") != self._learner_id:
                    continue
                action_type = action.get("action_type")
                if action_type == 2:
                    attack_count += 1
                elif action_type == 3:
                    capture_count += 1

        avg_attacks = attack_count / total_turns if total_turns else 0.0
        capture_rate = capture_count / attack_count if attack_count else 0.0
        return {
            "avg_attacks_per_turn": avg_attacks,
            "capture_success_rate": capture_rate,
        }

    def card_trade_timing(self) -> dict[str, float]:
        """Return card trade timing heuristics."""
        all_hand_sizes: list[int] = []
        trades_early = 0
        trades_mid = 0
        trades_late = 0
        total_early = 0
        total_mid = 0
        total_late = 0

        for game in self._game_results:
            all_hand_sizes.extend(game.card_trade_hand_sizes)
            total_turns = game.n_turns
            for turn in range(total_turns):
                phase = _phase_for_turn(turn, total_turns)
                if phase == "early":
                    total_early += 1
                elif phase == "mid":
                    total_mid += 1
                else:
                    total_late += 1
            for turn in game.card_trade_turns:
                phase = _phase_for_turn(turn, total_turns)
                if phase == "early":
                    trades_early += 1
                elif phase == "mid":
                    trades_mid += 1
                else:
                    trades_late += 1

        avg_hand = sum(all_hand_sizes) / len(all_hand_sizes) if all_hand_sizes else 0.0
        return {
            "avg_hand_size_at_trade": avg_hand,
            "trades_per_turn_early": trades_early / total_early if total_early else 0.0,
            "trades_per_turn_mid": trades_mid / total_mid if total_mid else 0.0,
            "trades_per_turn_late": trades_late / total_late if total_late else 0.0,
        }

    def fortification_patterns(self) -> dict[str, float]:
        """Return fortification movement statistics."""
        total_armies = 0
        total_distance = 0
        fortify_count = 0

        for game in self._game_results:
            for action in game.action_log:
                if action.get("player") != self._learner_id:
                    continue
                if action.get("action_type") != 4:
                    continue
                fortify_count += 1
                total_armies += action.get("param_c", 0)
                total_distance += _bfs_distance(
                    action.get("param_a", -1), action.get("param_b", -1)
                )

        avg_armies = total_armies / fortify_count if fortify_count else 0.0
        avg_distance = total_distance / fortify_count if fortify_count else 0.0
        return {
            "avg_armies_moved": avg_armies,
            "avg_fortify_distance": avg_distance,
        }

    def attack_distance_distribution(self) -> dict[str, dict[str, int | float]]:
        """Return BFS attack-distance counts and mean, bucketed by game phase."""
        phase_counters: dict[str, Counter[str]] = {
            "early": Counter(),
            "mid": Counter(),
            "late": Counter(),
            "all": Counter(),
        }
        for game in self._game_results:
            total_len = len(game.action_log)
            for i, action in enumerate(game.action_log):
                if action.get("player") != self._learner_id:
                    continue
                if action.get("action_type") != 2:
                    continue
                dist = _bfs_distance(action.get("param_a", -1), action.get("param_b", -1))
                if dist <= 0:
                    continue
                phase = _phase_for_turn(i, total_len)
                phase_counters[phase][str(dist)] += 1
                phase_counters["all"][str(dist)] += 1
        return {phase: _counter_to_bucket(counter) for phase, counter in phase_counters.items()}

    def report(self) -> dict[str, Any]:
        """Return a flat, JSON-serializable dict with all heuristics."""
        continents = self.continent_priority()
        attacks = self.attack_aggressiveness()
        trades = self.card_trade_timing()
        fortifies = self.fortification_patterns()
        attack_dist = self.attack_distance_distribution()

        report: dict[str, Any] = {}
        for name, data in continents.items():
            safe_name = name.replace(" ", "_").lower()
            report[f"{safe_name}_reinforce_count"] = data["reinforce"]
            report[f"{safe_name}_attack_count"] = data["attack"]
            report[f"{safe_name}_reinforce_pct"] = data["reinforce_pct"]
            report[f"{safe_name}_attack_pct"] = data["attack_pct"]

        report.update(attacks)
        report.update(trades)
        report.update(fortifies)
        report["attack_dist_early_mean"] = attack_dist["early"]["mean"]
        report["attack_dist_mid_mean"] = attack_dist["mid"]["mean"]
        report["attack_dist_late_mean"] = attack_dist["late"]["mean"]
        report["attack_dist_all_mean"] = attack_dist["all"]["mean"]
        return report
