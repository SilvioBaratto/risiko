"""Per-player LLM sampling configuration for Risiko RL agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL = "gemma4:12b-mlx"


@dataclass(frozen=True)
class PlayerConfig:
    """Immutable LLM sampling profile for a single player slot."""

    player_id: int
    temperature: float
    top_p: float
    strategy_hint: str
    model: str = DEFAULT_MODEL
    trust_propensity: float = 0.5
    vengefulness: float = 0.5


DEFAULT_6P_PROFILES: tuple[PlayerConfig, ...] = (
    PlayerConfig(
        player_id=0,
        temperature=0.1,
        top_p=0.9,
        strategy_hint="Play greedily: maximise territory gain each turn.",
        trust_propensity=0.2,
        vengefulness=0.4,
    ),
    PlayerConfig(
        player_id=1,
        temperature=0.4,
        top_p=0.9,
        strategy_hint="Focus on securing full continents before expanding.",
        trust_propensity=0.6,
        vengefulness=0.3,
    ),
    PlayerConfig(
        player_id=2,
        temperature=0.7,
        top_p=0.85,
        strategy_hint="Eliminate the weakest player early; control cards.",
        trust_propensity=0.3,
        vengefulness=0.7,
    ),
    PlayerConfig(
        player_id=3,
        temperature=0.3,
        top_p=0.95,
        strategy_hint="Fortify borders and expand only when safe.",
        trust_propensity=0.7,
        vengefulness=0.2,
    ),
    PlayerConfig(
        player_id=4,
        temperature=0.9,
        top_p=0.8,
        strategy_hint="Play unpredictably; avoid predictable attack patterns.",
        trust_propensity=0.4,
        vengefulness=0.9,
    ),
    PlayerConfig(
        player_id=5,
        temperature=0.5,
        top_p=0.9,
        strategy_hint="Balance attack and defence; trade cards conservatively.",
        trust_propensity=0.5,
        vengefulness=0.5,
    ),
)


def load_profiles_from_yaml(path: Path | str) -> list[PlayerConfig]:
    """Load a list of PlayerConfig objects from a YAML file.

    Raises:
        ValueError: If any player_id is outside 0–5 or if duplicates exist.
    """
    raw: list[dict[str, Any]] = yaml.safe_load(Path(path).read_text())
    profiles = [_parse_profile(entry) for entry in raw]
    _validate_player_ids(profiles)
    return profiles


def _parse_profile(entry: dict[str, Any]) -> PlayerConfig:
    return PlayerConfig(
        player_id=entry["player_id"],
        temperature=entry["temperature"],
        top_p=entry["top_p"],
        strategy_hint=entry["strategy_hint"],
        model=entry.get("model", DEFAULT_MODEL),
        trust_propensity=entry.get("trust_propensity", 0.5),
        vengefulness=entry.get("vengefulness", 0.5),
    )


def _validate_player_ids(profiles: list[PlayerConfig]) -> None:
    seen: set[int] = set()
    for profile in profiles:
        if profile.player_id < 0 or profile.player_id > 5:
            raise ValueError(f"player_id {profile.player_id} out-of-range: must be 0–5")
        if profile.player_id in seen:
            raise ValueError(f"duplicate player_id {profile.player_id} in profiles")
        seen.add(profile.player_id)


__all__ = ["PlayerConfig", "DEFAULT_6P_PROFILES", "DEFAULT_MODEL", "load_profiles_from_yaml"]
