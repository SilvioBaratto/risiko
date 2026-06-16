"""Unit tests for src.agents.action_prompt.render_action_prompt().

Issue #57 — drives action_prompt.py to >= 95% line coverage.

Fixtures use RisikoEnv(n_players=3).reset(seed=42) for determinism.
No network calls, no LLM, no Azure credentials required.

Interface contract under test::

    render_action_prompt(
        obs: dict[str, np.ndarray],
        legal_actions: list[dict[str, int]],
        strategy_hint: str | None = None,
    ) -> str

Action dict keys: action_type, param_a, param_b, param_c, param_d (all int).
  0 TRADE        param_a/b/c = card hand indices.
  1 REINFORCE    param_a = territory id, param_b = army count.
  2 ATTACK       param_a = attacker, param_b = defender, param_c = dice.
  3 CAPTURE_MOVE param_a = armies to move.
  4 FORTIFY      param_a = source, param_b = dest, param_c = army count.
  5 SKIP         all params = 0.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.agents.action_prompt import render_action_prompt
from src.env import RisikoEnv
from src.utils.constants import CONTINENT_BONUSES

# ---------------------------------------------------------------------------
# Action-dict helpers (real format)
# ---------------------------------------------------------------------------


def _act(t: int, a: int = 0, b: int = 0, c: int = 0) -> dict:
    return {"action_type": t, "param_a": a, "param_b": b, "param_c": c, "param_d": 0}


def _skip() -> dict:
    return _act(5)


def _reinforce(territory: int = 0, count: int = 3) -> dict:
    return _act(1, territory, count)


def _attack(src: int = 0, dst: int = 5, dice: int = 2) -> dict:
    # territory 0 (Alaska) is adjacent to territory 5 (Northwest Territory)
    return _act(2, src, dst, dice)


def _capture_move(armies: int = 2) -> dict:
    return _act(3, armies)


def _fortify(src: int = 0, dst: int = 1, count: int = 2) -> dict:
    return _act(4, src, dst, count)


def _trade(a: int = 0, b: int = 1, c: int = 2) -> dict:
    return _act(0, a, b, c)


# ---------------------------------------------------------------------------
# Obs helpers
# ---------------------------------------------------------------------------


def _with_cards(base: dict, *symbols: int) -> dict:
    """Return a copy of base obs with card_matrix rows populated."""
    card_matrix = np.zeros((5, 4), dtype=np.int32)
    for i, sym in enumerate(symbols[:5]):
        card_matrix[i, sym] = 1
    return {**base, "cards": card_matrix}


def _with_eliminated(base: dict, *players: int) -> dict:
    """Return a copy of base obs with specific players marked eliminated."""
    elim = base["eliminated"].copy()
    for p in players:
        if p < len(elim):
            elim[p] = 1
    return {**base, "eliminated": elim}


def _with_phase(base: dict, phase: int) -> dict:
    """Return a copy of base obs with the phase field overridden."""
    return {**base, "phase": np.array(phase, dtype=np.int32)}


# ---------------------------------------------------------------------------
# Pytest fixtures (module-scoped for speed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env_reset():
    """Deterministic REINFORCE-phase obs from seed=42 (3 players)."""
    env = RisikoEnv(n_players=3)
    obs, info = env.reset(seed=42)
    return obs, info["legal_actions"]


@pytest.fixture(scope="module")
def obs(env_reset):
    """Observation dict from deterministic env reset."""
    return env_reset[0]


@pytest.fixture(scope="module")
def legal_actions(env_reset):
    """Legal actions list from deterministic env reset."""
    return env_reset[1]


# Lightweight synthetic obs for Hypothesis (avoids env overhead per case).
_SYNTH_OBS: dict = {
    "territory_owner": np.zeros(42, dtype=np.int32),
    "armies": np.ones(42, dtype=np.int32) * 5,
    "phase": np.array(2, dtype=np.int32),  # ATTACK
    "current_player": np.array(0, dtype=np.int32),
    "cards": np.zeros((5, 4), dtype=np.int32),
    "trade_count": np.array(0, dtype=np.int32),
    "reinforcements_remaining": np.array(0, dtype=np.int32),
    "eliminated": np.zeros(6, dtype=np.int32),
}


# ===========================================================================
# Criterion: strategy_hint line — placement and count
# ===========================================================================


class TestStrategyHintPlacement:
    """Verify strategy_hint is injected once, before the legal-actions section."""

    def test_when_hint_provided_then_strategy_line_appears_exactly_once(self, obs):
        hint = "Focus on continents before expanding."
        prompt = render_action_prompt(obs, [_skip()], strategy_hint=hint)
        assert prompt.count(f"Strategy: {hint}") == 1

    def test_when_hint_provided_then_strategy_line_precedes_legal_actions_section(self, obs):
        hint = "Expand aggressively from Australia."
        prompt = render_action_prompt(obs, [_skip()], strategy_hint=hint)
        assert f"Strategy: {hint}" in prompt, "Strategy line missing"
        assert "## Legal Actions" in prompt, "'## Legal Actions' heading missing"
        assert prompt.index(f"Strategy: {hint}") < prompt.index("## Legal Actions")

    def test_when_hint_is_none_then_strategy_line_is_absent(self, obs):
        prompt = render_action_prompt(obs, [_skip()], strategy_hint=None)
        assert "Strategy:" not in prompt

    def test_when_hint_provided_then_hint_text_is_verbatim_in_prompt(self, obs):
        hint = "Fortify borders and expand only when safe."
        prompt = render_action_prompt(obs, [_skip()], strategy_hint=hint)
        assert hint in prompt


# ===========================================================================
# Criterion: reply instruction contains correct index range; all indices shown
# ===========================================================================


class TestIndexRangeInReplyInstruction:
    """Verify the reply instruction encodes the valid 0..n-1 range."""

    def test_when_one_action_then_reply_range_is_0_to_0(self, obs):
        prompt = render_action_prompt(obs, [_skip()])
        assert "0..0" in prompt

    def test_when_three_actions_then_reply_range_is_0_to_2(self, obs):
        legal = [_skip(), _reinforce(), _attack()]
        prompt = render_action_prompt(obs, legal)
        assert "0..2" in prompt

    def test_when_five_actions_then_reply_range_is_0_to_4(self, obs):
        legal = [_skip()] * 5
        prompt = render_action_prompt(obs, legal)
        assert "0..4" in prompt

    def test_when_real_legal_actions_used_then_reply_range_matches_count(self, obs, legal_actions):
        prompt = render_action_prompt(obs, legal_actions)
        assert f"0..{len(legal_actions) - 1}" in prompt

    def test_when_legal_actions_provided_then_every_index_appears_in_prompt(self, obs):
        legal = [_reinforce(), _attack(), _fortify(), _skip()]
        prompt = render_action_prompt(obs, legal)
        for i in range(len(legal)):
            assert str(i) in prompt, f"Index {i} not found in prompt"


# ===========================================================================
# Criterion: every _describe_action branch covered
# (TRADE / REINFORCE / ATTACK / CAPTURE_MOVE / FORTIFY / SKIP / unknown)
# ===========================================================================


class TestDescribeActionBranches:
    """Cover every branch in _describe_action — one assertion per action type."""

    def test_when_trade_action_then_trade_label_in_prompt(self, obs):
        prompt = render_action_prompt(obs, [_trade()])
        assert "trade" in prompt.lower()

    def test_when_reinforce_action_then_reinforce_label_in_prompt(self, obs):
        prompt = render_action_prompt(obs, [_reinforce()])
        assert "reinforce" in prompt.lower()

    def test_when_attack_with_two_dice_then_dice_label_in_prompt(self, obs):
        prompt = render_action_prompt(obs, [_attack(dice=2)])
        assert "dice" in prompt.lower()

    def test_when_attack_with_one_die_then_die_label_in_prompt(self, obs):
        prompt = render_action_prompt(obs, [_attack(dice=1)])
        assert "die" in prompt.lower()

    def test_when_attack_action_then_attack_label_in_prompt(self, obs):
        prompt = render_action_prompt(obs, [_attack()])
        assert "attack" in prompt.lower()

    def test_when_capture_move_action_then_capture_move_label_in_prompt(self, obs):
        prompt = render_action_prompt(obs, [_capture_move(armies=3)])
        assert "capture_move" in prompt.lower() or "capture" in prompt.lower()

    def test_when_capture_move_action_then_army_count_in_description(self, obs):
        prompt = render_action_prompt(obs, [_capture_move(armies=5)])
        assert "5" in prompt

    def test_when_fortify_action_then_fortify_label_in_prompt(self, obs):
        prompt = render_action_prompt(obs, [_fortify()])
        assert "fortify" in prompt.lower()

    def test_when_skip_action_then_skip_label_in_prompt(self, obs):
        prompt = render_action_prompt(obs, [_skip()])
        assert "skip" in prompt.lower()

    def test_when_unknown_action_type_then_fallback_description_is_used(self, obs):
        unknown = {"action_type": 99, "param_a": 0, "param_b": 0, "param_c": 0, "param_d": 0}
        prompt = render_action_prompt(obs, [unknown])
        assert "type=99" in prompt

    def test_when_mixed_action_types_then_every_type_described(self, obs):
        legal = [_trade(), _reinforce(), _attack(), _capture_move(), _fortify(), _skip()]
        prompt = render_action_prompt(obs, legal)
        p = prompt.lower()
        assert "trade" in p
        assert "reinforce" in p
        assert "attack" in p
        assert "capture" in p
        assert "fortify" in p
        assert "skip" in p


# ===========================================================================
# Criterion: _format_cards covered
# ===========================================================================


class TestFormatCards:
    """Verify _format_cards renders all card symbols and the empty-hand case."""

    def test_when_player_has_infantry_card_then_infantry_in_status(self, obs):
        prompt = render_action_prompt(_with_cards(obs, 0), [_skip()])
        assert "infantry" in prompt.lower()

    def test_when_player_has_cavalry_card_then_cavalry_in_status(self, obs):
        prompt = render_action_prompt(_with_cards(obs, 1), [_skip()])
        assert "cavalry" in prompt.lower()

    def test_when_player_has_artillery_card_then_artillery_in_status(self, obs):
        prompt = render_action_prompt(_with_cards(obs, 2), [_skip()])
        assert "artillery" in prompt.lower()

    def test_when_player_has_wild_card_then_wild_in_status(self, obs):
        prompt = render_action_prompt(_with_cards(obs, 3), [_skip()])
        assert "wild" in prompt.lower()

    def test_when_player_has_mixed_cards_then_all_types_in_status(self, obs):
        prompt = render_action_prompt(_with_cards(obs, 0, 1, 2), [_skip()])
        assert "infantry" in prompt.lower()
        assert "cavalry" in prompt.lower()
        assert "artillery" in prompt.lower()

    def test_when_player_has_no_cards_then_none_appears_in_status(self, obs):
        prompt = render_action_prompt(obs, [_skip()])
        assert "none" in prompt.lower()


# ===========================================================================
# Criterion: board — own territories marked ★; continents show correct +bonus
# ===========================================================================


class TestBoardRendering:
    """Verify board section marks own territories and shows continent bonuses."""

    def test_when_prompt_rendered_then_board_section_is_present(self, obs):
        assert "## Board" in render_action_prompt(obs, [_skip()])

    def test_when_current_player_owns_territories_then_star_appears_in_board(self, obs):
        assert "★" in render_action_prompt(obs, [_skip()])

    def test_when_prompt_rendered_then_north_america_bonus_is_correct(self, obs):
        prompt = render_action_prompt(obs, [_skip()])
        assert f"+{CONTINENT_BONUSES['North America']} bonus" in prompt

    def test_when_prompt_rendered_then_asia_bonus_is_correct(self, obs):
        prompt = render_action_prompt(obs, [_skip()])
        assert f"+{CONTINENT_BONUSES['Asia']} bonus" in prompt

    def test_when_prompt_rendered_then_all_continents_appear_in_board(self, obs):
        prompt = render_action_prompt(obs, [_skip()])
        for continent in CONTINENT_BONUSES:
            assert continent in prompt, f"Continent '{continent}' missing"

    def test_when_prompt_rendered_then_representative_territory_names_appear(self, obs):
        prompt = render_action_prompt(obs, [_skip()])
        for name in ("Alaska", "Brazil", "Egypt", "Japan", "Eastern Australia"):
            assert name in prompt, f"Territory '{name}' missing from board"


# ===========================================================================
# Criterion: eliminated players — no KeyError; status reflects state
# ===========================================================================


class TestEliminatedPlayers:
    """Verify prompt renders correctly with eliminated players in obs."""

    def test_when_player_is_eliminated_then_prompt_renders_without_error(self, obs):
        prompt = render_action_prompt(_with_eliminated(obs, 1), [_skip()])
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_when_player_is_eliminated_then_status_shows_eliminated_list(self, obs):
        obs_e = {**obs, "eliminated": np.array([0, 1, 0, 0, 0, 0], dtype=np.int32)}
        prompt = render_action_prompt(obs_e, [_skip()])
        assert "eliminated_players" in prompt

    def test_when_no_eliminated_players_then_status_omits_eliminated_key(self, obs):
        obs_e = {**obs, "eliminated": np.zeros(6, dtype=np.int32)}
        prompt = render_action_prompt(obs_e, [_skip()])
        assert "eliminated_players" not in prompt


# ===========================================================================
# Criterion: status line — reinforcements shown in REINFORCE phase only
# ===========================================================================


class TestStatusLine:
    """Verify phase-sensitive status rendering."""

    def test_when_reinforce_phase_then_reinforcements_left_in_status(self, obs):
        # reset(seed=42) starts in REINFORCE phase (phase == 1)
        assert "reinforcements_left" in render_action_prompt(obs, [_skip()])

    def test_when_attack_phase_then_reinforcements_left_not_in_status(self, obs):
        obs_a = _with_phase(obs, 2)  # ATTACK
        assert "reinforcements_left" not in render_action_prompt(obs_a, [_skip()])

    def test_when_prompt_rendered_then_trade_count_in_status(self, obs):
        assert "trade_count" in render_action_prompt(obs, [_skip()])


# ===========================================================================
# Criterion: no secrets hardcoded in source (proxy via rendered output)
# ===========================================================================


class TestSecretsHygiene:
    """Confirm no credential material appears in the rendered prompt."""

    def test_when_prompt_rendered_then_azure_api_key_var_absent(self, obs):
        assert "AZURE_OPENAI_API_KEY" not in render_action_prompt(obs, [_skip()])

    def test_when_prompt_rendered_then_azure_base_url_var_absent(self, obs):
        assert "AZURE_OPENAI_BASE_URL" not in render_action_prompt(obs, [_skip()])

    def test_when_prompt_rendered_then_no_32_char_hex_key_pattern(self, obs):
        prompt = render_action_prompt(obs, [_skip()])
        assert not re.search(r"[0-9a-fA-F]{32,}", prompt)


# ===========================================================================
# Property-based tests (Hypothesis)
# Uses _SYNTH_OBS to avoid env overhead inside hypothesis runs.
# ===========================================================================


@given(st.integers(min_value=1, max_value=30))
def test_property_reply_range_always_matches_action_count(n: int) -> None:
    """For any n >= 1, the reply instruction must contain the token 0..n-1."""
    legal = [_skip() for _ in range(n)]
    assert f"0..{n - 1}" in render_action_prompt(_SYNTH_OBS, legal)


@given(st.integers(min_value=1, max_value=30))
def test_property_every_index_appears_for_any_action_count(n: int) -> None:
    """For any n >= 1, every index 0..n-1 must appear in the rendered prompt."""
    legal = [_skip() for _ in range(n)]
    prompt = render_action_prompt(_SYNTH_OBS, legal)
    for i in range(n):
        assert str(i) in prompt, f"Index {i} missing (n={n})"


@given(st.text(min_size=1, max_size=120))
def test_property_strategy_hint_appears_exactly_once_for_any_hint(hint: str) -> None:
    """For any non-empty hint, 'Strategy: <hint>' appears exactly once."""
    prompt = render_action_prompt(_SYNTH_OBS, [_skip()], strategy_hint=hint)
    assert prompt.count(f"Strategy: {hint}") == 1


@given(st.text(min_size=1, max_size=120))
def test_property_strategy_hint_always_precedes_legal_actions_section(hint: str) -> None:
    """For any non-empty hint, the strategy line precedes '## Legal Actions'."""
    prompt = render_action_prompt(_SYNTH_OBS, [_skip()], strategy_hint=hint)
    assert f"Strategy: {hint}" in prompt
    assert "## Legal Actions" in prompt
    assert prompt.index(f"Strategy: {hint}") < prompt.index("## Legal Actions")
