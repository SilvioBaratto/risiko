"""Tests for board renderer and replay exporter."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from visualization.render_game import ReplayExporter, render_ascii, render_matplotlib


@pytest.fixture
def sample_state():
    """A simple 3-player board state."""
    owner = np.zeros(42, dtype=np.int32)
    owner[9:42] = 1
    owner[38:42] = 2
    armies = np.ones(42, dtype=np.int32)
    armies[0:9] = 3
    return {"territory_owner": owner, "armies": armies}


@pytest.fixture
def sample_states():
    """A sequence of board states for replay testing."""
    states = []
    for i in range(3):
        owner = np.zeros(42, dtype=np.int32)
        owner[i * 14 : (i + 1) * 14] = i % 3
        armies = np.ones(42, dtype=np.int32) * (i + 1)
        states.append({"territory_owner": owner, "armies": armies})
    return states


class TestRenderAscii:
    """ASCII renderer returns a descriptive string."""

    def test_returns_string(self, sample_state: dict) -> None:
        text = render_ascii(sample_state)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_includes_territory_names(self, sample_state: dict) -> None:
        text = render_ascii(sample_state)
        assert "Alaska" in text
        assert "Brazil" in text

    def test_includes_army_counts(self, sample_state: dict) -> None:
        text = render_ascii(sample_state)
        assert "3" in text or "1" in text

    def test_includes_player_info(self, sample_state: dict) -> None:
        text = render_ascii(sample_state)
        assert "Player" in text or "P" in text

    def test_groups_by_continent(self, sample_state: dict) -> None:
        text = render_ascii(sample_state)
        assert "North America" in text
        assert "South America" in text


class TestRenderMatplotlib:
    """Matplotlib renderer returns a Figure."""

    def test_returns_figure(self, sample_state: dict) -> None:
        fig = render_matplotlib(sample_state)
        assert isinstance(fig, Figure)

    def test_figure_has_axes(self, sample_state: dict) -> None:
        fig = render_matplotlib(sample_state)
        assert len(fig.axes) > 0

    def test_multiple_calls_dont_error(self, sample_state: dict) -> None:
        fig1 = render_matplotlib(sample_state)
        fig2 = render_matplotlib(sample_state)
        assert fig1 is not fig2


class TestReplayExporter:
    """ReplayExporter records frames and exports GIFs."""

    def test_init_creates_empty_buffer(self) -> None:
        exporter = ReplayExporter()
        assert len(exporter) == 0

    def test_add_frame_increases_length(self, sample_state: dict) -> None:
        exporter = ReplayExporter()
        exporter.add_frame(sample_state)
        assert len(exporter) == 1

    def test_add_frame_stores_state(self, sample_state: dict) -> None:
        exporter = ReplayExporter()
        exporter.add_frame(sample_state)
        assert exporter._frames[0]["territory_owner"][0] == 0

    def test_export_gif_creates_file(self, sample_states: list[dict], tmp_path: Path) -> None:
        exporter = ReplayExporter()
        for state in sample_states:
            exporter.add_frame(state)
        path = tmp_path / "replay.gif"
        exporter.export_gif(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_export_gif_with_three_frames(self, sample_states: list[dict], tmp_path: Path) -> None:
        exporter = ReplayExporter()
        for state in sample_states:
            exporter.add_frame(state)
        path = tmp_path / "replay.gif"
        exporter.export_gif(path)
        assert path.exists()

    def test_export_gif_overwrites_existing(
        self,
        sample_states: list[dict],
        tmp_path: Path,
    ) -> None:
        exporter = ReplayExporter()
        for state in sample_states:
            exporter.add_frame(state)
        path = tmp_path / "replay.gif"
        path.write_text("old")
        exporter.export_gif(path)
        assert path.stat().st_size > 3


class TestRendererWithRealGameStates:
    """Renderers work with states from actual game trajectories."""

    def test_render_ascii_with_env_reset(self) -> None:
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        text = render_ascii(obs)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_render_matplotlib_with_env_reset(self) -> None:
        from src.env import RisikoEnv

        env = RisikoEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        fig = render_matplotlib(obs)
        assert isinstance(fig, Figure)

    def test_exporter_with_multi_agent_runner(self, tmp_path: Path) -> None:
        from src.agents.random_agent import RandomAgent
        from src.env import RisikoEnv
        from src.multi_agent import MultiAgentRunner

        env = RisikoEnv(n_players=2)
        agents = [RandomAgent(seed=1), RandomAgent(seed=2)]
        runner = MultiAgentRunner(env, agents, max_turns=5)
        result = runner.run_game(seed=42)

        exporter = ReplayExporter()
        for obs, _action, _reward in result.trajectories:
            exporter.add_frame(obs)
        path = tmp_path / "replay.gif"
        exporter.export_gif(path)
        assert path.exists()
