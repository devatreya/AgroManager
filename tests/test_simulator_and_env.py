from __future__ import annotations

import asyncio
import math

import pytest

from env import AgroManager
from grader import bankruptcy_aware, scalar_final_score, stewardship_weighted
from pipeline.farm_session import HostedFarmSession, normalize_tool_schema
from sim import FarmAction, FarmSimulator, PlotAction


def test_simulator_progression(sample_task):
    sim = FarmSimulator(sample_task)
    result = sim.step(
        FarmAction(
            capital_action="none",
            plots=[
                PlotAction("field_beans", "medium", "ipm"),
                PlotAction("barley", "medium", "ipm"),
                PlotAction("wheat", "medium", "ipm"),
                PlotAction("cover_crop", "low", "none"),
            ],
        )
    )
    assert sim.state.quarter_index == 2
    assert result.finished is False
    assert len(result.plot_pnl) == 4
    recent_weather = sim.weather_history(1)[0]
    assert "year" in recent_weather
    assert "quarter" in recent_weather


def test_reward_logic_matches_formula(sample_task):
    sim = FarmSimulator(sample_task)
    result = sim.step(
        FarmAction(
            capital_action="none",
            plots=[
                PlotAction("field_beans", "medium", "ipm"),
                PlotAction("barley", "medium", "ipm"),
                PlotAction("wheat", "medium", "ipm"),
                PlotAction("cover_crop", "low", "none"),
            ],
        )
    )
    healthy = sum(1 for plot in sim.state.plots if plot.soil_health > 0.55)
    expected = result.pnl * 1e-4 + min(0.4, healthy * 0.1)
    assert math.isclose(result.reward, expected, rel_tol=1e-6)


def test_cover_crop_improves_soil_more_than_extractive_crop(sample_task):
    restorative = FarmSimulator(sample_task)
    extractive = FarmSimulator(sample_task)

    before_restorative = restorative.state.plots[0].soil_health
    restorative.step(
        FarmAction(
            capital_action="none",
            plots=[PlotAction("cover_crop", "low", "none") for _ in range(4)],
        )
    )
    extractive.step(
        FarmAction(
            capital_action="none",
            plots=[PlotAction("oilseed_rape", "high", "spray") for _ in range(4)],
        )
    )
    assert restorative.state.plots[0].soil_health > before_restorative
    assert restorative.mean_final_soil() > extractive.mean_final_soil()


def test_task_generation_files_have_expected_counts():
    for split, expected in (("train", 64), ("validation", 16), ("test", 16)):
        tasks = AgroManager.list_tasks(split)
        assert len(tasks) == expected
        assert all(task["real_data_mode"] is True for task in tasks)


def test_grader_behaviour():
    trajectory = {
        "terminal_score": 0.8,
        "mean_final_soil": 0.7,
        "completion_rate": 1.0,
        "ever_bankrupt": False,
    }
    assert scalar_final_score(trajectory) == 0.8
    assert bankruptcy_aware({**trajectory, "ever_bankrupt": True}) == 0.4
    assert stewardship_weighted(trajectory) > 0.7


def test_tool_schema_normalization_flattens_commit_plan():
    tools = AgroManager.list_tools().tools
    assert len(tools) == 5
    commit_tool = next(tool for tool in tools if tool.name == "commit_plan")
    schema = normalize_tool_schema(commit_tool.input_schema, commit_tool.name)
    assert "$defs" not in schema
    assert "title" not in schema
    assert "plot_1" in schema["properties"]
    assert "crop" in schema["properties"]["plot_1"]["properties"]


def test_hosted_session_prefers_metadata_over_text_parsing():
    class DummyToolOutput:
        def __init__(self):
            self.blocks = [type("Block", (), {"type": "text", "text": "Quarter 1\nCash: GBP 100.00"})()]
            self.metadata = {"state": {"cash": 999.0}, "episode_metrics": {"terminal_score": 0.75}}
            self.reward = None
            self.finished = False

    class DummySession:
        async def call_tool(self, tool_name, payload):
            return DummyToolOutput()

    session = HostedFarmSession(split="validation", task_id="dummy")
    session.session = DummySession()
    result = asyncio.run(session.call_tool("read_farm_state", {}))
    assert result.state["cash"] == 999.0
    assert result.episode_metrics["terminal_score"] == 0.75
