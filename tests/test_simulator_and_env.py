from __future__ import annotations

import asyncio
import math

import pytest

from baselines.weather_aware_rotation import _score_plan, decide as weather_aware_rotation
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


def test_weather_aware_rotation_high_fertiliser_bonus_is_gated():
    plot = {"plot_id": 0, "crop": "field_beans"}
    soil_row = {"soil_health": 0.60, "nutrient_balance": 0.50}
    price_board = {
        "crop_prices_gbp_per_acre": {"wheat": 700.0},
        "fertiliser_costs_gbp_per_acre": {"high": 75.0},
    }
    favourable = _score_plan(
        plot=plot,
        soil_row=soil_row,
        crop="wheat",
        fertiliser="high",
        pest_control="ipm",
        rainfall=0.95,
        temperature=1.00,
        price_board=price_board,
        irrigation_owned=False,
    )
    dry_blocked = _score_plan(
        plot=plot,
        soil_row=soil_row,
        crop="wheat",
        fertiliser="high",
        pest_control="ipm",
        rainfall=0.85,
        temperature=1.00,
        price_board=price_board,
        irrigation_owned=False,
    )
    assert math.isclose(favourable - dry_blocked, 8.0, rel_tol=1e-6)


def test_weather_aware_rotation_irrigation_uses_quarter_index_horizon():
    state = {
        "quarter_index": 35,
        "quarter": 1,
        "cash": 150000.0,
        "irrigation_owned": False,
        "plots": [
            {"plot_id": i, "crop": "wheat"}
            for i in range(4)
        ],
    }
    soil_report = [
        {"plot_id": i, "soil_health": 0.50, "nutrient_balance": 0.55}
        for i in range(4)
    ]
    weather_history = [{"rainfall_index": 0.75, "temperature_index": 1.00}]
    price_board = {
        "irrigation_cost_gbp": 35000.0,
        "crop_prices_gbp_per_acre": {
            "wheat": 700.0,
            "barley": 620.0,
            "oilseed_rape": 760.0,
            "field_beans": 540.0,
            "cover_crop": 0.0,
            "fallow": 0.0,
        },
        "fertiliser_costs_gbp_per_acre": {"low": 20.0, "medium": 45.0, "high": 75.0},
    }

    payload = weather_aware_rotation(state, soil_report, weather_history, price_board)
    assert payload["capital_action"] == "none"
