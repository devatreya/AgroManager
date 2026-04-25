from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

from config import (
    BASE_DIRECT_COST_PER_ACRE,
    BASE_FERTILISER_COST_PER_ACRE,
    BASE_GROSS_REVENUE_PER_ACRE,
    BASE_PEST_CONTROL_COST_PER_ACRE,
    CAPITAL_ACTIONS,
    CLIMATE_NORMALS_FILE,
    CROPS,
    DEFAULT_READ_TOOL_SEQUENCE,
    ENV_CLASS_NAME,
    FERTILISER_LEVELS,
    IRRIGATION_COST,
    PEST_CONTROL_LEVELS,
    SOIL_COMPONENT_WEIGHTS,
    TASK_FILE_NAMES,
    TOTAL_ACRES,
    TOTAL_QUARTERS,
    load_json_data,
)
from openreward.environments import Environment, TextBlock, ToolOutput, tool

from sim import FarmAction, FarmSimulator, PlotAction


class SoilReportInput(BaseModel):
    plots: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])

    @field_validator("plots")
    @classmethod
    def validate_plots(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("plots may not be empty")
        for plot_id in value:
            if plot_id not in {0, 1, 2, 3}:
                raise ValueError("plot ids must be in 0..3")
        return value


class WeatherHistoryInput(BaseModel):
    lookback_quarters: int = Field(default=4, ge=1, le=16)


class PlotPlanInput(BaseModel):
    crop: str
    fertiliser: str
    pest_control: str

    @field_validator("crop")
    @classmethod
    def validate_crop(cls, value: str) -> str:
        if value not in CROPS:
            raise ValueError(f"crop must be one of {CROPS}")
        return value

    @field_validator("fertiliser")
    @classmethod
    def validate_fertiliser(cls, value: str) -> str:
        if value not in FERTILISER_LEVELS:
            raise ValueError(f"fertiliser must be one of {FERTILISER_LEVELS}")
        return value

    @field_validator("pest_control")
    @classmethod
    def validate_pest_control(cls, value: str) -> str:
        if value not in PEST_CONTROL_LEVELS:
            raise ValueError(f"pest_control must be one of {PEST_CONTROL_LEVELS}")
        return value


class CommitPlanInput(BaseModel):
    capital_action: str = Field(default="none")
    plot_1: PlotPlanInput
    plot_2: PlotPlanInput
    plot_3: PlotPlanInput
    plot_4: PlotPlanInput

    @field_validator("capital_action")
    @classmethod
    def validate_capital_action(cls, value: str) -> str:
        if value not in CAPITAL_ACTIONS:
            raise ValueError(f"capital_action must be one of {CAPITAL_ACTIONS}")
        return value


class AgroManager(Environment):
    def __init__(self, task_spec: dict[str, Any] = {}, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec=task_spec, secrets=secrets)
        self.simulator = FarmSimulator(task_spec)
        self._climate_normals = load_json_data(CLIMATE_NORMALS_FILE)

    @classmethod
    def name(cls) -> str:
        return ENV_CLASS_NAME

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "validation", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[dict[str, Any]]:
        if split not in TASK_FILE_NAMES:
            raise ValueError(f"Unknown split {split!r}")
        return list(load_json_data(TASK_FILE_NAMES[split]))

    def _state_metadata(self, tool_name: str) -> dict[str, Any]:
        return {
            "tool": tool_name,
            "state": self.simulator.current_state(),
        }

    def _block(self, text: str) -> list[TextBlock]:
        return [TextBlock(text=text)]

    def _recent_weather_lines(self) -> str:
        lines = []
        for record in self.task_spec.get("recent_weather_context", []):
            lines.append(
                f"- {record['year']}-Q{record['quarter']}: regime={record['regime']}, "
                f"rainfall_index={record['rainfall_index']:.2f}, "
                f"temperature_index={record['temperature_index']:.2f}"
            )
        return "\n".join(lines) or "- No recent context provided."

    def _climate_normal_lines(self) -> str:
        lines = []
        for record in self._climate_normals["quarters"]:
            lines.append(
                f"- Q{record['quarter']}: rainfall={record['rainfall_mm']:.1f} mm, "
                f"temperature={record['temperature_c']:.1f} C"
            )
        return "\n".join(lines)

    def _starting_state_lines(self) -> str:
        state = self.simulator.current_state()
        lines = [
            f"- Quarter {state['quarter']} of {TOTAL_QUARTERS}",
            f"- Cash: GBP {state['cash']:.2f}",
            f"- Irrigation owned: {state['irrigation_owned']}",
        ]
        for plot in state["plots"]:
            lines.append(
                f"- Plot {plot['plot_id'] + 1}: crop={plot['crop']}, "
                f"previous_crop={plot['previous_crop']}, soil_health={plot['soil_health']:.3f}"
            )
        return "\n".join(lines)

    def get_prompt(self) -> list[TextBlock]:
        scenario = self.task_spec["scenario_type"]
        prompt = f"""
You are operating a 400-acre Cambridgeshire arable farm benchmark. There are 4 plots of 100 acres each and the episode lasts 10 years at 4 quarters per year, for 40 total quarters.

Task:
- task id: {self.task_spec['task_id']}
- split: {self.task_spec['split']}
- scenario: {scenario}
- real_data_mode: true
- terminal score: max(0, ending_cash / starting_cash) * soil_factor * solvency_gate
- soil_factor clips mean final soil from [0.4, 1.2] and linearly scales it to [0, 1]
- solvency_gate = 1.0 if never bankrupt else 0.2
- quarterly dense reward = quarterly_pnl * 1e-4 + soil preservation shaping
- soil preservation shaping = 0.10 per plot above soil health 0.55, capped at 0.40
- completion bonus = +2.0 only if you reach quarter 40 without bankruptcy

Recent weather context:
{self._recent_weather_lines()}

Climate normals for Cambridge/NIAB-like context:
{self._climate_normal_lines()}

Starting farm state:
{self._starting_state_lines()}

Farm layout:
- Plot 1: 100 acres
- Plot 2: 100 acres
- Plot 3: 100 acres
- Plot 4: 100 acres
- Farm-level capital action each quarter: none or buy_irrigation

Crop economics (GBP gross revenue per acre / direct cost per acre):
- wheat: {BASE_GROSS_REVENUE_PER_ACRE['wheat']} / {BASE_DIRECT_COST_PER_ACRE['wheat']}
- barley: {BASE_GROSS_REVENUE_PER_ACRE['barley']} / {BASE_DIRECT_COST_PER_ACRE['barley']}
- oilseed_rape: {BASE_GROSS_REVENUE_PER_ACRE['oilseed_rape']} / {BASE_DIRECT_COST_PER_ACRE['oilseed_rape']}
- field_beans: {BASE_GROSS_REVENUE_PER_ACRE['field_beans']} / {BASE_DIRECT_COST_PER_ACRE['field_beans']}
- cover_crop: {BASE_GROSS_REVENUE_PER_ACRE['cover_crop']} / {BASE_DIRECT_COST_PER_ACRE['cover_crop']}
- fallow: {BASE_GROSS_REVENUE_PER_ACRE['fallow']} / {BASE_DIRECT_COST_PER_ACRE['fallow']}

Fertiliser effects:
- low: cost {BASE_FERTILISER_COST_PER_ACRE['low']} GBP/acre, yield x0.88
- medium: cost {BASE_FERTILISER_COST_PER_ACRE['medium']} GBP/acre, yield x1.00
- high: cost {BASE_FERTILISER_COST_PER_ACRE['high']} GBP/acre, yield x1.12
- higher fertiliser improves nutrients but degrades soil faster

Pest-control effects:
- none: cost {BASE_PEST_CONTROL_COST_PER_ACRE['none']} GBP/acre, weakest protection
- ipm: cost {BASE_PEST_CONTROL_COST_PER_ACRE['ipm']} GBP/acre, balanced protection
- spray: cost {BASE_PEST_CONTROL_COST_PER_ACRE['spray']} GBP/acre, strongest protection
- pest pressure is stochastic and weather-dependent while the episode is running

Soil dynamics:
- Soil health is an aggregate over organic matter, structure, pH, and nutrient balance
- Aggregate weights: {json.dumps(SOIL_COMPONENT_WEIGHTS)}
- Repeating extracting crops erodes soil
- Field beans, cover crops, and fallow restore soil
- Dry quarters without irrigation apply extra soil damage

Irrigation:
- Irrigation is farm-level only, never per-plot
- It can be purchased once for GBP {IRRIGATION_COST:,.0f}
- In dry quarters it gives +18 percent yield above the no-irrigation baseline and removes the dry-soil penalty
- It has no direct reward term; the value is indirect through P&L, solvency, and final soil

Action sequence:
- Every model response must be tool use only
- Every quarter you must follow this exact sequence:
  1. read_farm_state
  2. read_soil_report
  3. read_weather_history
  4. read_price_board
  5. commit_plan

Warning:
- Greedy short-term extraction loses over 40 quarters
- Runtime weather, prices, pests, and soil drift remain uncertain while you are acting
- You need long-horizon tradeoffs, not quarter-by-quarter myopia
""".strip()
        return [TextBlock(text=prompt)]

    @tool
    def read_farm_state(self) -> ToolOutput:
        state = self.simulator.current_state()
        lines = [
            f"Quarter {state['quarter']} of {TOTAL_QUARTERS}",
            f"Cash: GBP {state['cash']:.2f}",
            f"Irrigation owned: {state['irrigation_owned']}",
        ]
        for plot in state["plots"]:
            lines.append(
                f"Plot {plot['plot_id'] + 1}: crop={plot['crop']}, previous={plot['previous_crop']}, soil={plot['soil_health']:.3f}"
            )
        return ToolOutput(
            blocks=self._block("\n".join(lines)),
            metadata=self._state_metadata("read_farm_state"),
            finished=state["finished"],
        )

    @tool
    def read_soil_report(self, params: SoilReportInput) -> ToolOutput:
        report = self.simulator.soil_report(params.plots)
        lines = []
        for row in report:
            lines.append(
                f"Plot {row['plot_id'] + 1}: OM={row['organic_matter']:.3f}, "
                f"structure={row['structure']:.3f}, pH={row['ph']:.3f}, "
                f"nutrients={row['nutrient_balance']:.3f}, soil={row['soil_health']:.3f}"
            )
        state = self.simulator.current_state()
        metadata = self._state_metadata("read_soil_report")
        metadata["report"] = report
        return ToolOutput(
            blocks=self._block("\n".join(lines)),
            metadata=metadata,
            finished=state["finished"],
        )

    @tool
    def read_weather_history(self, params: WeatherHistoryInput) -> ToolOutput:
        history = self.simulator.weather_history(params.lookback_quarters)
        lines = []
        for record in history:
            lines.append(
                f"{record['year']}-Q{record['quarter']}: regime={record['regime']}, "
                f"rainfall_index={record['rainfall_index']:.2f}, "
                f"temperature_index={record['temperature_index']:.2f}"
            )
        state = self.simulator.current_state()
        metadata = self._state_metadata("read_weather_history")
        metadata["history"] = history
        return ToolOutput(
            blocks=self._block("\n".join(lines or ["No weather history available."])),
            metadata=metadata,
            finished=state["finished"],
        )

    @tool
    def read_price_board(self) -> ToolOutput:
        state = self.simulator.current_state()
        board = dict(state["price_board"])
        if state["irrigation_owned"]:
            board.pop("irrigation_cost_gbp", None)
        lines = ["Crop prices (GBP per acre):"]
        for crop, price in board["crop_prices_gbp_per_acre"].items():
            lines.append(f"- {crop}: {price:.2f}")
        lines.append("Fertiliser costs (GBP per acre):")
        for level, price in board["fertiliser_costs_gbp_per_acre"].items():
            lines.append(f"- {level}: {price:.2f}")
        if "irrigation_cost_gbp" in board:
            lines.append(f"Irrigation capital cost: GBP {board['irrigation_cost_gbp']:.2f}")
        metadata = self._state_metadata("read_price_board")
        metadata["price_board"] = board
        return ToolOutput(
            blocks=self._block("\n".join(lines)),
            metadata=metadata,
            finished=state["finished"],
        )

    @tool
    def commit_plan(self, params: CommitPlanInput) -> ToolOutput:
        action = FarmAction(
            capital_action=params.capital_action,
            plots=[
                PlotAction(**params.plot_1.model_dump()),
                PlotAction(**params.plot_2.model_dump()),
                PlotAction(**params.plot_3.model_dump()),
                PlotAction(**params.plot_4.model_dump()),
            ],
        )
        committed_quarter = self.simulator.state.quarter_index
        result = self.simulator.step(action)
        state = self.simulator.current_state()
        metadata = self._state_metadata("commit_plan")
        metadata["step"] = committed_quarter
        metadata["episode_metrics"] = dict(state["episode_metrics"])
        metadata["action"] = {
            "capital_action": params.capital_action,
            "plot_1": params.plot_1.model_dump(),
            "plot_2": params.plot_2.model_dump(),
            "plot_3": params.plot_3.model_dump(),
            "plot_4": params.plot_4.model_dump(),
        }
        metadata["result"] = result.to_dict()

        lines = [
            f"Quarter {committed_quarter} committed.",
            f"Reward: {result.reward:.6f}",
            f"Quarterly P&L: GBP {result.pnl:.2f}",
            f"Weather: regime={result.weather.regime}, rainfall_index={result.weather.rainfall_index:.2f}, temperature_index={result.weather.temperature_index:.2f}",
            f"Pest pressure: {result.pest_pressure:.3f}",
            f"Cash: GBP {state['cash']:.2f}",
        ]
        if result.irrigation_purchased:
            lines.append("Irrigation purchased this quarter.")
        if result.finished:
            lines.append(f"Terminal score: {result.terminal_score:.6f}")
        return ToolOutput(
            blocks=self._block("\n".join(lines)),
            metadata=metadata,
            reward=result.reward,
            finished=result.finished,
        )
