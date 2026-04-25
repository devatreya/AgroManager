from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from config import (
    ACRES_PER_PLOT,
    BANKRUPTCY_THRESHOLD,
    BASE_DIRECT_COST_PER_ACRE,
    BASE_FERTILISER_COST_PER_ACRE,
    BASE_GROSS_REVENUE_PER_ACRE,
    BASE_PEST_CONTROL_COST_PER_ACRE,
    CAPITAL_ACTIONS,
    CROP_SOIL_DELTA,
    CROPS,
    DEFAULT_STARTING_CASH,
    DRY_STRESS_SOIL_PENALTY,
    DROUGHT_THRESHOLD,
    FERTILISER_LEVELS,
    FERTILISER_NUTRIENT_BOOST,
    FERTILISER_SOIL_PENALTY,
    FERTILISER_YIELD_MULTIPLIER,
    IRRIGATION_COST,
    IRRIGATION_DRY_YIELD_UPLIFT,
    NUM_PLOTS,
    PEST_CONTROL_LEVELS,
    PEST_CONTROL_PRESSURE_MULTIPLIER,
    PEST_DRY_BONUS,
    PEST_PRESSURE_BASE,
    PEST_WET_BONUS,
    QUARTERLY_PNL_REWARD_SCALE,
    QUARTERS_PER_YEAR,
    REPEAT_CROP_SOIL_PENALTY,
    SOIL_COMPONENT_WEIGHTS,
    SOIL_MAX,
    SOIL_MIN,
    SOIL_SENSITIVITY,
    SOIL_SHAPING_REWARD_MAX,
    SOIL_SHAPING_REWARD_PER_PLOT,
    SOIL_SHAPING_THRESHOLD,
    TOTAL_QUARTERS,
    WEATHER_REGIME_STATS,
    WEATHER_REGIMES,
    WEATHER_TRANSITION_PRIORS,
    clamp,
    load_json_data,
    quarter_to_season,
    season_index,
    year_and_quarter,
    WEATHER_FILE,
    PRICE_FILE,
)


CROP_WEATHER_SENSITIVITY = {
    "wheat": {"dry": 0.18, "wet": 0.10, "heat": 0.08},
    "barley": {"dry": 0.14, "wet": 0.08, "heat": 0.06},
    "oilseed_rape": {"dry": 0.20, "wet": 0.16, "heat": 0.10},
    "field_beans": {"dry": 0.12, "wet": 0.18, "heat": 0.06},
    "cover_crop": {"dry": 0.04, "wet": 0.03, "heat": 0.02},
    "fallow": {"dry": 0.00, "wet": 0.00, "heat": 0.00},
}

RESTORATIVE_CROPS = {"field_beans", "cover_crop", "fallow"}


@dataclass
class PlotState:
    plot_id: int
    acreage: int
    crop: str
    previous_crop: str | None
    soil_components: dict[str, float]

    @property
    def soil_health(self) -> float:
        return sum(
            self.soil_components[name] * weight
            for name, weight in SOIL_COMPONENT_WEIGHTS.items()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plot_id": self.plot_id,
            "acreage": self.acreage,
            "crop": self.crop,
            "previous_crop": self.previous_crop,
            "soil_components": {
                name: round(value, 4)
                for name, value in self.soil_components.items()
            },
            "soil_health": round(self.soil_health, 4),
        }


@dataclass
class WeatherRecord:
    episode_quarter: int
    source_year: int
    source_quarter: int
    regime: str
    rainfall_index: float
    temperature_index: float
    rainfall_mm: float
    temperature_c: float

    def to_dict(self) -> dict[str, Any]:
        year, quarter = year_and_quarter(self.episode_quarter)
        return {
            "episode_quarter": self.episode_quarter,
            "episode_year": year,
            "episode_quarter_of_year": quarter,
            "source_year": self.source_year,
            "source_quarter": self.source_quarter,
            "source_season": quarter_to_season(self.source_quarter),
            "regime": self.regime,
            "rainfall_index": round(self.rainfall_index, 4),
            "temperature_index": round(self.temperature_index, 4),
            "rainfall_mm": round(self.rainfall_mm, 2),
            "temperature_c": round(self.temperature_c, 2),
        }


@dataclass
class FarmState:
    quarter_index: int
    cash: float
    irrigation_owned: bool
    ever_bankrupt: bool
    finished: bool
    starting_cash: float
    bankruptcy_threshold: float
    plots: list[PlotState]
    recent_weather_context: list[dict[str, Any]]
    realised_weather: list[WeatherRecord] = field(default_factory=list)
    price_board: dict[str, Any] = field(default_factory=dict)
    episode_metrics: dict[str, float | int | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        year, quarter = year_and_quarter(max(1, min(self.quarter_index, TOTAL_QUARTERS)))
        return {
            "quarter_index": self.quarter_index,
            "year": year,
            "quarter": quarter,
            "cash": round(self.cash, 2),
            "irrigation_owned": self.irrigation_owned,
            "ever_bankrupt": self.ever_bankrupt,
            "finished": self.finished,
            "starting_cash": self.starting_cash,
            "bankruptcy_threshold": self.bankruptcy_threshold,
            "plots": [plot.to_dict() for plot in self.plots],
            "recent_weather_context": self.recent_weather_context,
            "price_board": self.price_board,
            "episode_metrics": self.episode_metrics,
        }


@dataclass
class PlotAction:
    crop: str
    fertiliser: str
    pest_control: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop": self.crop,
            "fertiliser": self.fertiliser,
            "pest_control": self.pest_control,
        }


@dataclass
class FarmAction:
    capital_action: str
    plots: list[PlotAction]

    def to_dict(self) -> dict[str, Any]:
        return {
            "capital_action": self.capital_action,
            "plots": [plot.to_dict() for plot in self.plots],
        }


@dataclass
class StepResult:
    reward: float
    pnl: float
    plot_pnl: list[float]
    weather: WeatherRecord
    pest_pressure: float
    irrigation_purchased: bool
    bankrupt: bool
    finished: bool
    terminal_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": round(self.reward, 6),
            "pnl": round(self.pnl, 2),
            "plot_pnl": [round(value, 2) for value in self.plot_pnl],
            "weather": self.weather.to_dict(),
            "pest_pressure": round(self.pest_pressure, 4),
            "irrigation_purchased": self.irrigation_purchased,
            "bankrupt": self.bankrupt,
            "finished": self.finished,
            "terminal_score": round(self.terminal_score, 6),
        }


class FarmSimulator:
    def __init__(self, task_spec: dict[str, Any]) -> None:
        if not task_spec.get("real_data_mode", False):
            raise ValueError("AgroManager only supports real_data_mode=true tasks")

        self.task_spec = task_spec
        self.seed = int(task_spec["seed"])
        self.rng = random.Random(self.seed)

        self._weather_catalog = load_json_data(WEATHER_FILE)
        self._price_catalog = load_json_data(PRICE_FILE)

        self._weather_records_by_quarter = self._index_weather_records(self._weather_catalog)
        self._price_records_by_quarter = self._index_price_records(self._price_catalog)
        self._climate_normals = {
            int(record["quarter"]): record
            for record in self._weather_catalog.get("climate_normals", [])
        }

        initial_soil = task_spec["initial_soil_by_plot"]
        initial_crops = task_spec["initial_crop_by_plot"]

        self.state = FarmState(
            quarter_index=1,
            cash=float(task_spec.get("starting_cash", DEFAULT_STARTING_CASH)),
            irrigation_owned=False,
            ever_bankrupt=False,
            finished=False,
            starting_cash=float(task_spec.get("starting_cash", DEFAULT_STARTING_CASH)),
            bankruptcy_threshold=float(
                task_spec.get("bankruptcy_threshold", BANKRUPTCY_THRESHOLD)
            ),
            plots=[
                PlotState(
                    plot_id=index,
                    acreage=ACRES_PER_PLOT,
                    crop=initial_crops[index],
                    previous_crop=initial_crops[index],
                    soil_components={
                        key: clamp(float(value), SOIL_MIN, SOIL_MAX)
                        for key, value in initial_soil[index].items()
                    },
                )
                for index in range(NUM_PLOTS)
            ],
            recent_weather_context=list(task_spec.get("recent_weather_context", [])),
        )

        self._current_regime = str(task_spec.get("initial_weather_regime", "normal"))
        self.state.price_board = self._sample_price_board(self.state.quarter_index)
        self.state.episode_metrics = self.episode_metrics()

    def _index_weather_records(self, payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
        records = payload.get("records", [])
        indexed: dict[int, list[dict[str, Any]]] = {quarter: [] for quarter in range(1, 5)}
        for record in records:
            indexed[int(record["quarter"])].append(record)
        return indexed

    def _index_price_records(self, payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
        records = payload.get("records", [])
        indexed: dict[int, list[dict[str, Any]]] = {quarter: [] for quarter in range(1, 5)}
        for record in records:
            indexed[int(record["quarter"])].append(record)
        return indexed

    def current_state(self) -> dict[str, Any]:
        self.state.episode_metrics = self.episode_metrics()
        return self.state.to_dict()

    def terminal_score(self) -> float:
        mean_soil = self.mean_final_soil()
        soil_clipped = clamp(mean_soil, 0.4, 1.2)
        soil_factor = (soil_clipped - 0.4) / 0.8
        solvency_gate = 1.0 if not self.state.ever_bankrupt else 0.2
        cash_factor = max(0.0, self.state.cash / self.state.starting_cash)
        return cash_factor * soil_factor * solvency_gate

    def mean_final_soil(self) -> float:
        return sum(plot.soil_health for plot in self.state.plots) / len(self.state.plots)

    def episode_metrics(self) -> dict[str, float | int | bool]:
        quarters_completed = self.state.quarter_index - 1
        return {
            "terminal_score": round(self.terminal_score(), 6),
            "ending_cash": round(self.state.cash, 2),
            "mean_final_soil": round(self.mean_final_soil(), 4),
            "ever_bankrupt": self.state.ever_bankrupt,
            "quarters_completed": quarters_completed,
            "completion_rate": round(quarters_completed / TOTAL_QUARTERS, 4),
            "finished": self.state.finished,
            "completed_all_quarters": quarters_completed >= TOTAL_QUARTERS and not self.state.ever_bankrupt,
        }

    def weather_history(self, lookback_quarters: int) -> list[dict[str, Any]]:
        realised = [record.to_dict() for record in self.state.realised_weather]
        combined = self.state.recent_weather_context + realised
        return combined[-lookback_quarters:]

    def soil_report(self, plots: list[int]) -> list[dict[str, Any]]:
        report = []
        for plot_id in plots:
            plot = self.state.plots[plot_id]
            report.append(
                {
                    "plot_id": plot.plot_id,
                    "soil_health": round(plot.soil_health, 4),
                    "organic_matter": round(plot.soil_components["organic_matter"], 4),
                    "structure": round(plot.soil_components["structure"], 4),
                    "ph": round(plot.soil_components["ph"], 4),
                    "nutrient_balance": round(plot.soil_components["nutrient_balance"], 4),
                    "crop": plot.crop,
                    "previous_crop": plot.previous_crop,
                }
            )
        return report

    def _sample_price_board(self, episode_quarter: int) -> dict[str, Any]:
        quarter = season_index(episode_quarter)
        records = self._price_records_by_quarter[quarter]
        anchor = self.rng.choice(records)
        volatility = float(self.task_spec.get("price_volatility", 1.0))
        fertiliser_multiplier = float(
            self.task_spec.get("fertiliser_cost_multiplier", 1.0)
        )
        irrigation_multiplier = float(
            self.task_spec.get("irrigation_cost_multiplier", 1.0)
        )

        crop_prices = {}
        crop_price_multiplier = {}
        for crop, base_revenue in BASE_GROSS_REVENUE_PER_ACRE.items():
            anchor_multiplier = float(anchor["crop_price_multiplier"].get(crop, 1.0))
            noise = clamp(self.rng.gauss(1.0, 0.04 * volatility), 0.82, 1.22)
            multiplier = anchor_multiplier * noise
            crop_price_multiplier[crop] = round(multiplier, 4)
            crop_prices[crop] = round(base_revenue * multiplier, 2)

        fert_anchor = float(anchor["fertiliser_price_multiplier"])
        fert_noise = clamp(self.rng.gauss(1.0, 0.03 * volatility), 0.85, 1.20)
        fertiliser_price_multiplier = fert_anchor * fertiliser_multiplier * fert_noise
        fertiliser_costs = {
            level: round(cost * fertiliser_price_multiplier, 2)
            for level, cost in BASE_FERTILISER_COST_PER_ACRE.items()
        }

        capital_anchor = float(anchor.get("capital_price_multiplier", 1.0))
        irrigation_cost = IRRIGATION_COST * irrigation_multiplier * capital_anchor

        return {
            "episode_quarter": episode_quarter,
            "source_year": int(anchor["year"]),
            "source_quarter": int(anchor["quarter"]),
            "source_season": quarter_to_season(int(anchor["quarter"])),
            "crop_prices_gbp_per_acre": crop_prices,
            "crop_price_multiplier": crop_price_multiplier,
            "fertiliser_costs_gbp_per_acre": fertiliser_costs,
            "fertiliser_price_multiplier": round(fertiliser_price_multiplier, 4),
            "pest_control_costs_gbp_per_acre": BASE_PEST_CONTROL_COST_PER_ACRE,
            "irrigation_cost_gbp": round(irrigation_cost, 2),
        }

    def _sample_regime(self) -> str:
        priors = dict(WEATHER_TRANSITION_PRIORS[self._current_regime])
        dry_bias = float(self.task_spec.get("dry_bias", 0.0))
        if dry_bias:
            priors["dry"] += dry_bias
            priors["normal"] -= dry_bias * 0.6
            priors["wet"] -= dry_bias * 0.4
        total = sum(max(0.001, value) for value in priors.values())
        roll = self.rng.random()
        running = 0.0
        for regime in WEATHER_REGIMES:
            running += max(0.001, priors[regime]) / total
            if roll <= running:
                return regime
        return "normal"

    def _sample_weather(self, episode_quarter: int) -> WeatherRecord:
        quarter = season_index(episode_quarter)
        regime = self._sample_regime()
        candidates = [
            record
            for record in self._weather_records_by_quarter[quarter]
            if record["regime"] == regime
        ]
        if not candidates:
            candidates = self._weather_records_by_quarter[quarter]
        anchor = self.rng.choice(candidates)
        normals = self._climate_normals[quarter]
        regime_stats = WEATHER_REGIME_STATS[regime]

        rainfall_index = clamp(
            self.rng.gauss(
                float(anchor["rainfall_index"]),
                regime_stats["rainfall_std"] * 0.35,
            ),
            0.30,
            2.20,
        )
        temperature_index = clamp(
            self.rng.gauss(
                float(anchor["temperature_index"]),
                regime_stats["temperature_std"] * 0.35,
            ),
            0.60,
            1.50,
        )

        rainfall_mm = float(normals["rainfall_mm"]) * rainfall_index
        temperature_c = float(normals["temperature_c"]) * temperature_index
        self._current_regime = regime

        return WeatherRecord(
            episode_quarter=episode_quarter,
            source_year=int(anchor["year"]),
            source_quarter=int(anchor["quarter"]),
            regime=regime,
            rainfall_index=rainfall_index,
            temperature_index=temperature_index,
            rainfall_mm=rainfall_mm,
            temperature_c=temperature_c,
        )

    def _sample_pest_pressure(self, weather: WeatherRecord, actions: list[PlotAction]) -> float:
        pressure = PEST_PRESSURE_BASE
        if weather.regime == "wet":
            pressure += PEST_WET_BONUS
        elif weather.regime == "dry":
            pressure += PEST_DRY_BONUS
        repeat_bonus = sum(
            0.03
            for plot, action in zip(self.state.plots, actions)
            if plot.crop == action.crop and action.crop not in RESTORATIVE_CROPS
        ) / max(1, len(actions))
        pressure += repeat_bonus
        return clamp(self.rng.gauss(pressure, 0.07), 0.05, 1.0)

    def _validate_action(self, action: FarmAction) -> None:
        if action.capital_action not in CAPITAL_ACTIONS:
            raise ValueError(f"Invalid capital_action {action.capital_action!r}")
        if len(action.plots) != NUM_PLOTS:
            raise ValueError(f"Expected {NUM_PLOTS} plot actions")
        for index, plot_action in enumerate(action.plots):
            if plot_action.crop not in CROPS:
                raise ValueError(f"Invalid crop for plot {index}: {plot_action.crop!r}")
            if plot_action.fertiliser not in FERTILISER_LEVELS:
                raise ValueError(
                    f"Invalid fertiliser for plot {index}: {plot_action.fertiliser!r}"
                )
            if plot_action.pest_control not in PEST_CONTROL_LEVELS:
                raise ValueError(
                    f"Invalid pest_control for plot {index}: {plot_action.pest_control!r}"
                )

    def _weather_yield_multiplier(
        self,
        crop: str,
        weather: WeatherRecord,
        irrigation_owned: bool,
    ) -> float:
        sensitivity = CROP_WEATHER_SENSITIVITY[crop]
        rainfall_stress = max(0.0, 1.0 - weather.rainfall_index)
        wet_stress = max(0.0, weather.rainfall_index - 1.0)
        heat_stress = abs(weather.temperature_index - 1.0)
        multiplier = 1.0
        multiplier -= sensitivity["dry"] * rainfall_stress
        multiplier -= sensitivity["wet"] * wet_stress
        multiplier -= sensitivity["heat"] * heat_stress

        if weather.rainfall_index < DROUGHT_THRESHOLD and irrigation_owned and crop not in {"fallow"}:
            multiplier *= 1.0 + IRRIGATION_DRY_YIELD_UPLIFT
        return clamp(multiplier, 0.55, 1.20)

    def _soil_yield_multiplier(self, plot: PlotState) -> float:
        return clamp(0.75 + 0.25 * plot.soil_health, 0.72, 1.08)

    def _pest_yield_multiplier(self, pest_control: str, pest_pressure: float) -> float:
        under_pressure = PEST_CONTROL_PRESSURE_MULTIPLIER[pest_control]
        return clamp(1.0 - pest_pressure * (1.0 - under_pressure), 0.65, 1.00)

    def _rotation_yield_multiplier(self, crop: str, previous_crop: str | None) -> float:
        if previous_crop is None:
            return 1.0
        if crop == previous_crop and crop not in RESTORATIVE_CROPS:
            return 0.93
        if previous_crop in {"cover_crop", "field_beans"} and crop in {"wheat", "barley", "oilseed_rape"}:
            return 1.04
        return 1.0

    def _update_soil(
        self,
        plot: PlotState,
        action: PlotAction,
        weather: WeatherRecord,
        irrigation_owned: bool,
    ) -> None:
        crop_delta = CROP_SOIL_DELTA[action.crop]
        repeat_penalty = (
            REPEAT_CROP_SOIL_PENALTY
            if plot.crop == action.crop and action.crop not in RESTORATIVE_CROPS
            else 0.0
        )
        dry_penalty = (
            DRY_STRESS_SOIL_PENALTY
            if weather.rainfall_index < DROUGHT_THRESHOLD and not irrigation_owned
            else 0.0
        )

        for component, sensitivity in SOIL_SENSITIVITY.items():
            base_delta = crop_delta * sensitivity
            if component == "nutrient_balance":
                base_delta += FERTILISER_NUTRIENT_BOOST[action.fertiliser]
            if component in {"organic_matter", "structure"}:
                base_delta += repeat_penalty * sensitivity
                base_delta += dry_penalty * sensitivity
            base_delta += FERTILISER_SOIL_PENALTY[action.fertiliser] * sensitivity
            base_delta += self.rng.gauss(0.0, 0.004)
            plot.soil_components[component] = clamp(
                plot.soil_components[component] + base_delta,
                SOIL_MIN,
                SOIL_MAX,
            )

    def step(self, action: FarmAction) -> StepResult:
        if self.state.finished:
            raise ValueError("Episode already finished")
        self._validate_action(action)

        episode_quarter = self.state.quarter_index
        irrigation_purchased = False
        irrigation_charge = 0.0

        if action.capital_action == "buy_irrigation":
            if self.state.irrigation_owned:
                raise ValueError("Irrigation has already been purchased")
            irrigation_purchased = True
            irrigation_charge = float(self.state.price_board["irrigation_cost_gbp"])
            self.state.irrigation_owned = True

        weather = self._sample_weather(episode_quarter)
        pest_pressure = self._sample_pest_pressure(weather, action.plots)

        plot_pnl: list[float] = []
        next_plot_states: list[PlotState] = []
        crop_prices = self.state.price_board["crop_prices_gbp_per_acre"]
        fertiliser_costs = self.state.price_board["fertiliser_costs_gbp_per_acre"]

        for plot, plot_action in zip(self.state.plots, action.plots):
            gross_price = float(crop_prices[plot_action.crop])
            yield_multiplier = (
                FERTILISER_YIELD_MULTIPLIER[plot_action.fertiliser]
                * self._weather_yield_multiplier(
                    plot_action.crop,
                    weather,
                    self.state.irrigation_owned,
                )
                * self._soil_yield_multiplier(plot)
                * self._pest_yield_multiplier(plot_action.pest_control, pest_pressure)
                * self._rotation_yield_multiplier(plot_action.crop, plot.crop)
            )

            revenue = plot.acreage * gross_price * yield_multiplier
            direct_cost = plot.acreage * BASE_DIRECT_COST_PER_ACRE[plot_action.crop]
            fertiliser_cost = plot.acreage * float(
                fertiliser_costs[plot_action.fertiliser]
            )
            pest_cost = plot.acreage * BASE_PEST_CONTROL_COST_PER_ACRE[
                plot_action.pest_control
            ]
            pnl = revenue - direct_cost - fertiliser_cost - pest_cost
            plot_pnl.append(pnl)

            next_plot = PlotState(
                plot_id=plot.plot_id,
                acreage=plot.acreage,
                crop=plot_action.crop,
                previous_crop=plot.crop,
                soil_components=dict(plot.soil_components),
            )
            self._update_soil(
                next_plot,
                plot_action,
                weather,
                self.state.irrigation_owned,
            )
            next_plot_states.append(next_plot)

        pnl = sum(plot_pnl) - irrigation_charge
        self.state.cash += pnl
        self.state.plots = next_plot_states
        self.state.realised_weather.append(weather)

        bankrupt = self.state.cash <= self.state.bankruptcy_threshold
        if bankrupt:
            self.state.ever_bankrupt = True

        self.state.quarter_index += 1
        finished = bankrupt or (self.state.quarter_index - 1) >= TOTAL_QUARTERS
        self.state.finished = finished

        reward = pnl * QUARTERLY_PNL_REWARD_SCALE
        healthy_plots = sum(
            1 for plot in self.state.plots if plot.soil_health > SOIL_SHAPING_THRESHOLD
        )
        reward += min(
            SOIL_SHAPING_REWARD_MAX,
            healthy_plots * SOIL_SHAPING_REWARD_PER_PLOT,
        )
        if finished and not bankrupt and (self.state.quarter_index - 1) >= TOTAL_QUARTERS:
            reward += 2.0

        if not finished:
            self.state.price_board = self._sample_price_board(self.state.quarter_index)
        self.state.episode_metrics = self.episode_metrics()

        return StepResult(
            reward=reward,
            pnl=pnl,
            plot_pnl=plot_pnl,
            weather=weather,
            pest_pressure=pest_pressure,
            irrigation_purchased=irrigation_purchased,
            bankrupt=bankrupt,
            finished=finished,
            terminal_score=self.terminal_score(),
        )
