from __future__ import annotations

from typing import Any

from config import (
    BASE_DIRECT_COST_PER_ACRE,
    BASE_PEST_CONTROL_COST_PER_ACRE,
    CROPS,
    FERTILISER_LEVELS,
    PEST_CONTROL_LEVELS,
)


def build_commit_payload(
    capital_action: str,
    plot_plans: list[dict[str, str]],
) -> dict[str, Any]:
    if len(plot_plans) != 4:
        raise ValueError("Expected exactly four plot plans")
    payload = {"capital_action": capital_action}
    for index, plan in enumerate(plot_plans, start=1):
        payload[f"plot_{index}"] = plan
    return payload


def mean_recent_rainfall(weather_history: list[dict[str, Any]]) -> float:
    if not weather_history:
        return 1.0
    return sum(float(row["rainfall_index"]) for row in weather_history) / len(weather_history)


def mean_recent_temperature(weather_history: list[dict[str, Any]]) -> float:
    if not weather_history:
        return 1.0
    return sum(float(row["temperature_index"]) for row in weather_history) / len(weather_history)


def gross_margin_per_acre(
    crop: str,
    fertiliser: str,
    pest_control: str,
    price_board: dict[str, Any],
) -> float:
    revenue = float(price_board["crop_prices_gbp_per_acre"][crop])
    direct = BASE_DIRECT_COST_PER_ACRE[crop]
    fertiliser_cost = float(price_board["fertiliser_costs_gbp_per_acre"][fertiliser])
    pest_cost = BASE_PEST_CONTROL_COST_PER_ACRE[pest_control]
    return revenue - direct - fertiliser_cost - pest_cost


def valid_plot_plan(plan: dict[str, str]) -> dict[str, str]:
    if plan["crop"] not in CROPS:
        raise ValueError(f"Invalid crop {plan['crop']!r}")
    if plan["fertiliser"] not in FERTILISER_LEVELS:
        raise ValueError(f"Invalid fertiliser {plan['fertiliser']!r}")
    if plan["pest_control"] not in PEST_CONTROL_LEVELS:
        raise ValueError(f"Invalid pest_control {plan['pest_control']!r}")
    return plan
