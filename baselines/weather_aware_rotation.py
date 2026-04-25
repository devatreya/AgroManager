from __future__ import annotations

from typing import Any

from config import TOTAL_QUARTERS

from .common import (
    build_commit_payload,
    gross_margin_per_acre,
    mean_recent_rainfall,
    mean_recent_temperature,
)


def _candidate_plans() -> list[tuple[str, str, str]]:
    candidates = []
    for crop in ("wheat", "barley", "oilseed_rape", "field_beans", "cover_crop", "fallow"):
        if crop in {"cover_crop", "fallow"}:
            candidates.append((crop, "low", "none"))
            candidates.append((crop, "medium", "none"))
            continue
        for fertiliser in ("low", "medium", "high"):
            for pest_control in ("ipm", "spray"):
                candidates.append((crop, fertiliser, pest_control))
        candidates.append((crop, "medium", "none"))
    return candidates


ALL_CANDIDATES = _candidate_plans()


def _score_plan(
    plot: dict[str, Any],
    soil_row: dict[str, Any],
    crop: str,
    fertiliser: str,
    pest_control: str,
    rainfall: float,
    temperature: float,
    price_board: dict[str, Any],
    irrigation_owned: bool,
) -> float:
    margin = gross_margin_per_acre(crop, fertiliser, pest_control, price_board)
    soil = float(soil_row["soil_health"])
    nutrients = float(soil_row["nutrient_balance"])
    repeated = crop == plot["crop"]

    score = margin
    if soil < 0.50:
        if crop == "cover_crop":
            score += 95.0
        elif crop == "field_beans":
            score += 65.0
        elif crop == "fallow":
            score += 40.0
        else:
            score -= 95.0
    elif soil < 0.60:
        if crop in {"cover_crop", "field_beans"}:
            score += 40.0
        if crop in {"oilseed_rape", "wheat"}:
            score -= 30.0

    if (
        nutrients < 0.55
        and fertiliser == "high"
        and rainfall >= 0.90
        and temperature <= 1.05
        and soil >= 0.55
    ):
        score += 8.0
    if nutrients > 0.90 and fertiliser == "high":
        score -= 14.0

    if repeated and crop not in {"cover_crop", "fallow"}:
        score -= 48.0

    if rainfall < 0.82 and crop == "oilseed_rape":
        score -= 42.0
    if rainfall < 0.82 and crop == "barley":
        score += 14.0
    if rainfall < 0.82 and crop == "field_beans":
        score += 8.0
    if rainfall > 1.18 and crop == "oilseed_rape":
        score -= 30.0
    if rainfall > 1.18 and crop == "barley":
        score += 6.0
    if rainfall > 1.18 and crop == "field_beans":
        score -= 18.0

    if temperature > 1.08 and fertiliser == "high":
        score -= 10.0
    if not irrigation_owned and rainfall < 0.78 and crop in {"wheat", "oilseed_rape"}:
        score -= 22.0

    return score


def decide(
    state: dict[str, Any],
    soil_report: list[dict[str, Any]],
    weather_history: list[dict[str, Any]],
    price_board: dict[str, Any],
) -> dict[str, Any]:
    rainfall = mean_recent_rainfall(weather_history)
    temperature = mean_recent_temperature(weather_history)
    soil_by_plot = {row["plot_id"]: row for row in soil_report}

    capital_action = "none"
    dry_trend = rainfall < 0.82
    quarter_index = int(state.get("quarter_index", 1))
    remaining_quarters = TOTAL_QUARTERS - quarter_index + 1
    if (
        not state["irrigation_owned"]
        and dry_trend
        and state["cash"] > price_board.get("irrigation_cost_gbp", 10**9) + 70_000
        and remaining_quarters >= 12
    ):
        capital_action = "buy_irrigation"

    plans = []
    for plot in state["plots"]:
        soil_row = soil_by_plot[plot["plot_id"]]
        best = max(
            ALL_CANDIDATES,
            key=lambda candidate: _score_plan(
                plot=plot,
                soil_row=soil_row,
                crop=candidate[0],
                fertiliser=candidate[1],
                pest_control=candidate[2],
                rainfall=rainfall,
                temperature=temperature,
                price_board=price_board,
                irrigation_owned=state["irrigation_owned"] or capital_action == "buy_irrigation",
            ),
        )
        plans.append(
            {
                "crop": best[0],
                "fertiliser": best[1],
                "pest_control": best[2],
            }
        )
    return build_commit_payload(capital_action, plans)
