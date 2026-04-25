from __future__ import annotations

from typing import Any

from .common import build_commit_payload, mean_recent_rainfall


ROTATION_NEXT = {
    "wheat": "field_beans",
    "field_beans": "barley",
    "barley": "oilseed_rape",
    "oilseed_rape": "wheat",
    "cover_crop": "wheat",
    "fallow": "barley",
}


def decide(
    state: dict[str, Any],
    soil_report: list[dict[str, Any]],
    weather_history: list[dict[str, Any]],
    price_board: dict[str, Any],
) -> dict[str, Any]:
    dry_trend = mean_recent_rainfall(weather_history) < 0.80
    capital_action = "none"
    if (
        not state["irrigation_owned"]
        and dry_trend
        and state["cash"] > price_board.get("irrigation_cost_gbp", 10**9) + 90_000
    ):
        capital_action = "buy_irrigation"

    soil_by_plot = {row["plot_id"]: row for row in soil_report}
    plans = []
    for plot in state["plots"]:
        soil = soil_by_plot[plot["plot_id"]]["soil_health"]
        if soil < 0.52:
            crop = "cover_crop"
            fertiliser = "low"
            pest = "none"
        else:
            crop = ROTATION_NEXT.get(plot["crop"], "wheat")
            if dry_trend and crop == "oilseed_rape":
                crop = "barley"
            fertiliser = "medium"
            pest = "ipm"
            if crop in {"cover_crop", "fallow"}:
                fertiliser = "low"
                pest = "none"
        plans.append(
            {
                "crop": crop,
                "fertiliser": fertiliser,
                "pest_control": pest,
            }
        )
    return build_commit_payload(capital_action, plans)
