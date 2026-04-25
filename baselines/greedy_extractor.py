from __future__ import annotations

from typing import Any

from .common import build_commit_payload, gross_margin_per_acre, mean_recent_rainfall


EXTRACTIVE_CROPS = ("oilseed_rape", "wheat", "barley", "field_beans")


def decide(
    state: dict[str, Any],
    soil_report: list[dict[str, Any]],
    weather_history: list[dict[str, Any]],
    price_board: dict[str, Any],
) -> dict[str, Any]:
    del soil_report
    dry_trend = mean_recent_rainfall(weather_history) < 0.85
    capital_action = "none"
    if (
        not state["irrigation_owned"]
        and dry_trend
        and state["cash"] > price_board.get("irrigation_cost_gbp", 10**9) + 60_000
    ):
        capital_action = "buy_irrigation"

    plans = []
    for _plot in state["plots"]:
        best_crop = max(
            EXTRACTIVE_CROPS,
            key=lambda crop: gross_margin_per_acre(
                crop,
                "high",
                "spray",
                price_board,
            ),
        )
        plans.append(
            {
                "crop": best_crop,
                "fertiliser": "high",
                "pest_control": "spray",
            }
        )
    return build_commit_payload(capital_action, plans)
