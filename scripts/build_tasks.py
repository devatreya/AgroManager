from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (
    CURRENT_PRICES_FILE,
    DEFAULT_STARTING_CASH,
    RECENT_WEATHER_FILE,
    SCENARIO_MIX,
    SPLIT_SIZES,
    TASK_FILE_NAMES,
    project_root,
    load_json_data,
)


ROTATION_TEMPLATES = {
    "standard": ["wheat", "barley", "oilseed_rape", "field_beans"],
    "drought_stressed": ["barley", "field_beans", "wheat", "cover_crop"],
    "input_cost_shock": ["wheat", "barley", "field_beans", "fallow"],
    "recovery": ["cover_crop", "field_beans", "wheat", "barley"],
}


def scenario_counts(split_size: int) -> dict[str, int]:
    counts = {}
    running_total = 0
    keys = list(SCENARIO_MIX)
    for scenario in keys[:-1]:
        count = int(round(split_size * SCENARIO_MIX[scenario]))
        counts[scenario] = count
        running_total += count
    counts[keys[-1]] = split_size - running_total
    return counts


def make_initial_soil(rng: random.Random, scenario: str, rainfall_context: float) -> list[dict[str, float]]:
    soil = []
    base = 0.86
    if scenario == "drought_stressed":
        base -= 0.08
    elif scenario == "recovery":
        base -= 0.10
    elif scenario == "input_cost_shock":
        base -= 0.03
    base -= max(0.0, 1.0 - rainfall_context) * 0.05
    for _ in range(4):
        organic_matter = max(0.35, min(1.05, rng.gauss(base, 0.06)))
        structure = max(0.35, min(1.00, rng.gauss(base - 0.02, 0.07)))
        ph = max(0.55, min(1.05, rng.gauss(0.88, 0.04)))
        nutrients = max(0.35, min(1.05, rng.gauss(base + 0.01, 0.06)))
        soil.append(
            {
                "organic_matter": round(organic_matter, 4),
                "structure": round(structure, 4),
                "ph": round(ph, 4),
                "nutrient_balance": round(nutrients, 4),
            }
        )
    return soil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="medium", choices=["medium"])
    parser.add_argument("--output-dir", default=str(project_root() / "data" / "processed"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recent_weather = load_json_data(RECENT_WEATHER_FILE)["records"]
    current_prices = load_json_data(CURRENT_PRICES_FILE)

    for split, split_size in SPLIT_SIZES.items():
        rng = random.Random(f"{split}:23")
        tasks = []
        counts = scenario_counts(split_size)
        for scenario, count in counts.items():
            for local_index in range(count):
                context_start = rng.randint(0, max(0, len(recent_weather) - 4))
                context = recent_weather[context_start : context_start + 4]
                mean_rainfall = sum(row["rainfall_index"] for row in context) / len(context)
                recent_price_span = (
                    max(
                        row["crop_price_multiplier"]["wheat"]
                        for row in current_prices["recent_records"]
                    )
                    - min(
                        row["crop_price_multiplier"]["wheat"]
                        for row in current_prices["recent_records"]
                    )
                )
                dry_bias = max(0.0, (1.0 - mean_rainfall) * 0.10)
                price_volatility = 0.90 + recent_price_span * 0.9
                fertiliser_cost_multiplier = 1.0
                irrigation_cost_multiplier = 1.0

                if scenario == "drought_stressed":
                    dry_bias += 0.08
                    price_volatility += 0.05
                    fertiliser_cost_multiplier += 0.05
                    irrigation_cost_multiplier += 0.04
                elif scenario == "input_cost_shock":
                    price_volatility += 0.08
                    fertiliser_cost_multiplier += 0.32
                    irrigation_cost_multiplier += 0.10
                elif scenario == "recovery":
                    dry_bias = max(0.0, dry_bias - 0.02)
                    fertiliser_cost_multiplier -= 0.04

                initial_crops = list(ROTATION_TEMPLATES[scenario])
                rng.shuffle(initial_crops)

                task = {
                    "task_id": f"{split}_{scenario}_{local_index:03d}",
                    "seed": rng.randint(0, 10**9),
                    "split": split,
                    "scenario_type": scenario,
                    "real_data_mode": True,
                    "starting_cash": DEFAULT_STARTING_CASH,
                    "initial_weather_regime": context[-1]["regime"],
                    "dry_bias": round(min(0.18, dry_bias), 4),
                    "price_volatility": round(min(1.35, price_volatility), 4),
                    "fertiliser_cost_multiplier": round(min(1.60, fertiliser_cost_multiplier), 4),
                    "irrigation_cost_multiplier": round(min(1.25, irrigation_cost_multiplier), 4),
                    "initial_soil_by_plot": make_initial_soil(rng, scenario, mean_rainfall),
                    "initial_crop_by_plot": initial_crops,
                    "recent_weather_context": context,
                }
                tasks.append(task)

        tasks.sort(key=lambda item: item["task_id"])
        (output_dir / TASK_FILE_NAMES[split]).write_text(
            json.dumps(tasks, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {output_dir / TASK_FILE_NAMES[split]}")


if __name__ == "__main__":
    main()
