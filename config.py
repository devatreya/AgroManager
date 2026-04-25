from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_NAME = "AgroManager"
ENV_CLASS_NAME = "AgroManager"
OPENREWARD_ENV_ID = "devatreya/AgroManager"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "AgroManager-qwen25-7b"

NUM_PLOTS = 4
ACRES_PER_PLOT = 100
TOTAL_ACRES = NUM_PLOTS * ACRES_PER_PLOT
TOTAL_YEARS = 10
QUARTERS_PER_YEAR = 4
TOTAL_QUARTERS = TOTAL_YEARS * QUARTERS_PER_YEAR

DEFAULT_STARTING_CASH = 150_000.0
BANKRUPTCY_THRESHOLD = -60_000.0

DEFAULT_READ_TOOL_SEQUENCE = [
    "read_farm_state",
    "read_soil_report",
    "read_weather_history",
    "read_price_board",
]
COMMIT_TOOL_NAME = "commit_plan"
DEFAULT_MAX_TOOL_CALLS = TOTAL_QUARTERS * (len(DEFAULT_READ_TOOL_SEQUENCE) + 1) + 40

CROPS = (
    "wheat",
    "barley",
    "oilseed_rape",
    "field_beans",
    "cover_crop",
    "fallow",
)
FERTILISER_LEVELS = ("low", "medium", "high")
PEST_CONTROL_LEVELS = ("none", "ipm", "spray")
CAPITAL_ACTIONS = ("none", "buy_irrigation")
WEATHER_REGIMES = ("normal", "dry", "wet")
SPLITS = ("train", "validation", "test")

PLOT_TOOL_NAMES = ("plot_1", "plot_2", "plot_3", "plot_4")

BASE_GROSS_REVENUE_PER_ACRE = {
    "wheat": 700.0,
    "barley": 620.0,
    "oilseed_rape": 760.0,
    "field_beans": 540.0,
    "cover_crop": 0.0,
    "fallow": 0.0,
}

BASE_DIRECT_COST_PER_ACRE = {
    "wheat": 420.0,
    "barley": 380.0,
    "oilseed_rape": 470.0,
    "field_beans": 300.0,
    "cover_crop": 45.0,
    "fallow": 10.0,
}

BASE_FERTILISER_COST_PER_ACRE = {
    "low": 20.0,
    "medium": 45.0,
    "high": 75.0,
}

FERTILISER_YIELD_MULTIPLIER = {
    "low": 0.88,
    "medium": 1.00,
    "high": 1.12,
}

FERTILISER_SOIL_PENALTY = {
    "low": 0.000,
    "medium": -0.005,
    "high": -0.020,
}

FERTILISER_NUTRIENT_BOOST = {
    "low": -0.005,
    "medium": 0.010,
    "high": 0.030,
}

BASE_PEST_CONTROL_COST_PER_ACRE = {
    "none": 0.0,
    "ipm": 18.0,
    "spray": 40.0,
}

PEST_CONTROL_PRESSURE_MULTIPLIER = {
    "none": 0.74,
    "ipm": 0.95,
    "spray": 1.00,
}

IRRIGATION_COST = 35_000.0
IRRIGATION_DRY_YIELD_UPLIFT = 0.18

SOIL_COMPONENT_WEIGHTS = {
    "organic_matter": 0.45,
    "structure": 0.20,
    "ph": 0.15,
    "nutrient_balance": 0.20,
}

SOIL_SENSITIVITY = {
    "organic_matter": 1.20,
    "structure": 0.90,
    "ph": 0.60,
    "nutrient_balance": 1.00,
}

CROP_SOIL_DELTA = {
    "wheat": -0.050,
    "barley": -0.040,
    "oilseed_rape": -0.060,
    "field_beans": 0.020,
    "cover_crop": 0.060,
    "fallow": 0.030,
}

REPEAT_CROP_SOIL_PENALTY = -0.030
DRY_STRESS_SOIL_PENALTY = -0.018
SOIL_MIN = 0.20
SOIL_MAX = 1.30

WEATHER_TRANSITION_PRIORS = {
    "normal": {"normal": 0.70, "dry": 0.18, "wet": 0.12},
    "dry": {"normal": 0.35, "dry": 0.55, "wet": 0.10},
    "wet": {"normal": 0.45, "dry": 0.05, "wet": 0.50},
}

WEATHER_REGIME_STATS = {
    "normal": {
        "rainfall_mean": 1.00,
        "rainfall_std": 0.10,
        "temperature_mean": 1.00,
        "temperature_std": 0.08,
    },
    "dry": {
        "rainfall_mean": 0.58,
        "rainfall_std": 0.12,
        "temperature_mean": 1.12,
        "temperature_std": 0.10,
    },
    "wet": {
        "rainfall_mean": 1.48,
        "rainfall_std": 0.15,
        "temperature_mean": 0.91,
        "temperature_std": 0.09,
    },
}

DROUGHT_THRESHOLD = 0.70
PEST_PRESSURE_BASE = 0.28
PEST_WET_BONUS = 0.15
PEST_DRY_BONUS = 0.05

SOIL_SHAPING_THRESHOLD = 0.55
SOIL_SHAPING_REWARD_PER_PLOT = 0.10
SOIL_SHAPING_REWARD_MAX = 0.40
QUARTERLY_PNL_REWARD_SCALE = 1e-4
COMPLETION_BONUS = 2.0

SCENARIO_MIX = {
    "standard": 0.50,
    "drought_stressed": 0.25,
    "input_cost_shock": 0.15,
    "recovery": 0.10,
}

SPLIT_SIZES = {
    "train": 64,
    "validation": 16,
    "test": 16,
}

TASK_FILE_NAMES = {
    "train": "scenario_tasks_train.json",
    "validation": "scenario_tasks_validation.json",
    "test": "scenario_tasks_test.json",
}

WEATHER_FILE = "quarterly_weather.json"
PRICE_FILE = "quarterly_prices.json"
CLIMATE_NORMALS_FILE = "climate_normals.json"
RECENT_WEATHER_FILE = "recent_weather.json"
CURRENT_PRICES_FILE = "current_prices.json"

DEFAULT_DATA_DIR = Path("data") / "processed"
ORWD_DATA_DIR = Path("/orwd_data")

BASELINE_NAMES = (
    "weather_aware_rotation",
)

MODAL_VOLUME_NAMES = {
    "art": "agromanager-art",
    "hf_cache": "agromanager-hf-cache",
    "results": "agromanager-results",
}

MODAL_VOLUME_MOUNTS = {
    "art": "/vol/art",
    "hf_cache": "/vol/hf-cache",
    "results": "/vol/results",
}

MODAL_ENV = {
    "PYTHONPATH": "/root/project",
    "HF_HOME": MODAL_VOLUME_MOUNTS["hf_cache"],
    "WANDB_DIR": f"{MODAL_VOLUME_MOUNTS['results']}/wandb",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}

REQUIRED_PROCESSED_FILES = (
    TASK_FILE_NAMES["train"],
    TASK_FILE_NAMES["validation"],
    TASK_FILE_NAMES["test"],
    WEATHER_FILE,
    PRICE_FILE,
    CLIMATE_NORMALS_FILE,
    RECENT_WEATHER_FILE,
    CURRENT_PRICES_FILE,
)


def project_root() -> Path:
    return Path(__file__).resolve().parent


def data_search_paths() -> list[Path]:
    env_override = os.getenv("AGROMANAGER_DATA_DIR")
    candidates = []
    if env_override:
        candidates.append(Path(env_override))
    candidates.append(project_root() / DEFAULT_DATA_DIR)
    candidates.append(ORWD_DATA_DIR / "processed")
    candidates.append(ORWD_DATA_DIR)
    return candidates


def resolve_data_path(file_name: str) -> Path:
    for base in data_search_paths():
        candidate = base / file_name
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in data_search_paths())
    raise FileNotFoundError(
        f"Unable to locate {file_name}. Looked in: {searched}. "
        "Populate data/processed locally or upload files into /orwd_data."
    )


@lru_cache(maxsize=None)
def load_json_data(file_name: str) -> Any:
    with resolve_data_path(file_name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def softmax_weights(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        return {}
    max_weight = max(weights.values())
    exp_values = {key: math.exp(value - max_weight) for key, value in weights.items()}
    total = sum(exp_values.values()) or 1.0
    return {key: value / total for key, value in exp_values.items()}


def quarter_to_season(quarter: int) -> str:
    return f"Q{quarter}"


def season_index(quarter_index: int) -> int:
    return ((quarter_index - 1) % QUARTERS_PER_YEAR) + 1


def year_and_quarter(quarter_index: int) -> tuple[int, int]:
    year = ((quarter_index - 1) // QUARTERS_PER_YEAR) + 1
    quarter = season_index(quarter_index)
    return year, quarter


def canonical_state_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    plots = []
    for plot in state.get("plots", []):
        plots.append(
            {
                "plot_id": plot.get("plot_id"),
                "crop": plot.get("crop"),
                "previous_crop": plot.get("previous_crop"),
                "soil_health": plot.get("soil_health"),
                "soil_components": plot.get("soil_components", {}),
            }
        )
    return {
        "quarter_index": state.get("quarter_index"),
        "year": state.get("year"),
        "quarter": state.get("quarter"),
        "cash": state.get("cash"),
        "irrigation_owned": state.get("irrigation_owned"),
        "finished": state.get("finished"),
        "ever_bankrupt": state.get("ever_bankrupt"),
        "plots": plots,
        "episode_metrics": state.get("episode_metrics", {}),
    }
