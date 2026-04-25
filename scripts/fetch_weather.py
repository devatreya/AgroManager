from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import certifi
import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import CLIMATE_NORMALS_FILE, RECENT_WEATHER_FILE, WEATHER_FILE, project_root


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CAMBRIDGE = {"latitude": 52.2053, "longitude": 0.1218}


def fetch_chunk(start_date: str, end_date: str) -> pd.DataFrame:
    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": CAMBRIDGE["latitude"],
            "longitude": CAMBRIDGE["longitude"],
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "Europe/London",
        },
        timeout=120,
        verify=certifi.where(),
    )
    response.raise_for_status()
    payload = response.json()
    daily = payload["daily"]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "temperature_c": daily["temperature_2m_mean"],
            "rainfall_mm": daily["precipitation_sum"],
        }
    )


def classify_regime(rainfall_index: float, temperature_index: float) -> str:
    regime_centres = {
        "normal": (1.00, 1.00),
        "dry": (0.58, 1.12),
        "wet": (1.48, 0.91),
    }
    return min(
        regime_centres,
        key=lambda name: abs(rainfall_index - regime_centres[name][0])
        + 0.35 * abs(temperature_index - regime_centres[name][1]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(project_root() / "data" / "processed"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for start_year in range(1991, 2026, 5):
        end_year = min(start_year + 4, 2025)
        chunks.append(
            fetch_chunk(f"{start_year}-01-01", f"{end_year}-12-31")
        )

    daily = pd.concat(chunks, ignore_index=True)
    daily["year"] = daily["date"].dt.year
    daily["quarter"] = daily["date"].dt.quarter

    quarterly = (
        daily.groupby(["year", "quarter"], as_index=False)
        .agg(rainfall_mm=("rainfall_mm", "sum"), temperature_c=("temperature_c", "mean"))
        .sort_values(["year", "quarter"])
    )

    climate_normals = (
        quarterly[(quarterly["year"] >= 1991) & (quarterly["year"] <= 2020)]
        .groupby("quarter", as_index=False)
        .agg(rainfall_mm=("rainfall_mm", "mean"), temperature_c=("temperature_c", "mean"))
    )

    normals_by_quarter = {
        int(row.quarter): row
        for row in climate_normals.itertuples(index=False)
    }

    quarterly["rainfall_index"] = quarterly.apply(
        lambda row: row["rainfall_mm"] / normals_by_quarter[int(row["quarter"])].rainfall_mm,
        axis=1,
    )
    quarterly["temperature_index"] = quarterly.apply(
        lambda row: row["temperature_c"] / normals_by_quarter[int(row["quarter"])].temperature_c,
        axis=1,
    )
    quarterly["regime"] = quarterly.apply(
        lambda row: classify_regime(row["rainfall_index"], row["temperature_index"]),
        axis=1,
    )

    historical = quarterly[(quarterly["year"] >= 2000) & (quarterly["year"] <= 2023)]
    recent = quarterly[(quarterly["year"] >= 2024) & (quarterly["year"] <= 2025)]

    climate_payload = {
        "source": "Open-Meteo historical archive",
        "location": CAMBRIDGE,
        "baseline_years": [1991, 2020],
        "quarters": [
            {
                "quarter": int(row.quarter),
                "rainfall_mm": round(float(row.rainfall_mm), 2),
                "temperature_c": round(float(row.temperature_c), 2),
            }
            for row in climate_normals.itertuples(index=False)
        ],
    }

    weather_payload = {
        "source": "Open-Meteo historical archive",
        "location": CAMBRIDGE,
        "records": [
            {
                "year": int(row.year),
                "quarter": int(row.quarter),
                "rainfall_mm": round(float(row.rainfall_mm), 2),
                "temperature_c": round(float(row.temperature_c), 2),
                "rainfall_index": round(float(row.rainfall_index), 4),
                "temperature_index": round(float(row.temperature_index), 4),
                "regime": row.regime,
            }
            for row in historical.itertuples(index=False)
        ],
        "climate_normals": climate_payload["quarters"],
    }

    recent_payload = {
        "source": "Open-Meteo historical archive",
        "location": CAMBRIDGE,
        "records": [
            {
                "year": int(row.year),
                "quarter": int(row.quarter),
                "rainfall_mm": round(float(row.rainfall_mm), 2),
                "temperature_c": round(float(row.temperature_c), 2),
                "rainfall_index": round(float(row.rainfall_index), 4),
                "temperature_index": round(float(row.temperature_index), 4),
                "regime": row.regime,
            }
            for row in recent.itertuples(index=False)
        ],
    }

    (output_dir / WEATHER_FILE).write_text(
        json.dumps(weather_payload, indent=2), encoding="utf-8"
    )
    (output_dir / CLIMATE_NORMALS_FILE).write_text(
        json.dumps(climate_payload, indent=2), encoding="utf-8"
    )
    (output_dir / RECENT_WEATHER_FILE).write_text(
        json.dumps(recent_payload, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output_dir / WEATHER_FILE}")
    print(f"Wrote {output_dir / CLIMATE_NORMALS_FILE}")
    print(f"Wrote {output_dir / RECENT_WEATHER_FILE}")


if __name__ == "__main__":
    main()
