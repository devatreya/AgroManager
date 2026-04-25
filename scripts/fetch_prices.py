from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import certifi
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import CURRENT_PRICES_FILE, PRICE_FILE, project_root


DEFRA_CSV_URL = "https://assets.publishing.service.gov.uk/media/69c257847e02b81c0d1c76e8/API_20260326.csv"
DEFRA_ODS_URL = "https://assets.publishing.service.gov.uk/media/646e1646ab40bf00101969c9/API-monthlydataset-25apr23.ods"

LEGACY_OUTPUT_SERIES = {
    "wheat": "Wheat",
    "barley": "Barley",
    "oilseed_rape": "Oilseed Rape (non set aside)",
    "field_beans": "Beans (Green)",
}

LEGACY_INPUT_SERIES = {
    "fertiliser": "Fertilisers and soil improvers",
    "capital": "Machinery and other equipment",
}

MODERN_SERIES = {
    "wheat": "wheat",
    "barley": "barley",
    "oilseed_rape": "oilseed_rape",
    "field_beans": "field_beans",
    "fertiliser": "fertilisers_and_soil_improvers",
    "capital": "machinery_and_other_equipment",
}


def load_legacy_monthly(path: str, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, engine="odf", header=None)
    header_row = None
    for row_index in range(min(20, len(raw.index))):
        if isinstance(raw.iat[row_index, 3], pd.Timestamp):
            header_row = row_index
            break
    if header_row is None:
        raise ValueError(f"Could not detect date header row in {sheet_name}")
    dates = pd.to_datetime(raw.iloc[header_row, 3:], errors="coerce")
    records = []
    for row_index in range(header_row + 2, len(raw.index)):
        label = raw.iat[row_index, 0]
        if not isinstance(label, str):
            continue
        values = pd.to_numeric(raw.iloc[row_index, 3 : 3 + len(dates)], errors="coerce")
        for date, value in zip(dates, values):
            if pd.isna(date) or pd.isna(value):
                continue
            records.append({"date": date, "series": label.strip(), "index": float(value)})
    return pd.DataFrame.from_records(records)


def load_modern_monthly(csv_url: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_url, parse_dates=["date"])
    return frame.rename(columns={"category": "series"})


def chain_link(legacy: pd.Series, modern: pd.Series) -> pd.Series:
    overlap = legacy.index.intersection(modern.index)
    if overlap.empty:
        raise ValueError("No overlap found while chain-linking price series")
    scale = modern.loc[overlap].mean() / legacy.loc[overlap].mean()
    combined = pd.concat([legacy[legacy.index < modern.index.min()] * scale, modern])
    return combined.sort_index()


def build_quarterly_payload(monthly: pd.DataFrame) -> list[dict[str, object]]:
    monthly = monthly.copy()
    monthly["year"] = monthly["date"].dt.year
    monthly["quarter"] = monthly["date"].dt.quarter
    quarterly = (
        monthly.groupby(["year", "quarter"], as_index=False)
        .agg(multiplier=("multiplier", "mean"))
        .sort_values(["year", "quarter"])
    )
    return [
        {
            "year": int(row.year),
            "quarter": int(row.quarter),
            "multiplier": round(float(row.multiplier), 4),
        }
        for row in quarterly.itertuples(index=False)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(project_root() / "data" / "processed"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_path = tmp_path / "defra_current.csv"
        ods_path = tmp_path / "defra_legacy.ods"

        for url, path in ((DEFRA_CSV_URL, csv_path), (DEFRA_ODS_URL, ods_path)):
            response = requests.get(url, timeout=120, verify=certifi.where())
            response.raise_for_status()
            path.write_bytes(response.content)

        legacy_outputs = pd.concat(
            [
                load_legacy_monthly(str(ods_path), "Outputs_monthly_2001-2010"),
                load_legacy_monthly(str(ods_path), "Outputs_monthly_2011_onwards"),
            ],
            ignore_index=True,
        )
        legacy_inputs = pd.concat(
            [
                load_legacy_monthly(str(ods_path), "Inputs_monthly_2001-2010"),
                load_legacy_monthly(str(ods_path), "Inputs_monthly_2011_onwards"),
            ],
            ignore_index=True,
        )
        modern = load_modern_monthly(str(csv_path))

        linked_series: dict[str, pd.Series] = {}
        for key, legacy_label in LEGACY_OUTPUT_SERIES.items():
            legacy_series = (
                legacy_outputs[legacy_outputs["series"] == legacy_label]
                .set_index("date")["index"]
                .sort_index()
            )
            modern_series = (
                modern[modern["series"] == MODERN_SERIES[key]]
                .set_index("date")["index"]
                .sort_index()
            )
            linked_series[key] = chain_link(legacy_series, modern_series)

        for key, legacy_label in LEGACY_INPUT_SERIES.items():
            legacy_series = (
                legacy_inputs[legacy_inputs["series"] == legacy_label]
                .set_index("date")["index"]
                .sort_index()
            )
            modern_series = (
                modern[modern["series"] == MODERN_SERIES[key]]
                .set_index("date")["index"]
                .sort_index()
            )
            linked_series[key] = chain_link(legacy_series, modern_series)

        quarterly_tables = {}
        for key, series in linked_series.items():
            monthly = series.rename("index").reset_index()
            monthly["multiplier"] = monthly["index"] / 100.0
            quarterly_tables[key] = pd.DataFrame(build_quarterly_payload(monthly))

        merged = quarterly_tables["wheat"][["year", "quarter", "multiplier"]].rename(
            columns={"multiplier": "wheat"}
        )
        for crop in ("barley", "oilseed_rape", "field_beans"):
            merged = merged.merge(
                quarterly_tables[crop][["year", "quarter", "multiplier"]].rename(
                    columns={"multiplier": crop}
                ),
                on=["year", "quarter"],
                how="inner",
            )
        merged = merged.merge(
            quarterly_tables["fertiliser"][["year", "quarter", "multiplier"]].rename(
                columns={"multiplier": "fertiliser"}
            ),
            on=["year", "quarter"],
            how="inner",
        )
        merged = merged.merge(
            quarterly_tables["capital"][["year", "quarter", "multiplier"]].rename(
                columns={"multiplier": "capital"}
            ),
            on=["year", "quarter"],
            how="inner",
        )
        merged = merged.sort_values(["year", "quarter"])

        historical = merged[(merged["year"] >= 2000) & (merged["year"] <= 2023)]
        recent = merged[(merged["year"] >= 2024) & (merged["year"] <= 2025)]

    payload = {
        "source": {
            "legacy": DEFRA_ODS_URL,
            "modern": DEFRA_CSV_URL,
        },
        "notes": {
            "field_beans_proxy_pre_2014": "Linked from historical DEFRA beans series to modern field_beans series across the overlap period.",
        },
        "records": [
            {
                "year": int(row.year),
                "quarter": int(row.quarter),
                "crop_price_multiplier": {
                    "wheat": round(float(row.wheat), 4),
                    "barley": round(float(row.barley), 4),
                    "oilseed_rape": round(float(row.oilseed_rape), 4),
                    "field_beans": round(float(row.field_beans), 4),
                    "cover_crop": 1.0,
                    "fallow": 1.0,
                },
                "fertiliser_price_multiplier": round(float(row.fertiliser), 4),
                "capital_price_multiplier": round(float(row.capital), 4),
            }
            for row in historical.itertuples(index=False)
        ],
    }

    latest = recent.sort_values(["year", "quarter"]).iloc[-1]
    current_payload = {
        "source": payload["source"],
        "as_of_year": int(latest["year"]),
        "as_of_quarter": int(latest["quarter"]),
        "crop_price_multiplier": {
            "wheat": round(float(latest["wheat"]), 4),
            "barley": round(float(latest["barley"]), 4),
            "oilseed_rape": round(float(latest["oilseed_rape"]), 4),
            "field_beans": round(float(latest["field_beans"]), 4),
            "cover_crop": 1.0,
            "fallow": 1.0,
        },
        "fertiliser_price_multiplier": round(float(latest["fertiliser"]), 4),
        "capital_price_multiplier": round(float(latest["capital"]), 4),
        "recent_records": [
            {
                "year": int(row.year),
                "quarter": int(row.quarter),
                "crop_price_multiplier": {
                    "wheat": round(float(row.wheat), 4),
                    "barley": round(float(row.barley), 4),
                    "oilseed_rape": round(float(row.oilseed_rape), 4),
                    "field_beans": round(float(row.field_beans), 4),
                    "cover_crop": 1.0,
                    "fallow": 1.0,
                },
                "fertiliser_price_multiplier": round(float(row.fertiliser), 4),
                "capital_price_multiplier": round(float(row.capital), 4),
            }
            for row in recent.itertuples(index=False)
        ],
    }

    (output_dir / PRICE_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / CURRENT_PRICES_FILE).write_text(
        json.dumps(current_payload, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output_dir / PRICE_FILE}")
    print(f"Wrote {output_dir / CURRENT_PRICES_FILE}")


if __name__ == "__main__":
    main()
