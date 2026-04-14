"""
eccc_download.py – Download ECCC Stanhope hourly data via bulk CSV API.
Station: STANHOPE, Climate ID: 8300590, Station ID: 6545
Saves to data/raw/eccc_stanhope.parquet
"""

import io
import sys
from pathlib import Path

import pandas as pd

try:
    from urllib.request import urlopen, Request
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_OUT = PROJECT_ROOT / "data" / "raw"

STATION_ID = 6545          # ECCC internal ID used in bulk-download URL
YEARS = range(2022, 2026)  # 2022-2025

BASE_URL = (
    "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    "?format=csv&stationID={station_id}&Year={year}&Month={month}"
    "&Day=14&timeframe=1&submit=Download+Data"
)


def download_month(year: int, month: int) -> pd.DataFrame | None:
    """Download one month of hourly data from ECCC."""
    url = BASE_URL.format(station_id=STATION_ID, year=year, month=month)
    req = Request(url, headers={"User-Agent": "Parks-Canada-DataPipeline/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8-sig")
        df = pd.read_csv(io.StringIO(raw))
        return df if len(df) > 0 else None
    except Exception as exc:
        print(f"  [WARN] {year}-{month:02d}: {exc}")
        return None


def main():
    RAW_OUT.mkdir(parents=True, exist_ok=True)

    frames = []
    for year in YEARS:
        for month in range(1, 13):
            print(f"  Downloading {year}-{month:02d} ...", end=" ")
            df = download_month(year, month)
            if df is not None and len(df) > 0:
                frames.append(df)
                print(f"{len(df)} rows")
            else:
                print("no data")

    if not frames:
        print("[ERR] No ECCC data downloaded. Check network / station ID.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Standardize column names to match our pipeline
    col_map = {}
    for col in combined.columns:
        cl = col.lower().strip()
        if "date/time" in cl:
            col_map[col] = "datetime"
        elif cl.startswith("temp") and "flag" not in cl:
            col_map[col] = "temp_c"
        elif cl.startswith("dew point") and "flag" not in cl:
            col_map[col] = "dew_point_c"
        elif cl.startswith("rel hum") and "flag" not in cl:
            col_map[col] = "rh_pct"
        elif cl.startswith("precip") and "flag" not in cl:
            col_map[col] = "rain_mm"
        elif cl.startswith("wind dir") and "flag" not in cl:
            col_map[col] = "wind_dir_deg"
        elif cl.startswith("wind spd") and "flag" not in cl:
            col_map[col] = "wind_speed_kmh"
        elif cl.startswith("stn press") and "flag" not in cl:
            col_map[col] = "stn_pressure_kpa"

    combined = combined.rename(columns=col_map)

    # Parse timestamp to UTC (ECCC hourly is in LST = UTC-4 for PEI)
    if "datetime" in combined.columns:
        combined["timestamp_utc"] = (
            pd.to_datetime(combined["datetime"])
            .dt.tz_localize("Etc/GMT+4")
            .dt.tz_convert("UTC")
        )
        combined = combined.drop(columns=["datetime"])

    combined["station"] = "Stanhope (ECCC)"

    # Keep only useful columns
    keep = [
        "timestamp_utc", "station", "temp_c", "rh_pct", "dew_point_c",
        "rain_mm", "wind_dir_deg", "wind_speed_kmh", "stn_pressure_kpa",
    ]
    keep = [c for c in keep if c in combined.columns]
    combined = combined[keep]

    for col in combined.columns:
        if col not in ("timestamp_utc", "station"):
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.set_index("timestamp_utc").sort_index()

    out_path = RAW_OUT / "eccc_stanhope.parquet"
    combined.to_parquet(out_path)
    print(f"\n✓ ECCC Stanhope saved → {out_path}")
    print(f"  Rows: {len(combined):,}  |  Range: {combined.index.min()} → {combined.index.max()}")
    print(f"  Columns: {list(combined.columns)}")
    non_null = combined.drop(columns="station").notna().mean() * 100
    print(f"  Data availability:\n{non_null.round(1).to_string()}")


if __name__ == "__main__":
    main()
