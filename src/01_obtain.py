"""
01_obtain.py – Obtain
Load raw HOBOlink CSVs, standardize column names, verify structure.
Outputs one combined Parquet per station into data/raw/.
"""

import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_HOBO = PROJECT_ROOT / "Data" / "Raw"
RAW_OUT = PROJECT_ROOT / "data" / "raw"

STATIONS = [
    "Cavendish",
    "Greenwich",
    "North Rustico Wharf",
    "Stanley Bridge Wharf",
    "Tracadie Wharf",
]

# ---------------------------------------------------------------------------
# Column-name standardisation map
# ---------------------------------------------------------------------------
# Each key is a regex matched against the raw HOBOlink header (case-insensitive).
# First match wins, so order matters (e.g. "Accumulated Rain" before "Rain").
COLUMN_MAP = [
    (r"^Date$",                          "date"),
    (r"^Time$",                          "time"),
    (r"Accumulated Rain",                "accum_rain_mm"),
    (r"Rain\b",                          "rain_mm"),
    (r"Solar Radiation",                 "solar_rad_wm2"),
    (r"Barometric Pressure",             "baro_pressure_kpa"),
    (r"Water Pressure",                  "water_pressure_kpa"),
    (r"Diff Pressure",                   "diff_pressure_kpa"),
    (r"Water Temperature",               "water_temp_c"),
    (r"Water Level",                     "water_level_m"),
    (r"Wind Direction",                  "wind_dir_deg"),
    (r"Gust Speed.*m/s",                 "wind_gust_ms"),
    (r"Wind gust\s*speed|Wind Gust Speed", "wind_gust_kmh"),
    (r"Wind Speed.*m/s",                 "wind_speed_ms"),
    (r"Avg Wind speed|Average [Ww]ind [Ss]peed|Average wind speed", "wind_speed_kmh"),
    (r"Battery",                         "battery_v"),
    (r"Dew Point",                       "dew_point_c"),
    (r"^RH\b|^RH ",                      "rh_pct"),
    (r"Temperature\b",                   "temp_c"),
]


def standardize_column(raw_col: str) -> str:
    """Map a raw HOBOlink column header to a standardized name."""
    for pattern, std_name in COLUMN_MAP:
        if re.search(pattern, raw_col, re.IGNORECASE):
            return std_name
    return raw_col  # keep original if unmatched


def load_hobo_csv(filepath: Path) -> pd.DataFrame:
    """Load a single HOBOlink CSV and standardize its columns."""
    df = pd.read_csv(filepath, low_memory=False)

    # Standardize column names
    new_cols = []
    seen = {}
    for col in df.columns:
        std = standardize_column(col)
        # Handle duplicate standard names (e.g. Greenwich has two temp sensors)
        count = seen.get(std, 0)
        seen[std] = count + 1
        if count > 0:
            std = f"{std}_{count + 1}"
        new_cols.append(std)

    df.columns = new_cols
    return df


def load_station(station_name: str) -> pd.DataFrame:
    """Load all PEINP CSVs for one station across all years."""
    station_dir = RAW_HOBO / station_name
    if not station_dir.exists():
        print(f"  [WARN] Directory not found: {station_dir}")
        return pd.DataFrame()

    frames = []
    for csv_path in sorted(station_dir.rglob("PEINP_*.csv")):
        try:
            df = load_hobo_csv(csv_path)
            df["source_file"] = csv_path.name
            frames.append(df)
        except Exception as exc:
            print(f"  [ERR]  {csv_path.name}: {exc}")

    if not frames:
        print(f"  [WARN] No PEINP CSVs found for {station_name}")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["station"] = station_name
    return combined


def verify_structure(df: pd.DataFrame, station_name: str) -> None:
    """Print structural summary for verification."""
    print(f"\n{'='*60}")
    print(f"  {station_name}")
    print(f"{'='*60}")
    print(f"  Rows        : {len(df):,}")
    print(f"  Columns     : {list(df.columns)}")
    print(f"  Date range  : {df['date'].iloc[0]}  →  {df['date'].iloc[-1]}")
    print(f"  dtypes      :")
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = non_null / len(df) * 100
        print(f"    {col:30s}  {str(df[col].dtype):10s}  {pct:5.1f}% present")


def main():
    RAW_OUT.mkdir(parents=True, exist_ok=True)

    for station in STATIONS:
        print(f"\n>>> Loading {station} ...")
        df = load_station(station)

        if df.empty:
            continue

        verify_structure(df, station)

        # Coerce measurement columns to numeric (handles mixed str/float)
        skip = {"date", "time", "station", "source_file"}
        for col in df.columns:
            if col not in skip:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Persist as Parquet for downstream scripts
        out_path = RAW_OUT / f"{station.replace(' ', '_').lower()}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    print("\n✓ 01_obtain complete.")


if __name__ == "__main__":
    main()
