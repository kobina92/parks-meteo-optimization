"""
02_scrub.py – Scrub
Read raw Parquets from 01_obtain, then:
  1. Parse timestamps and convert to UTC.
  2. Coerce measurement columns to numeric.
  3. Flag / remove outliers.
  4. Resample irregular (2-10 min) data to hourly intervals.
  5. Report missing-value summary.
Outputs one cleaned hourly Parquet per station into data/scrubbed/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_IN = PROJECT_ROOT / "data" / "raw"
SCRUBBED_OUT = PROJECT_ROOT / "data" / "scrubbed"

# ---------------------------------------------------------------------------
# Core weather columns needed downstream (PCA / FWI)
# ---------------------------------------------------------------------------
WEATHER_COLS = [
    "temp_c",
    "rh_pct",
    "dew_point_c",
    "rain_mm",
    "wind_dir_deg",
    "wind_speed_kmh",
    "wind_gust_kmh",
    "solar_rad_wm2",
]

# Physically plausible bounds for outlier flagging
VALID_RANGES = {
    "temp_c":         (-50, 50),
    "rh_pct":         (0, 100),
    "dew_point_c":    (-60, 40),
    "rain_mm":        (0, 100),       # per observation (2-min bucket)
    "wind_dir_deg":   (0, 360),
    "wind_speed_kmh": (0, 200),
    "wind_gust_kmh":  (0, 250),
    "solar_rad_wm2":  (0, 1400),
}

# Hourly aggregation rules
AGG_RULES = {
    "temp_c":         "mean",
    "rh_pct":         "mean",
    "dew_point_c":    "mean",
    "rain_mm":        "sum",
    "wind_dir_deg":   "mean",
    "wind_speed_kmh": "mean",
    "wind_gust_kmh":  "max",
    "solar_rad_wm2":  "mean",
    # Keep extras if they exist
    "wind_speed_ms":  "mean",
    "wind_gust_ms":   "max",
    "baro_pressure_kpa": "mean",
    "water_level_m":  "mean",
    "water_temp_c":   "mean",
    "battery_v":      "mean",
    "accum_rain_mm":  "last",
}


def parse_timestamp_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine 'date' + 'time' columns into a timezone-aware UTC timestamp.
    HOBOlink format: date=MM/DD/YYYY, time=HH:MM:SS -0400
    The offset (e.g. -0400) represents the local timezone.
    """
    raw_dt = df["date"].astype(str) + " " + df["time"].astype(str)

    # Parse with UTC=False first so pandas reads the offset literally
    ts = pd.to_datetime(raw_dt, format="mixed", dayfirst=False, utc=True)

    df = df.copy()
    df["timestamp_utc"] = ts
    df = df.drop(columns=["date", "time"], errors="ignore")
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Force all measurement columns to numeric, coercing errors to NaN."""
    skip = {"timestamp_utc", "station", "source_file"}
    for col in df.columns:
        if col in skip:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def flag_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Replace out-of-range values with NaN and return counts."""
    counts = {}
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n = mask.sum()
        if n > 0:
            df.loc[mask, col] = np.nan
            counts[col] = int(n)
    return df, counts


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to hourly intervals using column-specific aggregation.
    Index must be 'timestamp_utc'.
    """
    df = df.set_index("timestamp_utc").sort_index()

    # Drop non-numeric / metadata cols before resampling
    meta_cols = {"station", "source_file"}
    station_name = df.get("station", pd.Series()).iloc[0] if "station" in df.columns else ""
    df = df.drop(columns=meta_cols & set(df.columns), errors="ignore")

    # Build agg dict only for columns that exist
    agg = {col: rule for col, rule in AGG_RULES.items() if col in df.columns}

    # For any remaining numeric columns not in AGG_RULES, default to mean
    for col in df.select_dtypes(include="number").columns:
        if col not in agg:
            agg[col] = "mean"

    hourly = df.resample("1h").agg(agg)

    # Restore station name
    hourly["station"] = station_name
    return hourly


def missing_summary(df: pd.DataFrame, station: str) -> pd.DataFrame:
    """Return a summary table of missing values per column."""
    total = len(df)
    records = []
    for col in df.columns:
        if col == "station":
            continue
        n_miss = df[col].isna().sum()
        records.append({
            "station": station,
            "column": col,
            "total_hours": total,
            "missing": int(n_miss),
            "pct_missing": round(n_miss / total * 100, 2) if total else 0,
        })
    return pd.DataFrame(records)


def impute_gaps(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing sensor values using a tiered strategy:
      1. Gaps ≤ 3 hours  → linear interpolation
      2. Gaps ≤ 24 hours → forward-fill (last known value)
      3. Gaps > 24 hours → left as NaN (sensor failure)
    Returns the dataframe and a dict of {col: n_filled}.
    """
    filled_counts = {}
    skip = {"station"}
    for col in df.columns:
        if col in skip or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        before_na = df[col].isna().sum()
        # Tier 1: linear interpolation for short gaps (≤ 3 hours)
        df[col] = df[col].interpolate(method="time", limit=3)
        # Tier 2: forward-fill for medium gaps (≤ 24 hours beyond interp)
        df[col] = df[col].ffill(limit=24)
        after_na = df[col].isna().sum()
        n_filled = before_na - after_na
        if n_filled > 0:
            filled_counts[col] = int(n_filled)
    return df, filled_counts


def scrub_station(parquet_path: Path) -> pd.DataFrame | None:
    """Full scrub pipeline for one station file."""
    station = parquet_path.stem.replace("_", " ").title()
    print(f"\n>>> Scrubbing {station} ...")

    df = pd.read_parquet(parquet_path)
    print(f"  Raw rows: {len(df):,}")

    # 1 – Parse timestamps to UTC
    df = parse_timestamp_utc(df)

    # 2 – Coerce to numeric
    df = coerce_numeric(df)

    # 3 – Drop duplicate timestamps
    before = len(df)
    df = df.drop_duplicates(subset="timestamp_utc", keep="first")
    dupes = before - len(df)
    if dupes:
        print(f"  Dropped {dupes:,} duplicate timestamps")

    # 4 – Outlier flagging
    df, outlier_counts = flag_outliers(df)
    if outlier_counts:
        print(f"  Outliers replaced with NaN: {outlier_counts}")

    # 5 – Hourly resample
    hourly = resample_hourly(df)
    print(f"  Hourly rows: {len(hourly):,}")

    # 6 – Impute missing values (tiered strategy)
    hourly, filled = impute_gaps(hourly)
    if filled:
        print(f"  Imputed values: {filled}")

    # 7 – Missing-value report (post-imputation)
    miss = missing_summary(hourly, station)
    print(f"  Missing-value summary:")
    for _, row in miss.iterrows():
        if row["pct_missing"] > 0:
            print(f"    {row['column']:30s}  {row['pct_missing']:6.2f}% missing")

    return hourly, miss


def main():
    SCRUBBED_OUT.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(RAW_IN.glob("*.parquet"))
    if not parquet_files:
        print("[ERR] No parquet files in data/raw/. Run 01_obtain.py first.")
        sys.exit(1)

    all_missing = []

    for pq in parquet_files:
        # ECCC data is already hourly – just impute and pass through
        if pq.stem == "eccc_stanhope":
            print(f"\n>>> Processing ECCC Stanhope (already hourly) ...")
            eccc = pd.read_parquet(pq)
            print(f"  Rows: {len(eccc):,}")
            eccc, filled = impute_gaps(eccc)
            if filled:
                print(f"  Imputed values: {filled}")
            station = "Stanhope (ECCC)"
            miss = missing_summary(eccc, station)
            all_missing.append(miss)
            out_path = SCRUBBED_OUT / pq.name
            eccc.to_parquet(out_path)
            print(f"  Saved → {out_path}")
            continue

        result = scrub_station(pq)
        if result is None:
            continue

        hourly, miss = result
        all_missing.append(miss)

        # Save scrubbed hourly data
        out_path = SCRUBBED_OUT / pq.name
        hourly.to_parquet(out_path)
        print(f"  Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Save combined missing-value report
    if all_missing:
        miss_df = pd.concat(all_missing, ignore_index=True)
        miss_path = SCRUBBED_OUT / "missing_value_report.csv"
        miss_df.to_csv(miss_path, index=False)
        print(f"\n  Missing-value report → {miss_path}")

    print("\n✓ 02_scrub complete.")


if __name__ == "__main__":
    main()
