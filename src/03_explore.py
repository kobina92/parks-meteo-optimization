"""
03_explore.py – Explore
Read scrubbed hourly Parquets and generate statistical visualisations
that inform the Redundancy Analysis (PCA) and FWI calculation downstream.

Outputs saved to outputs/figures/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for script use

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRUBBED_IN = PROJECT_ROOT / "data" / "scrubbed"
FIG_OUT = PROJECT_ROOT / "outputs" / "figures"

# Core columns for analysis
CORE_COLS = ["temp_c", "rh_pct", "rain_mm", "wind_speed_kmh", "solar_rad_wm2"]
CORE_LABELS = {
    "temp_c":         "Temperature (°C)",
    "rh_pct":         "Relative Humidity (%)",
    "rain_mm":        "Precipitation (mm/h)",
    "wind_speed_kmh": "Wind Speed (km/h)",
    "solar_rad_wm2":  "Solar Radiation (W/m²)",
}


def load_all_stations() -> pd.DataFrame:
    """Load every scrubbed station Parquet into one DataFrame."""
    frames = []
    for pq in sorted(SCRUBBED_IN.glob("*.parquet")):
        if pq.stem == "missing_value_report":
            continue
        df = pd.read_parquet(pq)
        frames.append(df)
    return pd.concat(frames, ignore_index=False)


# ===================================================================
# Plot 1 – Time-series overlay (one subplot per variable)
# ===================================================================
def plot_timeseries_overlay(df: pd.DataFrame) -> None:
    cols = [c for c in CORE_COLS if c in df.columns]
    n = len(cols)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(16, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    stations = df["station"].unique()
    palette = sns.color_palette("tab10", len(stations))

    for ax, col in zip(axes, cols):
        for i, stn in enumerate(stations):
            sub = df[df["station"] == stn]
            ax.plot(sub.index, sub[col], label=stn,
                    alpha=0.6, linewidth=0.4, color=palette[i])
        ax.set_ylabel(CORE_LABELS.get(col, col))
        ax.legend(fontsize=7, loc="upper right", ncol=len(stations))
        ax.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30)
    fig.suptitle("Hourly Weather – All Stations", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "01_timeseries_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 01_timeseries_overlay.png")


# ===================================================================
# Plot 2 – Pairwise correlation heatmap across stations (per variable)
# ===================================================================
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = [c for c in CORE_COLS if c in df.columns]

    for col in cols:
        pivot = df.pivot_table(index=df.index, columns="station", values=col)
        if pivot.shape[1] < 2:
            continue

        corr = pivot.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0.5, vmax=1.0, square=True, ax=ax,
                    linewidths=0.5)
        ax.set_title(f"Inter-Station Correlation – {CORE_LABELS.get(col, col)}")
        fig.tight_layout()
        safe = col.replace("/", "_")
        fig.savefig(FIG_OUT / f"02_correlation_{safe}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved 02_correlation_{col}.png")


# ===================================================================
# Plot 3 – Box plots (distribution comparison per station)
# ===================================================================
def plot_boxplots(df: pd.DataFrame) -> None:
    cols = [c for c in CORE_COLS if c in df.columns]
    n = len(cols)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
    if n == 1:
        axes = [axes]

    df_reset = df.reset_index(drop=True)

    for ax, col in zip(axes, cols):
        sns.boxplot(data=df_reset, x="station", y=col, hue="station",
                    ax=ax, palette="Set2", fliersize=1, legend=False)
        ax.set_title(CORE_LABELS.get(col, col), fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35, labelsize=8)

    fig.suptitle("Variable Distributions by Station", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "03_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 03_boxplots.png")


# ===================================================================
# Plot 4 – Missing-data heatmap (hours × station)
# ===================================================================
def plot_missing_heatmap(df: pd.DataFrame) -> None:
    stations = df["station"].unique()
    cols = [c for c in CORE_COLS if c in df.columns]
    if not cols:
        return

    records = []
    for stn in stations:
        sub = df[df["station"] == stn]
        total = len(sub)
        for col in cols:
            pct = sub[col].isna().sum() / total * 100 if total else 0
            records.append({"station": stn, "variable": col, "pct_missing": pct})

    miss = pd.DataFrame(records).pivot(index="station", columns="variable",
                                       values="pct_missing")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(miss, annot=True, fmt=".1f", cmap="YlOrRd",
                ax=ax, linewidths=0.5)
    ax.set_title("Missing Data (% of hourly records)")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "04_missing_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved 04_missing_heatmap.png")


# ===================================================================
# Plot 5 – Monthly mean temperature by station
# ===================================================================
def plot_monthly_means(df: pd.DataFrame) -> None:
    if "temp_c" not in df.columns:
        return

    df_m = df.copy()
    df_m["month"] = df_m.index.to_period("M").to_timestamp()
    monthly = df_m.groupby(["month", "station"])["temp_c"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.lineplot(data=monthly, x="month", y="temp_c", hue="station",
                 marker="o", ax=ax, palette="tab10")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.set_xlabel("")
    ax.set_title("Monthly Mean Temperature by Station")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "05_monthly_temp.png", dpi=150)
    plt.close(fig)
    print("  Saved 05_monthly_temp.png")


# ===================================================================
# Plot 6 – Descriptive statistics table (saved as CSV)
# ===================================================================
def save_descriptive_stats(df: pd.DataFrame) -> None:
    cols = [c for c in CORE_COLS if c in df.columns]
    stats = df.groupby("station")[cols].describe().round(2)
    out_path = FIG_OUT.parent / "descriptive_stats.csv"
    stats.to_csv(out_path)
    print(f"  Saved descriptive_stats.csv  ({len(stats)} rows)")


# ===================================================================
# Main
# ===================================================================
def main():
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    print(">>> Loading scrubbed data ...")
    df = load_all_stations()
    print(f"  Total rows : {len(df):,}")
    print(f"  Stations   : {df['station'].nunique()} – {list(df['station'].unique())}")
    print(f"  Date range : {df.index.min()} → {df.index.max()}")

    print("\n>>> Generating visualisations ...")
    plot_timeseries_overlay(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df)
    plot_missing_heatmap(df)
    plot_monthly_means(df)
    save_descriptive_stats(df)

    print("\n✓ 03_explore complete.")


if __name__ == "__main__":
    main()
