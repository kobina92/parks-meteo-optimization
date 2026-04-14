"""
uncertainty.py – Probabilistic Uncertainty Analysis (LO 7)
Uses Kernel Density Estimation to quantify the risk that removing
a station would result in losing critical micro-climate data.

For each station, estimates the probability that the station captures
conditions NOT well-represented by the remaining network.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRUBBED = PROJECT_ROOT / "data" / "scrubbed"
FIG_OUT = PROJECT_ROOT / "outputs" / "figures"

CORE_COLS = ["temp_c", "rain_mm", "wind_speed_kmh"]


def load_all_hourly() -> pd.DataFrame:
    frames = []
    for pq in sorted(SCRUBBED.glob("*.parquet")):
        if "fwi" in pq.stem or pq.stem == "missing_value_report":
            continue
        df = pd.read_parquet(pq)
        frames.append(df)
    return pd.concat(frames, ignore_index=False)


def compute_residuals(df: pd.DataFrame, target_station: str,
                      col: str) -> np.ndarray:
    """
    Compute residuals between target station and network mean
    (all other stations). Large residuals indicate unique micro-climate.
    """
    target = df[df["station"] == target_station]
    others = df[df["station"] != target_station]

    # Compute hourly network mean (excluding target)
    net_mean = others.groupby(others.index)[col].mean()

    # Align and compute residuals
    merged = target[[col]].join(net_mean, rsuffix="_net", how="inner").dropna()
    if len(merged) < 50:
        return np.array([])

    residuals = merged[col] - merged[f"{col}_net"]
    return residuals.values


def kde_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each station and variable, fit KDE to residuals and compute
    probability of extreme deviations (|residual| > 2σ of network).
    """
    stations = [s for s in df["station"].unique()]
    cols = [c for c in CORE_COLS if c in df.columns]

    records = []
    for stn in stations:
        for col in cols:
            resid = compute_residuals(df, stn, col)
            if len(resid) < 50:
                records.append({
                    "station": stn, "variable": col,
                    "p_extreme": np.nan, "mean_resid": np.nan,
                    "std_resid": np.nan, "n_obs": len(resid),
                })
                continue

            # Fit KDE
            kde = stats.gaussian_kde(resid)
            sigma = resid.std()
            mean = resid.mean()

            # P(|residual| > 2σ) = probability of extreme micro-climate
            x = np.linspace(resid.min() - sigma, resid.max() + sigma, 1000)
            density = kde(x)
            mask = np.abs(x - mean) > 2 * sigma
            dx = x[1] - x[0]
            p_extreme = np.sum(density[mask]) * dx

            records.append({
                "station": stn, "variable": col,
                "p_extreme": round(p_extreme, 4),
                "mean_resid": round(mean, 3),
                "std_resid": round(sigma, 3),
                "n_obs": len(resid),
            })

    return pd.DataFrame(records)


def plot_kde_ridgeline(df: pd.DataFrame, col: str = "temp_c"):
    """KDE ridge plot of residuals for each station."""
    stations = [s for s in df["station"].unique()]

    fig, axes = plt.subplots(len(stations), 1, figsize=(10, 2.5 * len(stations)),
                             sharex=True)
    if len(stations) == 1:
        axes = [axes]

    colors = sns.color_palette("tab10", len(stations))

    for ax, stn, color in zip(axes, stations, colors):
        resid = compute_residuals(df, stn, col)
        if len(resid) < 50:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center")
            ax.set_ylabel(stn, fontsize=8, rotation=0, labelpad=80)
            continue

        kde = stats.gaussian_kde(resid)
        x = np.linspace(resid.min() - 1, resid.max() + 1, 300)
        ax.fill_between(x, kde(x), alpha=0.4, color=color)
        ax.plot(x, kde(x), color=color, lw=1.5)

        # Mark 2σ bounds
        sigma = resid.std()
        mean = resid.mean()
        ax.axvline(mean - 2 * sigma, ls="--", color="red", alpha=0.5, lw=0.8)
        ax.axvline(mean + 2 * sigma, ls="--", color="red", alpha=0.5, lw=0.8)
        ax.set_ylabel(stn, fontsize=8, rotation=0, labelpad=100, ha="right")
        ax.set_yticks([])

    axes[-1].set_xlabel(f"Residual ({col}) vs Network Mean")
    fig.suptitle(f"KDE of Station Residuals – {col}", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_OUT / f"10_kde_residuals_{col}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 10_kde_residuals_{col}.png")


def plot_risk_heatmap(risk_df: pd.DataFrame):
    """Heatmap showing P(extreme) for each station × variable."""
    pivot = risk_df.pivot(index="station", columns="variable", values="p_extreme")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, vmin=0, vmax=0.15)
    ax.set_title("Probability of Extreme Micro-Climate Deviation\n(Risk of Data Loss if Station Removed)")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "11_risk_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved 11_risk_heatmap.png")


def main():
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    print(">>> Loading scrubbed data ...")
    df = load_all_hourly()
    print(f"  Stations: {list(df['station'].unique())}")

    print("\n>>> KDE Uncertainty Analysis ...")
    risk = kde_analysis(df)
    print("\n  Risk Assessment (P of extreme deviation):")
    print(risk.to_string(index=False))

    risk.to_csv(FIG_OUT.parent / "uncertainty_risk.csv", index=False)

    print("\n>>> Generating KDE plots ...")
    for col in CORE_COLS:
        if col in df.columns:
            plot_kde_ridgeline(df, col)

    plot_risk_heatmap(risk)

    # Summary recommendation
    print("\n" + "=" * 60)
    print("  UNCERTAINTY SUMMARY")
    print("=" * 60)
    for stn in risk["station"].unique():
        sub = risk[risk["station"] == stn]
        max_p = sub["p_extreme"].max()
        if max_p > 0.05:
            verdict = "HIGH RISK – captures unique micro-climate"
        elif max_p > 0.02:
            verdict = "MODERATE RISK – some unique signal"
        else:
            verdict = "LOW RISK – removable without significant data loss"
        print(f"  {stn:25s} P_max={max_p:.4f}  → {verdict}")

    print("\n✓ Uncertainty analysis complete.")


if __name__ == "__main__":
    main()
