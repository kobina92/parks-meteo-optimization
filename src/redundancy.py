"""
redundancy.py – Station Redundancy Analysis (LO 3)
  1. Benchmark all Park stations against ECCC Stanhope (correlation).
  2. PCA on shared weather variables across all 6 stations.
  3. K-Means clustering to identify overlapping variance.
  4. Hierarchical clustering dendrogram.
Outputs figures and a summary CSV to outputs/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRUBBED = PROJECT_ROOT / "data" / "scrubbed"
FIG_OUT = PROJECT_ROOT / "outputs" / "figures"

CORE_COLS = ["temp_c", "rh_pct", "rain_mm", "wind_speed_kmh", "solar_rad_wm2"]


def load_all_hourly() -> pd.DataFrame:
    """Load all 6 scrubbed station Parquets into one DataFrame."""
    frames = []
    for pq in sorted(SCRUBBED.glob("*.parquet")):
        if "fwi" in pq.stem or pq.stem == "missing_value_report":
            continue
        df = pd.read_parquet(pq)
        frames.append(df)
    return pd.concat(frames, ignore_index=False)


# ===================================================================
# 1. ECCC Benchmark – Pairwise correlation with Stanhope
# ===================================================================
def benchmark_vs_eccc(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation of each Park station vs ECCC Stanhope."""
    cols = [c for c in CORE_COLS if c in df.columns]
    stations = [s for s in df["station"].unique() if "ECCC" not in s]

    eccc = df[df["station"] == "Stanhope (ECCC)"]

    records = []
    for stn in stations:
        park = df[df["station"] == stn]
        row = {"station": stn}
        for col in cols:
            if col not in eccc.columns or eccc[col].isna().all():
                row[f"{col}_corr"] = np.nan
                continue
            merged = park[[col]].join(eccc[[col]], rsuffix="_eccc", how="inner").dropna()
            if len(merged) > 30:
                row[f"{col}_corr"] = merged.iloc[:, 0].corr(merged.iloc[:, 1])
            else:
                row[f"{col}_corr"] = np.nan
        records.append(row)

    result = pd.DataFrame(records)

    # Determine redundancy flag (>0.9 on temp AND at least 2 other vars)
    corr_cols = [c for c in result.columns if c.endswith("_corr")]
    for idx, row in result.iterrows():
        vals = [row[c] for c in corr_cols if not np.isnan(row[c])]
        high = sum(1 for v in vals if v > 0.9)
        result.loc[idx, "high_corr_count"] = high
        result.loc[idx, "redundancy_flag"] = "REDUNDANT" if high >= 3 else (
            "MARGINAL" if high >= 2 else "UNIQUE"
        )

    print("\n  ECCC Benchmark Results:")
    print(result.to_string(index=False))
    return result


# ===================================================================
# 2. PCA Analysis
# ===================================================================
def run_pca(df: pd.DataFrame) -> tuple[pd.DataFrame, PCA]:
    """Run PCA on daily-mean weather features across all stations."""
    cols = [c for c in CORE_COLS if c in df.columns]
    stations = df["station"].unique()

    # Build station × variable matrix (daily means → overall means)
    daily_records = []
    for stn in stations:
        sub = df[df["station"] == stn]
        # Resample to daily, then compute overall summary stats
        daily = sub[cols].resample("1D").mean().dropna()
        for _, day_row in daily.iterrows():
            rec = {"station": stn}
            for c in cols:
                rec[c] = day_row[c]
            daily_records.append(rec)

    feat_df = pd.DataFrame(daily_records).dropna()
    labels = feat_df["station"].values
    X = feat_df[cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(len(cols), 3))
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    pca_df["station"] = labels

    print(f"\n  PCA explained variance: {pca.explained_variance_ratio_.round(3)}")
    print(f"  Cumulative: {pca.explained_variance_ratio_.cumsum().round(3)}")
    return pca_df, pca


def plot_pca_scatter(pca_df: pd.DataFrame, pca: PCA):
    """Scatter plot of PC1 vs PC2 colored by station."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for stn in pca_df["station"].unique():
        sub = pca_df[pca_df["station"] == stn]
        ax.scatter(sub["PC1"], sub["PC2"], label=stn, alpha=0.3, s=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("PCA – All Stations (Daily Weather Features)")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "06_pca_scatter.png", dpi=150)
    plt.close(fig)
    print("  Saved 06_pca_scatter.png")


# ===================================================================
# 3. K-Means Clustering
# ===================================================================
def run_kmeans(pca_df: pd.DataFrame):
    """K-Means clustering on PCA components, with elbow method."""
    X = pca_df[["PC1", "PC2"]].values
    stations = pca_df["station"].values

    # Elbow curve
    inertias = []
    K_range = range(2, 7)
    for k in K_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(K_range), inertias, "bo-")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "07_kmeans_elbow.png", dpi=150)
    plt.close(fig)
    print("  Saved 07_kmeans_elbow.png")

    # Final clustering with K=3 (reasonable for 6 stations)
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    pca_df["cluster"] = km.fit_predict(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    for cl in sorted(pca_df["cluster"].unique()):
        sub = pca_df[pca_df["cluster"] == cl]
        ax.scatter(sub["PC1"], sub["PC2"], label=f"Cluster {cl}", alpha=0.3, s=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-Means Clusters (K=3) on PCA Space")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "08_kmeans_clusters.png", dpi=150)
    plt.close(fig)
    print("  Saved 08_kmeans_clusters.png")

    # Station-cluster summary
    summary = pca_df.groupby(["station", "cluster"]).size().reset_index(name="n_days")
    dominant = summary.loc[summary.groupby("station")["n_days"].idxmax()]
    print("\n  K-Means Station Assignments (dominant cluster):")
    print(dominant[["station", "cluster"]].to_string(index=False))
    return pca_df


# ===================================================================
# 4. Hierarchical Clustering Dendrogram
# ===================================================================
def plot_dendrogram(df: pd.DataFrame):
    """Hierarchical clustering on station-level mean features."""
    cols = [c for c in CORE_COLS if c in df.columns]
    stations = df["station"].unique()

    # Station-level mean vectors – drop cols with any NaN across stations
    means = []
    for stn in stations:
        sub = df[df["station"] == stn][cols].mean()
        means.append(sub)

    mean_df = pd.DataFrame(means, index=stations)
    mean_df = mean_df.dropna(axis=1)  # drop cols with NaN for any station

    X = mean_df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Z = linkage(X_scaled, method="ward")

    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, labels=list(mean_df.index), ax=ax, leaf_rotation=35, leaf_font_size=9)
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "09_dendrogram.png", dpi=150)
    plt.close(fig)
    print("  Saved 09_dendrogram.png")


def main():
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    print(">>> Loading scrubbed data ...")
    df = load_all_hourly()
    print(f"  Stations: {df['station'].nunique()} – {list(df['station'].unique())}")

    print("\n>>> ECCC Benchmarking ...")
    bench = benchmark_vs_eccc(df)
    bench.to_csv(FIG_OUT.parent / "eccc_benchmark.csv", index=False)

    print("\n>>> PCA Analysis ...")
    pca_df, pca_model = run_pca(df)
    plot_pca_scatter(pca_df, pca_model)

    print("\n>>> K-Means Clustering ...")
    pca_df = run_kmeans(pca_df)

    print("\n>>> Hierarchical Clustering ...")
    plot_dendrogram(df)

    print("\n✓ Redundancy analysis complete.")


if __name__ == "__main__":
    main()
