"""
build_pdf.py – Generate final analysis PDF from report text + figure outputs.
Uses matplotlib PdfPages so no LaTeX or external tools are needed.
"""
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "outputs" / "figures"
OUT_PDF = ROOT / "outputs" / "Final_Analysis_Report.pdf"

# ── Helper functions ─────────────────────────────────────────────────

def add_text_page(pdf, title, body, fontsize=10):
    """Add a page with a title and wrapped body text."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    # Title
    fig.text(0.05, 0.95, title, fontsize=16, fontweight="bold",
             va="top", fontfamily="serif")
    # Body
    wrapped = "\n".join(
        "\n".join(textwrap.wrap(line, width=95)) if line.strip() else ""
        for line in body.strip().splitlines()
    )
    fig.text(0.05, 0.90, wrapped, fontsize=fontsize, va="top",
             fontfamily="serif", linespacing=1.45,
             transform=fig.transFigure, wrap=False)
    pdf.savefig(fig)
    plt.close(fig)


def add_figure_page(pdf, img_path, caption=""):
    """Add a page showing a saved PNG with an optional caption."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    img = mpimg.imread(str(img_path))
    ax = fig.add_axes([0.05, 0.15, 0.9, 0.75])
    ax.imshow(img, aspect="equal")
    ax.axis("off")
    if caption:
        fig.text(0.5, 0.10, caption, fontsize=11, ha="center",
                 fontfamily="serif", style="italic")
    pdf.savefig(fig)
    plt.close(fig)


def add_title_page(pdf):
    """Add a cover / title page."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.65,
             "Parks Canada\nMeteorological Network Optimization",
             fontsize=26, ha="center", va="center", fontweight="bold",
             fontfamily="serif", linespacing=1.6)
    fig.text(0.5, 0.50,
             "Final Analysis Report",
             fontsize=18, ha="center", va="center",
             fontfamily="serif", color="#444444")
    fig.text(0.5, 0.42,
             "DATA-3210 – Meteorological Network Optimization\n& Predictive Modeling",
             fontsize=12, ha="center", va="center",
             fontfamily="serif", color="#666666", linespacing=1.5)
    fig.text(0.5, 0.32,
             "April 2026",
             fontsize=12, ha="center", va="center",
             fontfamily="serif", color="#888888")
    pdf.savefig(fig)
    plt.close(fig)


# ── Report content ───────────────────────────────────────────────────

SECTION_A = """
Parks Canada operates five autonomous weather stations across PEI National Park (PEINP): Cavendish, Greenwich, North Rustico Wharf, Stanley Bridge Wharf, and Tracadie Wharf. These stations record meteorological variables at 2–10 minute intervals via the HOBOlink telemetry platform.

ECCC Stanhope (Climate ID 8300590) serves as the independent reference station.

Data Ingestion:
~185 raw HOBOlink CSV files were loaded and standardized using regex-based column mapping. Sensor-specific column names were normalized to: temp_c, rh_pct, rain_mm, wind_speed_kmh, solar_rad_wm2, etc. All measurement columns were coerced to numeric types.

ECCC Reference Data:
35,064 hourly records were downloaded from ECCC via the bulk CSV API (Station ID 6545, 2022–2025).

Cleaning & Imputation:
• Timestamps converted from HOBOlink format to UTC.
• Outliers flagged using physically plausible bounds (e.g. temp: -50 to 50°C).
• Irregular observations resampled to hourly frequency.
• Three-tier gap filling: (1) linear interpolation ≤3h, (2) forward fill ≤24h, (3) NaN if >24h.

Total cleaned dataset: 156,971 hourly records across 6 stations.

Key finding: Stanley Bridge and Tracadie have NO humidity sensors (100% missing RH). Greenwich has 34% missing temperature. Wind speed gaps are 18–27% across the network.
"""

SECTION_B = """
ECCC Benchmark Correlation:
Each park station was correlated against ECCC Stanhope on overlapping hourly data.

• Cavendish:  temp r=0.991, RH r=0.921, wind r=0.800  → MARGINAL
• Greenwich:  temp r=0.975, RH r=0.833, wind r=0.767  → UNIQUE
• N. Rustico: temp r=0.993, RH r=0.913, wind r=0.497  → MARGINAL
• Stanley Br: temp r=0.960, RH=N/A,     wind r=0.799  → UNIQUE
• Tracadie:   temp r=0.997, RH=N/A,     wind r=0.887  → UNIQUE

No station meets the REDUNDANT threshold (≥3 variables with r>0.9 vs ECCC).

PCA:
Three components explain 83.2% of variance (PC1=39.9%, PC2=22.9%, PC3=20.4%). PCA was computed on daily-averaged features for stations with complete data (Cavendish, Greenwich, North Rustico).

K-Means (K=3):
Cavendish and North Rustico share the same dominant cluster. Greenwich occupies a distinct cluster, confirming it captures a different environmental signal.

Hierarchical Clustering:
Ward-linkage dendrogram shows Cavendish + N. Rustico as the closest pair (distance ~0.9). The wharf stations (Tracadie, Stanley Bridge) form a separate branch driven by wind exposure differences.
"""

SECTION_C = """
The Canadian Forest Fire Weather Index System (CFFDRS) was implemented following Van Wagner (1987). All six standard components were calculated: FFMC, DMC, DC, ISI, BUI, FWI.

Eligible stations: Only Cavendish and Greenwich have all four required inputs (temperature, RH, wind speed, rainfall). ECCC Stanhope was computed as a validation reference.

Results:
• Cavendish: 1,051 days computed, mean FWI=1.14, max FWI=16.66
• Greenwich: 602 days computed, mean FWI=1.28, max FWI=11.53
• Stanhope (ECCC): 1,445 days computed, mean FWI=1.34, max FWI=14.85

ECCC Validation:
• Cavendish vs ECCC Stanhope FWI correlation: r = 0.933 (884 overlapping days)
• Greenwich vs ECCC Stanhope FWI correlation: r = 0.936 (602 overlapping days)

Both correlations exceed 0.93, demonstrating strong agreement with the government reference and confirming that park-station data quality is sufficient for operational fire risk assessment.
"""

SECTION_D = """
For each station, hourly residuals were computed as:
    r(i,t) = x(i,t) - mean(x(-i,t))
where x(i,t) is the station reading and mean(x(-i,t)) is the network mean excluding that station.

A Gaussian KDE was fitted to each station's residual distribution. P(extreme) = P(|r| > 2σ).

Results – P(Extreme Deviation):
               temp_c   rain_mm   wind_speed   Max Risk
Cavendish       0.045    0.025      0.053       0.053 HIGH
Greenwich       0.012    0.024      0.049       0.049 MODERATE
N. Rustico      0.049    0.023      0.046       0.049 MODERATE
Stanley Br.     0.014    0.024      0.068       0.068 HIGH
Tracadie        0.040    0.025      0.054       0.054 HIGH
Stanhope        0.044    0.026      0.046       0.046 MODERATE

Wind speed is the dominant driver of uniqueness. Every station has P(extreme) > 0.04 for wind, reflecting localized coastal exposure effects. Stanley Bridge has the highest single-variable risk at 0.068.

No station has P(extreme) < 0.02 across all variables — every station captures micro-climate signals not represented by the rest of the network.
"""

CONCLUSIONS = """
1. No station is fully redundant. The combined evidence from ECCC correlations, PCA/clustering, and KDE uncertainty consistently shows each station contributes unique variance.

2. Cavendish and North Rustico are the most similar pair (same cluster, merge distance ~0.9), but both capture unique wind patterns.

3. Greenwich is irreplaceable for FWI. Despite 34% missing temperature, it provides the only FWI-capable measurements east of Cavendish.

4. Stanley Bridge and Tracadie lack RH sensors but provide unique wind and water-level monitoring critical for coastal resource management.

5. FWI outputs validate well against ECCC Stanhope (r > 0.93).

Recommendations:
• IMMEDIATE: Install RH sensors at Stanley Bridge and Tracadie.
• SHORT-TERM: Address 20–27% wind speed gaps through sensor maintenance.
• MEDIUM-TERM: If budget forces station reduction, North Rustico is the most expendable — but this increases micro-climate risk by ~5%.
• OVERALL: Retain all five stations.
"""

# ── Figure ordering ──────────────────────────────────────────────────

FIGURES = [
    ("01_timeseries_overlay.png",           "Figure 1: Hourly temperature overlay – all stations (2022–2025)"),
    ("04_missing_heatmap.png",              "Figure 2: Missing data (% of hourly records) after imputation"),
    ("02_correlation_temp_c.png",           "Figure 3: Inter-station temperature correlation matrix"),
    ("03_boxplots.png",                     "Figure 4: Variable distributions by station"),
    ("06_pca_scatter.png",                  "Figure 5: PCA scatter – daily weather features"),
    ("08_kmeans_clusters.png",              "Figure 6: K-Means clusters (K=3) in PCA space"),
    ("09_dendrogram.png",                   "Figure 7: Hierarchical clustering dendrogram (Ward linkage)"),
    ("07_kmeans_elbow.png",                 "Figure 8: K-Means elbow plot"),
    ("10_kde_residuals_temp_c.png",         "Figure 9: KDE residuals – temperature"),
    ("10_kde_residuals_wind_speed_kmh.png", "Figure 10: KDE residuals – wind speed"),
    ("11_risk_heatmap.png",                 "Figure 11: P(Extreme Deviation) – risk assessment heatmap"),
]

# ── Build PDF ────────────────────────────────────────────────────────

def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(OUT_PDF)) as pdf:
        # Cover
        add_title_page(pdf)

        # Section A
        add_text_page(pdf, "Section A – Data Pipeline & QA/QC", SECTION_A)

        # Figures 1-4 (EDA)
        for fname, caption in FIGURES[:4]:
            fp = FIG_DIR / fname
            if fp.exists():
                add_figure_page(pdf, fp, caption)

        # Section B
        add_text_page(pdf, "Section B – Station Redundancy Analysis", SECTION_B)

        # Figures 5-8 (PCA / Clustering)
        for fname, caption in FIGURES[4:8]:
            fp = FIG_DIR / fname
            if fp.exists():
                add_figure_page(pdf, fp, caption)

        # Section C
        add_text_page(pdf, "Section C – Fire Weather Index (FWI)", SECTION_C)

        # Section D
        add_text_page(pdf, "Section D – Probabilistic Uncertainty", SECTION_D)

        # Figures 9-11 (KDE / risk)
        for fname, caption in FIGURES[8:]:
            fp = FIG_DIR / fname
            if fp.exists():
                add_figure_page(pdf, fp, caption)

        # Conclusions
        add_text_page(pdf, "Conclusions & Recommendations", CONCLUSIONS)

    print(f"PDF written: {OUT_PDF}  ({OUT_PDF.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
