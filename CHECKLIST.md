# Execution Pipeline Checklist

## A — Data Pipeline & QA/QC

- [x] Ingest raw HOBOlink CSVs (185 files, 5 stations) → `data/raw/` Parquet
- [x] Download ECCC Stanhope hourly data (Climate ID 8300590) → `data/raw/eccc_stanhope.parquet`
- [x] UTC timestamp conversion
- [x] Outlier flagging (physical-bounds checks)
- [x] Hourly resampling (2–10 min → 60 min)
- [x] Tiered gap-filling (interpolation ≤ 3 h → ffill ≤ 24 h → NaN)
- [x] Generate `descriptive_stats.csv`
- [x] Generate `missing_value_report.csv`

## B — Station Redundancy Analysis

- [x] Pearson correlation vs ECCC Stanhope (temp, RH, wind)
- [x] ECCC benchmark classification (Redundant / Marginal / Unique)
- [x] PCA (3 components, 83.2% cumulative variance)
- [x] K-Means clustering (K = 3, elbow method)
- [x] Ward-linkage hierarchical clustering + dendrogram
- [x] Generate `eccc_benchmark.csv`

## C — Fire Weather Index (FWI)

- [x] CFFDRS implementation (Van Wagner 1987)
- [x] Daily noon-local aggregation (FFMC, DMC, DC, ISI, BUI, FWI)
- [x] Compute FWI for Cavendish, Greenwich, and ECCC Stanhope
- [x] Validate against ECCC (r > 0.93)

## D — Probabilistic Uncertainty (KDE)

- [x] Compute per-station hourly residuals vs network mean
- [x] Fit Gaussian KDE per variable per station
- [x] Calculate P(|residual| > 2σ) extreme-deviation risk
- [x] Generate `uncertainty_risk.csv`
- [x] Generate risk heatmap figure

## E — Figures & Visualizations

- [x] 01 — Time-series overlay
- [x] 02 — Correlation matrices (temp, RH, wind, rain, solar)
- [x] 03 — Boxplots
- [x] 04 — Missing-data heatmap
- [x] 05 — Monthly temperature
- [x] 06 — PCA scatter
- [x] 07 — K-Means elbow
- [x] 08 — K-Means clusters
- [x] 09 — Dendrogram
- [x] 10 — KDE residual distributions (temp, wind, rain)
- [x] 11 — Risk heatmap

## F — Deliverables

- [x] `README.md` — Project overview & quick start
- [x] `report.md` — Full technical report (Sections A–D)
- [x] `presentation.md` — Marp slide deck
- [x] `analysis.ipynb` — Documented analysis notebook
- [x] `outputs/analysis.html` — Notebook HTML export
- [x] `outputs/Final_Analysis_Report.pdf` — PDF report
- [x] GitHub repository (public) — https://github.com/kobina92/parks-meteo-optimization
