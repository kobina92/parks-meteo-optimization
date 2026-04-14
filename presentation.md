---
marp: true
theme: default
paginate: true
---

# Parks Canada Weather Station Optimization
### DATA-3210 – Meteorological Network Optimization
**April 2026**

---

# Agenda

1. Project Overview & Data Sources
2. Data Pipeline & QA/QC
3. Redundancy Analysis (PCA & Clustering)
4. Fire Weather Index (FWI)
5. Probabilistic Uncertainty (KDE)
6. Conclusions & Recommendations

---

# 1. Project Overview

- **5 autonomous PEINP stations** + ECCC Stanhope reference
- Cavendish · Greenwich · North Rustico · Stanley Bridge · Tracadie
- HOBOlink telemetry: 2–10 min intervals, 2022–2025
- **Goal:** determine if any station can be removed without data loss

![bg right:40% 90%](outputs/figures/01_timeseries_overlay.png)

---

# 2. Data Pipeline

```
cleaning.py
  → 01_obtain.py   (185 CSVs → standardized Parquet)
  → eccc_download.py (ECCC API → Parquet)
  → 02_scrub.py     (UTC, outliers, hourly resample, impute)
```

**156,971 hourly records** across 6 stations

| Imputation Tier | Method | Max Gap |
|-----------------|--------|---------|
| Tier 1 | Linear interpolation | ≤ 3 h |
| Tier 2 | Forward fill | ≤ 24 h |
| Tier 3 | Leave NaN | > 24 h |

---

# 2b. Missing Data Post-Imputation

![width:900px](outputs/figures/04_missing_heatmap.png)

- Stanley Bridge & Tracadie: **no RH sensor** (100% missing)
- Greenwich: 34% temp missing

---

# 3. Redundancy – ECCC Benchmark

| Station | Temp r | RH r | Wind r | Flag |
|---------|--------|------|--------|------|
| Cavendish | 0.991 | 0.921 | 0.800 | MARGINAL |
| Greenwich | 0.975 | 0.833 | 0.767 | UNIQUE |
| N. Rustico | 0.993 | 0.913 | 0.497 | MARGINAL |
| Stanley Bridge | 0.960 | — | 0.799 | UNIQUE |
| Tracadie | 0.997 | — | 0.887 | UNIQUE |

**No station meets the REDUNDANT threshold** (≥3 vars with r > 0.9)

---

# 3b. PCA & K-Means Clustering

| PC1 (39.9%) | PC2 (22.9%) | PC3 (20.4%) | Cumulative: 83.2% |
|---|---|---|---|

**K-Means (K=3):** Cavendish & N. Rustico cluster together; Greenwich is distinct.

![bg right:50% 95%](outputs/figures/08_kmeans_clusters.png)

---

# 3c. Hierarchical Dendrogram

![width:800px](outputs/figures/09_dendrogram.png)

- **Closest pair:** Cavendish + North Rustico (distance ~0.9)
- Wharf stations form a separate branch (wind-driven)

---

# 4. Fire Weather Index (FWI)

CFFDRS implementation (Van Wagner, 1987): FFMC → DMC → DC → ISI → BUI → FWI

**Eligible:** Cavendish & Greenwich (require temp, RH, wind, rain)

| Station | Mean FWI | Max FWI | FWI r vs ECCC |
|---------|----------|---------|---------------|
| Cavendish | 1.14 | 16.66 | **0.933** |
| Greenwich | 1.28 | 11.53 | **0.936** |

![bg right:45% 90%](outputs/figures/05_monthly_temp.png)

---

# 5. Probabilistic Uncertainty (KDE)

Residual: $r_{i,t} = x_{i,t} - \bar{x}_{-i,t}$

P(extreme) = P(|residual| > 2σ)

![width:700px](outputs/figures/11_risk_heatmap.png)

**Every station:** P(extreme wind) > 0.04
**Highest risk:** Stanley Bridge (0.068 wind)

---

# 6. Conclusions

1. **No station is fully redundant** – all capture unique micro-climate signals
2. **Cavendish & N. Rustico** are most similar but still distinct in wind
3. **Greenwich** is irreplaceable for eastern FWI coverage
4. **Stanley Bridge & Tracadie** need RH sensors for FWI expansion
5. FWI validates at r > 0.93 vs ECCC Stanhope

---

# 6b. Recommendations

| Priority | Action |
|----------|--------|
| **Immediate** | Install RH sensors at Stanley Bridge & Tracadie |
| **Short-term** | Fix 20–27% wind data gaps (maintenance) |
| **Medium-term** | If forced to cut: North Rustico is most expendable |
| **Overall** | **Retain all 5 stations** |

> KDE shows every station captures P(extreme) > 0.04 micro-climate deviations that would be permanently lost if removed.

---

# Thank You

**Repository:** `parks-meteo-optimization/`
**Pipeline:** `python cleaning.py`
**Notebook:** `analysis.ipynb`
