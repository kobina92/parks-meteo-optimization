# Technical Report: Parks Canada Meteorological Network Optimization

**DATA-3210 – Meteorological Network Optimization & Predictive Modeling**  
**Date:** April 13, 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Section A – Data Pipeline & QA/QC](#2-section-a--data-pipeline--qaqc)
3. [Section B – Station Redundancy Analysis](#3-section-b--station-redundancy-analysis)
4. [Section C – Fire Weather Index (FWI)](#4-section-c--fire-weather-index-fwi)
5. [Section D – Probabilistic Uncertainty Analysis](#5-section-d--probabilistic-uncertainty-analysis)
6. [Conclusions & Recommendations](#6-conclusions--recommendations)
7. [References](#7-references)

---

## 1. Introduction

Parks Canada operates five autonomous weather stations across Prince Edward Island National Park (PEINP) to support ecological monitoring, fire risk assessment, and resource allocation. These stations—**Cavendish**, **Greenwich**, **North Rustico Wharf**, **Stanley Bridge Wharf**, and **Tracadie Wharf**—record meteorological variables at 2–10 minute intervals via the HOBOlink telemetry platform.

This report addresses four analytical objectives:

- **A.** Build an automated data pipeline with quality assurance and gap-filling.
- **B.** Assess inter-station redundancy using PCA and clustering, benchmarked against ECCC Stanhope.
- **C.** Calculate the Canadian Forest Fire Weather Index (FWI) for fire-capable stations.
- **D.** Quantify the probabilistic risk of micro-climate data loss if any station were removed.

Environment Canada and Climate Change (ECCC) Stanhope station (Climate ID 8300590) serves as the independent reference throughout.

---

## 2. Section A – Data Pipeline & QA/QC

### 2.1 Data Ingestion (`01_obtain.py`)

Raw HOBOlink CSV files (~185 files across 5 station folders) were ingested and standardized. Key challenges:

- **Inconsistent headers**: Sensor-specific column names varied across stations (e.g., `Temperature - PTemp` vs `Temperature - Water`). A regex-based `COLUMN_MAP` normalized these to a canonical schema: `temp_c`, `rh_pct`, `rain_mm`, `wind_speed_kmh`, `solar_rad_wm2`, etc.
- **Mixed types**: Some columns contained non-numeric overflow values (e.g., solar radiation logged as strings). All measurement columns were coerced via `pd.to_numeric(errors='coerce')` before Parquet serialization.
- **Non-PEINP files**: Stanley Bridge 2022 contained a Solinst water logger CSV in a different format. The glob pattern `PEINP_*.csv` correctly excluded it.

Output: 5 Parquet files in `data/raw/`, one per station.

### 2.2 ECCC Reference Download (`eccc_download.py`)

ECCC Stanhope hourly data (2022–2025) was downloaded programmatically via the bulk CSV API:

```
https://climate.weather.gc.ca/climate_data/bulk_data_e.html
  ?format=csv&stationID=6545&Year={year}&Month={month}&timeframe=1
```

Column names were mapped to the same canonical schema. Timestamps were converted from AST (UTC-4) to UTC. **35,064 hourly records** were retrieved. Notable data gap: `stn_pressure_kpa` had 0% availability from ECCC.

### 2.3 Cleaning & Imputation (`02_scrub.py`)

**Timestamp normalization**: HOBOlink timestamps (`MM/DD/YYYY HH:MM:SS -0400`) were parsed and converted to UTC.

**Outlier flagging**: Physically implausible values were masked using domain-informed bounds:

| Variable | Min | Max |
|----------|-----|-----|
| temp_c | −50°C | 50°C |
| rh_pct | 0% | 100% |
| rain_mm | 0 mm | 100 mm |
| wind_speed_kmh | 0 km/h | 200 km/h |
| solar_rad_wm2 | 0 W/m² | 1400 W/m² |

**Hourly resampling**: Irregular 2–10 minute observations were aggregated to hourly resolution using physically appropriate rules (mean for temperature/RH, sum for rain, max for gusts).

**Tiered gap-filling strategy**:

| Tier | Method | Max Gap | Rationale |
|------|--------|---------|-----------|
| 1 | Linear interpolation | ≤ 3 hours | Smooth short-lived sensor dropouts |
| 2 | Forward fill | ≤ 24 hours | Carry last valid reading during brief outages |
| 3 | Leave as NaN | > 24 hours | Avoid fabricating data during extended failures |

### 2.4 Missing Data Summary (Post-Imputation)

| Station | temp_c | rh_pct | rain_mm | wind_speed_kmh | solar_rad_wm2 |
|---------|--------|--------|---------|----------------|---------------|
| Cavendish | 6.5% | 6.5% | 0.0% | 20.5% | 6.5% |
| Greenwich | 33.9% | 26.7% | 0.0% | 10.1% | 7.4% |
| North Rustico Wharf | 3.0% | 3.0% | 0.0% | 26.7% | 0.0% |
| Stanley Bridge Wharf | 7.5% | **100%** | 0.0% | 18.3% | 1.5% |
| Tracadie Wharf | 6.2% | **100%** | 0.0% | 17.1% | 6.2% |
| Stanhope (ECCC) | 1.3% | 1.3% | 1.3% | 1.3% | **100%** |

**Critical finding**: Stanley Bridge and Tracadie have no RH sensors (100% missing). This eliminates them from FWI calculations which require temperature, RH, wind, and rain.

Total cleaned dataset: **156,971 hourly records** across 6 stations (5 park + 1 ECCC).

---

## 3. Section B – Station Redundancy Analysis

### 3.1 ECCC Benchmark Correlation

Each park station was correlated against ECCC Stanhope on overlapping hourly data:

| Station | temp_c r | rh_pct r | rain_mm r | wind_speed r | Flag |
|---------|----------|----------|-----------|--------------|------|
| Cavendish | 0.991 | 0.921 | 0.379 | 0.800 | MARGINAL |
| Greenwich | 0.975 | 0.833 | 0.517 | 0.767 | UNIQUE |
| North Rustico Wharf | 0.993 | 0.913 | 0.473 | 0.497 | MARGINAL |
| Stanley Bridge Wharf | 0.960 | N/A | 0.389 | 0.799 | UNIQUE |
| Tracadie Wharf | 0.997 | N/A | 0.492 | 0.887 | UNIQUE |

A station is flagged REDUNDANT if ≥3 variables exceed r = 0.9 vs ECCC. No station met this criterion. Cavendish and North Rustico are MARGINAL (2 high-correlation variables). Greenwich, Stanley Bridge, and Tracadie are UNIQUE.

### 3.2 PCA

Principal Component Analysis was performed on daily-averaged weather features (`temp_c`, `rh_pct`, `rain_mm`, `wind_speed_kmh`, `solar_rad_wm2`) for stations with complete data (Cavendish, Greenwich, North Rustico).

| Component | Explained Variance | Cumulative |
|-----------|--------------------|------------|
| PC1 | 39.9% | 39.9% |
| PC2 | 22.9% | 62.8% |
| PC3 | 20.4% | 83.2% |

Three components capture 83.2% of total variance. The PCA scatter plot shows substantial overlap between stations, but with distinct tails—particularly for Greenwich, which extends further into high-PC2 space (driven by RH variance).

### 3.3 K-Means Clustering (K=3)

K-Means was applied to the first two principal components:

| Dominant Cluster | Stations |
|------------------|----------|
| Cluster 0 | Greenwich |
| Cluster 1 | Cavendish, North Rustico Wharf |

Cavendish and North Rustico share the same dominant cluster, confirming they are the most similar pair. Greenwich occupies a distinct cluster.

### 3.4 Hierarchical Clustering

Ward-linkage hierarchical clustering on station-mean feature vectors (dropping columns with >50% missing) shows:

- **Nearest merge**: Cavendish + North Rustico (distance ~0.9)
- **Second merge**: Tracadie + Stanley Bridge (distance ~2.4)
- **Final merge**: The two wharf stations join the Cavendish/Rustico cluster before linking to the Stanhope/Greenwich branch

This dendrogram structure confirms that the wharf stations, despite lacking RH, capture a distinct environmental signal driven by coastal wind exposure.

---

## 4. Section C – Fire Weather Index (FWI)

### 4.1 Implementation

The Canadian Forest Fire Weather Index System (CFFDRS) was implemented following Van Wagner (1987). All six standard components were calculated:

- **Moisture codes**: Fine Fuel Moisture Code (FFMC), Duff Moisture Code (DMC), Drought Code (DC)
- **Fire behaviour indices**: Initial Spread Index (ISI), Buildup Index (BUI), Fire Weather Index (FWI)

Start-up values: FFMC=85, DMC=6, DC=15 (standard spring defaults). Day-length adjustments for latitude ~46°N (PEI) were applied to DMC and DC.

**Eligible stations**: Only Cavendish and Greenwich have all four required inputs (temperature, RH, wind speed, rainfall). ECCC Stanhope was computed as a validation reference.

### 4.2 Daily Aggregation

Hourly data was aggregated to daily noon-local (12:00 LST = 16:00 UTC) values:
- Temperature, RH, wind speed: noon reading (or daily mean if noon missing)
- Rainfall: 24-hour accumulation

### 4.3 Results

| Station | Days | FWI Mean | FWI Max | FFMC Mean | DC Max |
|---------|------|----------|---------|-----------|--------|
| Cavendish | 1,051 | 1.14 | 16.66 | 48.19 | 554.37 |
| Greenwich | 602 | 1.28 | 11.53 | 56.27 | 43.35 |
| Stanhope (ECCC) | 1,445 | 1.34 | 14.85 | 56.87 | 508.35 |

### 4.4 ECCC Validation

FWI correlations between park stations and ECCC Stanhope on overlapping days:

| Station | FWI r | Overlapping Days |
|---------|-------|------------------|
| Cavendish | **0.933** | 884 |
| Greenwich | **0.936** | 602 |

Both correlations exceed 0.93, demonstrating strong agreement with the government reference. Greenwich's shorter record (due to higher missing data) still produces valid FWI values when inputs are available.

---

## 5. Section D – Probabilistic Uncertainty Analysis

### 5.1 Methodology

For each station, hourly residuals were computed as:

$$r_{i,t} = x_{i,t} - \bar{x}_{-i,t}$$

where $x_{i,t}$ is the station's reading at hour $t$ and $\bar{x}_{-i,t}$ is the network mean excluding station $i$.

A Gaussian Kernel Density Estimate (KDE) was fitted to each station's residual distribution. The probability of extreme deviation was computed as:

$$P(\text{extreme}) = P(|r| > 2\sigma_{\text{network}})$$

where $\sigma_{\text{network}}$ is the standard deviation of the full network's residuals for that variable.

### 5.2 Results: P(Extreme Deviation) by Station

| Station | temp_c | rain_mm | wind_speed_kmh | Max Risk |
|---------|--------|---------|----------------|----------|
| Cavendish | 0.045 | 0.025 | 0.053 | 0.053 (HIGH) |
| Greenwich | 0.012 | 0.024 | 0.049 | 0.049 (MODERATE) |
| North Rustico Wharf | 0.049 | 0.023 | 0.046 | 0.049 (MODERATE) |
| Stanley Bridge Wharf | 0.014 | 0.024 | **0.068** | **0.068 (HIGH)** |
| Tracadie Wharf | 0.040 | 0.025 | 0.054 | 0.054 (HIGH) |
| Stanhope (ECCC) | 0.044 | 0.026 | 0.046 | 0.046 (MODERATE) |

### 5.3 Interpretation

- **Wind speed** is the dominant driver of uniqueness across all stations. Every station has P(extreme) > 0.04 for wind, reflecting localized coastal exposure effects.
- **Stanley Bridge Wharf** shows the highest single-variable risk (0.068 for wind), despite being flagged as having no RH sensor. Its coastal wind regime is statistically distinct from the network.
- **Greenwich** has the lowest overall risk (max 0.049) because its temperature residuals are tightly centered—but it remains classified as UNIQUE due to its distinct humidity profile (Section 3.1).
- No station has P(extreme) < 0.02 across all variables, meaning **every station captures some micro-climate signal not represented by the rest of the network**.

---

## 6. Conclusions & Recommendations

### Summary of Findings

1. **No station is fully redundant.** The combined evidence from ECCC benchmark correlations (Section B), PCA/clustering (Section B), and KDE uncertainty (Section D) consistently shows that each station contributes unique variance.

2. **Cavendish and North Rustico Wharf** are the most similar pair (same K-Means cluster, hierarchical merge distance ~0.9, both MARGINAL vs ECCC). If forced to remove one station, **North Rustico** would cause the least data loss.

3. **Greenwich** is irreplaceable for FWI computation. Despite 34% missing temperature, it provides the only FWI-capable measurements east of Cavendish.

4. **Stanley Bridge and Tracadie** lack RH sensors but provide unique wind and water-level monitoring critical for coastal resource management.

5. **FWI outputs validate well** against ECCC Stanhope (r > 0.93), confirming that park-station data quality is sufficient for operational fire risk assessment.

### Recommendations

| Priority | Action |
|----------|--------|
| **Immediate** | Install RH sensors at Stanley Bridge and Tracadie to enable FWI calculation at all stations. |
| **Short-term** | Address the 20–27% wind speed data gaps across the network through sensor maintenance and backup power. |
| **Medium-term** | If budget requires station reduction, North Rustico is the most expendable—but this increases micro-climate risk by ~5% for temperature. |
| **Retain** | All five stations. The KDE analysis shows every station captures statistically significant micro-climate deviations (P > 0.04) that would be lost if removed. |

---

## 7. References

- Van Wagner, C.E. (1987). *Development and Structure of the Canadian Forest Fire Weather Index System*. Forestry Technical Report 35, Canadian Forestry Service.
- Environment and Climate Change Canada. Climate Data Online: Stanhope, PE (Climate ID 8300590). https://climate.weather.gc.ca
- Onset Computer Corporation. HOBOlink Remote Monitoring Platform. https://www.onsetcomp.com
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
