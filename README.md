# Parks Canada – Meteorological Network Optimization

DATA-3210 semester project: QA/QC pipeline, redundancy analysis, Fire Weather Index, and probabilistic uncertainty modelling for Parks Canada's PEI National Park autonomous weather station network.

## Directory Structure

```
parks-meteo-optimization/
├── cleaning.py                  # Top-level pipeline (run this first)
├── analysis.ipynb               # Documented EDA, PCA, FWI, and KDE notebook
├── requirements.txt             # Python dependencies
├── Data/                        # Raw HOBOlink CSVs (not tracked in git)
│   ├── Cavendish/
│   ├── Greenwich/
│   ├── North Rustico Wharf/
│   ├── Stanley Bridge Wharf/
│   └── Tracadie Wharf/
├── data/                        # Pipeline-generated Parquet files (not tracked)
│   ├── raw/                     # Output of 01_obtain.py
│   └── scrubbed/                # Output of 02_scrub.py (+ FWI daily files)
├── outputs/                     # Figures, CSVs, and analysis artifacts
│   ├── figures/
│   ├── eccc_benchmark.csv
│   ├── uncertainty_risk.csv
│   └── descriptive_stats.csv
└── src/
    ├── 01_obtain.py             # Load and standardize raw CSVs
    ├── 02_scrub.py              # UTC conversion, outlier flagging, resampling, imputation
    ├── 03_explore.py            # Exploratory visualisation suite
    ├── eccc_download.py         # ECCC Stanhope bulk-CSV API download
    ├── fwi.py                   # CFFDRS Fire Weather Index (Van Wagner 1987)
    ├── redundancy.py            # PCA, K-Means, hierarchical clustering + ECCC benchmark
    └── uncertainty.py           # KDE probabilistic risk assessment
```

## Quick Start

### 1. Create a virtual environment and install dependencies

```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the cleaning pipeline

```bash
python cleaning.py
```

This executes three steps sequentially:
1. **Obtain** – Reads all `PEINP_*.csv` files from `Data/`, standardizes column names, saves to `data/raw/` as Parquet.
2. **ECCC Download** – Fetches hourly data from ECCC Stanhope (Climate ID 8300590) via the bulk CSV API and saves to `data/raw/eccc_stanhope.parquet`.
3. **Scrub** – Converts timestamps to UTC, flags outliers, resamples to hourly frequency, applies tiered gap-filling (interpolation ≤ 3 h, forward fill ≤ 24 h), and writes cleaned data to `data/scrubbed/`.

Use `python cleaning.py --skip-eccc` to skip the ECCC download if the data is already cached.

### 3. Run analysis scripts

```bash
# Exploratory plots
python src/03_explore.py

# Fire Weather Index
python src/fwi.py

# PCA / Clustering redundancy
python src/redundancy.py

# KDE uncertainty
python src/uncertainty.py
```

### 4. Open the analysis notebook

```bash
jupyter notebook analysis.ipynb
```

## Data Sources

| Source | Stations | Period | Frequency |
|--------|----------|--------|-----------|
| HOBOlink (Parks Canada) | Cavendish, Greenwich, North Rustico Wharf, Stanley Bridge Wharf, Tracadie Wharf | 2022–2025 | 2–10 min |
| ECCC Stanhope | Climate ID 8300590 (Station ID 6545) | 2022–2025 | Hourly |

## Key Findings

- **No station is fully redundant.** Wind speed diverges significantly across stations despite high temperature correlations (>0.94).
- **Cavendish & North Rustico** are the most similar pair but each captures a unique wind regime.
- **Greenwich** is essential for FWI despite having 34% missing temperature data.
- **Stanley Bridge & Tracadie** lack RH sensors but provide unique tidal and water level monitoring.

## Dependencies

- Python ≥ 3.10
- pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
