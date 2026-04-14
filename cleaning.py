"""
cleaning.py – Data Ingestion and Cleaning Pipeline
===================================================
Consolidates the full OSEMN pipeline for Parks Canada weather station data:
  1. OBTAIN  – Load raw HOBOlink CSVs, standardize column names.
  2. SCRUB   – UTC conversion, outlier removal, hourly resampling, imputation.
  3. ECCC    – Download ECCC Stanhope reference data via bulk API.

Usage:
    python cleaning.py              # Run full pipeline
    python cleaning.py --skip-eccc  # Skip ECCC download (if already cached)
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"


def run_step(label: str, script: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(SRC / script)],
        cwd=str(SRC.parent),
    )
    if result.returncode != 0:
        print(f"[ERR] {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    skip_eccc = "--skip-eccc" in sys.argv

    print("=" * 60)
    print("  PARKS CANADA WEATHER – DATA CLEANING PIPELINE")
    print("=" * 60)

    run_step("[1/3] OBTAIN – Loading raw HOBOlink CSVs", "01_obtain.py")

    if not skip_eccc:
        run_step("[2/3] ECCC – Downloading Stanhope reference", "eccc_download.py")
    else:
        print("\n[2/3] ECCC – Skipped (--skip-eccc)")

    run_step("[3/3] SCRUB – Cleaning, resampling, imputing", "02_scrub.py")

    print("\n" + "=" * 60)
    print("  ✓ CLEANING PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
