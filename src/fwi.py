"""
fwi.py – Canadian Forest Fire Weather Index System (CFFDRS)
Calculates daily FWI moisture codes and indices for Cavendish & Greenwich.
Cross-references against ECCC Stanhope published values.

Implements the six standard CFFDRS components:
  Moisture codes : FFMC, DMC, DC
  Fire behaviour : ISI, BUI, FWI

Reference: Van Wagner, C.E. 1987. "Development and Structure of the Canadian
           Forest Fire Weather Index System." Forestry Technical Report 35.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRUBBED = PROJECT_ROOT / "data" / "scrubbed"
OUTPUT = PROJECT_ROOT / "data" / "scrubbed"

# Default startup values (standard CFFDRS spring startup)
FFMC_START = 85.0
DMC_START = 6.0
DC_START = 15.0

# Day-length adjustment factors for DMC (Le) and DC (Lf) by month
# Latitude ~46°N (PEI)
DMC_DL = {1: 6.5, 2: 7.5, 3: 9.0, 4: 12.8, 5: 13.9, 6: 13.9,
           7: 12.4, 8: 10.9, 9: 9.4, 10: 8.0, 11: 7.0, 12: 6.0}

DC_FL = {1: -1.6, 2: -1.6, 3: -1.6, 4: 0.9, 5: 3.8, 6: 5.8,
          7: 6.4, 8: 5.0, 9: 2.4, 10: 0.4, 11: -1.6, 12: -1.6}


# -----------------------------------------------------------------------
# FFMC – Fine Fuel Moisture Code
# -----------------------------------------------------------------------
def calc_ffmc(temp, rh, wind, rain, ffmc_prev):
    """Calculate today's FFMC from yesterday's FFMC and today's weather."""
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)

    if rain > 0.5:
        rf = rain - 0.5
        if mo <= 150.0:
            mr = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        else:
            mr = (mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) *
                  (1.0 - math.exp(-6.93 / rf)) +
                  0.0015 * (mo - 150.0) ** 2 * rf ** 0.5)
        mo = min(mr, 250.0)

    ed = (0.942 * rh ** 0.679 + 11.0 * math.exp((rh - 100.0) / 10.0) +
          0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh)))

    if mo > ed:
        ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + \
             0.0694 * wind ** 0.5 * (1.0 - (rh / 100.0) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * temp)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    else:
        ew = (0.618 * rh ** 0.753 + 10.0 * math.exp((rh - 100.0) / 10.0) +
              0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh)))
        if mo < ew:
            k1 = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + \
                 0.0694 * wind ** 0.5 * (1.0 - ((100.0 - rh) / 100.0) ** 8)
            kw = k1 * 0.581 * math.exp(0.0365 * temp)
            m = ew - (ew - mo) * 10.0 ** (-kw)
        else:
            m = mo

    ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    return max(0.0, min(ffmc, 101.0))


# -----------------------------------------------------------------------
# DMC – Duff Moisture Code
# -----------------------------------------------------------------------
def calc_dmc(temp, rh, rain, dmc_prev, month):
    """Calculate today's DMC."""
    if temp < -1.1:
        temp = -1.1

    le = DMC_DL[month]

    if rain > 1.5:
        re = 0.92 * rain - 1.27
        mo = 20.0 + math.exp(5.6348 - dmc_prev / 43.43)
        if dmc_prev <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc_prev)
        elif dmc_prev <= 65.0:
            b = 14.0 - 1.3 * math.log(dmc_prev)
        else:
            b = 6.2 * math.log(dmc_prev) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)
        pr = 244.72 - 43.43 * math.log(mr - 20.0)
        dmc_prev = max(pr, 0.0)

    if temp > -1.1:
        k = 1.894 * (temp + 1.1) * (100.0 - rh) * le * 1e-6
    else:
        k = 0.0

    return max(dmc_prev + 100.0 * k, 0.0)


# -----------------------------------------------------------------------
# DC – Drought Code
# -----------------------------------------------------------------------
def calc_dc(temp, rain, dc_prev, month):
    """Calculate today's DC."""
    if temp < -2.8:
        temp = -2.8

    lf = DC_FL[month]

    if rain > 2.8:
        rd = 0.83 * rain - 1.27
        qo = 800.0 * math.exp(-dc_prev / 400.0)
        qr = qo + 3.937 * rd
        dr = 400.0 * math.log(800.0 / qr)
        dc_prev = max(dr, 0.0)

    v = 0.36 * (temp + 2.8) + lf
    v = max(v, 0.0)

    return max(dc_prev + 0.5 * v, 0.0)


# -----------------------------------------------------------------------
# ISI – Initial Spread Index
# -----------------------------------------------------------------------
def calc_isi(wind, ffmc):
    """Calculate ISI from wind and FFMC."""
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    fw = math.exp(0.05039 * wind)
    ff = 91.9 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / (4.93e7))
    return 0.208 * fw * ff


# -----------------------------------------------------------------------
# BUI – Buildup Index
# -----------------------------------------------------------------------
def calc_bui(dmc, dc):
    """Calculate BUI from DMC and DC."""
    if dmc <= 0.4 * dc:
        bui = 0.8 * dmc * dc / (dmc + 0.4 * dc) if (dmc + 0.4 * dc) > 0 else 0.0
    else:
        bui = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * \
              (0.92 + (0.0114 * dmc) ** 1.7)
    return max(bui, 0.0)


# -----------------------------------------------------------------------
# FWI – Fire Weather Index
# -----------------------------------------------------------------------
def calc_fwi(isi, bui):
    """Calculate FWI from ISI and BUI."""
    if bui <= 80.0:
        fd = 0.626 * bui ** 0.809 + 2.0
    else:
        fd = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui))

    b = 0.1 * isi * fd
    if b > 1.0:
        fwi = math.exp(2.72 * (0.434 * math.log(b)) ** 0.647)
    else:
        fwi = b
    return fwi


# -----------------------------------------------------------------------
# Daily aggregation helpers
# -----------------------------------------------------------------------
def aggregate_to_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly data to daily values for FWI:
      - temp_c  : noon value (12:00 UTC ≈ 08:00 ADT) or daily mean
      - rh_pct  : noon value or daily mean
      - wind_speed_kmh : noon value or daily mean
      - rain_mm : 24-hour cumulative total
    """
    df = hourly_df.copy()
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")

    daily = pd.DataFrame()
    daily["temp_c"] = df["temp_c"].resample("1D").mean()
    daily["rh_pct"] = df["rh_pct"].resample("1D").mean()
    daily["rain_mm"] = df["rain_mm"].resample("1D").sum()

    if "wind_speed_kmh" in df.columns:
        daily["wind_speed_kmh"] = df["wind_speed_kmh"].resample("1D").mean()
    else:
        daily["wind_speed_kmh"] = 0.0

    daily = daily.dropna(subset=["temp_c", "rh_pct"])
    daily["month"] = daily.index.month
    return daily


def compute_fwi_series(daily: pd.DataFrame) -> pd.DataFrame:
    """Run the FWI system day-by-day for a daily weather DataFrame."""
    ffmc_prev = FFMC_START
    dmc_prev = DMC_START
    dc_prev = DC_START

    records = []
    for dt, row in daily.iterrows():
        t = row["temp_c"]
        h = max(0.0, min(row["rh_pct"], 100.0))
        w = max(row["wind_speed_kmh"], 0.0)
        r = max(row["rain_mm"], 0.0)
        m = int(row["month"])

        ffmc = calc_ffmc(t, h, w, r, ffmc_prev)
        dmc = calc_dmc(t, h, r, dmc_prev, m)
        dc = calc_dc(t, r, dc_prev, m)
        isi = calc_isi(w, ffmc)
        bui = calc_bui(dmc, dc)
        fwi = calc_fwi(isi, bui)

        records.append({
            "date": dt, "FFMC": ffmc, "DMC": dmc, "DC": dc,
            "ISI": isi, "BUI": bui, "FWI": fwi,
        })

        ffmc_prev = ffmc
        dmc_prev = dmc
        dc_prev = dc

    result = pd.DataFrame(records).set_index("date")
    return result


def main():
    stations = ["cavendish", "greenwich"]

    for stn in stations:
        path = SCRUBBED / f"{stn}.parquet"
        print(f"\n>>> Computing FWI for {stn.title()} ...")
        hourly = pd.read_parquet(path)

        daily = aggregate_to_daily(hourly)
        print(f"  Daily rows: {len(daily):,}")

        fwi_df = compute_fwi_series(daily)
        print(f"  FWI computed: {len(fwi_df):,} days")
        print(f"  FWI range: {fwi_df['FWI'].min():.2f} – {fwi_df['FWI'].max():.2f}")
        print(f"  FWI mean:  {fwi_df['FWI'].mean():.2f}")

        out = OUTPUT / f"{stn}_fwi_daily.parquet"
        fwi_df.to_parquet(out)
        print(f"  Saved → {out}")

    # ECCC Stanhope FWI for validation
    eccc_path = SCRUBBED / "eccc_stanhope.parquet"
    if eccc_path.exists():
        print(f"\n>>> Computing FWI for ECCC Stanhope (validation) ...")
        eccc = pd.read_parquet(eccc_path)
        eccc_daily = aggregate_to_daily(eccc)
        eccc_fwi = compute_fwi_series(eccc_daily)
        print(f"  FWI computed: {len(eccc_fwi):,} days")

        out = OUTPUT / "eccc_stanhope_fwi_daily.parquet"
        eccc_fwi.to_parquet(out)
        print(f"  Saved → {out}")

    print("\n✓ FWI calculation complete.")


if __name__ == "__main__":
    main()
