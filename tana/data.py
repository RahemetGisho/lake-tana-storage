"""NASA POWER download (with caching) plus a clearly-labelled synthetic generator for smoke tests."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
PARAMS = {
    "PRECTOTCORR": "rain_mm_day",             # corrected precipitation, mm/day
    "ALLSKY_SFC_SW_DWN": "solar_kwh_m2_day",  # all-sky surface shortwave irradiance, kWh/m2/day
    "WS50M": "wind_ms_50m",                   # wind speed at 50 m, m/s
}


def fetch_nasa_power(lat: float, lon: float, start_year: int, end_year: int,
                     cache_dir: str | Path = "data", force: bool = False) -> pd.DataFrame:
    """Monthly NASA POWER series for one grid cell. Cached as CSV so reruns are offline."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"nasa_power_monthly_{lat:.2f}_{lon:.2f}_{start_year}_{end_year}.csv"
    if cache.exists() and not force:
        return _finish(pd.read_csv(cache, index_col=0, parse_dates=True))

    query = {
        "parameters": ",".join(PARAMS),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_year,
        "end": end_year,
        "format": "JSON",
    }
    last_err = None
    for attempt in range(4):
        try:
            r = requests.get(POWER_URL, params=query, timeout=90,
                             headers={"User-Agent": "lake-tana-storage/1.0"})
            r.raise_for_status()
            payload = r.json()
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"NASA POWER request failed after retries: {last_err}")

    block = payload["properties"]["parameter"]
    cols = {}
    for api_name, name in PARAMS.items():
        series = {}
        for key, val in block[api_name].items():
            if key.endswith("13"):  # POWER appends an annual mean as "month 13"
                continue
            ts = pd.Timestamp(year=int(key[:4]), month=int(key[4:]), day=1)
            series[ts] = np.nan if val <= -990 else val
        cols[name] = pd.Series(series)
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "month"
    df.to_csv(cache)
    with open(cache.with_suffix(".meta.json"), "w") as fh:
        json.dump({"source": "NASA POWER monthly point API", "query": query,
                   "fetched_utc": pd.Timestamp.now(tz="UTC").isoformat()}, fh, indent=2)
    return _finish(df)


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp()
    df = df.asfreq("MS")
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        print(f"[data] filled {n_missing} missing monthly values with calendar-month means")
        for c in df.columns:
            clim = df[c].groupby(df.index.month).transform("mean")
            df[c] = df[c].fillna(clim)
    df["days"] = df.index.days_in_month
    df["rain_mm"] = df["rain_mm_day"] * df["days"]
    return df


def synthetic_monthly(start_year: int, end_year: int, seed: int = 7) -> pd.DataFrame:
    """Plausible-looking highland climate for SMOKE TESTS ONLY. Never report results from this."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")
    rain_clim = np.array([0.1, 0.1, 0.5, 1.3, 3.0, 7.0, 11.0, 11.5, 6.5, 2.5, 0.7, 0.2])
    solar_clim = np.array([6.0, 6.4, 6.5, 6.3, 5.8, 4.8, 4.0, 4.0, 5.0, 5.8, 5.9, 5.8])
    wind_clim = np.array([5.8, 5.9, 5.6, 4.8, 4.1, 4.0, 4.4, 4.2, 3.9, 4.0, 4.7, 5.5])
    years = idx.year.values - start_year
    year_factor = np.exp(rng.normal(0, 0.12, years.max() + 1))[years]
    noise = rng.gamma(shape=6.0, scale=1 / 6.0, size=len(idx))
    rain = rain_clim[idx.month - 1] * year_factor * noise
    wet = (idx.month >= 6) & (idx.month <= 9)
    solar = solar_clim[idx.month - 1] * (1 - 0.06 * (year_factor - 1) * wet) + rng.normal(0, 0.15, len(idx))
    wind = wind_clim[idx.month - 1] * (1 + 0.04 * (1 - year_factor)) + rng.normal(0, 0.2, len(idx))
    df = pd.DataFrame({"rain_mm_day": rain, "solar_kwh_m2_day": solar, "wind_ms_50m": wind}, index=idx)
    df.index.name = "month"
    return _finish(df)


DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_nasa_power_daily(lat: float, lon: float, start_year: int, end_year: int,
                           cache_dir: str | Path = "data", force: bool = False) -> pd.DataFrame:
    """Daily all-sky solar irradiance (kWh/m2/day), one request per year, cached as CSV."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"nasa_power_daily_solar_{lat:.2f}_{lon:.2f}_{start_year}_{end_year}.csv"
    if cache.exists() and not force:
        return _finish_daily(pd.read_csv(cache, index_col=0, parse_dates=True))
    parts = []
    for yr in range(start_year, end_year + 1):
        query = {"parameters": "ALLSKY_SFC_SW_DWN", "community": "RE", "longitude": lon, "latitude": lat,
                 "start": f"{yr}0101", "end": f"{yr}1231", "format": "JSON"}
        last_err = None
        for attempt in range(4):
            try:
                r = requests.get(DAILY_URL, params=query, timeout=120, headers={"User-Agent": "lake-tana-storage/1.0"})
                r.raise_for_status()
                block = r.json()["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"NASA POWER daily request failed for {yr}: {last_err}")
        s = pd.Series({pd.Timestamp(k): (np.nan if v <= -990 else v) for k, v in block.items()})
        parts.append(s)
        print(f"[data] daily solar {yr} ok")
    df = pd.concat(parts).sort_index().to_frame("solar_kwh_m2_day")
    df.index.name = "date"
    df.to_csv(cache)
    return _finish_daily(df)


def _finish_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).normalize()
    df = df.asfreq("D")
    n = int(df["solar_kwh_m2_day"].isna().sum())
    if n:
        print(f"[data] filled {n} missing daily solar values with calendar-month means")
        clim = df["solar_kwh_m2_day"].groupby(df.index.month).transform("mean")
        df["solar_kwh_m2_day"] = df["solar_kwh_m2_day"].fillna(clim)
    return df


def synthetic_daily(start_year: int, end_year: int, seed: int = 11) -> pd.DataFrame:
    """Fake daily solar for SMOKE TESTS ONLY: monthly mean with persistent cloudy spells, worse in Jun-Sep."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    clear = np.array([6.6, 6.9, 6.9, 6.7, 6.4, 5.9, 5.3, 5.3, 6.0, 6.5, 6.5, 6.4])
    cloud_p = np.array([0.05, 0.06, 0.1, 0.2, 0.3, 0.55, 0.75, 0.75, 0.5, 0.2, 0.08, 0.05])
    vals = np.zeros(len(idx))
    cloudy = False
    for i, d in enumerate(idx):
        p = cloud_p[d.month - 1]
        stay = 0.55 + 0.4 * p
        cloudy = (rng.random() < stay) if cloudy else (rng.random() < p * 0.6)
        base = clear[d.month - 1]
        vals[i] = base * (rng.uniform(0.35, 0.75) if cloudy else rng.uniform(0.9, 1.03))
    df = pd.DataFrame({"solar_kwh_m2_day": vals}, index=idx)
    df.index.name = "date"
    return _finish_daily(df)
