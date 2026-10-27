"""Solar and wind as backup for hydropower shortfalls, from the same NASA POWER cell.

Monthly means hide sub-monthly variability, so the wind capacity factor is computed by integrating a
power curve over a Weibull (k=2) wind distribution whose mean is the monthly mean speed. These are
screening-level estimates, good for showing the shape of the argument, not for sizing a plant.
"""
from __future__ import annotations

from math import gamma

import numpy as np
import pandas as pd

from .config import Config

WET = [6, 7, 8, 9]       # June-September: the main rainy season (kiremt)
DRY = [12, 1, 2]         # December-February: the driest, sunniest, windiest stretch


def seasonal_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["rain_mm", "solar_kwh_m2_day", "wind_ms_50m"]
    g = df[cols].groupby(df.index.month)
    out = pd.concat({"mean": g.mean(), "p10": g.quantile(0.10), "p90": g.quantile(0.90)}, axis=1)
    return out


def classify_rainy_seasons(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    jjas = df[df.index.month.isin(WET)].groupby(df[df.index.month.isin(WET)].index.year)["rain_mm"].sum()
    thr = jjas.quantile(cfg.weak_season_quantile)
    return pd.DataFrame({"jjas_rain_mm": jjas, "weak": jjas <= thr})


def condition_stats(df: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """Mean solar and wind in the dry season, in weak rainy seasons, and in other rainy seasons."""
    yr, mo = df.index.year, df.index.month
    weak_years = set(seasons.index[seasons.weak])
    sel = {
        "Dry season (Dec-Feb)": np.isin(mo, DRY),
        "Rainy season, weak years": np.isin(mo, WET) & np.isin(yr, list(weak_years)),
        "Rainy season, other years": np.isin(mo, WET) & ~np.isin(yr, list(weak_years)),
    }
    rows = {}
    for label, mask in sel.items():
        sub = df[mask]
        rows[label] = {
            "solar_kwh_m2_day": sub["solar_kwh_m2_day"].mean(),
            "wind_ms_50m": sub["wind_ms_50m"].mean(),
            "rain_mm_month": sub["rain_mm"].mean(),
            "months": len(sub),
        }
    return pd.DataFrame(rows).T


def pv_gwh(solar_kwh_m2_day, days, mwp: float, cfg: Config):
    """Monthly PV energy (GWh). Peak-sun-hours x capacity x performance ratio."""
    return mwp * np.asarray(solar_kwh_m2_day) * np.asarray(days) * cfg.pv_perf_ratio / 1000.0


def wind_capacity_factor(v50, cfg: Config) -> np.ndarray:
    v_hub = np.asarray(v50, dtype=float) * (cfg.hub_height_m / 50.0) ** cfg.wind_shear
    c = v_hub / gamma(1.5)                                   # Weibull scale for k = 2
    v = np.linspace(0.0, 40.0, 801)[None, :]
    c2 = c[:, None] if c.ndim else np.array([[c]])
    pdf = (2.0 / c2) * (v / c2) * np.exp(-(v / c2) ** 2)
    ramp = (v ** 3 - cfg.cut_in ** 3) / (cfg.rated ** 3 - cfg.cut_in ** 3)
    power = np.where(v < cfg.cut_in, 0.0, np.where(v < cfg.rated, ramp, np.where(v <= cfg.cut_out, 1.0, 0.0)))
    cf = np.trapezoid(pdf * power, v[0], axis=1) if hasattr(np, "trapezoid") else np.trapz(pdf * power, v[0], axis=1)
    return cf * cfg.wind_losses


def wind_gwh(v50, days, mw: float, cfg: Config):
    return mw * wind_capacity_factor(np.atleast_1d(v50), cfg) * 24.0 * np.asarray(days) / 1000.0


def supply_per_mw(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """GWh delivered per installed MW each month, for solar and for wind."""
    return pd.DataFrame({
        "solar_gwh_per_mw": pv_gwh(df["solar_kwh_m2_day"].values, df["days"].values, 1.0, cfg),
        "wind_gwh_per_mw": wind_gwh(df["wind_ms_50m"].values, df["days"].values, 1.0, cfg),
    }, index=df.index)


def gap_cover(shortfall_gwh: pd.Series, per_mw: pd.DataFrame, mw_grid=None) -> pd.DataFrame:
    """Share of hydropower shortfall energy covered, month by month, for solar-only, wind-only and a 50/50 mix."""
    if mw_grid is None:
        mw_grid = np.arange(0, 305, 5)
    sf = shortfall_gwh.reindex(per_mw.index).fillna(0.0).values
    total = sf.sum()
    rec = []
    for mw in mw_grid:
        solar = mw * per_mw["solar_gwh_per_mw"].values
        wind = mw * per_mw["wind_gwh_per_mw"].values
        mix = 0.5 * mw * (per_mw["solar_gwh_per_mw"].values + per_mw["wind_gwh_per_mw"].values)
        rec.append({
            "mw": mw,
            "solar": np.minimum(sf, solar).sum() / total if total > 0 else np.nan,
            "wind": np.minimum(sf, wind).sum() / total if total > 0 else np.nan,
            "mix": np.minimum(sf, mix).sum() / total if total > 0 else np.nan,
        })
    return pd.DataFrame(rec)


def worst_month_requirement(shortfall_gwh: pd.Series, per_mw: pd.DataFrame) -> pd.DataFrame:
    """Installed MW (50/50 solar + wind, total) needed to cover each shortfall month completely."""
    sf = shortfall_gwh[shortfall_gwh > 0]
    if sf.empty:
        return pd.DataFrame(columns=["shortfall_gwh", "mw_needed_mix"])
    per = 0.5 * (per_mw["solar_gwh_per_mw"] + per_mw["wind_gwh_per_mw"]).reindex(sf.index)
    return pd.DataFrame({"shortfall_gwh": sf, "mw_needed_mix": sf / per})


def complementarity(df: pd.DataFrame, inflow: pd.Series) -> dict:
    """Correlation of deseasonalised rainfall with deseasonalised solar and wind."""
    def anom(s):
        return s - s.groupby(s.index.month).transform("mean")
    a_rain, a_sol, a_wind = anom(df["rain_mm"]), anom(df["solar_kwh_m2_day"]), anom(df["wind_ms_50m"])
    return {
        "corr_rain_solar": float(a_rain.corr(a_sol)),
        "corr_rain_wind": float(a_rain.corr(a_wind)),
        "corr_seasonal_inflow_solar": float(
            inflow.groupby(inflow.index.month).mean().corr(df.groupby(df.index.month)["solar_kwh_m2_day"].mean())),
        "corr_seasonal_inflow_wind": float(
            inflow.groupby(inflow.index.month).mean().corr(df.groupby(df.index.month)["wind_ms_50m"].mean())),
    }
