"""Inputs for the dashboard: data loading plus a disk cache for the slow rolling forecasts.

The SARIMA and boosting forecasts take about a minute to compute, so they are computed once, stored in
data/dashboard/forecasts.npz, and reloaded instantly. Commit that folder to deploy the app without waiting.
A fingerprint of the inputs invalidates the cache automatically when the data or settings change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BatteryConfig, Config
from .data import fetch_nasa_power, fetch_nasa_power_daily, synthetic_daily, synthetic_monthly
from .forecast import MODEL_ORDER, rolling_forecasts

CACHE_VERSION = "v1"


def load_inputs(cfg: Config, bc: BatteryConfig, synthetic: bool, data_dir: str | Path):
    """Monthly data (required) and daily solar (optional: None if it cannot be loaded)."""
    if synthetic:
        monthly = synthetic_monthly(cfg.start_year, cfg.end_year)
        daily = synthetic_daily(bc.daily_start_year, cfg.end_year)
        return monthly, daily
    monthly = fetch_nasa_power(cfg.lat, cfg.lon, cfg.start_year, cfg.end_year, data_dir)
    try:
        daily = fetch_nasa_power_daily(cfg.lat, cfg.lon, bc.daily_start_year, cfg.end_year, data_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[dashboard] daily solar unavailable, battery page disabled: {exc}")
        daily = None
    return monthly, daily


def fingerprint(cfg: Config, df: pd.DataFrame) -> str:
    key = "|".join(str(x) for x in (CACHE_VERSION, cfg.lat, cfg.lon, cfg.start_year, cfg.end_year,
                                     cfg.test_start, cfg.horizon, len(df), round(float(df["rain_mm"].sum()), 4)))
    return hashlib.md5(key.encode()).hexdigest()


def load_forecasts(cfg: Config, df: pd.DataFrame, synthetic: bool, data_dir: str | Path):
    """Return (forecast dict, built_now). Reads the cache when it matches the data, otherwise recomputes."""
    folder = Path(data_dir) / ("dashboard_synthetic" if synthetic else "dashboard")
    npz, meta = folder / "forecasts.npz", folder / "meta.json"
    fp = fingerprint(cfg, df)
    if npz.exists() and meta.exists():
        try:
            if json.loads(meta.read_text()).get("fingerprint") == fp:
                with np.load(npz) as z:
                    return {m: z[m] for m in MODEL_ORDER}, False
        except Exception:  # noqa: BLE001
            pass
    fcs = rolling_forecasts(df["rain_mm"], cfg.test_start, cfg.horizon)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **fcs)
        meta.write_text(json.dumps({"fingerprint": fp, "version": CACHE_VERSION}))
    except OSError as exc:
        print(f"[dashboard] could not write cache: {exc}")
    return fcs, True
