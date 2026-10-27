"""Toy reservoir: inflow follows lagged rainfall; compare release rules that use, or ignore, forecasts.

Units: volumes in million cubic metres (Mm3) per month, energy in GWh.
The reservoir is deliberately simple. It is a decision-making testbed, not a model of any real dam.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

RULES = ["blind", "climatology", "sarima", "gbm", "oracle"]
RULE_LABEL = {
    "blind": "No forecast",
    "climatology": "Climatology",
    "sarima": "SARIMA forecast",
    "gbm": "Boosting forecast",
    "oracle": "Perfect foresight",
}


@dataclass(frozen=True)
class ResParams:
    cap: float       # storage capacity, Mm3
    dead: float      # dead storage, Mm3
    demand: float    # firm monthly release target, Mm3
    rmax: float      # max turbine release, Mm3/month
    init: float      # starting storage, Mm3


def inflow_scale(rain_mm: pd.Series, cfg: Config) -> float:
    """Calibrate so that mean annual inflow equals the assumed value. Uses all data (a pure unit conversion)."""
    conv = _lagged(rain_mm.values, cfg.runoff_weights)
    return (cfg.mean_annual_inflow_mm3 / 12.0) / np.nanmean(conv)


def _lagged(rain: np.ndarray, w: tuple) -> np.ndarray:
    out = np.zeros(len(rain))
    for lag, wt in enumerate(w):
        shifted = np.roll(rain, lag)
        shifted[:lag] = rain[:lag]  # first months: assume steady rainfall before the record starts
        out += wt * shifted
    return out


def actual_inflow(rain_mm: pd.Series, cfg: Config, scale: float) -> pd.Series:
    return pd.Series(scale * _lagged(rain_mm.values, cfg.runoff_weights), index=rain_mm.index, name="inflow_mm3")


def make_params(cfg: Config, cap_months: float | None = None, demand_frac: float | None = None) -> ResParams:
    mean_m = cfg.mean_annual_inflow_mm3 / 12.0
    cap = (cap_months if cap_months is not None else cfg.cap_months) * mean_m
    demand = (demand_frac if demand_frac is not None else cfg.demand_frac) * mean_m
    return ResParams(cap=cap, dead=cfg.dead_frac * cap, demand=demand,
                     rmax=cfg.rmax_mult * demand, init=cfg.init_frac * cap)


def inflow_lookahead(rain_obs: np.ndarray, rain_fc: np.ndarray, t: int, cfg: Config, scale: float) -> np.ndarray:
    """Inflow estimates for months t..t+H-1 as seen at the start of month t.

    rain_obs: observed monthly rain (only indices < t are used).
    rain_fc : forecast rain for months t, t+1, ..., t+H-1 (made with data up to t-1).
    """
    H = len(rain_fc)
    w = cfg.runoff_weights
    est = np.zeros(H)
    for k in range(H):
        total = 0.0
        for lag, wt in enumerate(w):
            j = k - lag                      # position relative to t
            total += wt * (rain_fc[j] if j >= 0 else rain_obs[max(t + j, 0)])
        est[k] = scale * total
    return est


def lookahead_release(storage: float, inflow_hat: np.ndarray, p: ResParams) -> float:
    """Release rule that looks ahead. If the reservoir is projected to overflow within the horizon
    when releasing only the firm demand, release extra water through the turbines now (up to rmax)
    instead of letting it spill later. It never cuts supply below the firm demand."""
    s = storage
    excess = []
    for inf in inflow_hat:
        s = s + inf - p.demand
        excess.append(s - p.cap)
    k = int(np.argmax(excess))
    ex = excess[k]
    if ex <= 0:
        return p.demand
    return float(min(p.rmax, p.demand + ex / (k + 1)))


def simulate(inflow: np.ndarray, p: ResParams, inflow_hat: np.ndarray | None = None) -> pd.DataFrame:
    """Month-by-month mass balance. inflow_hat (n, H) enables the look-ahead rule; None = no forecast."""
    n = len(inflow)
    s = p.init
    rows = []
    for t in range(n):
        inf = float(inflow[t])
        target = p.demand if inflow_hat is None else lookahead_release(s, inflow_hat[t], p)
        available = max(s + inf - p.dead, 0.0)
        release = min(target, available)
        unmet = max(p.demand - release, 0.0)
        s_new = s + inf - release
        spill = max(s_new - p.cap, 0.0)
        s_new = min(s_new, p.cap)
        rows.append((s, inf, release, unmet, spill, s_new))
        s = s_new
    df = pd.DataFrame(rows, columns=["storage_start", "inflow", "release", "unmet", "spill", "storage_end"])
    return df


def summarise(sim: pd.DataFrame, p: ResParams, cfg: Config) -> dict:
    shortfall = sim["unmet"] > 0.01 * p.demand
    return {
        "shortfall_months": int(shortfall.sum()),
        "unmet_mm3": float(sim["unmet"].sum()),
        "unmet_gwh": float(sim["unmet"].sum() * cfg.gwh_per_mm3),
        "spill_mm3": float(sim["spill"].sum()),
        "spill_months": int((sim["spill"] > 0.01 * p.demand).sum()),
        "generation_gwh": float(sim["release"].sum() * cfg.gwh_per_mm3),
        "mass_balance_error": float(
            (sim["storage_start"].iloc[0] + sim["inflow"].sum() - sim["release"].sum()
             - sim["spill"].sum() - sim["storage_end"].iloc[-1])),
    }


def build_lookaheads(rain: pd.Series, fcs: dict[str, np.ndarray], test_start: str,
                     cfg: Config, scale: float) -> dict[str, np.ndarray]:
    """Turn rainfall forecasts into per-month inflow look-ahead arrays for every rule that uses one."""
    vals = rain.values.astype(float)
    n = len(vals)
    t0 = rain.index.get_loc(pd.Timestamp(test_start))
    H = cfg.horizon
    out = {}
    sources = {k: v for k, v in fcs.items() if k in ("climatology", "sarima", "gbm")}
    for name, arr in sources.items():
        out[name] = np.array([inflow_lookahead(vals, arr[i], t0 + i, cfg, scale) for i in range(n - t0)])
    # oracle: the true future rainfall (padded with climatology past the end of the record)
    oracle = np.zeros((n - t0, H))
    clim = fcs["climatology"]
    for i in range(n - t0):
        t = t0 + i
        fut = np.array([vals[t + k] if t + k < n else clim[i, min(k, H - 1)] for k in range(H)])
        oracle[i] = inflow_lookahead(vals, fut, t, cfg, scale)
    out["oracle"] = oracle
    return out


def run_all_rules(inflow_test: np.ndarray, p: ResParams, lookaheads: dict[str, np.ndarray],
                  cfg: Config) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    sims = {"blind": simulate(inflow_test, p, None)}
    for name in ("climatology", "sarima", "gbm", "oracle"):
        sims[name] = simulate(inflow_test, p, lookaheads[name])
    table = pd.DataFrame({RULE_LABEL[k]: summarise(v, p, cfg) for k, v in sims.items()}).T
    return sims, table


def capacity_sensitivity(inflow_test: np.ndarray, lookaheads: dict[str, np.ndarray], cfg: Config,
                         cap_months=(3, 4, 5, 6, 7, 8, 10, 12)) -> pd.DataFrame:
    rec = []
    for cm in cap_months:
        p = make_params(cfg, cap_months=cm)
        sims, tab = run_all_rules(inflow_test, p, lookaheads, cfg)
        for rule, sim in sims.items():
            s = summarise(sim, p, cfg)
            rec.append({"cap_months": cm, "rule": rule, **s})
    return pd.DataFrame(rec)
