"""Battery dispatch next to solar PV: does a day-ahead solar forecast improve the overnight top-up decision?

The decision. A battery is reserved for the evening peak (critical hours). Solar normally charges it during the
day, but on a cloudy day it will not fill. The night before, the operator can top the battery up from hydropower
or the grid. Topping up when the sun would have filled the battery anyway wastes hydro energy; not topping up
before a cloudy day pushes the evening peak back onto the hydropower system, which is costlier.

   cost = peak_weight x (evening peak energy hydro must supply) + overnight_weight x (hydro energy drawn to top up)

Policies compared:  solar only (never top up) | always top up | calendar rule (top up Jun-Sep) |
                    forecast-aware with climatology / persistence / AR(1) forecasts | perfect day-ahead foresight.

HONESTY NOTES (repeated in the README): the load profile is synthetic, the hourly PV shape is a clear-sky bell
scaled to the real daily total, and intra-day cloud variability is not modelled. The experiment shows how the
decision logic behaves, not how a real plant would perform.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BatteryConfig

POLICIES = ["solar_only", "always_full", "calendar", "clim", "persistence", "ar1", "oracle"]
POLICY_LABEL = {
    "solar_only": "Solar only",
    "always_full": "Always top up",
    "calendar": "Top up Jun-Sep",
    "clim": "Forecast: climatology",
    "persistence": "Forecast: persistence",
    "ar1": "Forecast: AR(1)",
    "oracle": "Perfect foresight",
}

_LOAD_SHAPE = np.array([0.60, 0.58, 0.56, 0.56, 0.58, 0.70, 0.95, 1.10, 1.05, 0.95, 0.90, 0.90,
                        0.90, 0.90, 0.90, 0.92, 0.98, 1.10, 1.75, 1.80, 1.75, 1.60, 1.10, 0.80])


def load_profile(bc: BatteryConfig) -> np.ndarray:
    """Synthetic hourly load (MW): quiet night, morning bump, strong evening peak. Mean = load_mean_mw."""
    return bc.load_mean_mw * _LOAD_SHAPE / _LOAD_SHAPE.mean()


def pv_shape(bc: BatteryConfig) -> np.ndarray:
    w = np.zeros(24)
    span = bc.sunset - bc.sunrise
    for h in range(bc.sunrise, bc.sunset):
        w[h] = np.sin(np.pi * (h + 0.5 - bc.sunrise) / span)
    return w / w.sum()


def pv_profile(solar_kwh_m2_day: float, bc: BatteryConfig, shape: np.ndarray) -> np.ndarray:
    """Hourly PV energy (MWh = MW average over the hour) from a daily irradiance total."""
    daily_mwh = bc.pv_mwp * solar_kwh_m2_day * bc.pv_perf_ratio
    return daily_mwh * shape


def _t_crit(bc):
    return min(bc.critical_hours)


def soc_before_peak(pv: np.ndarray, load: np.ndarray, soc_start: float, bc: BatteryConfig) -> float:
    """State of charge when the critical window opens (battery only charges before it)."""
    soc = soc_start
    for h in range(_t_crit(bc)):
        surplus = pv[h] - load[h]
        if surplus > 0:
            charge_in = min(surplus, bc.power_mw, (bc.cap_mwh - soc) / bc.eta_charge)
            soc += charge_in * bc.eta_charge
    return soc


def plan_top_up(pv_expected: np.ndarray, load: np.ndarray, soc0: float, bc: BatteryConfig) -> float:
    """Smallest overnight top-up (MWh stored) that leaves the battery ready for the evening peak, given the
    PV expected tomorrow. Bisection works because the state of charge at the peak is non-decreasing in the top-up."""
    need = sum(max(load[h] - pv_expected[h], 0.0) for h in bc.critical_hours)
    target = min(bc.cap_mwh, need / bc.eta_discharge + bc.margin_mwh)
    room = bc.cap_mwh - soc0
    if room <= 1e-9 or soc_before_peak(pv_expected, load, soc0, bc) >= target - 1e-9:
        return 0.0
    if soc_before_peak(pv_expected, load, soc0 + room, bc) < target - 1e-9:
        return room
    lo, hi = 0.0, room
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if soc_before_peak(pv_expected, load, soc0 + mid, bc) >= target - 1e-9:
            hi = mid
        else:
            lo = mid
    return hi


def simulate_day(pv: np.ndarray, load: np.ndarray, soc0: float, top_up: float, bc: BatteryConfig,
                 detail: bool = False) -> dict:
    """Hourly day. Overnight top-up first, then PV charges the battery, and the battery discharges only in the
    critical window. Returns energy flows (MWh)."""
    top_up = float(min(max(top_up, 0.0), bc.cap_mwh - soc0))
    soc = soc0 + top_up
    grid_draw = top_up / bc.eta_charge
    curtailed = 0.0
    delivered = 0.0
    crit_unmet = 0.0
    charged_from_pv = 0.0
    soc_path, dis_path = [], []
    for h in range(24):
        surplus = pv[h] - load[h]
        discharge = 0.0
        if surplus > 0:
            charge_in = min(surplus, bc.power_mw, (bc.cap_mwh - soc) / bc.eta_charge)
            soc += charge_in * bc.eta_charge
            charged_from_pv += charge_in
            curtailed += surplus - charge_in
        elif h in bc.critical_hours:
            deficit = -surplus
            discharge = min(deficit, bc.power_mw, soc * bc.eta_discharge)
            soc -= discharge / bc.eta_discharge
            delivered += discharge
            crit_unmet += deficit - discharge
        soc_path.append(soc)
        dis_path.append(discharge)
    out = {"soc_end": soc, "grid_draw": grid_draw, "top_up": top_up, "curtailed": curtailed,
           "delivered": delivered, "crit_unmet": crit_unmet, "charged_from_pv": charged_from_pv,
           "pv_total": float(pv.sum())}
    if detail:
        out["soc_path"] = np.array(soc_path)
        out["discharge_path"] = np.array(dis_path)
    return out


def daily_solar_forecasts(solar: pd.Series, test_start: str) -> pd.DataFrame:
    """Day-ahead forecasts for every day, using only data from earlier days.
    clim = mean of earlier days in the same calendar month; persistence = yesterday;
    ar1 = clim + phi x yesterday's anomaly, with phi fitted on days before test_start."""
    s = solar.astype(float)
    clim = s.groupby(s.index.month).transform(lambda x: x.shift(1).expanding().mean())
    clim = clim.bfill()
    persistence = s.shift(1)
    train = s[s.index < pd.Timestamp(test_start)]
    tclim = train.groupby(train.index.month).transform("mean")
    anom = (train - tclim).values
    phi = float(np.corrcoef(anom[1:], anom[:-1])[0, 1])
    ar1 = clim + phi * (s.shift(1) - clim)
    out = pd.DataFrame({"actual": s, "clim": clim, "persistence": persistence, "ar1": ar1, "oracle": s})
    out = out.bfill()
    out.attrs["phi"] = phi
    return out


def score_daily_forecasts(fc: pd.DataFrame, test_start: str) -> pd.DataFrame:
    t = fc[fc.index >= pd.Timestamp(test_start)]
    rows = []
    for m in ("clim", "persistence", "ar1"):
        err = t[m] - t["actual"]
        rows.append({"model": m, "rmse": float(np.sqrt((err ** 2).mean())), "mae": float(err.abs().mean()),
                     "bias": float(err.mean())})
    df = pd.DataFrame(rows).set_index("model")
    df["skill_vs_clim"] = 1 - df["rmse"] / df.loc["clim", "rmse"]
    return df


def run_policy(policy: str, fc: pd.DataFrame, test_start: str, bc: BatteryConfig) -> pd.DataFrame:
    shape = pv_shape(bc)
    load = load_profile(bc)
    days = fc[fc.index >= pd.Timestamp(test_start)]
    soc = 0.0
    rec = []
    for date, row in days.iterrows():
        pv_actual = pv_profile(row["actual"], bc, shape)
        if policy == "solar_only":
            q = 0.0
        elif policy == "always_full":
            q = bc.cap_mwh - soc
        elif policy == "calendar":
            q = bc.cap_mwh - soc if date.month in (6, 7, 8, 9) else 0.0
        else:
            pv_exp = pv_profile(row[policy], bc, shape)
            q = plan_top_up(pv_exp, load, soc, bc)
        res = simulate_day(pv_actual, load, soc, q, bc)
        soc = res["soc_end"]
        cost = bc.peak_weight * res["crit_unmet"] + bc.overnight_weight * res["grid_draw"]
        rec.append({"date": date, "solar": row["actual"], "top_up": res["top_up"], "grid_draw": res["grid_draw"],
                    "crit_unmet": res["crit_unmet"], "curtailed": res["curtailed"], "delivered": res["delivered"],
                    "pv_total": res["pv_total"], "cost": cost})
    return pd.DataFrame(rec).set_index("date")


def run_all_policies(fc: pd.DataFrame, test_start: str, bc: BatteryConfig) -> tuple[dict, pd.DataFrame]:
    runs = {p: run_policy(p, fc, test_start, bc) for p in POLICIES}
    rows = {}
    for p, d in runs.items():
        rows[POLICY_LABEL[p]] = {
            "peak_hydro_mwh": d["crit_unmet"].sum(),
            "overnight_grid_mwh": d["grid_draw"].sum(),
            "weighted_cost": d["cost"].sum(),
            "curtailed_pv_mwh": d["curtailed"].sum(),
            "peak_covered_pct": 100 * d["delivered"].sum() / (d["delivered"].sum() + d["crit_unmet"].sum()),
        }
    table = pd.DataFrame(rows).T
    best_static = table.loc[[POLICY_LABEL[p] for p in ("solar_only", "always_full", "calendar")], "weighted_cost"].min()
    table["saving_vs_best_static_pct"] = 100 * (best_static - table["weighted_cost"]) / best_static
    return runs, table


def example_days(fc: pd.DataFrame, test_start: str):
    """A clear day and a cloudy day in the rainy season, for the dispatch figure."""
    t = fc[(fc.index >= pd.Timestamp(test_start)) & fc.index.month.isin([7, 8])]
    return t["actual"].idxmax(), t["actual"].idxmin()


def day_detail(policy: str, date: pd.Timestamp, fc: pd.DataFrame, bc: BatteryConfig) -> dict:
    """One day starting from an empty battery, with hourly paths, for illustration."""
    shape, load = pv_shape(bc), load_profile(bc)
    row = fc.loc[date]
    pv_actual = pv_profile(row["actual"], bc, shape)
    if policy == "solar_only":
        q = 0.0
    elif policy == "always_full":
        q = bc.cap_mwh
    else:
        q = plan_top_up(pv_profile(row[policy], bc, shape), load, 0.0, bc)
    res = simulate_day(pv_actual, load, 0.0, q, bc, detail=True)
    res.update({"pv": pv_actual, "load": load, "solar": float(row["actual"]), "forecast": float(row[policy]) if policy in row else np.nan})
    return res


def sensitivity(fc: pd.DataFrame, test_start: str, bc: BatteryConfig,
                pv_sizes=(20, 30, 40, 60, 80), overnight_weights=(0.0, 0.25, 0.5, 1.0, 1.5)) -> pd.DataFrame:
    """How the value of forecasts changes with PV size and with the cost of overnight energy.
    An overnight weight near 0 means hydro would otherwise spill, so topping up is almost free."""
    from dataclasses import replace
    rec = []
    for kind, values in (("pv_mwp", pv_sizes), ("overnight_weight", overnight_weights)):
        for v in values:
            b = replace(bc, **{kind: v})
            _, t = run_all_policies(fc, test_start, b)
            for pol in POLICIES:
                rec.append({"param": kind, "value": v, "policy": pol,
                            "weighted_cost": t.loc[POLICY_LABEL[pol], "weighted_cost"],
                            "saving_vs_best_static_pct": t.loc[POLICY_LABEL[pol], "saving_vs_best_static_pct"]})
    return pd.DataFrame(rec)
