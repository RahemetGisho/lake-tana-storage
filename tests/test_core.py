import numpy as np
import pandas as pd

from tana import backup, reservoir
from tana.config import Config
from tana.data import synthetic_monthly
from tana.forecast import rolling_forecasts

CFG = Config()


def test_mass_balance_closes_for_every_rule():
    df = synthetic_monthly(1990, 2025)
    scale = reservoir.inflow_scale(df.rain_mm, CFG)
    inflow = reservoir.actual_inflow(df.rain_mm, CFG, scale)
    p = reservoir.make_params(CFG)
    sim = reservoir.simulate(inflow.values, p)
    assert abs(reservoir.summarise(sim, p, CFG)["mass_balance_error"]) < 1e-6
    assert (sim.storage_end <= p.cap + 1e-9).all() and (sim.storage_end >= p.dead - 1e-9).all()


def test_inflow_calibration_hits_assumed_mean():
    df = synthetic_monthly(1990, 2025)
    scale = reservoir.inflow_scale(df.rain_mm, CFG)
    inflow = reservoir.actual_inflow(df.rain_mm, CFG, scale)
    assert abs(inflow.mean() * 12 - CFG.mean_annual_inflow_mm3) / CFG.mean_annual_inflow_mm3 < 0.01


def test_lookahead_never_cuts_below_firm_demand():
    p = reservoir.make_params(CFG)
    for s in np.linspace(p.dead, p.cap, 7):
        for inf in (0.0, p.demand, 3 * p.demand):
            r = reservoir.lookahead_release(s, np.array([inf, inf, inf]), p)
            assert p.demand - 1e-9 <= r <= p.rmax + 1e-9


def test_forecasts_do_not_leak_the_future():
    """Change the data from the test start onward; forecasts for the first test month must not move."""
    df = synthetic_monthly(1990, 2025)
    y = df.rain_mm
    a = rolling_forecasts(y, "2016-01", 3)
    y2 = y.copy()
    y2.loc["2016-01-01":] = y2.loc["2016-01-01":] * 5 + 100
    b = rolling_forecasts(y2, "2016-01", 3)
    for m in a:
        assert np.allclose(a[m][0], b[m][0]), f"{m} used information from the test period"


def test_wind_capacity_factor_is_sane_and_monotone():
    cf = backup.wind_capacity_factor(np.array([2.0, 4.0, 6.0, 8.0, 10.0]), CFG)
    assert (cf >= 0).all() and (cf <= 1).all()
    assert np.all(np.diff(cf) > 0)


def test_pv_energy_units():
    # 100 MWp x 6 peak-sun-hours x 30 days x 0.78 PR = 14.04 GWh
    assert abs(backup.pv_gwh(6.0, 30, 100.0, CFG) - 14.04) < 1e-6


# ------------------------------------------------------------------ battery
from dataclasses import replace  # noqa: E402

from tana import battery as B  # noqa: E402
from tana.config import BatteryConfig  # noqa: E402
from tana.data import synthetic_daily  # noqa: E402

BC = BatteryConfig()


def _fc():
    d = synthetic_daily(2010, 2025)
    return B.daily_solar_forecasts(d["solar_kwh_m2_day"], "2016-01-01")


def test_battery_day_energy_balance_and_limits():
    shape, load = B.pv_shape(BC), B.load_profile(BC)
    for solar in (1.5, 3.0, 5.5, 6.8):
        pv = B.pv_profile(solar, BC, shape)
        for soc0, q in ((0.0, 0.0), (10.0, 25.0), (0.0, 50.0)):
            r = B.simulate_day(pv, load, soc0, q, BC)
            stored = r["top_up"] + r["charged_from_pv"] * BC.eta_charge
            released = r["delivered"] / BC.eta_discharge
            assert abs((soc0 + stored - released) - r["soc_end"]) < 1e-9
            assert -1e-9 <= r["soc_end"] <= BC.cap_mwh + 1e-9


def test_pv_shape_conserves_daily_energy():
    pv = B.pv_profile(5.0, BC, B.pv_shape(BC))
    assert abs(pv.sum() - BC.pv_mwp * 5.0 * BC.pv_perf_ratio) < 1e-9


def test_daily_forecasts_do_not_leak_the_future():
    d = synthetic_daily(2010, 2025)["solar_kwh_m2_day"]
    a = B.daily_solar_forecasts(d, "2016-01-01")
    d2 = d.copy()
    d2.loc["2020-06-01":] = d2.loc["2020-06-01":] * 0.2
    b = B.daily_solar_forecasts(d2, "2016-01-01")
    before = a.index < pd.Timestamp("2020-06-01")
    for col in ("clim", "persistence", "ar1"):
        assert np.allclose(a.loc[before, col], b.loc[before, col]), f"{col} leaked future data"
    assert abs(a.loc[pd.Timestamp("2020-06-01"), "clim"] - b.loc[pd.Timestamp("2020-06-01"), "clim"]) < 1e-9


def test_perfect_foresight_is_never_worse_than_static_rules():
    fc = _fc()
    fc = fc[fc.index < pd.Timestamp("2018-01-01")]
    _, t = B.run_all_policies(fc, "2016-01-01", BC)
    oracle = t.loc[B.POLICY_LABEL["oracle"], "weighted_cost"]
    for p in ("solar_only", "always_full", "calendar"):
        assert oracle <= t.loc[B.POLICY_LABEL[p], "weighted_cost"] + 1e-6


def test_plan_top_up_is_zero_when_sun_will_fill_battery():
    shape, load = B.pv_shape(BC), B.load_profile(BC)
    big_pv = B.pv_profile(7.0, replace(BC, pv_mwp=120.0), shape)
    assert B.plan_top_up(big_pv, load, 0.0, BC) == 0.0
    no_pv = np.zeros(24)
    assert B.plan_top_up(no_pv, load, 0.0, BC) > 0.9 * BC.cap_mwh
