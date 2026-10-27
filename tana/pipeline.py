"""End-to-end pipeline: data -> forecasts -> reservoir rules -> solar/wind backup -> figures + tables."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import backup, battery as bat, plotting, reservoir
from .config import BatteryConfig, Config
from .data import fetch_nasa_power, fetch_nasa_power_daily, synthetic_daily, synthetic_monthly
from .forecast import MODEL_LABEL, MODEL_ORDER, rolling_forecasts, score_forecasts

SPIN_UP_MONTHS = 24   # ignore the first two years of the long simulation while storage settles


def load_data(cfg: Config, synthetic: bool = False, data_dir: str = "data") -> pd.DataFrame:
    if synthetic:
        return synthetic_monthly(cfg.start_year, cfg.end_year)
    return fetch_nasa_power(cfg.lat, cfg.lon, cfg.start_year, cfg.end_year, data_dir)


def analyse(cfg: Config, df: pd.DataFrame) -> dict:
    """All computations, no file output. Returns every intermediate so notebooks can inspect it."""
    rain = df["rain_mm"]
    # ---- 1. forecasting ----
    fcs = rolling_forecasts(rain, cfg.test_start, cfg.horizon)
    scores = score_forecasts(fcs, rain, cfg.test_start)

    # ---- 2. reservoir ----
    scale = reservoir.inflow_scale(rain, cfg)
    inflow = reservoir.actual_inflow(rain, cfg, scale)
    t0 = rain.index.get_loc(pd.Timestamp(cfg.test_start))
    test_index = rain.index[t0:]
    inflow_test = inflow.values[t0:]
    p = reservoir.make_params(cfg)
    look = reservoir.build_lookaheads(rain, fcs, cfg.test_start, cfg, scale)
    sims, table = reservoir.run_all_rules(inflow_test, p, look, cfg)
    sens = reservoir.capacity_sensitivity(inflow_test, look, cfg)

    # ---- 3. solar / wind backup (same grid cell) ----
    seasons = backup.classify_rainy_seasons(df, cfg)
    cond = backup.condition_stats(df, seasons)
    per_mw = backup.supply_per_mw(df, cfg)
    long_sim = reservoir.simulate(inflow.values, p, None)
    shortfall_gwh = pd.Series(long_sim["unmet"].values * cfg.gwh_per_mm3, index=df.index, name="shortfall_gwh")
    shortfall_gwh.iloc[:SPIN_UP_MONTHS] = 0.0
    cover = backup.gap_cover(shortfall_gwh, per_mw)
    req = backup.worst_month_requirement(shortfall_gwh, per_mw)
    compl = backup.complementarity(df, inflow)

    return dict(df=df, rain=rain, fcs=fcs, scores=scores, scale=scale, inflow=inflow, test_index=test_index,
                params=p, sims=sims, table=table, sens=sens, seasons=seasons, cond=cond, per_mw=per_mw,
                long_sim=long_sim, shortfall_gwh=shortfall_gwh, cover=cover, req=req, compl=compl,
                lookaheads=look)


def load_daily(cfg: Config, bc: BatteryConfig, synthetic: bool = False, data_dir: str = "data") -> pd.DataFrame:
    if synthetic:
        return synthetic_daily(bc.daily_start_year, cfg.end_year)
    return fetch_nasa_power_daily(cfg.lat, cfg.lon, bc.daily_start_year, cfg.end_year, data_dir)


def analyse_battery(cfg: Config, bc: BatteryConfig, daily: pd.DataFrame) -> dict:
    solar = daily["solar_kwh_m2_day"]
    fc = bat.daily_solar_forecasts(solar, cfg.test_start)
    fscores = bat.score_daily_forecasts(fc, cfg.test_start)
    runs, table = bat.run_all_policies(fc, cfg.test_start, bc)
    sens = bat.sensitivity(fc, cfg.test_start, bc)
    clear_day, cloudy_day = bat.example_days(fc, cfg.test_start)
    return dict(fc=fc, fscores=fscores, runs=runs, table=table, sens=sens, phi=fc.attrs["phi"],
                clear_day=clear_day, cloudy_day=cloudy_day, bc=bc)


def make_battery_figures(cfg: Config, B: dict, outdir: Path, synthetic: bool) -> list[Path]:
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    bc = B["bc"]
    dc = bat.day_detail("ar1", B["clear_day"], B["fc"], bc)
    dk = bat.day_detail("ar1", B["cloudy_day"], B["fc"], bc)
    return [
        plotting.save(plotting.fig_battery_days(dc, dk, bc, (f"Clear day {B['clear_day']:%d %b %Y}", f"Cloudy day {B['cloudy_day']:%d %b %Y}")),
                      fig_dir / "fig8_battery_days.png", synthetic),
        plotting.save(plotting.fig_battery_scorecard(B["table"], bc), fig_dir / "fig9_battery_policies.png", synthetic),
        plotting.save(plotting.fig_battery_months(B["runs"]), fig_dir / "fig10_battery_by_month.png", synthetic),
        plotting.save(plotting.fig_battery_sensitivity(B["sens"]), fig_dir / "fig11_battery_sensitivity.png", synthetic),
    ]


def make_figures(cfg: Config, R: dict, outdir: Path, synthetic: bool) -> list[Path]:
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    mean_monthly = cfg.mean_annual_inflow_mm3 / 12
    paths = [
        plotting.save(plotting.fig_climate(R["df"], R["seasons"], cfg.test_start), fig_dir / "fig1_climate.png", synthetic),
        plotting.save(plotting.fig_forecast(R["rain"], R["fcs"], R["scores"], cfg.test_start, cfg.horizon),
                      fig_dir / "fig2_forecast_skill.png", synthetic),
        plotting.save(plotting.fig_rules(R["sims"], R["params"], R["test_index"]), fig_dir / "fig3_reservoir_rules.png", synthetic),
        plotting.save(plotting.fig_scorecard(R["table"]), fig_dir / "fig4_scorecard.png", synthetic),
        plotting.save(plotting.fig_sensitivity(R["sens"], mean_monthly), fig_dir / "fig5_sensitivity.png", synthetic),
        plotting.save(plotting.fig_backup(R["cond"], R["seasons"], R["per_mw"]), fig_dir / "fig6_backup_conditions.png", synthetic),
        plotting.save(plotting.fig_gap(R["shortfall_gwh"], R["per_mw"], R["cover"], R["req"]), fig_dir / "fig7_gap_cover.png", synthetic),
    ]
    return paths


def _md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _battery_md(B: dict, cfg: Config) -> str:
    bc = B["bc"]
    t = B["table"].copy()
    t = t[["peak_hydro_mwh", "overnight_grid_mwh", "weighted_cost", "curtailed_pv_mwh", "saving_vs_best_static_pct"]]
    t.columns = ["Evening peak left to hydro (MWh)", "Overnight top-up (MWh)", "Weighted cost", "Curtailed PV (MWh)", "Saving vs best static (%)"]
    t.index.name = "Policy"
    f = B["fscores"].copy()
    f.index = ["Climatology", "Persistence", "AR(1)"]
    f.columns = ["RMSE (kWh/m2/day)", "MAE", "Bias", "Skill vs climatology"]
    f.index.name = "Day-ahead solar forecast"
    return f"""## 5. Battery dispatch (SYNTHETIC load, read the caveats)

A {bc.pv_mwp:.0f} MWp solar plant with a {bc.cap_mwh:.0f} MWh / {bc.power_mw:.1f} MW battery is reserved for the evening peak of a synthetic feeder
(mean {bc.load_mean_mw:.0f} MW). Each night the operator decides how much to top the battery up from hydro. Cost = {bc.peak_weight:g} x evening peak
energy hydro must supply + {bc.overnight_weight:g} x overnight top-up energy. Daily solar is real NASA POWER data; the load profile is invented and the
hourly PV shape is a clear-sky bell scaled to the real daily total.

{_md_table(f, "{:,.3f}")}

{_md_table(t, "{:,.1f}")}
"""


def write_results(cfg: Config, R: dict, outdir: Path, synthetic: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    tdir = outdir / "tables"
    tdir.mkdir(exist_ok=True)
    R["scores"].to_csv(tdir / "forecast_scores.csv", index=False)
    R["table"].to_csv(tdir / "reservoir_rules.csv")
    R["sens"].to_csv(tdir / "capacity_sensitivity.csv", index=False)
    R["cond"].to_csv(tdir / "solar_wind_conditions.csv")
    R["cover"].to_csv(tdir / "gap_cover.csv", index=False)
    R["req"].to_csv(tdir / "worst_month_requirement.csv")
    B = R.get("battery")
    if B is not None:
        B["table"].to_csv(tdir / "battery_policies.csv")
        B["fscores"].to_csv(tdir / "battery_daily_solar_forecasts.csv")
        B["sens"].to_csv(tdir / "battery_sensitivity.csv", index=False)

    sc = R["scores"]
    h1 = sc[sc.horizon == 1].set_index("model")
    tab = R["table"]
    spill_blind = tab.loc[reservoir.RULE_LABEL["blind"], "spill_mm3"]
    best_label = min([reservoir.RULE_LABEL["sarima"], reservoir.RULE_LABEL["gbm"]], key=lambda k: tab.loc[k, "spill_mm3"])
    cover = R["cover"]
    cov100 = float(cover.loc[cover.mw == 100, "mix"].iloc[0]) if cover["mix"].notna().any() else float("nan")
    cov200 = float(cover.loc[cover.mw == 200, "mix"].iloc[0]) if cover["mix"].notna().any() else float("nan")
    req = R["req"]

    summary = {
        "synthetic_test_data": synthetic,
        "config": asdict(cfg),
        "n_months": int(len(R["df"])),
        "test_months": int(len(R["test_index"])),
        "forecast_rmse_1mo_mm": {m: float(h1.loc[m, "rmse_mm"]) for m in MODEL_ORDER},
        "forecast_skill_vs_climatology_1mo": {m: float(h1.loc[m, "skill_vs_climatology"]) for m in MODEL_ORDER},
        "reservoir_rules": json.loads(tab.to_json(orient="index")),
        "spill_change_vs_no_forecast_pct": {
            k: float(100 * (tab.loc[k, "spill_mm3"] - spill_blind) / spill_blind) if spill_blind > 0 else None
            for k in tab.index},
        "best_forecast_rule_by_spill": best_label,
        "gap_cover_100mw_mix": cov100,
        "gap_cover_200mw_mix": cov200,
        "hydro_shortfall_months_long_run": int((R["shortfall_gwh"] > 0).sum()),
        "complementarity": R["compl"],
        "weak_rainy_season_years": [int(y) for y in R["seasons"].index[R["seasons"].weak]],
    }
    if B is not None:
        summary["battery"] = {
            "policies": json.loads(B["table"].to_json(orient="index")),
            "daily_solar_forecast_skill": json.loads(B["fscores"].to_json(orient="index")),
            "ar1_phi": B["phi"],
            "synthetic_load_and_hourly_pv_shape": True,
        }
    with open(outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    banner = ("> **SYNTHETIC TEST DATA. These numbers are not real. Delete this folder and run without `--synthetic`.**\n\n"
              if synthetic else "")
    skill_rows = sc.pivot(index="model", columns="horizon", values="rmse_mm").loc[MODEL_ORDER]
    skill_rows.index = [MODEL_LABEL[m] for m in skill_rows.index]
    skill_rows.columns = [f"{h}-month RMSE (mm)" for h in skill_rows.columns]
    skill_rows.index.name = "Model"
    rules = tab[["shortfall_months", "unmet_mm3", "spill_mm3", "spill_months", "generation_gwh"]].copy()
    rules.columns = ["Shortfall months", "Unmet (Mm3)", "Spill (Mm3)", "Spill months", "Energy (GWh)"]
    rules.index.name = "Release rule"
    cond = R["cond"].copy()
    cond.index.name = "Condition"
    md = f"""{banner}# Results

Location: {cfg.lat:.2f}N, {cfg.lon:.2f}E (NASA POWER grid cell). Record: {cfg.start_year}-{cfg.end_year}.
Training data before {cfg.test_start}; {len(R['test_index'])} held-out months.

## 1. Rainfall forecast error (held-out months)

{_md_table(skill_rows)}

Skill against climatology at 1 month: SARIMA {h1.loc['sarima','skill_vs_climatology']*100:+.1f}%, boosting {h1.loc['gbm','skill_vs_climatology']*100:+.1f}%, seasonal naive {h1.loc['seasonal_naive','skill_vs_climatology']*100:+.1f}%. Positive means better than the long-run monthly average.

## 2. Reservoir release rules (held-out months)

Capacity {R['params'].cap:,.0f} Mm3, firm demand {R['params'].demand:,.0f} Mm3/month, max release {R['params'].rmax:,.0f} Mm3/month.

{_md_table(rules, "{:,.1f}")}

## 3. Solar and wind under dry and weak-rain conditions

{_md_table(cond)}

## 4. Covering the hydropower gap

Hydropower shortfall months in the {cfg.start_year}-{cfg.end_year} simulation under no-forecast operation: {int((R['shortfall_gwh'] > 0).sum())}.
A 100 MW portfolio (50 MW solar, 50 MW wind) covers {cov100*100:.0f}% of the shortfall energy; 200 MW covers {cov200*100:.0f}%.
{'Worst single month needs about ' + format(req['mw_needed_mix'].max(), ',.0f') + ' MW of 50/50 solar and wind to cover it fully.' if len(req) else 'No shortfall months occurred, so there is no gap to cover under these settings.'}

{_battery_md(B, cfg) if B is not None else ""}
## Assumptions to state when you present this

Inflow is modelled as lagged rainfall, scaled to an assumed mean annual inflow. Reservoir size, demand and the plant are illustrative.
The point of the experiment is the comparison between release rules, not the absolute volumes.
"""
    (outdir / "results.md").write_text(md)


def run(cfg: Config | None = None, synthetic: bool = False, outdir: str = "results", data_dir: str = "data",
        battery: bool = True, bc: BatteryConfig | None = None):
    cfg = cfg or Config()
    bc = bc or BatteryConfig()
    out = Path(outdir + ("_synthetic" if synthetic else ""))
    df = load_data(cfg, synthetic, data_dir)
    R = analyse(cfg, df)
    figs = make_figures(cfg, R, out, synthetic)
    if battery:
        try:
            daily = load_daily(cfg, bc, synthetic, data_dir)
            R["battery"] = analyse_battery(cfg, bc, daily)
            figs += make_battery_figures(cfg, R["battery"], out, synthetic)
        except Exception as exc:  # noqa: BLE001
            print(f"[battery] skipped: {exc}\n[battery] the rest of the project is unaffected; rerun when the daily download works")
    write_results(cfg, R, out, synthetic)
    return R, out, figs
