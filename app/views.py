"""One function per dashboard page. Each takes the shared context built in dashboard.py."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tana import backup, battery as bat, reservoir
from tana.forecast import MODEL_LABEL, MODEL_ORDER
from tana.reservoir import RULE_LABEL, RULES

from . import ui

REPO_URL = "https://github.com/RahemetGisho/lake-tana-storage"
GITHUB = "https://github.com/RahemetGisho"
LINKEDIN = "https://www.linkedin.com/in/rahemet-gisho-15a081300/"
EMAIL = "kullefst@gmail.com"
CLIMATE_APP = "https://rahemetgisho-climate-challenge-week0-appmain-dashboard-d-2g8ijp.streamlit.app/"
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


# ----------------------------------------------------------------------------- helpers
def _pct(a: float, b: float) -> float:
    return 100 * (a - b) / b if b else float("nan")


def _sim_rules(ctx, cap_months, demand_frac, rmax_mult, horizon):
    cfg = replace(
        ctx.cfg, cap_months=cap_months, demand_frac=demand_frac, rmax_mult=rmax_mult
    )
    p = reservoir.make_params(cfg)
    look = {k: v[:, :horizon] for k, v in ctx.look_full.items()}
    sims, table = reservoir.run_all_rules(ctx.inflow_test, p, look, cfg)
    return cfg, p, look, sims, table


@st.cache_data(show_spinner=False)
def _long_shortfall(_ctx, cap_months: float, demand_frac: float) -> pd.Series:
    cfg = replace(_ctx.cfg, cap_months=cap_months, demand_frac=demand_frac)
    p = reservoir.make_params(cfg)
    sim = reservoir.simulate(_ctx.inflow.values, p, None)
    s = pd.Series(sim["unmet"].values * cfg.gwh_per_mm3, index=_ctx.df.index)
    s.iloc[:24] = 0.0
    return s


@st.cache_data(show_spinner=False)
def _battery_runs(_ctx, pv, cap, power, overnight_w, peak_w):
    bc = replace(
        _ctx.bc,
        pv_mwp=pv,
        cap_mwh=cap,
        power_mw=power,
        overnight_weight=overnight_w,
        peak_weight=peak_w,
    )
    runs, table = bat.run_all_policies(_ctx.dfc, _ctx.cfg.test_start, bc)
    return runs, table


@st.cache_data(show_spinner=False)
def _battery_pv_sweep(_ctx, cap, power, overnight_w, peak_w, sizes):
    rows = []
    for pv in sizes:
        bc = replace(
            _ctx.bc,
            pv_mwp=pv,
            cap_mwh=cap,
            power_mw=power,
            overnight_weight=overnight_w,
            peak_weight=peak_w,
        )
        costs = {
            p: bat.run_policy(p, _ctx.dfc, _ctx.cfg.test_start, bc)["cost"].sum()
            for p in ("solar_only", "always_full", "calendar", "clim", "ar1", "oracle")
        }
        best_static = min(costs["solar_only"], costs["always_full"], costs["calendar"])
        for p in ("clim", "ar1", "oracle"):
            rows.append(
                {
                    "pv": pv,
                    "policy": p,
                    "saving": 100 * (best_static - costs[p]) / best_static,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- overview
def overview(ctx):
    cfg = ctx.cfg
    ui.hero(
        "IRENA Youth Forum 2027 prototype",
        "When the rains arrive, what should a reservoir do?",
        "Ethiopia's reservoirs are its energy storage, and a rainfall forecast is a decision about when to release water. "
        "This lab tests that idea on real NASA data for the Lake Tana highlands, and asks how much of the gap solar, wind "
        "and a battery can fill.",
    )

    p = reservoir.make_params(cfg)
    sims, table = reservoir.run_all_rules(ctx.inflow_test, p, ctx.look_full, cfg)
    blind, sar = table.loc[RULE_LABEL["blind"]], table.loc[RULE_LABEL["sarima"]]
    h1 = ctx.scores[(ctx.scores.model == "sarima") & (ctx.scores.horizon == 1)].iloc[0]
    sf = _long_shortfall(ctx, cfg.cap_months, cfg.demand_frac)
    cover = (
        float(
            np.minimum(
                sf.values,
                100
                * 0.5
                * (ctx.per_mw.solar_gwh_per_mw + ctx.per_mw.wind_gwh_per_mw).values,
            ).sum()
            / sf.sum()
        )
        if sf.sum() > 0
        else float("nan")
    )
    ui.kpis(
        [
            (
                "Spilled water avoided",
                f"{-_pct(sar.spill_mm3, blind.spill_mm3):.0f}%",
                "looking 3 months ahead vs reacting only to current storage",
                ui.TEAL,
            ),
            (
                "Extra energy generated",
                f"{_pct(sar.generation_gwh, blind.generation_gwh):+.0f}%",
                "same water, released earlier through the turbines",
                ui.AMBER,
            ),
            (
                "Forecast skill vs seasonal average",
                f"{100 * h1.skill_vs_climatology:+.1f}%",
                "SARIMA, 1 month ahead: a monthly average is hard to beat",
                ui.NAVY,
            ),
            (
                "Shortfall covered by 100 MW",
                f"{100 * cover:.0f}%" if cover == cover else "n/a",
                "50 MW solar + 50 MW wind, simulated hydropower gap",
                ui.CORAL,
            ),
        ]
    )
    st.caption(
        f"Test years {ctx.test_index[0]:%Y}\u2013{ctx.test_index[-1]:%Y} ({len(ctx.test_index)} months) at "
        f"{cfg.lat:.2f}\u00b0N, {cfg.lon:.2f}\u00b0E. Defaults: reservoir of {cfg.cap_months:g} months of mean inflow, "
        f"firm demand {cfg.demand_frac:.0%} of mean inflow."
    )

    ui.section("How it works", "Predict, decide, test, then ask what fills the gap")
    ui.steps(
        [
            (
                "Predict",
                "Forecast monthly rainfall 1 to 3 months ahead with a seasonal naive rule, climatology, SARIMA and gradient boosting, tested on years the models never saw.",
            ),
            (
                "Decide",
                "A toy reservoir turns rainfall into inflow. Release rules that look ahead release extra water before a projected overflow instead of spilling it.",
            ),
            (
                "Test",
                "Count spilled water, unmet demand, months in shortfall and energy generated, then compare every rule with perfect foresight.",
            ),
            (
                "Back up",
                "Estimate how much of a hydropower shortfall solar and wind cover, and whether a day-ahead solar forecast improves a battery top-up.",
            ),
        ]
    )

    ui.section("What the data says", "Computed live from the data, not typed in")
    best_model = ctx.scores[ctx.scores.horizon == 1].sort_values("rmse_mm").iloc[0]
    clim_row = table.loc[RULE_LABEL["climatology"]]
    ui.callout(
        f"<b>Looking ahead works.</b> Releasing extra water ahead of projected overflows cut spill from "
        f"{blind.spill_mm3:,.0f} to {sar.spill_mm3:,.0f} Mm\u00b3 and raised generation from {blind.generation_gwh:,.0f} to "
        f"{sar.generation_gwh:,.0f} GWh over the test years.",
        "good",
    )
    ui.callout(
        f"<b>The forecast models add little.</b> Plain climatology (the monthly average) achieved "
        f"{clim_row.spill_mm3:,.0f} Mm\u00b3 of spill, against {sar.spill_mm3:,.0f} for SARIMA. At monthly resolution the value "
        f"comes from anticipating the rainy season, not from model skill. Best 1-month model: {MODEL_LABEL[best_model.model]}.",
        "warn",
    )
    ui.callout(
        f"<b>It has a cost.</b> The look-ahead rules had {int(sar.shortfall_months - blind.shortfall_months):+d} month(s) of shortfall "
        f"and {_pct(sar.unmet_mm3, blind.unmet_mm3):+.0f}% unmet demand versus no forecast, because early releases leave less water "
        "when a weak season follows.",
        "bad",
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for r in ("blind", "sarima", "oracle"):
            fig.add_trace(
                go.Scatter(
                    x=ctx.test_index,
                    y=sims[r].storage_end / p.cap * 100,
                    name=RULE_LABEL[r],
                    line=dict(
                        color=ui.RULE_COLOR[r],
                        width=2.4 if r != "oracle" else 1.6,
                        dash="dot" if r == "oracle" else "solid",
                    ),
                )
            )
        fig.update_yaxes(title="Storage (% of capacity)", range=[0, 105])
        st.plotly_chart(
            ui.style_fig(fig, 340).update_layout(
                title="Reservoir storage under three rules"
            ),
            width="stretch",
        )
    with c2:
        labels = [RULE_LABEL[r] for r in RULES]
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=[table.loc[l, "spill_mm3"] for l in labels],
                marker_color=[ui.RULE_COLOR[r] for r in RULES],
                text=[f"{table.loc[l, 'spill_mm3']:,.0f}" for l in labels],
                textposition="outside",
            )
        )
        fig.update_yaxes(title="Spilled water (Mm\u00b3)")
        st.plotly_chart(
            ui.style_fig(fig, 340, "closest").update_layout(
                title="Spill by release rule", showlegend=False
            ),
            width="stretch",
        )
    st.info(
        "Open **Reservoir lab** in the sidebar to change the reservoir size, demand and look-ahead and watch the result move."
    )


# ----------------------------------------------------------------------------- reservoir lab
def reservoir_lab(ctx):
    cfg = ctx.cfg
    ui.section(
        "Reservoir lab",
        "Change the reservoir and the operating rule. Everything recalculates instantly.",
    )
    c1, c2, c3, c4 = st.columns(4)
    cap = c1.slider(
        "Capacity (months of mean inflow)", 3.0, 12.0, float(cfg.cap_months), 0.5
    )
    dem = (
        c2.slider(
            "Firm demand (% of mean inflow)",
            50,
            95,
            int(round(cfg.demand_frac * 100)),
            1,
        )
        / 100
    )
    rmx = c3.slider(
        "Max turbine release (x demand)", 1.1, 2.0, float(cfg.rmax_mult), 0.05
    )
    hor = c4.slider("Look-ahead (months)", 1, cfg.horizon, cfg.horizon)
    rules = st.multiselect(
        "Rules to show",
        RULES,
        default=["blind", "sarima", "oracle"],
        format_func=lambda r: RULE_LABEL[r],
    )
    if "blind" not in rules:
        rules = ["blind"] + rules

    cfg2, p, look, sims, table = _sim_rules(ctx, cap, dem, rmx, hor)
    focus = next(
        (r for r in ("sarima", "gbm", "climatology", "oracle") if r in rules), "sarima"
    )
    focus = st.selectbox(
        "Compare against \u2018No forecast\u2019",
        [r for r in RULES if r != "blind"],
        index=[r for r in RULES if r != "blind"].index(focus),
        format_func=lambda r: RULE_LABEL[r],
    )
    b, f = table.loc[RULE_LABEL["blind"]], table.loc[RULE_LABEL[focus]]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Spill (Mm\u00b3)",
        f"{f.spill_mm3:,.0f}",
        f"{_pct(f.spill_mm3, b.spill_mm3):+.0f}% vs no forecast",
        delta_color="inverse",
    )
    m2.metric(
        "Unmet demand (Mm\u00b3)",
        f"{f.unmet_mm3:,.0f}",
        f"{f.unmet_mm3 - b.unmet_mm3:+,.0f} vs no forecast",
        delta_color="inverse",
    )
    m3.metric(
        "Months in shortfall",
        f"{int(f.shortfall_months)}",
        f"{int(f.shortfall_months - b.shortfall_months):+d} vs no forecast",
        delta_color="inverse",
    )
    m4.metric(
        "Energy (GWh)",
        f"{f.generation_gwh:,.0f}",
        f"{_pct(f.generation_gwh, b.generation_gwh):+.1f}% vs no forecast",
    )

    fig = go.Figure()
    for r in rules:
        fig.add_trace(
            go.Scatter(
                x=ctx.test_index,
                y=sims[r].storage_end / p.cap * 100,
                name=RULE_LABEL[r],
                line=dict(
                    color=ui.RULE_COLOR[r], width=2.6 if r in ("blind", focus) else 1.5
                ),
            )
        )
    fig.add_hline(
        y=p.dead / p.cap * 100,
        line=dict(color="black", width=1, dash="dot"),
        annotation_text="dead storage",
    )
    fig.update_yaxes(title="Storage (% of capacity)", range=[0, 105])
    st.plotly_chart(
        ui.style_fig(fig, 380).update_layout(title="Storage through the test years"),
        width="stretch",
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(
            x=ctx.test_index,
            y=sims["blind"].spill,
            name="No forecast",
            marker_color=ui.CORAL,
        )
        fig.add_bar(
            x=ctx.test_index,
            y=sims[focus].spill,
            name=RULE_LABEL[focus],
            marker_color=ui.RULE_COLOR[focus],
        )
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="Mm\u00b3 / month")
        st.plotly_chart(
            ui.style_fig(fig, 320).update_layout(title="Water lost over the spillway"),
            width="stretch",
        )
    with c2:
        fig = go.Figure()
        fig.add_bar(
            x=ctx.test_index,
            y=sims["blind"].unmet,
            name="No forecast",
            marker_color=ui.CORAL,
        )
        fig.add_bar(
            x=ctx.test_index,
            y=sims[focus].unmet,
            name=RULE_LABEL[focus],
            marker_color=ui.RULE_COLOR[focus],
        )
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="Mm\u00b3 / month")
        st.plotly_chart(
            ui.style_fig(fig, 320).update_layout(
                title="Demand the reservoir could not meet"
            ),
            width="stretch",
        )

    show = table[
        ["shortfall_months", "unmet_mm3", "spill_mm3", "spill_months", "generation_gwh"]
    ].copy()
    show.columns = [
        "Shortfall months",
        "Unmet (Mm\u00b3)",
        "Spill (Mm\u00b3)",
        "Spill months",
        "Energy (GWh)",
    ]
    st.dataframe(show.style.format("{:,.1f}"), width="stretch")
    st.caption(
        f"Capacity {p.cap:,.0f} Mm\u00b3 \u00b7 firm demand {p.demand:,.0f} Mm\u00b3/month \u00b7 max release {p.rmax:,.0f} Mm\u00b3/month. "
        "Mass balance is verified by an automated test."
    )

    with st.expander("How does reservoir size change the answer?"):
        sens = _sens(ctx, dem, rmx, hor)
        c1, c2 = st.columns(2)
        for col, key, title in (
            (c1, "spill_mm3", "Spill vs reservoir size"),
            (c2, "unmet_mm3", "Unmet demand vs reservoir size"),
        ):
            fig = go.Figure()
            for r in rules:
                d = sens[sens.rule == r]
                fig.add_trace(
                    go.Scatter(
                        x=d.cap_months,
                        y=d[key],
                        mode="lines+markers",
                        name=RULE_LABEL[r],
                        line=dict(color=ui.RULE_COLOR[r], width=2.2),
                    )
                )
            fig.update_xaxes(title="Capacity (months of mean inflow)")
            fig.update_yaxes(title="Mm\u00b3 over test years")
            col.plotly_chart(
                ui.style_fig(fig, 320).update_layout(title=title), width="stretch"
            )


@st.cache_data(show_spinner=False)
def _sens(_ctx, dem, rmx, hor):
    cfg = replace(_ctx.cfg, demand_frac=dem, rmax_mult=rmx)
    look = {k: v[:, :hor] for k, v in _ctx.look_full.items()}
    return reservoir.capacity_sensitivity(_ctx.inflow_test, look, cfg)


# ----------------------------------------------------------------------------- forecasts
def forecasts(ctx):
    cfg = ctx.cfg
    ui.section(
        "Rainfall forecasts",
        "Every forecast is made using only data available at the time. Models are refitted once a year.",
    )
    c1, c2 = st.columns([1, 2])
    h = c1.radio(
        "Forecast horizon",
        list(range(1, cfg.horizon + 1)),
        format_func=lambda x: f"{x} month{'s' if x > 1 else ''} ahead",
        horizontal=True,
    )
    models = c2.multiselect(
        "Models",
        MODEL_ORDER,
        default=["climatology", "sarima"],
        format_func=lambda m: MODEL_LABEL[m],
    )

    y = ctx.df["rain_mm"]
    n = len(y) - (ctx.t0 + h - 1)
    idx = y.index[ctx.t0 + h - 1 : ctx.t0 + h - 1 + n]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=y.values[ctx.t0 + h - 1 :],
            name="Observed",
            line=dict(color="black", width=2.2),
        )
    )
    for m in models:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=ctx.fcs[m][:n, h - 1],
                name=MODEL_LABEL[m],
                line=dict(color=ui.MODEL_COLOR[m], width=1.8),
            )
        )
    fig.update_yaxes(title="Rainfall (mm / month)")
    st.plotly_chart(
        ui.style_fig(fig, 400).update_layout(
            title=f"{h}-month-ahead forecasts on held-out years"
        ),
        width="stretch",
    )

    sc = ctx.scores[ctx.scores.horizon == h].set_index("model").loc[MODEL_ORDER]
    c1, c2 = st.columns(2)
    for col, key, title in (
        (c1, "rmse_mm", "Error, all months (RMSE)"),
        (c2, "rmse_wet_mm", "Error, rainy season Jun-Sep (RMSE)"),
    ):
        fig = go.Figure(
            go.Bar(
                x=[MODEL_LABEL[m] for m in MODEL_ORDER],
                y=sc[key].values,
                marker_color=[ui.MODEL_COLOR[m] for m in MODEL_ORDER],
                text=[f"{v:.1f}" for v in sc[key].values],
                textposition="outside",
            )
        )
        fig.update_yaxes(title="mm / month")
        col.plotly_chart(
            ui.style_fig(fig, 320, "closest").update_layout(
                title=title, showlegend=False
            ),
            width="stretch",
        )

    tab = sc[
        ["rmse_mm", "mae_mm", "rmse_wet_mm", "bias_mm", "skill_vs_climatology"]
    ].copy()
    tab["skill_vs_climatology"] *= 100
    tab.index = [MODEL_LABEL[m] for m in tab.index]
    tab.columns = [
        "RMSE (mm)",
        "MAE (mm)",
        "RMSE rainy season (mm)",
        "Bias (mm)",
        "Skill vs climatology (%)",
    ]
    st.dataframe(tab.style.format("{:,.2f}"), width="stretch")
    s_sar = float(sc.loc["sarima", "skill_vs_climatology"] * 100)
    s_gbm = float(sc.loc["gbm", "skill_vs_climatology"] * 100)
    ui.callout(
        f"<b>Read the baseline first.</b> Monthly rainfall is strongly seasonal, so the long-run average for each calendar month is a hard "
        f"baseline. At {h} month(s), SARIMA's skill against it is {s_sar:+.1f}% and gradient boosting's is {s_gbm:+.1f}%. A small or "
        "negative gain is a valid finding, and it is why the reservoir result is credited to anticipating the season, not to model skill.",
        "warn",
    )


# ----------------------------------------------------------------------------- backup
def solar_wind(ctx):
    cfg = ctx.cfg
    df = ctx.df
    ui.section(
        "Solar and wind backup",
        "How sun and wind behave when the reservoir is under pressure, and how much of the gap they can fill.",
    )
    st_ = backup.seasonal_table(df)
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Rainfall (mm / month)",
            "Solar (kWh / m\u00b2 / day)",
            "Wind at 50 m (m / s)",
        ),
    )
    for i, (col, c) in enumerate(
        (
            ("rain_mm", ui.TEAL),
            ("solar_kwh_m2_day", ui.AMBER),
            ("wind_ms_50m", ui.NAVY),
        ),
        1,
    ):
        m = st_[("mean", col)]
        fig.add_trace(
            go.Scatter(
                x=MONTHS,
                y=st_[("p90", col)].values,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=i,
        )
        fig.add_trace(
            go.Scatter(
                x=MONTHS,
                y=st_[("p10", col)].values,
                fill="tonexty",
                line=dict(width=0),
                fillcolor="rgba(150,160,170,.25)",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=i,
        )
        fig.add_trace(
            go.Scatter(
                x=MONTHS, y=m.values, line=dict(color=c, width=3), showlegend=False
            ),
            row=1,
            col=i,
        )
    st.plotly_chart(
        ui.style_fig(fig, 340).update_layout(
            title="Seasonal cycle (line = mean, band = 10th to 90th percentile)"
        ),
        width="stretch",
    )

    cond = ctx.cond.copy()
    dry, weak, other = cond.iloc[0], cond.iloc[1], cond.iloc[2]
    ui.callout(
        f"<b>Solar</b> averages {dry.solar_kwh_m2_day:.2f} kWh/m\u00b2/day in the dry season against {other.solar_kwh_m2_day:.2f} in a normal rainy "
        f"season. <b>Wind</b> barely changes ({dry.wind_ms_50m:.2f} vs {other.wind_ms_50m:.2f} m/s). In weak rainy seasons, solar was "
        f"{weak.solar_kwh_m2_day:.2f} and wind {weak.wind_ms_50m:.2f}: <b>not noticeably sunnier or windier</b> than other years. Solar is a steady "
        "complement, not a rescue that appears when the rains fail.",
        "warn",
    )
    c = cond.rename(
        columns={
            "solar_kwh_m2_day": "Solar (kWh/m\u00b2/day)",
            "wind_ms_50m": "Wind (m/s)",
            "rain_mm_month": "Rain (mm/month)",
            "months": "Months",
        }
    )
    st.dataframe(c.style.format("{:,.2f}"), width="stretch")

    ui.section(
        "Filling the hydropower gap",
        "Pick a reservoir scenario, then an installed portfolio.",
    )
    c1, c2, c3, c4 = st.columns(4)
    cap = c1.slider(
        "Reservoir (months of inflow)",
        3.0,
        12.0,
        float(cfg.cap_months),
        0.5,
        key="bk_cap",
    )
    dem = (
        c2.slider(
            "Firm demand (% of mean inflow)",
            50,
            95,
            int(round(cfg.demand_frac * 100)),
            1,
            key="bk_dem",
        )
        / 100
    )
    mw = c3.slider("Installed solar + wind (MW)", 0, 400, 100, 10)
    share = c4.slider("Solar share of the mix (%)", 0, 100, 50, 5) / 100
    sf = _long_shortfall(ctx, cap, dem)
    if sf.sum() <= 0:
        st.success(
            "No hydropower shortfall under this scenario, so there is no gap to fill. Try a smaller reservoir or a higher demand."
        )
        return

    per = ctx.per_mw
    supply = mw * (share * per.solar_gwh_per_mw + (1 - share) * per.wind_gwh_per_mw)
    covered = np.minimum(sf, supply)
    mwv = np.arange(0, 405, 5)
    cover_curve = [
        100
        * np.minimum(
            sf.values,
            m
            * (
                share * per.solar_gwh_per_mw.values
                + (1 - share) * per.wind_gwh_per_mw.values
            ),
        ).sum()
        / sf.sum()
        for m in mwv
    ]
    unit = (share * per.solar_gwh_per_mw + (1 - share) * per.wind_gwh_per_mw)[sf > 0]
    need = sf[sf > 0] / unit
    k1, k2, k3 = st.columns(3)
    k1.metric("Shortfall energy covered", f"{100 * covered.sum() / sf.sum():.0f}%")
    k2.metric(
        "Shortfall months",
        f"{int((sf > 0).sum())}",
        f"{int(((sf - supply) <= 1e-9)[sf > 0].sum())} fully covered",
        delta_color="off",
    )
    k3.metric(
        "Worst month needs",
        f"{need.max():,.0f} MW",
        "to cover it completely",
        delta_color="off",
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(
            go.Scatter(x=mwv, y=cover_curve, line=dict(color=ui.TEAL, width=3.2))
        )
        fig.add_vline(x=mw, line=dict(color=ui.AMBER, dash="dash"))
        fig.update_xaxes(title="Installed capacity (MW)")
        fig.update_yaxes(title="Shortfall covered (%)", range=[0, 102])
        st.plotly_chart(
            ui.style_fig(fig, 340, "closest").update_layout(
                title="Coverage vs installed capacity", showlegend=False
            ),
            width="stretch",
        )
    with c2:
        s = sf[sf > 0]
        fig = go.Figure()
        fig.add_bar(
            x=s.index, y=s.values, name="Hydropower shortfall", marker_color=ui.CORAL
        )
        fig.add_bar(
            x=s.index,
            y=covered[s.index].values,
            name="Covered by solar + wind",
            marker_color=ui.TEAL,
        )
        fig.update_layout(barmode="overlay")
        fig.update_yaxes(title="GWh / month")
        st.plotly_chart(
            ui.style_fig(fig, 340).update_layout(
                title="Shortfall months and what the portfolio delivers"
            ),
            width="stretch",
        )
    comp = backup.complementarity(df, ctx.inflow)
    st.caption(
        f"Deseasonalised correlation of rainfall with solar: {comp['corr_rain_solar']:+.2f}; with wind: {comp['corr_rain_wind']:+.2f}. "
        "Energy is compared month by month and ignores hour-by-hour firmness, so this is a screening estimate."
    )


# ----------------------------------------------------------------------------- battery
def battery_lab(ctx):
    bc0 = ctx.bc
    ui.section(
        "Battery lab",
        "Does a day-ahead solar forecast improve how much a battery is topped up overnight for the evening peak?",
    )
    ui.callout(
        "<b>Read this first.</b> The load profile is <b>synthetic</b>, and hourly solar is a clear-sky bell scaled to the real daily NASA total. "
        "Daily solar and its forecast skill are real. This tests decision logic, not plant performance.",
        "warn",
    )
    if ctx.dfc is None:
        st.error(
            "Daily solar data could not be loaded, so this page is unavailable. Run `python run_all.py` once with internet access, then reload."
        )
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    pv = c1.slider("Solar PV (MWp)", 10, 100, int(bc0.pv_mwp), 5)
    cap = c2.slider("Battery energy (MWh)", 10, 150, int(bc0.cap_mwh), 5)
    pw = c3.slider("Battery power (MW)", 2.5, 40.0, float(bc0.power_mw), 2.5)
    ow = c4.slider("Overnight energy cost", 0.0, 2.0, float(bc0.overnight_weight), 0.25)
    pk = c5.slider("Evening peak cost", 0.5, 4.0, float(bc0.peak_weight), 0.5)
    with st.spinner("Simulating ten years of days..."):
        runs, table = _battery_runs(ctx, pv, cap, pw, ow, pk)

    best_static = min(
        table.loc[bat.POLICY_LABEL[p], "weighted_cost"]
        for p in ("solar_only", "always_full", "calendar")
    )
    ar1 = table.loc[bat.POLICY_LABEL["ar1"]]
    orc = table.loc[bat.POLICY_LABEL["oracle"]]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best static rule cost", f"{best_static / 1e3:,.0f}k")
    k2.metric(
        "AR(1) forecast rule",
        f"{ar1.weighted_cost / 1e3:,.0f}k",
        f"{ar1.saving_vs_best_static_pct:+.1f}% vs best static",
        delta_color="normal",
    )
    k3.metric(
        "Perfect foresight",
        f"{orc.weighted_cost / 1e3:,.0f}k",
        f"{orc.saving_vs_best_static_pct:+.1f}% vs best static",
        delta_color="normal",
    )
    k4.metric("Evening peak covered (AR1)", f"{ar1.peak_covered_pct:.0f}%")

    labels = [bat.POLICY_LABEL[p] for p in bat.POLICIES]
    peak = [pk * table.loc[l, "peak_hydro_mwh"] / 1e3 for l in labels]
    night = [ow * table.loc[l, "overnight_grid_mwh"] / 1e3 for l in labels]
    fig = go.Figure()
    fig.add_bar(
        y=labels[::-1],
        x=peak[::-1],
        orientation="h",
        name="Evening peak hydro must supply",
        marker_color=ui.CORAL,
    )
    fig.add_bar(
        y=labels[::-1],
        x=night[::-1],
        orientation="h",
        name="Overnight top-up from hydro",
        marker_color=ui.TEAL,
    )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(
        title="Weighted cost over the test years (thousands, lower is better)"
    )
    st.plotly_chart(
        ui.style_fig(fig, 400, "closest").update_layout(
            title="What each top-up policy costs"
        ),
        width="stretch",
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for p in ("solar_only", "always_full", "calendar", "ar1", "oracle"):
            m = runs[p]["cost"].groupby(runs[p].index.month).mean()
            fig.add_trace(
                go.Scatter(
                    x=MONTHS,
                    y=m.values,
                    name=bat.POLICY_LABEL[p],
                    mode="lines+markers",
                    line=dict(color=ui.POLICY_COLOR[p], width=2.4),
                )
            )
        fig.update_yaxes(title="Mean daily cost")
        st.plotly_chart(
            ui.style_fig(fig, 340).update_layout(
                title="When does a solar forecast matter?"
            ),
            width="stretch",
        )
    with c2:
        fs = ctx.fscores.copy()
        fs.index = ["Climatology", "Persistence", "AR(1)"]
        fs.columns = ["RMSE", "MAE", "Bias", "Skill vs climatology"]
        st.markdown("**Day-ahead solar forecast error** (kWh/m\u00b2/day)")
        st.dataframe(fs.style.format("{:,.3f}"), width="stretch")
        ui.callout(
            f"The AR(1) forecast has real skill ({100 * fs.loc['AR(1)', 'Skill vs climatology']:+.0f}% vs climatology) but the "
            f"decision gain is {ar1.saving_vs_best_static_pct:+.1f}%. A skilled forecast helps only when it changes the decision."
        )

    st.markdown("##### Where forecasts pay off: solar size sweep")
    if st.button("Run the PV size sweep (about 15 seconds)"):
        with st.spinner("Running five PV sizes..."):
            sw = _battery_pv_sweep(ctx, cap, pw, ow, pk, (20, 30, 40, 60, 80))
        fig = go.Figure()
        for p in ("clim", "ar1", "oracle"):
            d = sw[sw.policy == p]
            fig.add_trace(
                go.Scatter(
                    x=d.pv,
                    y=d.saving,
                    mode="lines+markers",
                    name=bat.POLICY_LABEL[p],
                    line=dict(color=ui.POLICY_COLOR[p], width=2.6),
                )
            )
        fig.add_hline(y=0, line=dict(color="black", width=1))
        fig.update_xaxes(title="Solar PV (MWp)")
        fig.update_yaxes(title="Saving vs best static policy (%)")
        st.plotly_chart(
            ui.style_fig(fig, 340).update_layout(title="Forecast value vs solar size"),
            width="stretch",
        )
        st.caption(
            "Forecasts help when solar is too small to fill the battery on most days, and the gain vanishes once the sun fills it anyway."
        )

    st.markdown("##### Look at one day")
    mode = st.radio(
        "Day",
        ["Clearest July-August day", "Cloudiest July-August day", "Pick a date"],
        horizontal=True,
    )
    clear, cloudy = bat.example_days(ctx.dfc, ctx.cfg.test_start)
    test_days = ctx.dfc.index[ctx.dfc.index >= pd.Timestamp(ctx.cfg.test_start)]
    if mode.startswith("Clearest"):
        day = clear
    elif mode.startswith("Cloudiest"):
        day = cloudy
    else:
        day = pd.Timestamp(
            st.date_input(
                "Date",
                value=cloudy.date(),
                min_value=test_days[0].date(),
                max_value=test_days[-1].date(),
            )
        )
    pol = st.selectbox(
        "Policy",
        ["solar_only", "always_full", "clim", "persistence", "ar1", "oracle"],
        index=4,
        format_func=lambda p: bat.POLICY_LABEL[p],
    )
    bc = replace(
        bc0, pv_mwp=pv, cap_mwh=cap, power_mw=pw, overnight_weight=ow, peak_weight=pk
    )
    d = bat.day_detail(pol, day, ctx.dfc, bc)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    hrs = list(range(24))
    fig.add_bar(
        x=hrs, y=d["pv"], name="Solar PV (MW)", marker_color=ui.AMBER, opacity=0.8
    )
    fig.add_bar(
        x=hrs,
        y=d["discharge_path"],
        name="Battery discharge (MW)",
        marker_color=ui.TEAL,
    )
    fig.add_trace(
        go.Scatter(
            x=hrs,
            y=d["load"],
            name="Load (MW, synthetic)",
            line=dict(color="black", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hrs,
            y=d["soc_path"],
            name="Battery charge (MWh)",
            line=dict(color=ui.NAVY, width=2, dash="dash"),
        ),
        secondary_y=True,
    )
    fig.add_vrect(
        x0=min(bc.critical_hours) - 0.5,
        x1=max(bc.critical_hours) + 0.5,
        fillcolor=ui.CORAL,
        opacity=0.08,
        line_width=0,
    )
    fig.update_xaxes(title="Hour of day")
    fig.update_yaxes(title="MW", secondary_y=False)
    fig.update_yaxes(
        title="MWh", secondary_y=True, range=[0, bc.cap_mwh * 1.1], showgrid=False
    )
    st.plotly_chart(
        ui.style_fig(fig, 380).update_layout(
            title=f"{day:%d %B %Y}: {d['solar']:.1f} kWh/m\u00b2/day (forecast {d['forecast']:.1f})"
        ),
        width="stretch",
    )
    st.caption(
        f"Overnight top-up {d['top_up']:.0f} MWh \u00b7 evening peak left to hydro {d['crit_unmet']:.0f} MWh \u00b7 solar curtailed {d['curtailed']:.0f} MWh. The shaded band is the critical evening window. The battery starts the day empty here."
    )


# ----------------------------------------------------------------------------- method
def method(ctx):
    cfg, bc = ctx.cfg, ctx.bc
    ui.section("Method and limits", "What was assumed, so you can judge what to trust.")
    st.markdown("##### How the pieces work")
    ui.callout(
        "<b>Inflow</b> in month <i>t</i> is a scaled sum of rainfall now and in the two previous months (weights "
        f"{cfg.runoff_weights}), calibrated to an assumed mean annual inflow of {cfg.mean_annual_inflow_mm3:,.0f} Mm\u00b3. "
        "<b>Forecasts</b> are rolling-origin: each month uses only earlier data, and SARIMA and boosting are refitted once a year. "
        "<b>Look-ahead rules</b> release extra water through the turbines when the reservoir is projected to overflow within the horizon, "
        "never cutting supply below the firm demand."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Reservoir and plant assumptions")
        st.dataframe(
            pd.DataFrame(
                {
                    "Setting": [
                        "Location",
                        "Record",
                        "Test period starts",
                        "Look-ahead horizon",
                        "Reservoir capacity",
                        "Dead storage",
                        "Firm demand",
                        "Max turbine release",
                        "Energy per Mm\u00b3",
                        "PV performance ratio",
                        "Wind hub height",
                    ],
                    "Value": [
                        f"{cfg.lat:.2f}\u00b0N, {cfg.lon:.2f}\u00b0E",
                        f"{cfg.start_year}\u2013{cfg.end_year}",
                        cfg.test_start,
                        f"{cfg.horizon} months",
                        f"{cfg.cap_months:g} months of mean inflow",
                        f"{cfg.dead_frac:.0%} of capacity",
                        f"{cfg.demand_frac:.0%} of mean inflow",
                        f"{cfg.rmax_mult:g} x demand",
                        f"{cfg.gwh_per_mm3} GWh (about 100 m head, 90% efficiency)",
                        f"{cfg.pv_perf_ratio}",
                        f"{cfg.hub_height_m:g} m",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
    with c2:
        st.markdown("##### Battery assumptions (synthetic)")
        st.dataframe(
            pd.DataFrame(
                {
                    "Setting": [
                        "Solar PV",
                        "Battery",
                        "Battery power",
                        "Round-trip efficiency",
                        "Mean load",
                        "Critical hours",
                        "Cost of evening peak",
                        "Cost of overnight energy",
                    ],
                    "Value": [
                        f"{bc.pv_mwp:g} MWp",
                        f"{bc.cap_mwh:g} MWh",
                        f"{bc.power_mw:g} MW",
                        f"{bc.eta_charge * bc.eta_discharge:.0%}",
                        f"{bc.load_mean_mw:g} MW (invented profile)",
                        f"{min(bc.critical_hours)}:00\u2013{max(bc.critical_hours) + 1}:00",
                        f"{bc.peak_weight:g} per MWh",
                        f"{bc.overnight_weight:g} per MWh",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
    st.markdown("##### Limits to state out loud")
    ui.callout(
        "&bull; One 0.5\u00b0 NASA POWER grid cell: a satellite and reanalysis product, not a gauge record, and it does not resolve the lake.<br>"
        "&bull; Inflow is a lagged-rainfall rule, not a hydrological model. Reservoir size, demand and plant are illustrative.<br>"
        f"&bull; About {len(ctx.test_index) // 12} test years: differences between rules are described, not statistically tested.<br>"
        "&bull; No grid data, transmission limits or electricity prices were used. Wind uses a Weibull approximation of monthly means.<br>"
        "&bull; The battery load is synthetic and intra-day clouds are not modelled.",
        "warn",
    )
    st.markdown("##### Reproduce it")
    st.code(
        "git clone "
        + REPO_URL
        + ".git\ncd lake-tana-storage\npip install -r requirements.txt\npython run_all.py\nstreamlit run app/dashboard.py",
        language="bash",
    )
    st.caption(
        "Data: NASA Prediction Of Worldwide Energy Resources (POWER). Please cite NASA POWER when you use these results."
    )


# ----------------------------------------------------------------------------- about
def about(ctx):
    ui.hero(
        "About me",
        "Rahemet Hussen Gisho",
        "Software engineering student in Adama, Ethiopia, building forecasting and data systems on African climate, financial and energy data.",
    )
    left, right = st.columns([1, 2.4])
    with left:
        from pathlib import Path

        photo = Path(__file__).resolve().parents[1] / "assets" / "profile.jpg"
        if photo.exists():
            st.image(str(photo), width=170)
        else:
            st.markdown('<div class="avatar">RG</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="linkrow"><a href="{GITHUB}" target="_blank">GitHub: RahemetGisho</a>'
            f'<a href="{LINKEDIN}" target="_blank">LinkedIn</a>'
            f'<a href="mailto:{EMAIL}">{EMAIL}</a>'
            f'<a href="{REPO_URL}" target="_blank">This project\'s code</a>'
            f'<a href="{CLIMATE_APP}" target="_blank">African climate dashboard</a></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Languages**")
        ui.chips(["English", "Amharic", "Arabic"], teal=True)
    with right:
        st.markdown("##### Why I built this")
        st.write(
            "Most of Ethiopia's electricity comes from water, so a forecast of rain is also a decision about when to release it. "
            "I wanted to test that idea honestly: build the simulation, compare operating rules against perfect foresight, and report "
            "what did not work as clearly as what did. The most useful result was a humbling one: a simple seasonal average "
            "did as well as the forecasting models, so the value came from anticipating the rainy season."
        )
        st.markdown("##### What I want from the IRENA Youth Forum")
        st.write(
            "I have not worked with real grid or operating data yet. I want to learn how other countries plan storage and flexibility, "
            "what data operators actually have, and where someone with forecasting skills is useful. Afterwards I will extend this work "
            "with real Ethiopian data and share what I learn with Ethiopian data and energy communities."
        )
        st.markdown("##### What I work with")
        ui.chips(
            [
                "Forecasting: ARIMA/SARIMA, LSTM",
                "Bayesian change-point detection (PyMC)",
                "Optimisation",
                "Reservoir simulation",
                "Python",
                "pandas",
                "scikit-learn",
                "XGBoost",
                "dbt",
                "Dagster",
                "PostgreSQL",
                "Streamlit",
                "Flask",
                "React",
                "Node.js",
                "TypeScript",
                "pytest",
                "GitHub Actions",
            ]
        )

    st.markdown("##### Education and experience")
    c1, c2 = st.columns(2)
    with c1:
        ui.card(
            "B.Sc. Software Engineering",
            "Adama Science and Technology University \u00b7 2023 \u2013 present",
            "Data structures and algorithms, databases, probability and statistics, software testing, artificial intelligence.",
        )
        ui.card(
            "Kifiya AI Mastery Program (KAIM10), 10 Academy",
            "04/2026 \u2013 07/2026",
            "Weekly challenge-based training in machine learning, generative AI and data engineering.",
        )
        ui.card(
            "STEM Tutor, ASTU STEM Center",
            "Volunteering",
            "Tutoring students in STEM subjects.",
        )
    with c2:
        ui.card(
            "Software Engineering Intern, Adama City Administration",
            "ASTO Department",
            "Built \u201cWe Donate\u201d, a full-stack civic donation platform (React, Node.js, TypeScript, Prisma, PostgreSQL) with five user roles, "
            "payment gateway integration and end-to-end tests, plus a citizen service assistant over a 130-service catalog.",
        )
        ui.card(
            "Full Stack Developer Intern, ASTU ICT Center",
            "",
            "Built an Employee Performance Management System on the MERN stack with weighted scoring and automated calculation of points and averages.",
        )

    st.markdown("##### Other projects")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.card(
            "African Climate Trend Analysis",
            "10 Academy \u00b7 NASA POWER",
            f"Five-country comparison with ANOVA and a vulnerability ranking for Ethiopia's COP32 positioning. <a href='{CLIMATE_APP}' target='_blank'>Live dashboard</a>.",
        )
    with c2:
        ui.card(
            "Portfolio Optimisation and Forecasting",
            "GMF Investments",
            "LSTM cut Tesla forecast error from 17.7% to 3.7% MAPE; mean-variance allocation backtested against a 60/40 benchmark.",
        )
    with c3:
        ui.card(
            "Brent Oil Change-Point Analysis",
            "Bayesian, PyMC",
            "Located a mean shift in February 2005 and a volatility shift in August 2008; served through a Flask API and React dashboard.",
        )
