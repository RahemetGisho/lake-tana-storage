"""Lake Tana storage lab: an interactive companion to the analysis.

Run:     streamlit run app/dashboard.py
Offline smoke test with fake data:  TANA_SYNTHETIC=1 streamlit run app/dashboard.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import ui, views  # noqa: E402
from tana import backup, battery as bat, reservoir  # noqa: E402
from tana.config import BatteryConfig, Config  # noqa: E402
from tana.dashboard_data import load_forecasts, load_inputs  # noqa: E402
from tana.forecast import score_forecasts  # noqa: E402

st.set_page_config(
    page_title="Lake Tana storage lab",
    page_icon="\U0001f4a7",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYNTHETIC = os.environ.get("TANA_SYNTHETIC") == "1"
DATA_DIR = ROOT / "data"

PAGES = {
    "Overview": views.overview,
    "Reservoir lab": views.reservoir_lab,
    "Rainfall forecasts": views.forecasts,
    "Solar & wind backup": views.solar_wind,
    "Battery lab": views.battery_lab,
    "Method & limits": views.method,
    "About me": views.about,
}


@st.cache_resource(show_spinner=False)
def build_context(synthetic: bool):
    cfg, bc = Config(), BatteryConfig()
    df, daily = load_inputs(cfg, bc, synthetic, DATA_DIR)
    fcs, built_now = load_forecasts(cfg, df, synthetic, DATA_DIR)
    rain = df["rain_mm"]
    scores = score_forecasts(fcs, rain, cfg.test_start)
    scale = reservoir.inflow_scale(rain, cfg)
    inflow = reservoir.actual_inflow(rain, cfg, scale)
    t0 = rain.index.get_loc(pd.Timestamp(cfg.test_start))
    look_full = reservoir.build_lookaheads(rain, fcs, cfg.test_start, cfg, scale)
    seasons = backup.classify_rainy_seasons(df, cfg)
    dfc = fscores = None
    if daily is not None:
        dfc = bat.daily_solar_forecasts(daily["solar_kwh_m2_day"], cfg.test_start)
        fscores = bat.score_daily_forecasts(dfc, cfg.test_start)
    return SimpleNamespace(
        cfg=cfg,
        bc=bc,
        df=df,
        daily=daily,
        fcs=fcs,
        built_now=built_now,
        scores=scores,
        scale=scale,
        inflow=inflow,
        t0=t0,
        test_index=rain.index[t0:],
        inflow_test=inflow.values[t0:],
        look_full=look_full,
        seasons=seasons,
        cond=backup.condition_stats(df, seasons),
        per_mw=backup.supply_per_mw(df, cfg),
        dfc=dfc,
        fscores=fscores,
    )


def main():
    ui.inject_css()
    with st.sidebar:
        st.markdown(
            '<div class="brand">Lake Tana<br>storage lab<small>Forecasts, reservoirs and solar &amp; wind backup</small></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        page = st.radio(
            "Navigate", list(PAGES), label_visibility="collapsed", key="page"
        )
        st.divider()
        st.caption("Built by **Rahemet Hussen Gisho**")
        st.caption("Data: NASA POWER \u00b7 Prototype, not a dam model")
        st.markdown(f"[Code on GitHub]({views.REPO_URL})")

    if SYNTHETIC:
        st.markdown(
            '<div class="synthetic">SYNTHETIC TEST DATA: nothing on this page is a real result.</div>',
            unsafe_allow_html=True,
        )
    first = "build_ctx_done" not in st.session_state
    with st.spinner(
        "Loading data and forecasts. The first run can take about a minute..."
        if first
        else "Loading..."
    ):
        ctx = build_context(SYNTHETIC)
    st.session_state["build_ctx_done"] = True
    if ctx.built_now and not SYNTHETIC and first:
        st.toast(
            "Forecast cache built. Commit the data/ folder to make deployment instant."
        )

    PAGES[page](ctx)
    ui.footer()


main()
