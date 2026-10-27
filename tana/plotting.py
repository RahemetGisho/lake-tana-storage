"""Figures. One consistent style; every figure is watermarked when run on synthetic test data."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .forecast import MODEL_LABEL  # noqa: E402
from .reservoir import RULE_LABEL  # noqa: E402

NAVY, TEAL, AMBER, CORAL, SAND, GREY = "#16324f", "#1f8a8a", "#e0a030", "#d1495b", "#efe7d6", "#8a94a0"
COL = {
    "seasonal_naive": GREY, "climatology": "#b9a37a", "sarima": TEAL, "gbm": NAVY,
    "blind": CORAL, "oracle": "#6a994e",
}
MONTHS = list("JFMAMJJASOND")


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#c9ced6",
        "axes.grid": True, "grid.color": "#eceff3", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": "#2b3440", "xtick.color": "#2b3440", "ytick.color": "#2b3440",
        "font.size": 10, "legend.frameon": False, "figure.dpi": 110,
    })


_style()


def save(fig, path: Path, synthetic: bool):
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC TEST DATA\nNOT REAL RESULTS", ha="center", va="center", rotation=25,
                 fontsize=34, color="red", alpha=0.13, weight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_climate(df, seasons, test_start):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    m = np.arange(1, 13)
    g = df.groupby(df.index.month)
    for ax, col, title, unit, c in [
        (axes[0], "rain_mm", "Rainfall", "mm / month", TEAL),
        (axes[1], "solar_kwh_m2_day", "Solar irradiance", "kWh / m\u00b2 / day", AMBER),
        (axes[2], "wind_ms_50m", "Wind speed at 50 m", "m / s", NAVY),
    ]:
        ax.fill_between(m, g[col].quantile(0.1), g[col].quantile(0.9), color=c, alpha=0.18, lw=0)
        ax.plot(m, g[col].mean(), color=c, lw=2.4)
        ax.set_xticks(m, MONTHS)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.axvspan(5.5, 9.5, color=SAND, alpha=0.45, lw=0, zorder=0)
    axes[0].text(7.5, axes[0].get_ylim()[1] * 0.93, "main rains", ha="center", fontsize=9, color="#6b5f45")
    fig.suptitle("Rainfall is sharply seasonal; sun and wind vary far less through the year", x=0.01,
                 ha="left", fontsize=13, weight="bold", y=1.04)
    fig.tight_layout()
    return fig


def fig_forecast(y, fcs, scores, test_start, horizon):
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.42, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    t0 = y.index.get_loc(pd.Timestamp(test_start))
    idx = y.index[t0:]
    ax.plot(idx, y.values[t0:], color="black", lw=1.6, label="Observed", zorder=5)
    for m in ("climatology", "sarima", "gbm"):
        ax.plot(idx, fcs[m][:, 0][: len(idx)], color=COL[m], lw=1.4, alpha=0.95, label=MODEL_LABEL[m] + " (1 month ahead)")
    ax.set_title("One-month-ahead rainfall forecasts on the held-out years")
    ax.set_ylabel("mm / month")
    ax.legend(ncol=4, loc="upper left", fontsize=9)

    order = ["seasonal_naive", "climatology", "sarima", "gbm"]
    ax2 = fig.add_subplot(gs[1, 0])
    w = 0.2
    hs = np.arange(1, horizon + 1)
    for j, m in enumerate(order):
        vals = [scores[(scores.model == m) & (scores.horizon == h)]["rmse_mm"].iloc[0] for h in hs]
        ax2.bar(hs + (j - 1.5) * w, vals, w, color=COL[m], label=MODEL_LABEL[m])
    ax2.set_xticks(hs, [f"{h} mo" for h in hs])
    ax2.set_title("Error by horizon (RMSE, all months)")
    ax2.set_ylabel("mm / month")

    ax3 = fig.add_subplot(gs[1, 1])
    for j, m in enumerate(order):
        vals = [scores[(scores.model == m) & (scores.horizon == h)]["rmse_wet_mm"].iloc[0] for h in hs]
        ax3.bar(hs + (j - 1.5) * w, vals, w, color=COL[m], label=MODEL_LABEL[m])
    ax3.set_xticks(hs, [f"{h} mo" for h in hs])
    ax3.set_title("Error in the rainy season (Jun-Sep)")
    for a in (ax2, ax3):
        a.margins(y=0.12)
    h, l = ax3.get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.03), fontsize=9.5)
    return fig


def fig_rules(sims, p, test_index):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.4), gridspec_kw={"height_ratios": [1.5, 1]}, sharex=True)
    ax = axes[0]
    for k in ("blind", "climatology", "sarima", "gbm", "oracle"):
        ax.plot(test_index, sims[k]["storage_end"].values / p.cap * 100, color=COL.get(k, GREY),
                lw=2.2 if k in ("blind", "sarima") else 1.3, alpha=1 if k in ("blind", "sarima") else 0.8,
                label=RULE_LABEL[k])
    ax.axhline(p.dead / p.cap * 100, color="black", lw=0.8, ls=":")
    ax.text(test_index[-1], p.dead / p.cap * 100 - 5.5, "dead storage", fontsize=8.5, ha="right")
    ax.set_ylabel("Storage (% of capacity)")
    ax.set_ylim(0, 108)
    ax.set_title("Reservoir storage under each release rule")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=9)
    ax = axes[1]
    wd = 12
    x = mdates_num(test_index)
    ax.bar(x - wd / 2, sims["blind"]["spill"].values, wd, color=CORAL, label="Spill, no forecast")
    ax.bar(x + wd / 2, sims["sarima"]["spill"].values, wd, color=TEAL, label="Spill, SARIMA rule")
    ax.set_ylabel("Spilled water (Mm\u00b3 / month)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Water lost over the spillway")
    ax.xaxis_date()
    fig.tight_layout()
    return fig


def mdates_num(idx):
    import matplotlib.dates as mdates
    return mdates.date2num(idx.to_pydatetime())


def fig_scorecard(table):
    labels = list(table.index)
    cols = [COL.get(k, GREY) for k in ("blind", "climatology", "sarima", "gbm", "oracle")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    spec = [("spill_mm3", "Spilled water (Mm\u00b3)"), ("unmet_mm3", "Unmet demand (Mm\u00b3)"),
            ("shortfall_months", "Months in shortfall"), ("generation_gwh", "Energy generated (GWh)")]
    for ax, (key, title) in zip(axes, spec):
        v = table[key].values.astype(float)
        bars = ax.bar(range(len(v)), v, color=cols)
        ax.set_xticks(range(len(v)), [l.replace(" ", "\n", 1) for l in labels], fontsize=7.5)
        ax.set_title(title, fontsize=11)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{val:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def fig_sensitivity(sens, mean_monthly):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for rule in ("blind", "climatology", "sarima", "gbm", "oracle"):
        d = sens[sens.rule == rule]
        axes[0].plot(d.cap_months, d.spill_mm3, marker="o", color=COL[rule], lw=2, label=RULE_LABEL[rule])
        axes[1].plot(d.cap_months, d.unmet_mm3, marker="o", color=COL[rule], lw=2, label=RULE_LABEL[rule])
    axes[0].set_title("Spill vs reservoir size")
    axes[1].set_title("Unmet demand vs reservoir size")
    for a in axes:
        a.set_xlabel("Capacity (months of mean inflow)")
    axes[0].set_ylabel("Mm\u00b3 over test period")
    axes[0].legend(fontsize=8.5)
    fig.tight_layout()
    return fig


def fig_backup(cond, seasons, per_mw):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    labels = [l.replace(", ", "\n").replace(" (", "\n(") for l in cond.index]
    for ax, col, title, unit, c in [
        (axes[0], "solar_kwh_m2_day", "Solar irradiance", "kWh / m\u00b2 / day", AMBER),
        (axes[1], "wind_ms_50m", "Wind speed at 50 m", "m / s", NAVY),
    ]:
        v = cond[col].values.astype(float)
        bars = ax.bar(range(3), v, color=[c, CORAL, c], alpha=0.95)
        bars[2].set_alpha(0.55)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(3), labels, fontsize=8.5)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.margins(y=0.15)
    ax = axes[2]
    m = np.arange(1, 13)
    g = per_mw.groupby(per_mw.index.month).mean()
    ax.plot(m, g["solar_gwh_per_mw"], color=AMBER, lw=2.4, label="Solar")
    ax.plot(m, g["wind_gwh_per_mw"], color=NAVY, lw=2.4, label="Wind")
    ax.set_xticks(m, MONTHS)
    ax.set_title("Energy per installed MW")
    ax.set_ylabel("GWh / MW / month")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_gap(shortfall, per_mw, cover, req, cfg_note=""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0))
    ax = axes[0]
    ax.plot(cover.mw, cover.solar * 100, color=AMBER, lw=2.2, label="Solar only")
    ax.plot(cover.mw, cover.wind * 100, color=NAVY, lw=2.2, label="Wind only")
    ax.plot(cover.mw, cover["mix"] * 100, color=TEAL, lw=3, label="50/50 mix")
    ax.set_xlabel("Installed capacity (MW, total)")
    ax.set_ylabel("Hydropower shortfall covered (%)")
    ax.set_ylim(0, 102)
    ax.set_title("How much of the hydropower gap solar and wind can fill")
    ax.legend(loc="lower right")
    ax = axes[1]
    s = shortfall[shortfall > 0]
    if len(s):
        ax.bar(s.index, s.values, width=25, color=CORAL, label="Hydro shortfall")
        mix = 100 * 0.5 * (per_mw["solar_gwh_per_mw"] + per_mw["wind_gwh_per_mw"]).reindex(s.index)
        ax.bar(s.index, np.minimum(s.values, mix.values), width=25, color=TEAL, label="Covered by 100 MW (50 solar + 50 wind)")
        ax.legend(loc="upper left", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No shortfall months under these settings", ha="center", va="center", transform=ax.transAxes)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("GWh / month")
    ax.set_title("Shortfall months and what a 100 MW portfolio delivers")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- battery figures
def fig_battery_days(clear_day, cloudy_day, bc, labels):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    h = np.arange(24)
    for ax, d, label in zip(axes, (clear_day, cloudy_day), labels):
        ax.bar(h, d["pv"], width=0.85, color=AMBER, alpha=0.75, label="Solar PV")
        ax.plot(h, d["load"], color="black", lw=2, label="Load (synthetic)")
        ax.bar(h, d["discharge_path"], width=0.5, color=TEAL, label="Battery discharge")
        ax.axvspan(min(bc.critical_hours) - 0.5, max(bc.critical_hours) + 0.5, color=CORAL, alpha=0.10, lw=0)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Hour of day")
        ax.set_title(f"{label}: {d['solar']:.1f} kWh/m\u00b2/day")
        ax2 = ax.twinx()
        ax2.plot(h, d["soc_path"], color=NAVY, lw=1.8, ls="--", label="Battery charge (MWh)")
        ax.set_ylim(0, max(d["pv"].max(), d["load"].max(), 20) * 1.45)
        ax2.set_ylim(0, bc.cap_mwh * 1.45)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        ax.text(0.02, 0.97, f"day-ahead forecast {d['forecast']:.1f} vs actual {d['solar']:.1f} kWh/m\u00b2/day\n"
                f"overnight top-up {d['top_up']:.0f} MWh | evening peak left to hydro {d['crit_unmet']:.0f} MWh",
                transform=ax.transAxes, fontsize=9, va="top")
    axes[0].set_ylabel("MW")
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h1 + [plt.Line2D([0], [0], color=NAVY, ls="--")], l1 + ["Battery charge (MWh, right axis)"],
               ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return fig


def fig_battery_scorecard(table, bc):
    from .battery import POLICIES, POLICY_LABEL
    labels = [POLICY_LABEL[p] for p in POLICIES]
    peak = np.array([bc.peak_weight * table.loc[l, "peak_hydro_mwh"] for l in labels]) / 1e3
    night = np.array([bc.overnight_weight * table.loc[l, "overnight_grid_mwh"] for l in labels]) / 1e3
    fig, ax = plt.subplots(figsize=(11, 4.6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, peak, color=CORAL, label="Evening peak hydro must supply (weighted)")
    ax.barh(y, night, left=peak, color=TEAL, label="Overnight top-up drawn from hydro (weighted)")
    for yi, p, n, l in zip(y, peak, night, labels):
        sv = table.loc[l, "saving_vs_best_static_pct"]
        ax.text(p + n + 3, yi, f"{p + n:,.0f}k  ({sv:+.1f}% vs best static)", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Weighted cost over the test years (thousand units, lower is better)")
    ax.set_title("What each top-up policy costs")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=9)
    ax.margins(x=0.25)
    fig.tight_layout()
    return fig


def fig_battery_months(runs):
    from .battery import POLICY_LABEL
    fig, ax = plt.subplots(figsize=(11, 4.2))
    style = {"solar_only": (GREY, 2), "always_full": (CORAL, 2), "calendar": ("#b9a37a", 2),
             "ar1": (NAVY, 2.6), "oracle": ("#6a994e", 2)}
    for p, (c, lw) in style.items():
        m = runs[p]["cost"].groupby(runs[p].index.month).mean()
        ax.plot(m.index, m.values, color=c, lw=lw, marker="o", ms=4, label=POLICY_LABEL[p])
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_ylabel("Mean daily weighted cost")
    ax.set_title("When does a solar forecast matter? Cost by calendar month")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    return fig


def fig_battery_sensitivity(sens):
    from .battery import POLICY_LABEL
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    for ax, kind, xl in ((axes[0], "pv_mwp", "Solar PV capacity (MWp)"),
                         (axes[1], "overnight_weight", "Cost of overnight energy (relative to evening peak = 2)")):
        d = sens[sens.param == kind]
        for pol, c in (("clim", "#b9a37a"), ("ar1", NAVY), ("oracle", "#6a994e")):
            x = d[d.policy == pol]
            ax.plot(x.value, x.saving_vs_best_static_pct, marker="o", color=c, lw=2, label=POLICY_LABEL[pol])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel(xl)
    axes[0].set_ylabel("Saving vs best static policy (%)")
    axes[0].set_title("Value of forecasts vs PV size")
    axes[1].set_title("Value of forecasts vs overnight energy cost")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    return fig
