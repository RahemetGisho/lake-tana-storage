"""Look and feel: palette, CSS and small HTML components shared by every page."""

from __future__ import annotations

import streamlit as st

NAVY, TEAL, AMBER, CORAL = "#16324f", "#1f8a8a", "#e0a030", "#d1495b"
SAND, GREY, GREEN, TAN = "#efe7d6", "#8a94a0", "#6a994e", "#b9a37a"

RULE_COLOR = {
    "blind": CORAL,
    "climatology": TAN,
    "sarima": TEAL,
    "gbm": NAVY,
    "oracle": GREEN,
}
MODEL_COLOR = {"seasonal_naive": GREY, "climatology": TAN, "sarima": TEAL, "gbm": NAVY}
POLICY_COLOR = {
    "solar_only": GREY,
    "always_full": CORAL,
    "calendar": TAN,
    "clim": "#c9a96b",
    "persistence": "#7f9bb8",
    "ar1": NAVY,
    "oracle": GREEN,
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton, label { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }
#MainMenu, footer { visibility: hidden; }

/* hero */
.hero { background: linear-gradient(120deg, #16324f 0%, #1f6f86 55%, #1f8a8a 100%); color: #fff;
        border-radius: 20px; padding: 2.2rem 2.4rem; margin-bottom: 1.4rem; position: relative; overflow: hidden; }
.hero::after { content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(224,160,48,.35), rgba(224,160,48,0) 70%); }
.hero .kicker { text-transform: uppercase; letter-spacing: .14em; font-size: .75rem; font-weight: 600; opacity: .8; }
.hero h1 { font-size: 2.15rem; line-height: 1.15; margin: .35rem 0 .6rem; font-weight: 800; color: #fff; }
.hero p { font-size: 1.02rem; opacity: .92; max-width: 760px; margin: 0; }

/* section headings */
.sec { margin: 1.6rem 0 .8rem; }
.sec h2 { font-size: 1.45rem; font-weight: 700; margin: 0; color: #16324f; }
.sec p { color: #5b6675; margin: .25rem 0 0; font-size: .95rem; }

/* kpi cards */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: .4rem 0 1.2rem; }
.kpi { background: #fff; border: 1px solid #e6e9ee; border-radius: 16px; padding: 1.05rem 1.2rem;
       box-shadow: 0 1px 2px rgba(22,50,79,.05); border-top: 4px solid var(--accent, #1f8a8a); }
.kpi .label { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: #6b7686; font-weight: 600; }
.kpi .value { font-size: 2rem; font-weight: 800; color: #16324f; margin: .15rem 0; line-height: 1.1; }
.kpi .note { font-size: .84rem; color: #5b6675; }

/* callouts */
.callout { border-radius: 14px; padding: .95rem 1.15rem; margin: .8rem 0; font-size: .95rem; line-height: 1.5;
           border: 1px solid #e6e9ee; background: #f6f3ec; color: #16324f; }
.callout.good { background: #eaf6f4; border-color: #cbe8e3; }
.callout.warn { background: #fdf3e1; border-color: #f2dba8; }
.callout.bad { background: #fceaec; border-color: #f2c4ca; }
.callout b { font-weight: 700; }

/* step cards */
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin: .6rem 0; }
.step { background: #fff; border: 1px solid #e6e9ee; border-radius: 16px; padding: 1.1rem 1.2rem; }
.step .n { width: 30px; height: 30px; border-radius: 50%; background: #1f8a8a; color: #fff; font-weight: 700;
           display: inline-flex; align-items: center; justify-content: center; margin-bottom: .5rem; }
.step h4 { margin: 0 0 .25rem; font-size: 1.05rem; color: #16324f; }
.step p { margin: 0; font-size: .92rem; color: #5b6675; line-height: 1.45; }

/* chips */
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: .4rem 0 1rem; }
.chip { background: #eef3f7; color: #16324f; border-radius: 999px; padding: .28rem .8rem; font-size: .85rem; font-weight: 500; }
.chip.teal { background: #e3f3f1; color: #14605f; }

/* timeline / cards */
.card { background: #fff; border: 1px solid #e6e9ee; border-radius: 16px; padding: 1.05rem 1.25rem; margin-bottom: 12px; }
.card .t { font-weight: 700; color: #16324f; }
.card .s { color: #6b7686; font-size: .86rem; margin-bottom: .3rem; }
.card p { margin: .2rem 0 0; color: #3b4756; font-size: .93rem; line-height: 1.5; }
.avatar { width: 128px; height: 128px; border-radius: 50%; background: linear-gradient(135deg, #16324f, #1f8a8a);
          color: #fff; font-size: 2.6rem; font-weight: 800; display: flex; align-items: center; justify-content: center;
          margin-bottom: .8rem; box-shadow: 0 6px 18px rgba(22,50,79,.25); }
.linkrow a { display: block; margin: .25rem 0; color: #1f6f86; text-decoration: none; font-weight: 500; }
.linkrow a:hover { text-decoration: underline; }

/* sidebar */
section[data-testid="stSidebar"] { background: #f6f3ec; border-right: 1px solid #e6dfcf; }
.brand { font-weight: 800; font-size: 1.15rem; color: #16324f; line-height: 1.2; }
.brand small { display: block; font-weight: 500; color: #6b7686; font-size: .78rem; margin-top: .2rem; }
.foot { color: #8a94a0; font-size: .8rem; text-align: center; margin-top: 2.2rem; padding-top: 1rem; border-top: 1px solid #e6e9ee; }
.synthetic { background: #fceaec; color: #8a1f2d; border: 1px solid #f2c4ca; padding: .6rem 1rem; border-radius: 12px;
             font-weight: 600; margin-bottom: 1rem; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid #e6e9ee; border-radius: 14px; padding: .8rem 1rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="kicker">{kicker}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="sec"><h2>{title}</h2>{sub}</div>', unsafe_allow_html=True)


def kpis(items: list[tuple[str, str, str, str]]) -> None:
    """items: (label, value, note, accent colour)."""
    html = "".join(
        f'<div class="kpi" style="--accent:{c}"><div class="label">{l}</div><div class="value">{v}</div>'
        f'<div class="note">{n}</div></div>'
        for l, v, n, c in items
    )
    st.markdown(f'<div class="kpis">{html}</div>', unsafe_allow_html=True)


def callout(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def steps(items: list[tuple[str, str]]) -> None:
    html = "".join(
        f'<div class="step"><div class="n">{i}</div><h4>{h}</h4><p>{p}</p></div>'
        for i, (h, p) in enumerate(items, 1)
    )
    st.markdown(f'<div class="steps">{html}</div>', unsafe_allow_html=True)


def chips(items: list[str], teal: bool = False) -> None:
    cls = "chip teal" if teal else "chip"
    st.markdown(
        '<div class="chips">'
        + "".join(f'<span class="{cls}">{i}</span>' for i in items)
        + "</div>",
        unsafe_allow_html=True,
    )


def card(title: str, sub: str, body: str) -> None:
    st.markdown(
        f'<div class="card"><div class="t">{title}</div><div class="s">{sub}</div><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        '<div class="foot">Built by Rahemet Hussen Gisho &middot; NASA POWER data &middot; '
        "A decision-testing prototype, not a dam model</div>",
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 380, hover: str = "x unified"):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=8, t=48, b=8),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode=hover,
        title=dict(x=0, font=dict(size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor="#c9ced6")
    fig.update_yaxes(gridcolor="#eceff3", zeroline=False)
    return fig
