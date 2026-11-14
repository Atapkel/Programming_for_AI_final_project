"""Homepage for the Programming for AI final project: Kazakhstan economy explorer."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pages.analysis import show_analysis
from pages.predictions import show_predictions
from pages.visualization import show_visualization
from utils.data_utils import load_market_data
from utils.ui import (
    GOLD, NAVY, SKY, apply_theme, delta_html, footer, html, page_header, section, tiles,
)


st.set_page_config(
    page_title="Kazakhstan Economy | Programming for AI",
    page_icon="🇰🇿",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

BANKNOTE = Path(__file__).parent / "assets" / "tenge_100.jpg"
PAGES = {
    "Home": None,
    "Data Analysis": show_analysis,
    "Visualization": show_visualization,
    "Predictions": show_predictions,
}

with st.sidebar:
    st.image(BANKNOTE, width="stretch")
    html('<div class="kz-side-brand">Kazakhstan Economy Explorer</div>'
         '<div class="kz-side-sub">Programming for AI · Final project</div>'
         '<div class="kz-side-label">Menu</div>')
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")

if PAGES[page]:
    PAGES[page]()
    footer()
    st.stop()


def show_home():
    market = load_market_data()
    stamp = ""
    if market:
        stamp = f"Data as of {market['fx']['date'].iloc[-1]:%d.%m.%Y}"
    page_header(
        "Kazakhstan Economy Explorer",
        "Exchange rate, inflation, base rate and oil prices: historical data, "
        "interactive analysis and 12-month forecasts.",
        eyebrow="Macroeconomic dashboard",
        stamp=stamp,
    )
    if market is None:
        st.warning("Processed data not found. Run `python main.py` and choose **6**.")
        return

    fx, rates, brent, monthly = market["fx"], market["base_rate"], market["brent"], market["monthly"]
    last_fx, prev_fx = fx.iloc[-1], fx.iloc[-2]
    fx_change = last_fx["usd_kzt"] - prev_fx["usd_kzt"]
    last_rate = rates.iloc[-1]
    inflation = monthly.dropna(subset=["inflation_yoy"])
    last_infl, prev_infl = inflation.iloc[-1], inflation.iloc[-2]
    infl_change = last_infl["inflation_yoy"] - prev_infl["inflation_yoy"]
    last_oil, prev_oil = brent.iloc[-1], brent.iloc[-2]
    oil_change = last_oil["brent_usd"] - prev_oil["brent_usd"]

    tiles([
        {
            "label": "USD / KZT", "value": f"{last_fx['usd_kzt']:.2f}", "unit": "₸",
            # A falling USD/KZT means a stronger tenge, which is shown as good (green).
            "delta": delta_html(fx_change, f"{fx_change:+.2f} vs previous day", good_when_up=False),
            "note": f"Official rate, {last_fx['date']:%d.%m.%Y}",
        },
        {
            "label": "Base rate", "value": f"{last_rate['base_rate']:.2f}", "unit": "%",
            "delta": delta_html(last_rate["change_pp"], f"{last_rate['change_pp']:+.2f} p.p. ({last_rate['decision']})",
                                good_when_up=False),
            "note": f"Since {last_rate['date']:%d.%m.%Y}",
        },
        {
            "label": "Inflation, y/y", "value": f"{last_infl['inflation_yoy']:.1f}", "unit": "%",
            "delta": delta_html(infl_change, f"{infl_change:+.1f} p.p. vs previous month",
                                good_when_up=False),
            "note": f"{last_infl['date']:%B %Y}",
        },
        {
            "label": "Brent oil", "value": f"{last_oil['brent_usd']:.2f}", "unit": "$/bbl",
            "delta": delta_html(oil_change, f"{oil_change:+.2f} vs previous day"),
            "note": f"{last_oil['date']:%d.%m.%Y}",
        },
    ])

    left, right = st.columns(2)
    with left:
        recent = fx[fx["date"] >= fx["date"].iloc[-1] - pd.DateOffset(years=2)]
        figure = go.Figure(go.Scatter(
            x=recent["date"], y=recent["usd_kzt"], line=dict(color=NAVY, width=2),
            fill="tozeroy", fillcolor="rgba(11,60,125,.07)", name="USD/KZT",
        ))
        figure.update_layout(title="USD/KZT, last 2 years", height=320, showlegend=False)
        figure.update_yaxes(range=[recent["usd_kzt"].min() * 0.97, recent["usd_kzt"].max() * 1.02])
        st.plotly_chart(figure, width="stretch")
    with right:
        recent = monthly[monthly["date"] >= "2016-01-01"]
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=recent["date"], y=recent["base_rate"], name="Base rate",
                                    line=dict(color=NAVY, width=2.5, shape="hv")))
        figure.add_trace(go.Scatter(x=recent["date"], y=recent["inflation_yoy"],
                                    name="Inflation, y/y", line=dict(color=GOLD, width=2)))
        figure.update_layout(title="Base rate vs inflation, %", height=320, hovermode="x unified")
        st.plotly_chart(figure, width="stretch")

    section("Explore the project", "Choose a section in the menu on the left.")
    cards = [
        ("1", "Data Analysis", "Dataset structure, statistics, missing values and CSV downloads."),
        ("2", "Visualization", "Interactive charts for any indicator and date range."),
        ("3", "Predictions", "12-month forecasts from six models, tested on 36 months of data."),
    ]
    for column, (number, title, text) in zip(st.columns(3), cards):
        with column:
            html(f'<div class="kz-card"><div class="num">{number}</div>'
                 f'<h4>{title}</h4><p>{text}</p></div>')

    section("Data sources")
    sources = [
        ("Exchange & base rates", "Daily official USD/KZT rate and base rate decisions, National Bank of Kazakhstan."),
        ("Inflation & oil", "Monthly consumer price index (IMF) and daily Brent crude price (FRED)."),
        ("Annual indicators", "GDP, inflation, unemployment and exports, World Bank Open Data."),
    ]
    for column, (title, text) in zip(st.columns(3), sources):
        with column:
            html(f'<div class="kz-card" style="border-top:3px solid {SKY}"><h4>{title}</h4><p>{text}</p></div>')


show_home()
footer()
