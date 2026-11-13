"""Streamlit dashboard: Kazakhstan macroeconomic indicators and exchange rate.

Run:
    streamlit run app.py
"""

import json
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis import BREAKS
from src.database import DB_PATH, QUERIES
from src.preprocessing import PROCESSED_DIR, REGIMES

SERIES = {
    "usd_kzt_avg": ("USD/KZT, monthly average", "KZT per USD"),
    "brent_avg": ("Brent oil price", "USD per barrel"),
    "inflation_yoy": ("Inflation, % y/y", "%"),
    "inflation_mom": ("Inflation, % m/m", "%"),
    "base_rate": ("NBK base rate", "%"),
    "real_rate": ("Real rate (base rate − inflation)", "p.p."),
    "usd_kzt_change_pct": ("USD/KZT change, % m/m", "%"),
}
TARGET_LABELS = {"inflation_mom": "Inflation, % m/m", "usd_kzt": "USD/KZT, monthly average"}
REGIME_LABELS = {label: label.replace("_", " ") for _, label in REGIMES}

st.set_page_config(page_title="Kazakhstan Macro Dashboard", page_icon="📈", layout="wide")


@st.cache_data
def load_data():
    """Load every processed table the dashboard needs."""
    data = {
        "monthly": pd.read_parquet(PROCESSED_DIR / "monthly_macro.parquet"),
        "daily_fx": pd.read_parquet(PROCESSED_DIR / "usd_kzt_daily.parquet"),
        "base_rate": pd.read_parquet(PROCESSED_DIR / "base_rate.parquet"),
        "forecasts": pd.read_parquet(PROCESSED_DIR / "forecasts.parquet"),
        "backtest": pd.read_parquet(PROCESSED_DIR / "backtest_predictions.parquet"),
        "metrics": pd.read_csv(PROCESSED_DIR / "model_metrics.csv"),
        "errors_by_year": pd.read_csv(PROCESSED_DIR / "errors_by_year.csv"),
    }
    with open(PROCESSED_DIR / "forecast_summary.json", encoding="utf-8") as f:
        data["summary"] = json.load(f)
    return data


@st.cache_data
def run_sql(sql):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(sql, conn)


def add_break_lines(fig, start, end):
    for date, label in BREAKS.items():
        ts = pd.Timestamp(date)
        if start <= ts <= end:
            fig.add_vline(x=ts.timestamp() * 1000, line_dash="dash", line_color="red", opacity=0.5,
                          annotation_text=label, annotation_position="top left")


def kpi_row(monthly, daily_fx, base_rate):
    """Latest values with change vs 12 months earlier."""
    last_fx = daily_fx.iloc[-1]
    fx_year_ago = daily_fx.loc[daily_fx["date"] <= last_fx["date"] - pd.DateOffset(years=1), "usd_kzt"].iloc[-1]
    infl = monthly.dropna(subset=["inflation_yoy"]).iloc[-1]
    infl_year_ago = monthly.loc[monthly["date"] == infl["date"] - pd.DateOffset(years=1), "inflation_yoy"].iloc[0]
    last_rate = base_rate.iloc[-1]

    cols = st.columns(4)
    cols[0].metric(f"USD/KZT ({last_fx['date']:%d.%m.%Y})", f"{last_fx['usd_kzt']:.2f}",
                   f"{(last_fx['usd_kzt'] / fx_year_ago - 1) * 100:+.1f}% y/y", delta_color="inverse")
    cols[1].metric(f"Inflation y/y ({infl['date']:%m.%Y})", f"{infl['inflation_yoy']:.1f}%",
                   f"{infl['inflation_yoy'] - infl_year_ago:+.1f} p.p. y/y", delta_color="inverse")
    cols[2].metric(f"Base rate (since {last_rate['date']:%d.%m.%Y})", f"{last_rate['base_rate']:.2f}%",
                   f"{last_rate['change_pp']:+.2f} p.p. ({last_rate['decision']})", delta_color="off")
    cols[3].metric("Real rate", f"{infl['real_rate']:.1f} p.p.",
                   help="Base rate minus y/y inflation. Positive = tight monetary policy.")


def tab_time_series(monthly, selected, start, end):
    st.subheader("Indicators over time")
    normalize = st.toggle("Index to 100 at the start of the period", value=False,
                          help="Compare series with different units on one scale")
    fig = go.Figure()
    for col in selected:
        s = monthly.set_index("date")[col].dropna()
        if normalize and not s.empty and s.iloc[0] > 0:
            s = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=SERIES[col][0],
                                 line_shape="hv" if col == "base_rate" else "linear"))
    add_break_lines(fig, start, end)
    fig.update_layout(height=520, hovermode="x unified", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig)

    st.subheader("Correlation of monthly changes in the selected period")
    changes = pd.DataFrame({
        "USD/KZT change": monthly["usd_kzt_change_pct"],
        "Brent change": monthly["brent_change_pct"],
        "Inflation m/m": monthly["inflation_mom"],
        "Base rate change": monthly["base_rate"].diff(),
    }).corr().round(2)
    heat = go.Figure(go.Heatmap(z=changes.values, x=changes.columns, y=changes.index, zmin=-1, zmax=1,
                                colorscale="RdBu_r", text=changes.values, texttemplate="%{text}"))
    heat.update_layout(height=420)
    st.plotly_chart(heat)
    st.caption("Levels of trending series correlate spuriously, so the matrix uses monthly changes.")


def tab_exchange_rate(daily_fx, base_rate, start, end):
    st.subheader("Daily official USD/KZT rate")
    fx = daily_fx[(daily_fx["date"] >= start) & (daily_fx["date"] <= end)]
    fig = go.Figure(go.Scatter(x=fx["date"], y=fx["usd_kzt"], name="USD/KZT", line=dict(width=1)))
    filled = fx[fx["is_filled"]]
    if not filled.empty:
        fig.add_trace(go.Scatter(x=filled["date"], y=filled["usd_kzt"], mode="markers", name="interpolated",
                                 marker=dict(size=3, color="orange")))
    add_break_lines(fig, start, end)
    fig.update_layout(height=450, hovermode="x unified")
    st.plotly_chart(fig)

    left, right = st.columns(2)
    with left:
        st.markdown("**Largest daily moves (SQL)**")
        moves = run_sql(QUERIES["03_largest_daily_moves"][1])
        st.dataframe(moves, hide_index=True)
    with right:
        st.markdown("**Base rate decisions in the period**")
        decisions = base_rate[(base_rate["date"] >= start) & (base_rate["date"] <= end)
                              & (base_rate["decision"].isin(["hike", "cut", "first"]))]
        st.dataframe(decisions.assign(date=decisions["date"].dt.date), hide_index=True)


def tab_regimes():
    st.subheader("Exchange-rate regimes compared (SQL)")
    st.dataframe(run_sql(QUERIES["08_regime_comparison"][1]), hide_index=True)
    st.markdown(
        "- **managed rate** (2000 – Jul 2015): stable tenge with one-off devaluations in 2009 and 2014\n"
        "- **free float** (Aug 2015 – Jan 2022): tenge floated, inflation targeting introduced\n"
        "- **post 2022 shock**: sanctions on Russia, base rate jump, inflation peak of 21% in Feb 2023")
    st.subheader("Years ranked by tenge depreciation (SQL)")
    st.dataframe(run_sql(QUERIES["10_depreciation_ranking"][1]), hide_index=True)


def tab_forecast(data, monthly):
    target = st.radio("Indicator", list(TARGET_LABELS), format_func=TARGET_LABELS.get, horizontal=True)
    metrics = data["metrics"][data["metrics"]["target"] == target]
    all_models = metrics["model"].tolist()
    best = data["summary"][target]["best_model"]
    models = st.multiselect("Models", all_models, default=[m for m in all_models if m in (best, "naive")][:3])

    column = "inflation_mom" if target == "inflation_mom" else "usd_kzt_avg"
    history = monthly.set_index("date")[column].dropna().loc["2019":]
    fc = data["forecasts"][data["forecasts"]["target"] == target]
    bt = data["backtest"][data["backtest"]["target"] == target]

    st.subheader("Next 12 months")
    fig = go.Figure(go.Scatter(x=history.index, y=history.values, name="actual", line=dict(color="black")))
    for model in models:
        f = fc[fc["model"] == model]
        if f.empty:
            continue  # regression models are evaluated in the backtest only
        if f["lower"].notna().all():
            fig.add_trace(go.Scatter(x=list(f["date"]) + list(f["date"][::-1]),
                                     y=list(f["upper"]) + list(f["lower"][::-1]),
                                     fill="toself", opacity=0.2, line=dict(width=0),
                                     name=f"{model} 95% interval"))
        fig.add_trace(go.Scatter(x=f["date"], y=f["forecast"], name=f"{model} forecast", line=dict(dash="dash")))
    fig.update_layout(height=450, hovermode="x unified")
    st.plotly_chart(fig)

    if target == "inflation_mom":
        projected = data["summary"][target].get("projected_12m_inflation_pct", {})
        cols = st.columns(len(projected))
        for col, (model, value) in zip(cols, projected.items()):
            col.metric(f"Next 12 months inflation — {model}", f"{value:.1f}%")

    st.subheader("Backtest: one-step-ahead forecasts on the last 36 months")
    fig_bt = go.Figure()
    actual = bt.drop_duplicates("date")
    fig_bt.add_trace(go.Scatter(x=actual["date"], y=actual["actual"], name="actual", line=dict(color="black", width=3)))
    for model in models:
        b = bt[bt["model"] == model]
        fig_bt.add_trace(go.Scatter(x=b["date"], y=b["forecast"], name=model))
    fig_bt.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig_bt)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Model comparison** (sorted by MAE; MASE < 1 = better than the naive scale)")
        st.dataframe(metrics.drop(columns="target").round(3), hide_index=True)
    with right:
        st.markdown("**MAE by year of the test window**")
        yearly = data["errors_by_year"][data["errors_by_year"]["target"] == target]
        st.dataframe(yearly.drop(columns="target").dropna(axis=1, how="all").round(3), hide_index=True)


def tab_sql():
    st.subheader("SQL queries on data/macro.db")
    name = st.selectbox("Query", list(QUERIES), format_func=lambda q: f"{q} — {QUERIES[q][0]}")
    description, sql = QUERIES[name]
    st.code(sql.strip(), language="sql")
    st.dataframe(run_sql(sql), hide_index=True)


def main():
    st.title("Kazakhstan: exchange rate, inflation and base rate")
    try:
        data = load_data()
    except FileNotFoundError as err:
        st.error(f"Missing processed data ({err.filename}). Run `python main.py` and choose "
                 "'Run full pipeline' first.")
        st.stop()

    monthly_all = data["monthly"]
    with st.sidebar:
        st.header("Filters")
        years = st.slider("Period", int(monthly_all["date"].dt.year.min()), int(monthly_all["date"].dt.year.max()),
                          (2010, int(monthly_all["date"].dt.year.max())))
        selected = st.multiselect("Series", list(SERIES), default=["usd_kzt_avg", "inflation_yoy", "base_rate"],
                                  format_func=lambda c: SERIES[c][0])
        regimes = st.multiselect("Exchange-rate regime", list(REGIME_LABELS), default=list(REGIME_LABELS),
                                 format_func=REGIME_LABELS.get)
        st.caption("Sources: National Bank of Kazakhstan, IMF, FRED, World Bank")

    start, end = pd.Timestamp(f"{years[0]}-01-01"), pd.Timestamp(f"{years[1]}-12-31")
    monthly = monthly_all[(monthly_all["date"] >= start) & (monthly_all["date"] <= end)
                          & monthly_all["regime"].isin(regimes)]

    kpi_row(monthly_all, data["daily_fx"], data["base_rate"])
    tabs = st.tabs(["Time series", "Exchange rate", "Regimes", "Forecast", "SQL"])
    with tabs[0]:
        if not selected or monthly.empty:
            st.info("Select at least one series and regime.")
        else:
            tab_time_series(monthly, selected, start, end)
    with tabs[1]:
        tab_exchange_rate(data["daily_fx"], data["base_rate"], start, end)
    with tabs[2]:
        tab_regimes()
    with tabs[3]:
        tab_forecast(data, monthly_all)
    with tabs[4]:
        tab_sql()


main()
