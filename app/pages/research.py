"""EDA and SQL functionality shared by the unified application."""

import json
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import database
from src.analysis import BREAKS
from src.artifacts import ArtifactError, assert_idle, fingerprint
from src.dashboard_data import load_analysis_artifacts, load_market_data
from src.paths import FIGURES_DIR
from utils.data_utils import DATASETS, load_dataset
from utils.ui import page_header, section

SERIES = {"usd_kzt_avg": "USD/KZT · KZT per USD", "brent_avg": "Brent · USD/bbl",
          "inflation_mom": "Inflation · % m/m", "inflation_yoy": "Inflation · % y/y",
          "base_rate": "Base rate · %", "real_rate": "Real rate proxy · percentage points",
          "usd_kzt_change_pct": "FX change · % m/m"}


def download(frame, name):
    st.download_button("Download " + name.replace("_", " "), frame.to_csv(index=False).encode(),
                       file_name=name + ".csv", mime="text/csv", key="research_" + name)


@st.cache_data
def _sql(sql, path, version):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        result = pd.read_sql(sql, connection)
    if fingerprint([path]) != version:
        raise ArtifactError("Database changed while reading; reload the page")
    return result


def run_sql(sql):
    try:
        assert_idle(database.DB_PATH.parent)
        result = _sql(sql, str(database.DB_PATH), fingerprint([database.DB_PATH]))
        assert_idle(database.DB_PATH.parent)
        return result
    except (OSError, sqlite3.Error, pd.errors.DatabaseError, ArtifactError) as error:
        st.error(f"SQL data unavailable: {error}. Run `python -m src.database`.")
        return pd.DataFrame()


def show_sql():
    page_header("SQL Explorer", "Eleven reproducible queries over the local SQLite database.", eyebrow="Relational data")
    name = st.selectbox("Query", list(database.QUERIES), format_func=lambda q: database.QUERIES[q][0])
    description, sql = database.QUERIES[name]
    st.caption(description)
    with st.expander("SQL statement", expanded=True):
        st.code(sql.strip(), language="sql")
    with st.spinner("Reading SQLite…"):
        result = run_sql(sql)
    st.dataframe(result, hide_index=True, width="stretch")
    st.caption(f"{len(result):,} rows · read-only database · {database.DB_PATH.name}")
    if not result.empty:
        download(result, name)


def show_quality():
    page_header("Data Quality", "Missing cells, missing calendar periods and preprocessing decisions.", eyebrow="Data provenance")
    report, _ = load_analysis_artifacts()
    audit = pd.DataFrame(report["raw_audit"]).T.reset_index(names="source")
    section("Audit saved with preprocessing")
    st.dataframe(audit, hide_index=True, width="stretch")
    st.caption("Missing values counts cells; missing periods counts absent rows. Oil business-day gaps include exchange holidays, "
               "so they are not automatically data errors. Base-rate decisions are irregular events; a daily completeness score is inappropriate.")
    st.info("Forecasting uses past-only forward fill for FX gaps. EDA uses interpolation and marks filled days. "
            "The two monthly tables are separate. CPI gaps and duplicate monthly observations are rejected.")
    name = st.selectbox("Completeness dataset", list(DATASETS))
    raw = load_dataset(name)
    if "date" in raw and "Base interest" not in name:
        dates = pd.DatetimeIndex(raw.date.dropna()).normalize()
        frequency = "MS" if "Consumer" in name else "B" if "Brent" in name else "D"
        expected = pd.date_range(dates.min(), dates.max(), freq=frequency)
        missing = expected.difference(dates)
        counts = pd.DataFrame({"year": expected.year, "observed": expected.isin(dates)})
        counts = counts.groupby("year").agg(expected=("observed", "size"), observed=("observed", "sum")).reset_index()
        counts["missing"] = counts.expected - counts.observed
        counts["coverage_pct"] = 100 * counts.observed / counts.expected
        st.plotly_chart(px.bar(counts, x="year", y=["observed", "missing"], title=f"Calendar coverage · {frequency}"), width="stretch")
        st.dataframe(counts, hide_index=True)
        st.caption(f"Current source: {dates.min():%Y-%m-%d} – {dates.max():%Y-%m-%d}; "
                   f"{len(missing)} absent periods; {raw.date.duplicated().sum()} duplicate dates.")
        with st.expander("Absent dates"):
            st.dataframe(pd.DataFrame({"date": missing}), hide_index=True)
        download(counts, "calendar_completeness")
    elif "indicator" in raw:
        indicator = st.selectbox("Quality indicator", sorted(raw.indicator.unique()))
        single = raw[raw.indicator == indicator]
        expected = set(range(int(single.year.min()), int(single.year.max()) + 1))
        st.write("Missing years:", sorted(expected - set(single.year)))
        st.caption(f"{indicator}: {single.year.min()}–{single.year.max()}, {single.value.isna().sum()} missing values; mixed indicator units are not pooled.")
    else:
        st.caption("Irregular policy decisions: no uniform calendar expected.")
    show_storage_benchmarks(report)
    download(audit, "saved_quality_audit")
    st.download_button("Download full quality report", json.dumps(report, indent=2), "quality_report.json", "application/json")


def show_storage_benchmarks(report):
    """CSV vs Parquet, dtype optimisation and chunked loading, as measured in preprocessing."""
    section("Storage format and memory", "Measured during preprocessing; justifies Parquet and dtype choices")
    formats = report.get("csv_vs_parquet_usd_kzt_daily", {})
    rows = [{"format": fmt, **values} for fmt, values in formats.items() if isinstance(values, dict)]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if "parquet_size_saving_pct" in formats:
            st.caption(f"USD/KZT daily table. Parquet file is {formats['parquet_size_saving_pct']}% smaller and keeps column types "
                       f"without parsing. Timing: {formats.get('method', 'single pass')}. At this table size read times are "
                       "milliseconds for both formats, so size and type safety, not speed, are the deciding factors.")
        else:
            st.caption("Legacy measurement without warm-up; rerun preprocessing for comparable timings.")
    memory = report.get("memory_optimization")
    if memory:
        table = pd.DataFrame(memory).T.reset_index(names="table")
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Memory before/after dtype optimisation (float64→float32, smallest integer type, text→category).")
    chunked = report.get("chunked_loading")
    if chunked:
        rows = [{"source": name, "rows": r["rows"], "chunksize": r["chunksize"], "chunks": r["chunks"],
                 "full peak KB": r["full_read"]["peak_kb"], "chunked peak KB": r["chunked_read"]["peak_kb"],
                 "full result KB": r["full_read"]["result_kb"], "chunked result KB": r["chunked_read"]["result_kb"],
                 "full ms": r["full_read"]["time_ms"], "chunked ms": r["chunked_read"]["time_ms"],
                 "same values": r["same_values"]} for name, r in chunked.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Whole-file read vs pd.read_csv(chunksize=…) with per-chunk typing. Peak memory is the tracemalloc peak while loading. "
                   "The loaded table is always smaller after chunked typing; peak savings and speed depend on file size, and on files "
                   "this small the per-chunk overhead can outweigh them. The method scales to files that do not fit in memory.")


def add_events(figure, start, end):
    for date, label in BREAKS.items():
        stamp = pd.Timestamp(date)
        if start <= stamp <= end:
            figure.add_vline(x=stamp.timestamp() * 1000, line_dash="dash", opacity=.45,
                             annotation_text=label, annotation_position="top left")


def show_research():
    page_header("Advanced Data Analysis", "Trends, regimes, seasonality, volatility and descriptive relationships.", eyebrow="Economic research")
    market = load_market_data()
    if market is None:
        st.info("Processed data unavailable. Run `python -m src.preprocessing`.")
        return
    _, results = load_analysis_artifacts()
    monthly = market["monthly"].copy()
    first, last = monthly.date.min().date(), monthly.date.max().date()
    dates = st.date_input("Analysis date range", (max(first, pd.Timestamp("2010-01-01").date()), last), min_value=first, max_value=last)
    if len(dates) != 2:
        st.info("Choose both dates.")
        return
    start, end = map(pd.Timestamp, dates)
    monthly = monthly[monthly.date.between(start, end)].copy()
    if len(monthly) < 2:
        st.info("Choose a range with at least two monthly observations.")
        return
    st.caption(f"EDA monthly data: {monthly.date.min():%Y-%m} – {monthly.date.max():%Y-%m}. "
               "Filled FX observations are for descriptive plots; model experiments use the separate causal table.")
    tabs = st.tabs(["Trends", "Regimes", "Seasonality, rolling mean & volatility", "Correlations & lags", "Statistical findings"])
    with tabs[0]:
        selected = st.multiselect("Indicators", list(SERIES), default=["usd_kzt_avg", "brent_avg"], format_func=SERIES.get)
        normalize = st.toggle("Index selected series to 100", value=True)
        figure = go.Figure()
        for col in selected:
            s = monthly.set_index("date")[col].dropna()
            if normalize:
                if s.empty or s.iloc[0] <= 0:
                    st.warning(f"Cannot meaningfully index {SERIES[col]} from a nonpositive or missing first value.")
                    continue
                s = s / s.iloc[0] * 100
            figure.add_trace(go.Scatter(x=s.index, y=s, name=SERIES[col], line_shape="hv" if col == "base_rate" else "linear"))
        add_events(figure, start, end)
        figure.update_layout(height=430, yaxis_title="First available observation = 100" if normalize else "Native units (see legend)")
        st.plotly_chart(figure, width="stretch")
        if not normalize and len(selected) > 1:
            st.caption("Series use different native units; magnitudes on this axis should not be compared. Enable indexing for a common relative scale.")
        fx = market["fx"][market["fx"].date.between(start, end)]
        figure = px.line(fx, x="date", y="usd_kzt", title="Daily official FX · KZT per USD")
        filled = fx[fx.is_filled]
        figure.add_trace(go.Scatter(x=filled.date, y=filled.usd_kzt, mode="markers", name="Interpolated EDA days", marker=dict(color="orange", size=5)))
        add_events(figure, start, end)
        st.plotly_chart(figure, width="stretch")
        decisions = market["base_rate"]
        section("Base-rate decisions in selected period")
        st.dataframe(decisions[decisions.date.between(start, end)], hide_index=True)
    with tabs[1]:
        regimes = st.multiselect("Economic regimes", monthly.regime.dropna().unique().tolist(), default=monthly.regime.dropna().unique().tolist())
        rows = monthly[monthly.regime.isin(regimes)]
        comparison = rows.groupby("regime", observed=True).agg(months=("date", "size"), start=("date", "min"), end=("date", "max"),
            mean_fx_change_pct=("usd_kzt_change_pct", "mean"), sd_fx_change_pp=("usd_kzt_change_pct", "std"),
            mean_inflation_mom_pct=("inflation_mom", "mean"), mean_base_rate_pct=("base_rate", "mean")).reset_index()
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.plotly_chart(px.box(rows, x="regime", y="usd_kzt_change_pct", points="outliers", title="Monthly FX changes by regime · %"), width="stretch")
        st.caption("Descriptive regimes: managed rate through July 2015, free float from August 2015, post-shock from February 2022. "
                   "Different sample lengths and other changing conditions prevent causal interpretation.")
        download(comparison, "regime_comparison")
        with st.expander("Full-history SQL regime table and annual depreciation ranking"):
            st.dataframe(run_sql(database.QUERIES["08_regime_comparison"][1]), hide_index=True)
            st.dataframe(run_sql(database.QUERIES["10_depreciation_ranking"][1]), hide_index=True)
    with tabs[2]:
        seasonal = monthly.assign(month=monthly.date.dt.month)
        st.plotly_chart(px.box(seasonal, x="month", y="inflation_mom", title="Inflation by calendar month · % m/m"), width="stretch")
        st.dataframe(seasonal.groupby("month").inflation_mom.agg(["count", "mean", "std"]), width="stretch")
        section("Rolling mean", "Trailing moving average smooths month-to-month noise")
        col_a, col_b = st.columns(2)
        rolling_col = col_a.selectbox("Series", list(SERIES), format_func=SERIES.get, key="rolling_series")
        window = col_b.slider("Window, months", 3, 36, 12, key="rolling_window")
        series = monthly.set_index("date")[rolling_col]
        smoothed = series.rolling(window, min_periods=window).mean()
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=series.index, y=series, name="Monthly", line=dict(width=1), opacity=.55))
        figure.add_trace(go.Scatter(x=smoothed.index, y=smoothed, name=f"{window}-month rolling mean", line=dict(width=3)))
        add_events(figure, start, end)
        figure.update_layout(height=380, title=f"{SERIES[rolling_col]} · {window}-month trailing mean", hovermode="x unified")
        st.plotly_chart(figure, width="stretch")
        st.caption(f"Each point averages the current and previous {window - 1} months; the first {window - 1} months of the "
                   "selected range have no value.")
        volatility = monthly[["date", "usd_kzt_change_pct", "brent_change_pct"]].set_index("date").rolling(12, min_periods=12).std()
        st.plotly_chart(px.line(volatility, title="Trailing 12-month standard deviation of monthly changes · percentage points"), width="stretch")
        st.caption("Not annualized. Twelve complete months are required. Short samples and structural shocks can dominate volatility estimates.")
        download(volatility.reset_index(), "rolling_volatility")
        with st.expander("Saved full-history STL, ACF/PACF and rolling-mean figures"):
            st.caption(f"STL seasonal strength on full saved history: {results['inflation_seasonal_strength']:.3f}; independent of the date filter above.")
            for name in ("02_inflation_stl.png", "03_inflation_acf_pacf.png", "07_rolling_means.png"):
                path = FIGURES_DIR / name
                if path.exists():
                    st.image(str(path), width="stretch")
    with tabs[3]:
        changes = pd.DataFrame({"FX % change": monthly.usd_kzt_change_pct,
                                "Brent % change": monthly.brent_change_pct,
                                "Inflation % m/m": monthly.inflation_mom,
                                "Rate change p.p.": monthly.base_rate.diff()})
        corr = changes.corr(min_periods=12)
        st.plotly_chart(px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", title="Correlation of changes · selected period"), width="stretch")
        st.dataframe(changes.notna().astype(int).T.dot(changes.notna().astype(int)).rename_axis("Pairwise sample sizes"), width="stretch")
        lagrows = []
        for lag in range(13):
            paired = pd.concat([monthly.inflation_mom, monthly.usd_kzt_change_pct.shift(lag)], axis=1).dropna()
            lagrows.append({"lag_months": lag, "correlation": paired.iloc[:, 0].corr(paired.iloc[:, 1]) if len(paired) >= 12 else np.nan, "n_pairs": len(paired)})
        lags = pd.DataFrame(lagrows)
        st.plotly_chart(px.bar(lags, x="lag_months", y="correlation", hover_data=["n_pairs"], title="Earlier FX change vs later inflation"), width="stretch")
        st.caption("Associations, not causal pass-through. Trending levels can correlate spuriously; the main matrix uses changes. "
                   "Positive lag means FX change precedes inflation. At least 12 pairs are required; multiple lags are exploratory.")
        download(corr.reset_index(), "change_correlations")
        download(lags, "lag_correlations")
        with st.expander("Saved 2016+ level correlations (spurious-trend caveat)"):
            st.dataframe(pd.DataFrame(results["corr_levels_2016"]), width="stretch")
    with tabs[4]:
        st.info("The following tests use the full saved analysis sample, independent of the date filter. "
                "ADF H0: unit root; KPSS H0: stationarity. Rounded boundary p-values may reflect limits of the tabulated test distribution.")
        st.dataframe(pd.DataFrame(results["stationarity"]).T, width="stretch")
        for key, label in (("chow_tests_inflation_mom", "Inflation mean-shift tests"), ("chow_tests_fx_change", "FX-change mean-shift tests")):
            section(label)
            st.dataframe(pd.DataFrame(results[key]).T, width="stretch")
        st.warning("These exploratory F tests assess changes in the mean only. The whole event month belongs to the post-event group. "
                   "Classical p-values assume independent homoskedastic errors and are not autocorrelation-robust, multiplicity-adjusted or evidence of causation.")
        st.download_button("Download statistical findings", json.dumps(results, indent=2), "analysis_results.json", "application/json")
    download(monthly, "filtered_monthly_analysis")
