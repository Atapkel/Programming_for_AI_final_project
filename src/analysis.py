"""Exploratory time-series analysis of the monthly macro table.

Produces:
    reports/figures/*.png               - static charts for the report
    reports/figures/macro_interactive.html - interactive Plotly chart
    data/processed/analysis_results.json   - test statistics used in the report

Usage:
    python -m src.analysis
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from src.preprocessing import PROCESSED_DIR

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "reports" / "figures"

# Known shock dates marked on charts and tested as structural breaks
BREAKS = {
    "2009-02-04": "2009 devaluation",
    "2014-02-11": "2014 devaluation",
    "2015-08-20": "Tenge float",
    "2022-02-24": "2022 shock",
}


def load_monthly():
    """Monthly macro table indexed by month start."""
    path = PROCESSED_DIR / "monthly_macro.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run: python -m src.preprocessing")
    df = pd.read_parquet(path).set_index("date")
    df.index.freq = "MS"
    return df


def stationarity_tests(series):
    """ADF (H0: unit root) and KPSS (H0: stationary) for one series."""
    s = series.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS warns when p-value is outside its table
        adf_p = adfuller(s, autolag="AIC")[1]
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
    if adf_p < 0.05 and kpss_p > 0.05:
        verdict = "stationary"
    elif adf_p >= 0.05 and kpss_p <= 0.05:
        verdict = "non-stationary"
    else:
        verdict = "inconclusive"
    return {"adf_p": round(float(adf_p), 4), "kpss_p": round(float(kpss_p), 4), "verdict": verdict}


def stl_decompose(series, period=12):
    """STL decomposition and seasonal strength (Hyndman: 1 - Var(R) / Var(S + R))."""
    result = STL(series.dropna(), period=period, robust=True).fit()
    strength = max(0.0, 1 - np.var(result.resid) / np.var(result.seasonal + result.resid))
    return result, round(float(strength), 3)


def chow_test(series, break_date):
    """Chow test for a shift in the mean at break_date (model: y = const + e)."""
    s = series.dropna()
    before, after = s[s.index < break_date], s[s.index >= break_date]
    rss_pooled = ((s - s.mean()) ** 2).sum()
    rss_split = ((before - before.mean()) ** 2).sum() + ((after - after.mean()) ** 2).sum()
    k, n = 1, len(s)
    f_stat = ((rss_pooled - rss_split) / k) / (rss_split / (n - 2 * k))
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k)
    return {"mean_before": round(float(before.mean()), 3), "mean_after": round(float(after.mean()), 3),
            "f_stat": round(float(f_stat), 2), "p_value": round(float(p_value), 4)}


def pass_through_lags(df, max_lag=6):
    """Correlation of inflation (m/m) with the USD/KZT change k months earlier."""
    result = {}
    for lag in range(max_lag + 1):
        corr = df["inflation_mom"].corr(df["usd_kzt_change_pct"].shift(lag))
        result[lag] = round(float(corr), 3)
    return result


def correlations(df):
    """Correlation matrices in levels and in monthly changes (since 2016, after the float)."""
    recent = df.loc["2016":]
    levels = recent[["usd_kzt_avg", "brent_avg", "inflation_yoy", "base_rate"]].corr()
    changes = pd.DataFrame({
        "usd_kzt_change": recent["usd_kzt_change_pct"],
        "brent_change": recent["brent_change_pct"],
        "inflation_mom": recent["inflation_mom"],
        "base_rate_change": recent["base_rate"].diff(),
    }).corr()
    return levels.round(2), changes.round(2)


# ---------- figures ----------

def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_usd_kzt(daily):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily["date"], daily["usd_kzt"], lw=0.8, color="tab:blue")
    top = ax.get_ylim()[1]
    for i, (date, label) in enumerate(BREAKS.items()):
        ax.axvline(pd.Timestamp(date), color="tab:red", ls="--", lw=0.8)
        # alternate label heights so neighbouring breaks (2014, 2015) do not overlap
        y = top * (0.97 if i % 2 == 0 else 0.88)
        ax.text(pd.Timestamp(date), y, f" {label}", fontsize=8, color="tab:red", va="top")
    ax.set(title="Official USD/KZT exchange rate, 2000–2026", ylabel="KZT per USD", xlabel="")
    ax.grid(alpha=0.3)
    save(fig, "01_usd_kzt_breaks.png")


def plot_stl(stl_result):
    fig = stl_result.plot()
    fig.set_size_inches(11, 7)
    fig.suptitle("STL decomposition of monthly CPI inflation (% m/m)", y=1.01)
    save(fig, "02_inflation_stl.png")


def plot_acf_pacf(series):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    plot_acf(series.dropna(), lags=36, ax=axes[0], title="ACF — inflation % m/m")
    plot_pacf(series.dropna(), lags=36, ax=axes[1], method="ywm", title="PACF — inflation % m/m")
    save(fig, "03_inflation_acf_pacf.png")


def plot_heatmap(matrix, title, name):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix)), matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.index)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(j, i, f"{matrix.iat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    save(fig, name)


def plot_rate_vs_inflation(df):
    recent = df.loc["2015-09":]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(recent.index, recent["inflation_yoy"], label="Inflation, % y/y", color="tab:orange")
    ax.step(recent.index, recent["base_rate"], where="post", label="NBK base rate, %", color="tab:blue")
    ax.fill_between(recent.index, recent["base_rate"], recent["inflation_yoy"],
                    where=recent["real_rate"] > 0, alpha=0.15, color="tab:green", label="Positive real rate")
    ax.set(title="Base rate vs inflation since the 2015 float", ylabel="%")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "05_base_rate_vs_inflation.png")


def plot_interactive(df):
    """Interactive Plotly chart: exchange rate, inflation and base rate on a shared time axis."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("USD/KZT (monthly average) and Brent", "Inflation and base rate"),
                        specs=[[{"secondary_y": True}], [{}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["usd_kzt_avg"], name="USD/KZT"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["brent_avg"], name="Brent, USD"), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df["inflation_yoy"], name="Inflation % y/y"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["base_rate"], name="Base rate %", line_shape="hv"), row=2, col=1)
    for date in BREAKS:
        fig.add_vline(x=pd.Timestamp(date).timestamp() * 1000, line_dash="dash", line_color="red", opacity=0.5)
    fig.update_layout(height=700, title="Kazakhstan: exchange rate, oil, inflation and base rate",
                      hovermode="x unified")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(FIGURES_DIR / "macro_interactive.html", include_plotlyjs="cdn")


def run_analysis():
    """Run all tests, save figures and analysis_results.json; return the results dict."""
    df = load_monthly()
    daily = pd.read_parquet(PROCESSED_DIR / "usd_kzt_daily.parquet")
    inflation = df["inflation_mom"].dropna()

    series_to_test = {
        "usd_kzt_avg (level)": df["usd_kzt_avg"],
        "usd_kzt_change_pct": df["usd_kzt_change_pct"],
        "cpi_index (level)": df["cpi_index"],
        "inflation_mom": df["inflation_mom"],
        "inflation_yoy": df["inflation_yoy"],
        "brent_avg (level)": df["brent_avg"],
        "brent_change_pct": df["brent_change_pct"],
    }
    stl_result, seasonal_strength = stl_decompose(inflation)
    monthly_pattern = inflation.groupby(inflation.index.month).mean().round(3)
    levels_corr, changes_corr = correlations(df)

    results = {
        "stationarity": {name: stationarity_tests(s) for name, s in series_to_test.items()},
        "inflation_seasonal_strength": seasonal_strength,
        "inflation_mom_by_calendar_month": {int(k): float(v) for k, v in monthly_pattern.items()},
        "pass_through_corr_by_lag": pass_through_lags(df),
        "chow_tests_inflation_mom": {label: chow_test(inflation, date) for date, label in BREAKS.items()
                                     if pd.Timestamp(date) > inflation.index.min()},
        "chow_tests_fx_change": {label: chow_test(df["usd_kzt_change_pct"], date) for date, label in BREAKS.items()},
        "corr_levels_2016": levels_corr.to_dict(),
        "corr_changes_2016": changes_corr.to_dict(),
    }

    plot_usd_kzt(daily)
    plot_stl(stl_result)
    plot_acf_pacf(inflation)
    plot_heatmap(changes_corr, "Correlation of monthly changes (2016+)", "04_corr_changes.png")
    plot_rate_vs_inflation(df)
    plot_interactive(df)

    with open(PROCESSED_DIR / "analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    res = run_analysis()
    print("Stationarity (ADF p / KPSS p):")
    for name, r in res["stationarity"].items():
        print(f"  {name:22s} {r['adf_p']:.3f} / {r['kpss_p']:.3f}  -> {r['verdict']}")
    print(f"Inflation seasonal strength: {res['inflation_seasonal_strength']}")
    print("Pass-through corr(inflation_mom, usd_kzt_change lag k):", res["pass_through_corr_by_lag"])
    print("Chow tests (inflation_mom):")
    for label, r in res["chow_tests_inflation_mom"].items():
        print(f"  {label:18s} mean {r['mean_before']} -> {r['mean_after']}, F={r['f_stat']}, p={r['p_value']}")
    print(f"Figures saved to {FIGURES_DIR}")
