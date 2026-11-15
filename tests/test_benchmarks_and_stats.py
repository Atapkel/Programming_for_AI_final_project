"""Storage benchmarks, chunked loading, rolling means and the Diebold-Mariano test."""

import numpy as np
import pandas as pd
import pytest

from src import analysis, models, preprocessing


def daily_frame(rows=500):
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    return pd.DataFrame({"date": dates, "rate": np.linspace(400, 500, rows),
                         "is_filled": np.arange(rows) % 7 == 0})


def test_compare_formats_returns_equal_typed_tables_and_size_saving():
    result = preprocessing.compare_formats(daily_frame(), repeats=2)
    for fmt in ("csv", "parquet"):
        assert result[fmt]["types_preserved"]
        assert result[fmt]["read_ms"] > 0 and result[fmt]["write_ms"] > 0
    assert result["csv"]["memory_kb"] == result["parquet"]["memory_kb"]
    assert result["parquet_size_saving_pct"] == pytest.approx(
        (1 - result["parquet"]["size_kb"] / result["csv"]["size_kb"]) * 100, abs=0.1)
    assert "warm-up" in result["method"]


def test_chunked_read_matches_full_read(tmp_path):
    path = tmp_path / "daily.csv"
    frame = daily_frame(1234)[["date", "rate"]]
    frame.to_csv(path, index=False)
    chunked = preprocessing.read_csv_chunked(path, chunksize=100)
    assert len(chunked) == 1234 and chunked["date"].equals(frame["date"])
    assert chunked["rate"].dtype == "float32"
    np.testing.assert_allclose(chunked["rate"], frame["rate"], rtol=1e-6)
    report = preprocessing.compare_loading(path, chunksize=100)
    assert report["same_values"] and report["chunks"] == 13
    assert report["chunked_read"]["result_kb"] < report["full_read"]["result_kb"]


def test_rolling_means_need_a_full_window():
    index = pd.date_range("2020-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"usd_kzt_avg": np.arange(24.0), "inflation_mom": np.ones(24)}, index=index)
    rolled = analysis.rolling_means(df, window=12)
    assert rolled.iloc[:11].isna().all().all()
    assert rolled["usd_kzt_avg_rolling12"].iloc[11] == pytest.approx(5.5)
    summary = analysis.rolling_summary(df, rolled)
    assert summary["inflation_mom"]["rolling_mean"] == 1.0
    assert summary["usd_kzt_avg"]["month"] == "2021-12-01"


def test_diebold_mariano_sign_and_degenerate_cases():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 60)
    better = base * 0.3
    stat, p = models.diebold_mariano(better, base)
    assert stat < 0 and p < 0.01
    stat, p = models.diebold_mariano(base * 3, base)
    assert stat > 0 and p < 0.01
    assert all(np.isnan(models.diebold_mariano(base, base)))      # identical forecasts
    assert all(np.isnan(models.diebold_mariano(base[:2], base[:2] * 2)))  # too short


def test_metrics_include_diebold_mariano_against_naive():
    index = pd.date_range("2020-01-01", periods=12, freq="MS")
    actual = np.sin(np.arange(12))
    pred = pd.DataFrame({"actual": actual, "naive": actual + 1.0,
                         "good": actual + np.linspace(-0.2, 0.2, 12)}, index=index)
    metrics = models.compute_metrics(pred, {"naive": 0., "good": 0.}, pd.Series(np.arange(24.)), 1)
    metrics = metrics.set_index("model")
    assert np.isnan(metrics.loc["naive", "DM_p_vs_naive"])
    assert metrics.loc["good", "DM_stat_vs_naive"] < 0 and metrics.loc["good", "DM_p_vs_naive"] < 0.05
