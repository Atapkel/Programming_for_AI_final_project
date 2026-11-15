import shutil

import numpy as np
import pandas as pd
import pytest

from src import models
from src.artifacts import ArtifactError, verify_manifest
from src.evaluation import DIRECTORY, interval_scores, summarize, validate_outputs
from src.forecasting import (FEATURE_SETS, HORIZONS, direct_features, predict_ml,
                             statistical_forecasts, supervised_data, window_start)


@pytest.fixture
def history():
    index = pd.date_range("2005-01-01", periods=240, freq="MS")
    t = np.arange(len(index))
    fx = 140 + .5 * t + 4 * np.sin(t / 7)
    return pd.DataFrame({"usd_kzt_avg": fx, "usd_kzt_change_pct": pd.Series(fx).pct_change(fill_method=None).to_numpy() * 100,
                         "inflation_mom": .7 + .2 * np.cos(t / 4), "brent_change_pct": np.sin(t / 3),
                         "base_rate": np.where(index >= "2015-09-01", 12 + np.cos(t / 5), np.nan)}, index=index)


@pytest.mark.parametrize("target", models.TARGETS)
@pytest.mark.parametrize("horizon", HORIZONS)
def test_labels_and_features_stop_at_origin(history, target, horizon):
    cfg = models.TARGETS[target]
    cutoff = history.index[-15]
    prefix = history.loc[:cutoff]
    X, y, current = supervised_data(prefix, cfg, horizon, pd.Timestamp("2015-09-01"), "all", True)
    assert X.index[-1] + pd.DateOffset(months=horizon) == cutoff
    assert (X.index + pd.DateOffset(months=horizon) <= cutoff).all()
    assert current.index[0] == cutoff
    labels = history.loc[X.index + pd.DateOffset(months=horizon), cfg["column"]].to_numpy()
    if cfg["log"]:
        labels = 100 * (labels / history.loc[X.index, cfg["column"]].to_numpy() - 1)
    np.testing.assert_allclose(y, labels)
    changed = history.copy()
    changed.loc[changed.index > cutoff] = 999999
    pd.testing.assert_frame_equal(direct_features(history, cfg, horizon, "all").loc[:cutoff],
                                  direct_features(changed, cfg, horizon, "all").loc[:cutoff])


@pytest.mark.parametrize("target", models.TARGETS)
def test_future_ml_predicts_twelve_months_without_future_exog(history, target):
    cfg = models.TARGETS[target]
    for horizon in range(1, 13):
        forecasts, weights = predict_ml(history, cfg, horizon, pd.Timestamp("2015-09-01"))
        assert set(forecasts) == {"lag_regression", "gradient_boosting"}
        for row in forecasts.values():
            assert np.isfinite(row["predicted"])
            assert row["last_train_anchor"] + pd.DateOffset(months=horizon) <= history.index[-1]
            assert row["method"] == "direct"
            assert np.isnan(row["lower"]) and np.isnan(row["upper"])
        assert weights


def test_missing_current_predictor_is_not_filled(history):
    history.loc[history.index[-1], "brent_change_pct"] = np.nan
    with pytest.raises(ValueError, match="Predictors unavailable"):
        predict_ml(history, models.TARGETS["usd_kzt"], 12, history.index[0])


def test_ablation_uses_identical_training_pairs(history):
    cfg = models.TARGETS["usd_kzt"]
    history.loc["2018-03-01", "brent_change_pct"] = np.nan
    indices = []
    for features in FEATURE_SETS:
        X, y, current = supervised_data(history, cfg, 6, pd.Timestamp("2015-09-01"), features, True)
        indices.append(X.index)
    assert all(index.equals(indices[0]) for index in indices)
    assert len(indices[0]) >= 24


def test_training_window_boundaries(history):
    y = history.usd_kzt_avg
    assert window_start(y, "full") == y.index[0]
    assert window_start(y, "post2015") == pd.Timestamp("2015-09-01")
    assert len(y.loc[window_start(y, "rolling120"):]) == 120
    with pytest.raises(ValueError):
        window_start(y, "unknown")


def test_statistical_failure_has_no_invented_interval(history, monkeypatch):
    def fail(*args, **kwargs):
        raise models.ModelFitError("test failure")
    monkeypatch.setattr(models, "fit_sarima", fail)
    monkeypatch.setattr(models, "fit_ets", fail)
    rows = statistical_forecasts(history.usd_kzt_avg, models.TARGETS["usd_kzt"], (1, 1, 0))
    for row in rows:
        if row["model"] in ("sarima", "ets"):
            assert row["status"].startswith("naive_fallback")
            assert row["predicted"] == history.usd_kzt_avg.iloc[-1]
            assert np.isnan(row["lower"])
    seasonal = [row["predicted"] for row in rows if row["model"] == "seasonal_naive"]
    np.testing.assert_allclose(seasonal, history.usd_kzt_avg.iloc[-12:])


def test_interval_coverage_width_and_score():
    p = pd.DataFrame({"experiment": ["windows"] * 3, "target": ["x"] * 3, "window": ["full"] * 3,
                      "feature_set": ["primary"] * 3, "horizon": [1] * 3, "model": ["sarima"] * 3,
                      "actual": [1., 3., 9.], "lower": [0., 0., np.nan], "upper": [2., 2., np.nan]})
    result = interval_scores(p).iloc[0]
    assert result.n_total == 3 and result.n_intervals == 2
    assert result.coverage == .5 and result.mean_width == 2
    assert result.interval_score == 22
    p[["lower", "upper"]] = np.nan
    assert np.isnan(interval_scores(p).iloc[0].coverage)


def test_saved_metrics_recomputed_from_actual_predictions():
    validate_outputs(DIRECTORY)
    verify_manifest(DIRECTORY, "manifest.json")
    p = pd.read_parquet(DIRECTORY / "predictions.parquet")
    block = p.query("experiment == 'windows' and target == 'usd_kzt' and window == 'full' and horizon == 3 and model == 'gradient_boosting'")
    row = summarize(p).query("experiment == 'windows' and target == 'usd_kzt' and window == 'full' and horizon == 3 and model == 'gradient_boosting'").iloc[0]
    assert row.MAE == pytest.approx(np.mean(abs(block.predicted - block.actual)))
    assert row.RMSE == pytest.approx(np.sqrt(np.mean((block.predicted - block.actual) ** 2)))


@pytest.mark.parametrize("corruption", ["metric", "date", "actual", "missing_model", "future_label", "interval"])
def test_inconsistent_bundle_rejected(tmp_path, corruption):
    stage = tmp_path / "stage2"
    shutil.copytree(DIRECTORY, stage)
    if corruption == "metric":
        table = pd.read_csv(stage / "metrics.csv")
        table.loc[0, "MAE"] += 1
        table.to_csv(stage / "metrics.csv", index=False)
    else:
        table = pd.read_parquet(stage / "predictions.parquet")
        if corruption == "date":
            table.loc[0, "date"] += pd.DateOffset(months=1)
        elif corruption == "actual":
            table.loc[0, "actual"] += 1
        elif corruption == "missing_model":
            table = table.iloc[1:]
        elif corruption == "future_label":
            table.loc[0, "train_end"] = table.loc[0, "cutoff"] + pd.DateOffset(months=1)
        elif corruption == "interval":
            idx = table[table.model == "sarima"].index[0]
            table.loc[idx, "lower"] = table.loc[idx, "upper"] + 1
        table.to_parquet(stage / "predictions.parquet", index=False)
    with pytest.raises(ArtifactError):
        validate_outputs(stage)


def test_stage2_cache_invalidates_and_stale_snapshot_is_explicit(tmp_path):
    from src.dashboard_data import load_stage2
    parent = tmp_path / "processed"
    stage = parent / "stage2"
    shutil.copytree(DIRECTORY, stage)
    shutil.copy2(DIRECTORY.parent / "monthly_forecasting.parquet", parent)
    assert not load_stage2(stage)["stale"]
    (parent / "monthly_forecasting.parquet").write_bytes(b"changed input")
    assert load_stage2(stage)["stale"]
    (stage / "metrics.csv").write_text("tampered")
    with pytest.raises(ArtifactError):
        load_stage2(stage)


def test_failed_experiment_leaves_published_bundle_unchanged(tmp_path, monkeypatch):
    from src import evaluation
    path = tmp_path / "stage2"
    shutil.copytree(DIRECTORY, path)
    before = {file.name: file.read_bytes() for file in path.iterdir()}
    monkeypatch.setattr(evaluation, "DIRECTORY", path)
    def fail(*args, **kwargs):
        raise ValueError("experiment failed before publication")
    monkeypatch.setattr(evaluation, "evaluate", fail)
    with pytest.raises(ValueError, match="experiment failed"):
        evaluation.run_evaluation()
    assert before == {file.name: file.read_bytes() for file in path.iterdir()}


def test_metric_zero_denominators_and_failed_fit_eligibility():
    p = pd.DataFrame({"experiment": ["windows"] * 3, "target": ["x"] * 3, "window": ["full"] * 3,
                      "feature_set": ["primary"] * 3, "horizon": [1] * 3,
                      "model": ["naive", "seasonal_naive", "sarima"], "actual": [0.] * 3,
                      "predicted": [0.] * 3, "mase_scale": [0.] * 3,
                      "status": ["ok", "ok", "naive_fallback: test"], "n_train": [36] * 3,
                      "cutoff": [pd.Timestamp("2020-01-01")] * 3, "date": [pd.Timestamp("2020-02-01")] * 3})
    result = summarize(p).set_index("model")
    assert result.sMAPE.eq(0).all()
    assert result.MASE.isna().all()
    assert result.gain_vs_naive_pct.isna().all()
    assert not result.loc["sarima", "eligible"]
    assert result.loc["sarima", "fallbacks"] == 1


def test_rolling_engine_does_not_feed_future_data_to_fits(history, monkeypatch):
    """Perturb future truth through the complete orchestration, with cheap fit spies."""
    from src import evaluation
    order_cutoffs = []
    def select_order(train, cfg):
        order_cutoffs.append(train.index[-1])
        return (1, cfg["d"], 0)
    def stat(train, cfg, order, **kwargs):
        return [{"model": name, "horizon": h, "predicted": float(train.mean()),
                 "lower": np.nan, "upper": np.nan, "status": "ok", "method": "baseline_rule",
                 "n_train": len(train), "train_start": train.index[0], "train_end": train.index[-1],
                 "last_train_anchor": train.index[-1]}
                for name in ("naive", "seasonal_naive", "sarima", "ets") for h in range(1, 13)]
    def direct(prefix, cfg, horizon, start, feature_set="primary", common_support=False):
        X, y, current = supervised_data(prefix, cfg, horizon, start, feature_set, common_support)
        value = float(y.mean())
        if cfg["log"]:
            value = float(prefix[cfg["column"]].iloc[-1] * (1 + value / 100))
        row = {"predicted": value, "lower": np.nan, "upper": np.nan, "status": "ok",
               "method": "direct", "n_train": len(X), "train_start": X.index[0],
               "train_end": X.index[-1] + pd.DateOffset(months=horizon), "last_train_anchor": X.index[-1]}
        return {name: row for name in models.REGRESSORS}, []
    monkeypatch.setattr(models, "select_arima_order", select_order)
    monkeypatch.setattr(evaluation, "statistical_forecasts", stat)
    monkeypatch.setattr(evaluation, "predict_ml", direct)
    monkeypatch.setattr(evaluation, "permutation_experiment", lambda *args: [])
    cutoff = history.index[-1] - pd.DateOffset(months=13)
    original = evaluation.evaluate(history, origins_count=2)["predictions"]
    changed = history.copy()
    changed.loc[changed.index > cutoff] *= 1.5
    perturbed = evaluation.evaluate(changed, origins_count=2)["predictions"]
    assert all(date == cutoff for date in order_cutoffs)
    columns = [*evaluation.GROUPS, "cutoff", "predicted", "train_start", "train_end", "n_train"]
    pd.testing.assert_frame_equal(original.loc[original.cutoff == cutoff, columns].reset_index(drop=True),
                                  perturbed.loc[perturbed.cutoff == cutoff, columns].reset_index(drop=True))
    assert not np.array_equal(original.loc[original.cutoff == cutoff, "actual"],
                              perturbed.loc[perturbed.cutoff == cutoff, "actual"])
