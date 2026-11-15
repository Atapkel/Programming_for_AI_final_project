"""Reproducible Stage 2 rolling-origin experiments; run with python -m src.evaluation.

This is deliberately independent of the Stage 1 one-step reference artifacts.
No dashboard navigation fits a model. All publication is transactional.
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from src import models
from src.artifacts import ArtifactError, fingerprint, manifest, publish_files, write_json
from src.forecasting import (FEATURE_SETS, HORIZONS, MODEL_NAMES, WINDOWS, direct_features,
                             predict_ml, statistical_forecasts, supervised_data, window_start)
from src.paths import PROCESSED_DIR

DIRECTORY = PROCESSED_DIR / "stage2"
GROUPS = ["experiment", "target", "window", "feature_set", "horizon", "model"]
OUTPUTS = ("predictions.parquet", "metrics.csv", "intervals.csv", "future.parquet",
           "explainability.csv", "permutation.csv", "input.parquet", "diagnostics.json")


def summarize(predictions):
    rows = []
    for key, block in predictions.groupby(GROUPS, sort=True):
        error = block.predicted.to_numpy() - block.actual.to_numpy()
        denominator = np.abs(block.predicted.to_numpy()) + np.abs(block.actual.to_numpy())
        smape = np.divide(200 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator > 0)
        scaled = np.divide(np.abs(error), block.mase_scale.to_numpy(),
                           out=np.full_like(error, np.nan), where=block.mase_scale.to_numpy() > 0)
        rows.append(dict(zip(GROUPS, key)) | {
            "MAE": float(np.mean(np.abs(error))), "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "sMAPE": float(smape.mean()), "MASE": float(np.nanmean(scaled)) if np.isfinite(scaled).any() else np.nan,
            "n": len(block), "fallbacks": int(block.status.ne("ok").sum()),
            "eligible": bool(block.status.eq("ok").all()),
            "origin_start": block.cutoff.min(), "origin_end": block.cutoff.max(),
            "test_start": block.date.min(), "test_end": block.date.max(),
            "min_train": int(block.n_train.min()), "max_train": int(block.n_train.max()),
        })
    result = pd.DataFrame(rows)
    # The window-matched primary baselines also benchmark ablations.
    for baseline in ("naive", "seasonal_naive"):
        base = result.query("experiment == 'windows' and model == @baseline")
        base = base[["target", "window", "horizon", "MAE"]].rename(columns={"MAE": "baseline"})
        result = result.merge(base, on=["target", "window", "horizon"], how="left", validate="many_to_one")
        result[f"gain_vs_{baseline}_pct"] = np.where(result.baseline > 0, 100 * (1 - result.MAE / result.baseline), np.nan)
        result = result.drop(columns="baseline")
    return result


def interval_scores(predictions):
    """Marginal 95% model intervals; no calibration is fitted on evaluation errors."""
    rows = []
    for key, group in predictions.query("experiment == 'windows' and model == 'sarima'").groupby(GROUPS):
        block = group.dropna(subset=["lower", "upper"])
        n = len(block)
        if n:
            width = block.upper - block.lower
            score = (width + 40 * np.maximum(block.lower - block.actual, 0)
                     + 40 * np.maximum(block.actual - block.upper, 0))
            coverage = block.actual.between(block.lower, block.upper).mean()
        rows.append(dict(zip(GROUPS, key)) | {
            "nominal_coverage": .95, "n_intervals": n, "n_total": len(group),
            "coverage": float(coverage) if n else np.nan,
            "mean_width": float(width.mean()) if n else np.nan,
            "interval_score": float(score.mean()) if n else np.nan,
            "coverage_gap": float(coverage - .95) if n else np.nan,
            "kind": "model_based_marginal_prediction_interval",
        })
    return pd.DataFrame(rows)


def permutation_experiment(df, cfg, target):
    """Fixed model, final 24 target months, grouped 3-month block permutations.

    Training labels end before the holdout. Predictors are revealed sequentially.
    This separate diagnostic is NOT the rolling-origin model ranking. Correlated
    groups and synthetic block boundaries limit interpretation.
    """
    y = models.target_series(df, cfg)
    cutoff = y.index[-25]
    prefix = df.loc[:cutoff]
    start = window_start(y.loc[:cutoff], "post2015")
    X, labels, _ = supervised_data(prefix, cfg, 1, start, "all", True)
    anchors = y.index[-25:-1]
    holdout = direct_features(df.loc[:y.index[-1]], cfg, 1, "all").loc[anchors]
    actual = y.iloc[-24:].to_numpy()
    anchor_values = y.loc[anchors].to_numpy()
    rng = np.random.default_rng(42)
    permutations = [np.concatenate([np.arange(24).reshape(8, 3)[i] for i in rng.permutation(8)]) for _ in range(5)]
    rows = []
    for name, estimator in models.REGRESSORS.items():
        fitted = clone(estimator).fit(X, labels)
        def predict(features):
            output = fitted.predict(features)
            return anchor_values * (1 + output / 100) if cfg["log"] else output
        baseline = float(np.mean(np.abs(predict(holdout) - actual)))
        for group in sorted({col.split("__")[0] for col in X.columns}):
            columns = [col for col in X.columns if col.startswith(group + "__")]
            changes = []
            for permutation in permutations:
                shuffled = holdout.copy()
                shuffled[columns] = holdout[columns].iloc[permutation].to_numpy()
                changes.append(float(np.mean(np.abs(predict(shuffled) - actual)) - baseline))
            rows.append({"target": target, "model": name, "feature_group": group,
                         "delta_MAE": np.mean(changes), "repeat_std": np.std(changes),
                         "baseline_MAE": baseline, "n_holdout": 24, "block_months": 3, "repeats": 5,
                         "train_end": cutoff, "test_start": y.index[-24], "test_end": y.index[-1]})
    return rows


def evaluate(df, origins_count=36):
    if origins_count < 1:
        raise ValueError("At least one origin is required")
    predictions, future, explanations, permutation, order_records = [], [], [], [], []
    models.FIT_DIAGNOSTICS.clear()
    models._WARM_START.clear()
    started = time.perf_counter()
    for target, cfg in models.TARGETS.items():
        y = models.target_series(df, cfg)
        last = y.index[-1] - pd.DateOffset(months=max(HORIZONS))
        origins = pd.date_range(end=last, periods=origins_count, freq="MS")
        if origins[0] < y.index[36]:
            raise ValueError("Not enough observations for requested evaluation")
        for window in WINDOWS:
            initial = y.loc[:origins[0]]
            initial = initial.loc[window_start(initial, window):]
            order = models.select_arima_order(np.log(initial) if cfg["log"] else initial, cfg)
            order_records.append({"target": target, "window": window, "order": order,
                                  "selection_cutoff": str(origins[0].date())})
            print(f"{target} {window}: order={order}, {origins_count} origins", flush=True)
            for i, cutoff in enumerate(origins):
                past = y.loc[:cutoff]
                start = window_start(past, window)
                train = past.loc[start:]
                prefix = df.loc[:cutoff]
                scale = float(past.diff(cfg["mase_m"]).abs().dropna().mean())
                shared = {"experiment": "windows", "target": target, "window": window,
                          "feature_set": "primary", "cutoff": cutoff,
                          "origin": cutoff + pd.offsets.MonthBegin() + pd.Timedelta(days=14),
                          "mase_scale": scale}
                stat = statistical_forecasts(train, cfg, order, warm_key=(target, window, order))
                for row in stat:
                    if row["horizon"] in HORIZONS:
                        date = cutoff + pd.DateOffset(months=row["horizon"])
                        predictions.append(shared | row | {"date": date, "actual": y.loc[date]})
                for horizon in HORIZONS:
                    result, weights = predict_ml(prefix, cfg, horizon, start)
                    date = cutoff + pd.DateOffset(months=horizon)
                    for name, row in result.items():
                        predictions.append(shared | row | {"horizon": horizon, "model": name,
                                                           "date": date, "actual": y.loc[date]})
                    if i in (0, len(origins) - 1):
                        explanations.extend({"target": target, "window": window, "horizon": horizon,
                                             "feature_set": "primary", "cutoff": cutoff,
                                             "sample": "first_origin" if i == 0 else "last_origin", **weight}
                                            for weight in weights)
                if (i + 1) % 12 == 0:
                    print(f"  {i + 1}/{origins_count} complete ({time.perf_counter() - started:.1f}s)", flush=True)
            # Future origins and historical evaluation share the same frozen specification.
            cutoff = y.index[-1]
            prefix = df.loc[:cutoff]
            start = window_start(y, window)
            shared = {"target": target, "window": window, "feature_set": "primary", "cutoff": cutoff,
                      "origin": cutoff + pd.offsets.MonthBegin() + pd.Timedelta(days=14)}
            rows = statistical_forecasts(y.loc[start:], cfg, order, warm_key=None)
            for horizon in range(1, 13):
                result, weights = predict_ml(prefix, cfg, horizon, start)
                rows.extend({"model": name, "horizon": horizon, **row} for name, row in result.items())
                if horizon in HORIZONS:
                    explanations.extend({"target": target, "window": window, "horizon": horizon,
                                         "feature_set": "primary", "cutoff": cutoff,
                                         "sample": "future_fit", **weight} for weight in weights)
            future.extend(shared | row | {"date": cutoff + pd.DateOffset(months=row["horizon"])} for row in rows)
        print(f"{target}: feature ablation on common post-2015 support", flush=True)
        for i, cutoff in enumerate(origins):
            prefix = df.loc[:cutoff]
            past = y.loc[:cutoff]
            start = window_start(past, "post2015")
            for horizon in HORIZONS:
                for feature_set in FEATURE_SETS:
                    result, _ = predict_ml(prefix, cfg, horizon, start, feature_set, common_support=True)
                    for name, row in result.items():
                        date = cutoff + pd.DateOffset(months=horizon)
                        predictions.append({"experiment": "ablation", "target": target, "window": "post2015",
                                            "feature_set": feature_set, "cutoff": cutoff, "horizon": horizon,
                                            "origin": cutoff + pd.offsets.MonthBegin() + pd.Timedelta(days=14),
                                            "model": name, "date": date, "actual": y.loc[date],
                                            "mase_scale": float(past.diff(cfg["mase_m"]).abs().dropna().mean()), **row})
            if (i + 1) % 12 == 0:
                print(f"  ablation {i + 1}/{origins_count} ({time.perf_counter() - started:.1f}s)", flush=True)
        permutation.extend(permutation_experiment(df, cfg, target))
    predictions = pd.DataFrame(predictions)
    return {"predictions": predictions, "metrics": summarize(predictions),
            "intervals": interval_scores(predictions), "future": pd.DataFrame(future),
            "explainability": pd.DataFrame(explanations), "permutation": pd.DataFrame(permutation),
            "diagnostics": {"orders": order_records, "fits": models.FIT_DIAGNOSTICS,
                            "runtime_seconds": time.perf_counter() - started, "origins": origins_count}}


def validate_outputs(directory, origins_count=36):
    directory = Path(directory)
    p = pd.read_parquet(directory / "predictions.parquet")
    f = pd.read_parquet(directory / "future.parquet")
    source = pd.read_parquet(directory / "input.parquet").set_index("date")
    expected = len(models.TARGETS) * origins_count * len(HORIZONS) * (len(WINDOWS) * 6 + len(FEATURE_SETS) * 2)
    if len(p) != expected or p.duplicated(GROUPS + ["cutoff"]).any():
        raise ArtifactError("Missing or duplicate evaluation predictions")
    if set(p.target) != set(models.TARGETS) or set(f.target) != set(models.TARGETS):
        raise ArtifactError("Unknown or missing target")
    if set(p.horizon) != set(HORIZONS) or set(p.experiment) != {"windows", "ablation"}:
        raise ArtifactError("Unknown horizon or experiment")
    for target, cfg in models.TARGETS.items():
        for frame in (p, f):
            block = frame[frame.target == target]
            if not np.isfinite(block.predicted).all():
                raise ArtifactError("Non-finite prediction")
            if (block.train_end > block.cutoff).any() or (block.train_start > block.train_end).any():
                raise ArtifactError("Training after origin or inverted training period")
            expected_dates = pd.DatetimeIndex([c + pd.DateOffset(months=int(h)) for c, h in zip(block.cutoff, block.horizon)])
            if not np.array_equal(expected_dates, block.date):
                raise ArtifactError("Forecast target does not match its horizon")
            direct = block[block.method == "direct"]
            if any(a + pd.DateOffset(months=int(h)) > c for a, h, c in zip(direct.last_train_anchor, direct.horizon, direct.cutoff)):
                raise ArtifactError("Direct training label not yet available")
            bounds = block.dropna(subset=["lower", "upper"])
            if ((bounds.lower > bounds.upper) | (bounds.lower > bounds.predicted) | (bounds.upper < bounds.predicted)).any():
                raise ArtifactError("Invalid interval bounds")
        actual = source.loc[p.loc[p.target == target, "date"], cfg["column"]].to_numpy()
        if not np.allclose(actual, p.loc[p.target == target, "actual"]):
            raise ArtifactError("Actual observations differ from input snapshot")
    for target in models.TARGETS:
        for window in WINDOWS:
            for horizon in HORIZONS:
                block = p.query("target == @target and window == @window and horizon == @horizon and experiment == 'windows'")
                if set(block.model) != set(MODEL_NAMES) or not block.groupby("model").size().eq(origins_count).all():
                    raise ArtifactError("Incomplete model comparison")
                if block.groupby("cutoff").size().ne(6).any():
                    raise ArtifactError("Models do not share origins")
        ablation = p.query("target == @target and experiment == 'ablation'")
        if set(ablation.feature_set) != set(FEATURE_SETS) or set(ablation.window) != {"post2015"}:
            raise ArtifactError("Missing feature variant")
        counts = ablation.groupby(["feature_set", "horizon", "model"]).size()
        if len(counts) != len(FEATURE_SETS) * len(HORIZONS) * 2 or not counts.eq(origins_count).all():
            raise ArtifactError("Incomplete feature comparison")
        if ablation.groupby(["cutoff", "horizon"])["n_train"].nunique().ne(1).any():
            raise ArtifactError("Ablation training samples differ")
    if len(f) != 2 * 3 * 12 * 6 or f.duplicated(["target", "window", "model", "horizon"]).any():
        raise ArtifactError("Incomplete future forecast bundle")
    for (target, window), block in f.groupby(["target", "window"]):
        if window not in WINDOWS or set(block.model) != set(MODEL_NAMES) or set(block.horizon) != set(range(1, 13)):
            raise ArtifactError("Invalid future model, window or horizon")
        latest = source.dropna(subset=[models.TARGETS[target]["column"]]).index[-1]
        if not block.cutoff.eq(latest).all() or len(block) != 72:
            raise ArtifactError("Future origin does not match last observation")
    for frame in (p, f):
        if (frame.lower.isna() != frame.upper.isna()).any():
            raise ArtifactError("Incomplete interval bounds")
        bounded = frame.dropna(subset=["lower", "upper"])
        if not (bounded.model.eq("sarima") & bounded.status.eq("ok")).all():
            raise ArtifactError("Unsupported interval estimator")
        expected_origin = frame.cutoff + pd.offsets.MonthBegin() + pd.Timedelta(days=14)
        if not frame.origin.eq(expected_origin).all():
            raise ArtifactError("Incorrect assumed release origin")
    for name, recomputed in (("metrics", summarize(p)), ("intervals", interval_scores(p))):
        stored = pd.read_csv(directory / f"{name}.csv")
        # Normalize dates through CSV in both frames, then compare every field.
        from io import StringIO
        recomputed = pd.read_csv(StringIO(recomputed.to_csv(index=False)))
        try:
            pd.testing.assert_frame_equal(stored, recomputed, check_exact=False, rtol=1e-10, atol=1e-10)
        except AssertionError as error:
            raise ArtifactError(f"{name} do not match saved predictions") from error


def run_evaluation(origins_count=36):
    input_path = PROCESSED_DIR / "monthly_forecasting.parquet"
    before = fingerprint([input_path, PROCESSED_DIR / "preprocessing_manifest.json"])
    df = models.load_forecasting_monthly()
    result = evaluate(df, origins_count)
    with tempfile.TemporaryDirectory(prefix="macro-stage2-") as tmp:
        stage = Path(tmp)
        for name in ("predictions", "future"):
            result[name].to_parquet(stage / f"{name}.parquet", index=False)
        for name in ("metrics", "intervals", "explainability", "permutation"):
            result[name].to_csv(stage / f"{name}.csv", index=False)
        df.reset_index().to_parquet(stage / "input.parquet", index=False)
        write_json(stage / "diagnostics.json", result["diagnostics"])
        validate_outputs(stage, origins_count)
        if fingerprint([input_path, PROCESSED_DIR / "preprocessing_manifest.json"]) != before:
            raise ArtifactError("Model inputs changed during evaluation; results not published")
        run = manifest("stage2_evaluation", {"monthly_forecasting": input_path},
                       {name: stage / name for name in OUTPUTS},
                       horizons=list(HORIZONS), windows=list(WINDOWS), origins=origins_count,
                       target_specs=models.TARGETS,
                       ml_specs={"ridge": {"alpha": 1., "scaling": "StandardScaler fit on training pairs only"},
                                 "gradient_boosting": {"n_estimators": 200, "max_depth": 2,
                                                       "learning_rate": .05, "subsample": .8, "random_state": 42}},
                       feature_specs={"own_observation_lags": [0, 1, 2, 11], "extra_observation_lags": [0, 1, 2],
                                      "calendar": "known target month dummies, February through December",
                                      "ablation_support": "all-feature complete cases, identical anchors across variants"},
                       availability_policy={**models.AVAILABILITY_POLICY,
                           "origin": "15th of the month immediately after the last observed month t; target month is t+h",
                           "sources": "previous-month CPI/FX/Brent assumed available by origin; past announced base-rate decisions only; actual release calendars and vintages unavailable"},
                       strategy="Direct ML per horizon; statistical recursive paths; seasonal repetition",
                       scale="MASE: per-origin full past seasonal difference (12 inflation, 1 FX), common across windows",
                       window_policy="post2015 starts September 2015; rolling120 retains 10 seasonal cycles",
                       uncertainty="SARIMA marginal model-based 95% prediction intervals; parameter/revision uncertainty excluded",
                       selection="ARIMA AIC on first origin only per window; fixed ML hyperparameters; exploratory comparisons, no untouched final test")
        write_json(stage / "manifest.json", run)
        publish_files({DIRECTORY / name: stage / name for name in (*OUTPUTS, "manifest.json")}, PROCESSED_DIR.parent)
    print(f"Published Stage 2 run {run['run_id']} ({result['diagnostics']['runtime_seconds']:.1f}s)", flush=True)
    return result


if __name__ == "__main__":
    run_evaluation()
