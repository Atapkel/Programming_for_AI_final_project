"""Forecast monthly inflation and the USD/KZT rate and compare models with baselines.

Evaluation: expanding-window, one-step-ahead backtest over the last TEST_MONTHS months.
At every test month t the models are refit on earlier observation months (no
shuffling). Publication timing is explicitly assumed; historical data vintages
are not available. Forecast preprocessing never interpolates from future values.

Models:
    naive, seasonal_naive        - baselines
    sarima / arima, ets          - statistical models (statsmodels)
    lag_regression               - Ridge on lags + lagged exogenous changes
    gradient_boosting            - GradientBoostingRegressor on the same features

Outputs (data/processed/):
    model_metrics.csv            - MAE, RMSE, sMAPE, MASE, Diebold-Mariano vs naive, fit time
    errors_by_year.csv           - MAE per calendar year of the test window
    backtest_predictions.parquet - actual vs forecast for every test month
    forecasts.parquet            - next 12 months forecast (for the dashboard)
    forecast_summary.json        - chosen ARIMA orders, best model, projected inflation

Usage:
    python -m src.models
"""

import json
import time
import warnings
import logging
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.analysis import FIGURES_DIR, load_monthly
from src.preprocessing import PROCESSED_DIR
from src.preprocessing import run_preprocessing
from src.artifacts import assert_idle, verify_manifest, manifest, publish_files, write_json, fingerprint
from src.validation import validate_monthly_index

LOGGER = logging.getLogger(__name__)
FIT_DIAGNOSTICS = []
AVAILABILITY_POLICY = {
    "evaluation": "retrospective with final-vintage observations, not a real-time vintage backtest",
    "origin": "15th of the target month; only completed earlier months are inputs",
    "sources": "previous-month CPI/FX/Brent assumed available by this origin; actual release dates unavailable",
    "fx_gaps": "past-only daily forward fill; original source values preserved",
    "annual_and_crisis_features": "descriptive only, not supplied to forecasting models",
}


class ModelFitError(RuntimeError):
    """Bounded fitting attempts did not produce a reliable result."""


def load_forecasting_monthly():
    assert_idle(PROCESSED_DIR)
    assert_idle(PROCESSED_DIR.parent)
    path = PROCESSED_DIR / "monthly_forecasting.parquet"
    if path.exists():
        verify_manifest(PROCESSED_DIR, "preprocessing_manifest.json")
        df = pd.read_parquet(path)
    else:
        LOGGER.warning("Causal table absent; rebuilding from raw data in memory (legacy artifacts retained)")
        df = run_preprocessing(save=False)[0]["monthly_forecasting"]
    df = df.set_index("date")
    validate_monthly_index(df.index)
    df.index.freq = "MS"
    return df


def target_series(df, cfg):
    validate_monthly_index(df.index)
    y = df[cfg["column"]].loc[cfg["start"]:].dropna()
    validate_monthly_index(y.index)  # internal gaps must never compress calendar lags
    if not np.isfinite(y).all() or (cfg["log"] and (y <= 0).any()):
        raise ValueError("Target must be finite and positive for logarithmic models")
    return y

TEST_MONTHS = 36
HORIZON = 12

TARGETS = {
    "inflation_mom": {
        "column": "inflation_mom",       # series we forecast
        "reg_column": "inflation_mom",   # what the regression models predict
        "start": "2005-02",
        "exog": ["usd_kzt_change_pct", "brent_change_pct"],
        "log": False,
        "d": 0,
        "seasonal_order": (1, 0, 1, 12),
        "ets": {"trend": None, "seasonal": "add", "seasonal_periods": 12},
        "mase_m": 12,
        "unit": "% m/m",
    },
    "usd_kzt": {
        "column": "usd_kzt_avg",
        "reg_column": "usd_kzt_change_pct",  # trees cannot extrapolate levels -> predict % change
        "start": "2000-01",
        "exog": ["brent_change_pct"],
        "log": True,
        "d": 1,
        "seasonal_order": (0, 0, 0, 0),
        "ets": {"trend": "add", "damped_trend": True},
        "mase_m": 1,
        "unit": "KZT per USD",
    },
}

REGRESSORS = {
    "lag_regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42),
}


# ---------- data ----------

def make_features(df, cfg, lags=(1, 2, 3, 12)):
    """Lagged target, lagged exogenous changes and month dummies (all known at time t-1)."""
    validate_monthly_index(df.index)
    if any(not isinstance(k, int) or k < 1 for k in lags):
        raise ValueError("Forecast features require strictly positive integer lags")
    X = pd.DataFrame(index=df.index)
    for k in lags:
        X[f"{cfg['reg_column']}_lag{k}"] = df[cfg["reg_column"]].shift(k)
    for col in cfg["exog"]:
        X[f"{col}_lag1"] = df[col].shift(1)
    months = pd.get_dummies(pd.Categorical(df.index.month, categories=range(1, 13)),
                           prefix="m", drop_first=True, dtype=float)
    months.index = df.index
    return X.join(months)


# ---------- statistical models ----------

def _checked_sarimax(y, cfg, order, start_params=None):
    """Same specification/optimizer; at most one cold retry with a larger budget."""
    if order is None:
        raise ModelFitError("No converged ARIMA candidate was selected")
    for attempt, maxiter in enumerate((50, 200), start=1):
        messages, result = [], None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = SARIMAX(y, order=order, seasonal_order=cfg["seasonal_order"]).fit(
                    disp=False, maxiter=maxiter, start_params=start_params if attempt == 1 else None)
            messages = [str(w.message) for w in caught]
            converged = (bool(result.mle_retvals.get("converged", False))
                         and np.isfinite(result.aic) and np.isfinite(result.params).all())
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as err:
            converged, messages = False, [str(err)]
        record = {"target": cfg["column"], "n_observations": len(y), "order": list(order),
                  "attempt": attempt, "maxiter": maxiter, "converged": bool(converged),
                  "warnings": messages}
        FIT_DIAGNOSTICS.append(record)
        if converged:
            for message in messages:
                LOGGER.warning("SARIMAX %s order=%s: %s", cfg["column"], order, message)
            return result
        LOGGER.warning("SARIMAX failure target=%s n=%s order=%s attempt=%s/%s: %s",
                       cfg["column"], len(y), order, attempt, 2, messages or "optimizer did not converge")
    raise ModelFitError(f"SARIMAX {order} failed convergence after 2 attempts (n={len(y)})")


def select_arima_order(y, cfg, max_pq=2):
    """Pick (p, d, q) with the lowest AIC on the training data."""
    best_order, best_aic = None, np.inf
    for p in range(max_pq + 1):
        for q in range(max_pq + 1):
            try:
                res = _checked_sarimax(y, cfg, (p, cfg["d"], q))
            except ModelFitError:
                continue
            if res.aic < best_aic:
                best_order, best_aic = (p, cfg["d"], q), res.aic
    if best_order is None:
        LOGGER.error("No converged candidate for %s; backtest will explicitly mark naive fallback", cfg["column"])
    return best_order


_WARM_START = {}  # last fitted parameters per (target, order): speeds up backtest refits


def fit_sarima(y, cfg, order, warm_key=None):
    y_fit = np.log(y) if cfg["log"] else y
    start_params = _WARM_START.get(warm_key)
    res = _checked_sarimax(y_fit, cfg, order, start_params)
    if warm_key is not None:
        _WARM_START[warm_key] = res.params
    return res


def fit_ets(y, cfg):
    y_fit = np.log(y) if cfg["log"] else y
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ExponentialSmoothing(y_fit, **cfg["ets"]).fit()
    for warning in caught:
        LOGGER.warning("ETS %s: %s", cfg["column"], warning.message)
    if not result.mle_retvals.get("success", False) or not np.isfinite(result.fittedvalues).all():
        raise ModelFitError("ETS optimizer failed or produced non-finite values")
    return result


def undo_log(values, cfg):
    return np.exp(values) if cfg["log"] else values


# ---------- model functions: (history, X_train, reg_train, x_next, cfg, order) -> forecast ----------

def model_naive(history, *_):
    return history.iloc[-1]


def model_seasonal_naive(history, *_):
    return history.iloc[-12]


def model_sarima(history, X_train, reg_train, x_next, cfg, order):
    model = fit_sarima(history.reset_index(drop=True), cfg, order, warm_key=(cfg["column"], order))
    forecast = model.forecast(1)
    return float(undo_log(np.asarray(forecast), cfg)[0])


def model_ets(history, X_train, reg_train, x_next, cfg, order):
    forecast = fit_ets(history.reset_index(drop=True), cfg).forecast(1)
    return float(undo_log(np.asarray(forecast), cfg)[0])


def make_regression_model(estimator):
    def model(history, X_train, reg_train, x_next, cfg, order):
        fitted = clone(estimator).fit(X_train, reg_train)
        pred = float(fitted.predict(x_next)[0])
        if cfg["reg_column"] != cfg["column"]:      # predicted % change -> level
            return history.iloc[-1] * (1 + pred / 100)
        return pred
    return model


def get_models(cfg):
    stat_name = "sarima" if cfg["seasonal_order"][3] else "arima"
    models = {"naive": model_naive, "seasonal_naive": model_seasonal_naive,
              stat_name: model_sarima, "ets": model_ets}
    for name, estimator in REGRESSORS.items():
        models[name] = make_regression_model(estimator)
    return models


# ---------- evaluation ----------

def backtest(df, cfg, test_months=TEST_MONTHS):
    """Expanding-window one-step-ahead forecasts for the last test_months months."""
    _WARM_START.clear()
    y = target_series(df, cfg)
    if not isinstance(test_months, int) or test_months < 1 or len(y) - test_months < 36:
        raise ValueError("Backtest requires a positive test window and at least 36 training months")
    X = make_features(df, cfg).loc[y.index]
    reg_y = df[cfg["reg_column"]].loc[y.index]
    test_index = y.index[-test_months:]
    order = select_arima_order(np.log(y[:test_index[0]].iloc[:-1]) if cfg["log"] else y[:test_index[0]].iloc[:-1], cfg)

    models = get_models(cfg)
    preds = {name: [] for name in models}
    fit_seconds = {name: 0.0 for name in models}
    statuses = {name: [] for name in models}
    for t in test_index:
        history = y[y.index < t]
        train_rows = X.loc[history.index].dropna().index.intersection(reg_y.dropna().index)
        X_train, reg_train = X.loc[train_rows], reg_y.loc[train_rows]
        x_next = X.loc[[t]]
        if X_train.empty or not np.isfinite(x_next.to_numpy()).all():
            raise ValueError(f"Unavailable lagged features at {t}; no implicit future imputation")
        for name, model in models.items():
            start = time.perf_counter()
            try:
                prediction = model(history, X_train, reg_train, x_next, cfg, order)
                if not np.isfinite(prediction):
                    raise ModelFitError("Non-finite forecast")
                status = "ok"
            except ModelFitError as err:
                prediction, status = float(history.iloc[-1]), "fallback_naive"
                LOGGER.error("%s at %s: %s. Explicit naive fallback; model excluded from best selection.", name, t, err)
            preds[name].append(prediction)
            statuses[name].append(status)
            fit_seconds[name] += time.perf_counter() - start

    predictions = pd.DataFrame(preds, index=test_index)
    predictions.insert(0, "actual", y.loc[test_index])
    predictions.attrs["fit_status"] = statuses
    train_y = y[y.index < test_index[0]]
    return predictions, fit_seconds, order, train_y


def smape(actual, forecast):
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    denominator = np.abs(actual) + np.abs(forecast)
    ratio = np.divide(2 * np.abs(forecast - actual), denominator,
                      out=np.zeros_like(denominator, dtype=float), where=denominator != 0)
    return float(np.mean(ratio) * 100)


def diebold_mariano(errors_model, errors_base, horizon=1):
    """Diebold-Mariano test of equal absolute-error accuracy, HLN small-sample corrected.

    d_t = |e_model| - |e_base|. Negative statistic: the model has smaller errors.
    Two-sided p-value from Student t with n-1 df (Harvey, Leybourne, Newbold 1997).
    Returns (nan, nan) when the loss differential has no variance (identical forecasts).
    """
    d = np.abs(np.asarray(errors_model, dtype=float)) - np.abs(np.asarray(errors_base, dtype=float))
    n = len(d)
    if n < 3 or not np.isfinite(d).all():
        return np.nan, np.nan
    centered = d - d.mean()
    autocov = [np.dot(centered[k:], centered[:n - k]) / n for k in range(horizon)]
    variance = (autocov[0] + 2 * sum(autocov[1:])) / n
    if not variance > 0:
        return np.nan, np.nan
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = correction * d.mean() / np.sqrt(variance)
    return float(statistic), float(2 * stats.t.sf(abs(statistic), df=n - 1))


def compute_metrics(predictions, fit_seconds, train_y, m):
    """MAE, RMSE, sMAPE, MASE (scaled by the in-sample naive error with lag m),
    plus a Diebold-Mariano test of each model against the naive baseline."""
    if m < 1 or len(train_y) <= m or not np.isfinite(predictions.to_numpy()).all():
        raise ValueError("Metrics require finite predictions and sufficient training history")
    scale = np.mean(np.abs(train_y.values[m:] - train_y.values[:-m]))
    actual = predictions["actual"]
    rows = []
    for name in fit_seconds:
        err = predictions[name] - actual
        dm_stat, dm_p = (diebold_mariano(err, predictions["naive"] - actual)
                         if name != "naive" and "naive" in predictions else (np.nan, np.nan))
        rows.append({
            "model": name,
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "sMAPE_%": smape(actual, predictions[name]),
            "MASE": float(np.mean(np.abs(err)) / scale) if scale > 0 else np.nan,
            "DM_stat_vs_naive": dm_stat,
            "DM_p_vs_naive": dm_p,
            "fit_time_s": fit_seconds[name],
            "fallback_count": predictions.attrs.get("fit_status", {}).get(name, []).count("fallback_naive"),
        })
    result = pd.DataFrame(rows)
    result["eligible"] = result["fallback_count"].eq(0)
    return result.sort_values(["eligible", "MAE"], ascending=[False, True]).reset_index(drop=True)


def errors_by_year(predictions):
    """MAE of every model per calendar year of the test window."""
    abs_err = predictions.drop(columns="actual").sub(predictions["actual"], axis=0).abs()
    return abs_err.groupby(abs_err.index.year).mean()


# ---------- future forecast ----------

def future_forecast(df, cfg, order, horizon=HORIZON):
    """Forecast the next `horizon` months with models that support multi-step forecasts."""
    y = target_series(df, cfg)
    if not isinstance(horizon, int) or horizon < 1 or len(y) < 24:
        raise ValueError("Forecast requires a positive horizon and at least 24 observations")
    future_index = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
    rows, failures = [], {}

    def add(model, values, lower=None, upper=None):
        if len(values) != horizon or not np.isfinite(values).all():
            raise ModelFitError(f"Invalid {model} forecast")
        if lower is not None and (not np.isfinite(lower).all() or not np.isfinite(upper).all()
                                  or np.any(lower > values) or np.any(values > upper)):
            raise ModelFitError(f"Invalid {model} prediction intervals")
        for i, date in enumerate(future_index):
            rows.append({"model": model, "date": date, "forecast": float(values[i]),
                         "lower": None if lower is None else float(lower[i]),
                         "upper": None if upper is None else float(upper[i])})

    add("naive", np.repeat(y.iloc[-1], horizon))
    add("seasonal_naive", np.resize(y.iloc[-12:].values, horizon))

    stat_name = "sarima" if cfg["seasonal_order"][3] else "arima"
    try:
        sarima = fit_sarima(y.reset_index(drop=True), cfg, order).get_forecast(horizon)
        ci = np.asarray(sarima.conf_int(alpha=0.05))
        add(stat_name, undo_log(np.asarray(sarima.predicted_mean), cfg),
            undo_log(ci[:, 0], cfg), undo_log(ci[:, 1], cfg))
    except ModelFitError as err:
        failures[stat_name] = str(err)
        LOGGER.error("Future forecast unavailable for %s: %s", stat_name, err)
    try:
        add("ets", undo_log(np.asarray(fit_ets(y.reset_index(drop=True), cfg).forecast(horizon)), cfg))
    except ModelFitError as err:
        failures["ets"] = str(err)
        LOGGER.error("Future forecast unavailable for ETS: %s", err)
    result = pd.DataFrame(rows)
    result.attrs["failures"] = failures
    return result


def plot_backtest(predictions, target, unit, models_to_show):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(predictions.index, predictions["actual"], color="black", lw=2, label="actual")
    for name in models_to_show:
        ax.plot(predictions.index, predictions[name], lw=1.2, label=name)
    ax.set(title=f"One-step-ahead backtest: {target}", ylabel=unit)
    ax.legend(ncol=3)
    ax.grid(alpha=0.3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"06_backtest_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _calculate_models(output_dir):
    df = load_forecasting_monthly()
    FIT_DIAGNOSTICS.clear()
    all_metrics, all_errors, all_preds, all_forecasts, summary = [], [], [], [], {}

    for target, cfg in TARGETS.items():
        print(f"\n=== {target} ===")
        predictions, fit_seconds, order, train_y = backtest(df, cfg)
        metrics = compute_metrics(predictions, fit_seconds, train_y, cfg["mase_m"])
        yearly = errors_by_year(predictions)
        forecasts = future_forecast(df, cfg, order)
        print(metrics.round(3).to_string(index=False))
        print("MAE by year:\n" + yearly.round(3).to_string())

        best = metrics.iloc[0]["model"]
        plot_backtest(predictions, target, cfg["unit"],
                      [m for m in metrics["model"].head(3) if m != "actual"])

        all_metrics.append(metrics.assign(target=target))
        all_errors.append(yearly.reset_index(names="year").assign(target=target))
        long_preds = predictions.reset_index(names="date").melt(
            id_vars=["date", "actual"], var_name="model", value_name="forecast")
        long_preds["fit_status"] = [status for name in predictions.columns if name != "actual"
                                   for status in predictions.attrs["fit_status"][name]]
        all_preds.append(long_preds.assign(target=target))
        all_forecasts.append(forecasts.assign(target=target))
        last = target_series(df, cfg).index[-1]
        summary[target] = {
            "arima_order": list(order) if order is not None else None, "best_model": best,
            "test_period": [str(predictions.index[0].date()), str(predictions.index[-1].date())],
            "last_observation_month": str(last.date()),
            "information_cutoff": str((last + pd.offsets.MonthEnd()).date()),
            "forecast_origin": str((last + pd.offsets.MonthBegin() + pd.Timedelta(days=14)).date()),
            "origin_type": "assumed historical availability date; not actual training time",
            "forecast_period": [str(forecasts.date.min().date()), str(forecasts.date.max().date())],
            "forecast_horizon_months": HORIZON,
            "availability_policy": AVAILABILITY_POLICY,
            "forecast_failures": forecasts.attrs.get("failures", {}),
            "backtest_fallbacks": dict(zip(metrics.model, metrics.fallback_count.astype(int))),
            "unavailable_forecast_models": [name for name in metrics.model if name not in set(forecasts.model)],
        }

        if target == "inflation_mom":   # 12 monthly forecasts -> inflation over the next year
            projected = forecasts.groupby("model")["forecast"].apply(
                lambda f: (np.prod(1 + f / 100) - 1) * 100)
            summary[target]["projected_12m_inflation_pct"] = projected.round(2).to_dict()

    pd.concat(all_metrics).to_csv(output_dir / "model_metrics.csv", index=False)
    pd.concat(all_errors).to_csv(output_dir / "errors_by_year.csv", index=False)
    pd.concat(all_preds).to_parquet(output_dir / "backtest_predictions.parquet", index=False)
    pd.concat(all_forecasts).to_parquet(output_dir / "forecasts.parquet", index=False)
    df.reset_index().to_parquet(output_dir / "model_input.parquet", index=False)
    write_json(output_dir / "forecast_summary.json", summary)
    write_json(output_dir / "model_diagnostics.json", FIT_DIAGNOSTICS)
    return summary


def validate_model_outputs(directory):
    directory = Path(directory)
    summary = json.loads((directory / "forecast_summary.json").read_text())
    bt = pd.read_parquet(directory / "backtest_predictions.parquet")
    fc = pd.read_parquet(directory / "forecasts.parquet")
    metrics = pd.read_csv(directory / "model_metrics.csv")
    model_input = pd.read_parquet(directory / "model_input.parquet").set_index("date")
    validate_monthly_index(model_input.index)
    for frame in (bt, fc):
        if frame.empty or frame.duplicated(["target", "model", "date"]).any():
            raise ValueError("Empty or duplicate model output keys")
        if not np.isfinite(frame.forecast).all():
            raise ValueError("Non-finite model output")
    for target, cfg in TARGETS.items():
        actual = target_series(model_input, cfg)
        scores = metrics[metrics.target == target]
        expected_models = set(get_models(cfg))
        if (len(scores) != 6 or set(scores.model) != expected_models
                or set(bt[bt.target == target].model) != expected_models
                or not np.isfinite(scores[["MAE", "RMSE"]]).all().all()):
            raise ValueError("Invalid model metrics")
        eligible = scores[scores.eligible]
        if eligible.empty or summary[target]["best_model"] != eligible.sort_values("MAE").iloc[0].model:
            raise ValueError("Best model is not a reliable minimum-MAE candidate")
        for _, rows in bt[bt.target == target].groupby("model"):
            if not pd.DatetimeIndex(rows.date).equals(actual.index[-TEST_MONTHS:]):
                raise ValueError("Backtest timestamps do not match the holdout")
            if not np.allclose(rows.actual, actual.loc[rows.date]):
                raise ValueError("Backtest actual values do not match the input snapshot")
        subset = bt[bt.target == target]
        wide = subset.pivot(index="date", columns="model", values="forecast")
        wide["actual"] = subset.groupby("date").actual.first()
        wide.attrs["fit_status"] = {model: rows.sort_values("date").fit_status.tolist()
                                   for model, rows in subset.groupby("model")}
        if not set(subset.fit_status).issubset({"ok", "fallback_naive"}):
            raise ValueError("Unknown backtest fit status")
        training = actual.loc[actual.index < wide.index.min()]
        computed = compute_metrics(wide, dict.fromkeys(expected_models, 0.), training, cfg["mase_m"]).set_index("model")
        saved = scores.set_index("model").loc[computed.index]
        columns = ["MAE", "RMSE", "sMAPE_%", "MASE", "DM_stat_vs_naive", "DM_p_vs_naive",
                   "fallback_count", "eligible"]
        if not np.allclose(saved[columns].to_numpy(dtype=float), computed[columns].to_numpy(dtype=float), equal_nan=True):
            raise ValueError("Metrics/fallback flags do not match saved backtest predictions")
        expected = pd.date_range(actual.index[-1] + pd.offsets.MonthBegin(), periods=HORIZON, freq="MS")
        if not {"naive", "seasonal_naive"}.issubset(set(fc[fc.target == target].model)):
            raise ValueError("Missing future baseline predictions")
        for _, rows in fc[fc.target == target].groupby("model"):
            if not pd.DatetimeIndex(rows.date).equals(expected):
                raise ValueError("Forecast timestamps do not follow the last observation")
            if not rows.lower.isna().equals(rows.upper.isna()):
                raise ValueError("Incomplete forecast interval")
            intervals = rows.dropna(subset=["lower", "upper"])
            if not ((intervals.lower <= intervals.forecast) & (intervals.forecast <= intervals.upper)).all():
                raise ValueError("Unordered forecast intervals")


def run_models():
    """Train in isolation and publish a validated, fingerprinted result bundle."""
    global FIGURES_DIR
    original_figures = FIGURES_DIR
    _WARM_START.clear()
    input_files = [PROCESSED_DIR / "monthly_forecasting.parquet", PROCESSED_DIR / "preprocessing_manifest.json"]
    input_files.extend(sorted((PROCESSED_DIR.parent / "raw").glob("*.csv")))
    input_version = fingerprint(input_files)
    with tempfile.TemporaryDirectory(prefix="models-") as tmp:
        stage = Path(tmp)
        FIGURES_DIR = stage / "figures"
        try:
            summary = _calculate_models(stage)
        finally:
            FIGURES_DIR = original_figures
            _WARM_START.clear()
        validate_model_outputs(stage)
        if fingerprint(input_files) != input_version:
            raise ValueError("Model inputs changed during training; existing artifacts preserved")
        outputs = {p.name: p for p in stage.iterdir() if p.is_file()}
        inputs = {"model_input.parquet": stage / "model_input.parquet"}
        inputs.update({str(p.relative_to(PROCESSED_DIR.parent)): p for p in input_files if p.exists()})
        run = manifest("models", inputs, outputs,
                       configuration=TARGETS, availability_policy=AVAILABILITY_POLICY)
        write_json(stage / "model_manifest.json", run)
        files = {PROCESSED_DIR / name: path for name, path in outputs.items()}
        files.update({original_figures / p.name: p for p in (stage / "figures").iterdir()})
        files[PROCESSED_DIR / "model_manifest.json"] = stage / "model_manifest.json"
        publish_files(files, PROCESSED_DIR.parent)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_models(), indent=2))
