"""Forecast monthly inflation and the USD/KZT rate and compare models with baselines.

Evaluation: expanding-window, one-step-ahead backtest over the last TEST_MONTHS months.
At every test month t the models are refit on data strictly before t (no shuffling,
no look-ahead). Regressors use only lagged values, so they are known at forecast time.

Models:
    naive, seasonal_naive        - baselines
    sarima / arima, ets          - statistical models (statsmodels)
    lag_regression               - Ridge on lags + lagged exogenous changes
    gradient_boosting            - GradientBoostingRegressor on the same features

Outputs (data/processed/):
    model_metrics.csv            - MAE, RMSE, sMAPE, MASE, fit time per model
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.analysis import FIGURES_DIR, load_monthly
from src.preprocessing import PROCESSED_DIR

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
    X = pd.DataFrame(index=df.index)
    for k in lags:
        X[f"{cfg['reg_column']}_lag{k}"] = df[cfg["reg_column"]].shift(k)
    for col in cfg["exog"]:
        X[f"{col}_lag1"] = df[col].shift(1)
    months = pd.get_dummies(df.index.month, prefix="m", drop_first=True, dtype=float)
    months.index = df.index
    return X.join(months)


# ---------- statistical models ----------

def select_arima_order(y, cfg, max_pq=2):
    """Pick (p, d, q) with the lowest AIC on the training data."""
    best_order, best_aic = None, np.inf
    for p in range(max_pq + 1):
        for q in range(max_pq + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SARIMAX(y, order=(p, cfg["d"], q), seasonal_order=cfg["seasonal_order"]).fit(disp=False)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if res.aic < best_aic:
                best_order, best_aic = (p, cfg["d"], q), res.aic
    return best_order


_WARM_START = {}  # last fitted parameters per (target, order): speeds up backtest refits


def fit_sarima(y, cfg, order, warm_key=None):
    y_fit = np.log(y) if cfg["log"] else y
    start_params = _WARM_START.get(warm_key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = SARIMAX(y_fit, order=order, seasonal_order=cfg["seasonal_order"]).fit(
                disp=False, start_params=start_params)
        except (np.linalg.LinAlgError, ValueError):
            # numerical failure on some windows: refit without the stationarity constraints
            res = SARIMAX(y_fit, order=order, seasonal_order=cfg["seasonal_order"],
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    if warm_key is not None:
        _WARM_START[warm_key] = res.params
    return res


def fit_ets(y, cfg):
    y_fit = np.log(y) if cfg["log"] else y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExponentialSmoothing(y_fit, **cfg["ets"]).fit()


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
    y = df[cfg["column"]].loc[cfg["start"]:].dropna()
    X = make_features(df, cfg).loc[y.index]
    reg_y = df[cfg["reg_column"]].loc[y.index]
    test_index = y.index[-test_months:]
    order = select_arima_order(np.log(y[:test_index[0]].iloc[:-1]) if cfg["log"] else y[:test_index[0]].iloc[:-1], cfg)

    models = get_models(cfg)
    preds = {name: [] for name in models}
    fit_seconds = {name: 0.0 for name in models}
    for t in test_index:
        history = y[y.index < t]
        train_rows = X.loc[history.index].dropna().index
        X_train, reg_train = X.loc[train_rows], reg_y.loc[train_rows]
        x_next = X.loc[[t]]
        for name, model in models.items():
            start = time.perf_counter()
            preds[name].append(model(history, X_train, reg_train, x_next, cfg, order))
            fit_seconds[name] += time.perf_counter() - start

    predictions = pd.DataFrame(preds, index=test_index)
    predictions.insert(0, "actual", y.loc[test_index])
    train_y = y[y.index < test_index[0]]
    return predictions, fit_seconds, order, train_y


def smape(actual, forecast):
    return float(np.mean(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast))) * 100)


def compute_metrics(predictions, fit_seconds, train_y, m):
    """MAE, RMSE, sMAPE, MASE (scaled by the in-sample naive error with lag m)."""
    scale = np.mean(np.abs(train_y.values[m:] - train_y.values[:-m]))
    actual = predictions["actual"]
    rows = []
    for name in fit_seconds:
        err = predictions[name] - actual
        rows.append({
            "model": name,
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "sMAPE_%": smape(actual, predictions[name]),
            "MASE": float(np.mean(np.abs(err)) / scale),
            "fit_time_s": fit_seconds[name],
        })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def errors_by_year(predictions):
    """MAE of every model per calendar year of the test window."""
    abs_err = predictions.drop(columns="actual").sub(predictions["actual"], axis=0).abs()
    return abs_err.groupby(abs_err.index.year).mean()


# ---------- future forecast ----------

def future_forecast(df, cfg, order, horizon=HORIZON):
    """Forecast the next `horizon` months with models that support multi-step forecasts."""
    y = df[cfg["column"]].loc[cfg["start"]:].dropna()
    future_index = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
    rows = []

    def add(model, values, lower=None, upper=None):
        for i, date in enumerate(future_index):
            rows.append({"model": model, "date": date, "forecast": float(values[i]),
                         "lower": None if lower is None else float(lower[i]),
                         "upper": None if upper is None else float(upper[i])})

    add("naive", np.repeat(y.iloc[-1], horizon))
    add("seasonal_naive", np.resize(y.iloc[-12:].values, horizon))

    sarima = fit_sarima(y.reset_index(drop=True), cfg, order).get_forecast(horizon)
    ci = np.asarray(sarima.conf_int(alpha=0.05))
    stat_name = "sarima" if cfg["seasonal_order"][3] else "arima"
    add(stat_name, undo_log(np.asarray(sarima.predicted_mean), cfg),
        undo_log(ci[:, 0], cfg), undo_log(ci[:, 1], cfg))

    add("ets", undo_log(np.asarray(fit_ets(y.reset_index(drop=True), cfg).forecast(horizon)), cfg))
    return pd.DataFrame(rows)


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


def run_models():
    df = load_monthly()
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
        all_preds.append(long_preds.assign(target=target))
        all_forecasts.append(forecasts.assign(target=target))
        summary[target] = {"arima_order": list(order), "best_model": best,
                           "test_period": [str(predictions.index[0].date()), str(predictions.index[-1].date())]}

        if target == "inflation_mom":   # 12 monthly forecasts -> inflation over the next year
            projected = forecasts.groupby("model")["forecast"].apply(
                lambda f: (np.prod(1 + f / 100) - 1) * 100)
            summary[target]["projected_12m_inflation_pct"] = projected.round(2).to_dict()

    pd.concat(all_metrics).to_csv(PROCESSED_DIR / "model_metrics.csv", index=False)
    pd.concat(all_errors).to_csv(PROCESSED_DIR / "errors_by_year.csv", index=False)
    pd.concat(all_preds).to_parquet(PROCESSED_DIR / "backtest_predictions.parquet", index=False)
    pd.concat(all_forecasts).to_parquet(PROCESSED_DIR / "forecasts.parquet", index=False)
    with open(PROCESSED_DIR / "forecast_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_models(), indent=2))
