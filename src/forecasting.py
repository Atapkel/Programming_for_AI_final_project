"""Stage 2 adapters. Features are indexed by LAST OBSERVED MONTH, not target month.

ML uses direct horizon-specific regressions. FX labels are cumulative percentage
changes from the anchor; inflation labels are the target month's m/m inflation.
No future exogenous path is needed or estimated. Hyperparameters stay fixed.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone

from src import models
from src.validation import validate_monthly_index

HORIZONS = (1, 3, 6, 12)
WINDOWS = ("full", "post2015", "rolling120")
MODEL_NAMES = {
    "naive": "Naive", "seasonal_naive": "Seasonal Naive",
    "sarima": "ARIMA / SARIMA", "ets": "ETS",
    "lag_regression": "Ridge", "gradient_boosting": "Gradient Boosting",
}
FEATURE_SETS = ("ar", "ar_cross", "ar_brent", "ar_rate", "all",
                "without_cross", "without_brent", "without_rate")


def window_start(y, window):
    if window == "full":
        return y.index.min()
    if window == "post2015":
        return max(y.index.min(), pd.Timestamp("2015-09-01"))
    if window == "rolling120":
        return max(y.index.min(), y.index[-1] - pd.DateOffset(months=119))
    raise ValueError(f"Unknown training window: {window}")


def feature_groups(cfg, feature_set):
    cross = "inflation_mom" if cfg["log"] else "usd_kzt_change_pct"
    extras = {"cross": cross, "brent": "brent_change_pct", "rate": "base_rate"}
    if feature_set == "primary":
        return [cfg["reg_column"], *cfg["exog"]]
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    included = {"ar": [], "ar_cross": ["cross"], "ar_brent": ["brent"],
                "ar_rate": ["rate"], "all": list(extras),
                "without_cross": ["brent", "rate"],
                "without_brent": ["cross", "rate"],
                "without_rate": ["cross", "brent"]}[feature_set]
    return [cfg["reg_column"], *(extras[name] for name in included)]


def direct_features(df, cfg, horizon, feature_set="primary"):
    validate_monthly_index(df.index)
    if not isinstance(horizon, int) or not 1 <= horizon <= 12:
        raise ValueError("Horizon must be an integer in 1..12")
    result = pd.DataFrame(index=df.index)
    for col in feature_groups(cfg, feature_set):
        # Observation lag 0 is completed month t, available at the assumed origin.
        for lag in (0, 1, 2, 11) if col == cfg["reg_column"] else (0, 1, 2):
            result[f"{col}__lag{lag}"] = df[col].shift(lag)
    target_months = (df.index + pd.DateOffset(months=horizon)).month
    for month in range(2, 13):
        result[f"calendar__month{month}"] = (target_months == month).astype(float)
    return result.astype(float)


def supervised_data(prefix, cfg, horizon, start, feature_set="primary", common_support=False):
    """Labels must have been observed at cutoff; return only complete past rows.

    Ablations use complete-case support of ALL features, even when a particular
    feature is removed. This keeps training samples identical across feature sets.
    Lag inputs before window start are allowed; training anchors start at start.
    """
    X = direct_features(prefix, cfg, horizon, feature_set)
    values = prefix[cfg["column"]]
    labels = values.shift(-horizon)
    if cfg["log"]:
        labels = 100 * (labels / values - 1)
    target_dates = X.index + pd.DateOffset(months=horizon)
    valid = ((X.index >= start) & (target_dates <= prefix.index[-1])
             & np.isfinite(X).all(axis=1) & np.isfinite(labels))
    if common_support:
        valid &= np.isfinite(direct_features(prefix, cfg, horizon, "all")).all(axis=1)
    train_X, train_y = X.loc[valid], labels.loc[valid]
    current = X.iloc[[-1]]
    if len(train_X) < 24:
        raise ValueError(f"Insufficient complete training pairs: {len(train_X)}")
    if not np.isfinite(current).all(axis=None):
        missing = current.columns[~np.isfinite(current.iloc[0])].tolist()
        raise ValueError(f"Predictors unavailable at origin: {missing}; no future imputation permitted")
    return train_X, train_y, current


def predict_ml(prefix, cfg, horizon, start, feature_set="primary", common_support=False):
    X, y, current = supervised_data(prefix, cfg, horizon, start, feature_set, common_support)
    results, explanations = {}, []
    for name, estimator in models.REGRESSORS.items():
        fitted = clone(estimator).fit(X, y)
        predicted = float(fitted.predict(current)[0])
        if cfg["log"]:
            predicted = float(prefix[cfg["column"]].iloc[-1] * (1 + predicted / 100))
        if not np.isfinite(predicted) or (cfg["log"] and predicted <= 0):
            raise ValueError(f"Invalid forecast from {name}: {predicted}")
        results[name] = {"predicted": predicted, "lower": np.nan, "upper": np.nan,
                         "status": "ok", "method": "direct", "n_train": len(X),
                         "train_start": X.index[0],
                         "train_end": X.index[-1] + pd.DateOffset(months=horizon),
                         "last_train_anchor": X.index[-1]}
        weights = (fitted.named_steps["ridge"].coef_ if name == "lag_regression"
                   else fitted.feature_importances_)
        for feature, value in zip(X.columns, weights):
            explanations.append({"model": name, "feature": feature, "value": float(value),
                                 "kind": "standardized_coefficient" if name == "lag_regression"
                                 else "impurity_importance"})
    return results, explanations


def statistical_forecasts(train, cfg, order, steps=12, warm_key=None):
    """Forecast complete paths; failed fitted models are explicitly naive fallbacks."""
    paths = {
        "naive": np.repeat(train.iloc[-1], steps),
        "seasonal_naive": np.resize(train.iloc[-12:].to_numpy(), steps),
    }
    bounds, failures = {}, {}
    for name in ("sarima", "ets"):
        try:
            fit = (models.fit_sarima(train, cfg, order, warm_key=warm_key)
                   if name == "sarima" else models.fit_ets(train, cfg))
            paths[name] = np.asarray(models.undo_log(fit.forecast(steps), cfg), dtype=float)
            if not np.isfinite(paths[name]).all():
                raise models.ModelFitError("Non-finite forecast")
            if name == "sarima":
                interval = np.asarray(models.undo_log(fit.get_forecast(steps).conf_int(alpha=.05), cfg))
                if not np.isfinite(interval).all() or (interval[:, 0] > interval[:, 1]).any():
                    raise models.ModelFitError("Invalid prediction interval")
                bounds[name] = interval
        except models.ModelFitError as error:
            paths[name] = paths["naive"].copy()
            bounds.pop(name, None)
            failures[name] = str(error)
    return [{"model": name, "horizon": h, "predicted": float(path[h - 1]),
             "lower": float(bounds[name][h - 1, 0]) if name in bounds else np.nan,
             "upper": float(bounds[name][h - 1, 1]) if name in bounds else np.nan,
             "status": f"naive_fallback: {failures[name]}" if name in failures else "ok",
             "method": "model_recursive" if name in ("sarima", "ets") else "baseline_rule",
             "n_train": len(train), "train_start": train.index[0], "train_end": train.index[-1],
             "last_train_anchor": train.index[-1]}
            for name, path in paths.items() for h in range(1, steps + 1)]
