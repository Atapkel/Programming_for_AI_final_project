"""Saved experiments, uncertainty and direct ML future forecasts."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_data import load_stage2
from src.forecasting import MODEL_NAMES, WINDOWS
from utils.ui import page_header, section

TARGET_NAMES = {"inflation_mom": "Inflation, % m/m", "usd_kzt": "USD/KZT, monthly average"}
WINDOW_NAMES = {"full": "Full history · expanding", "post2015": "Since Sep 2015 · expanding",
                "rolling120": "Rolling 120 months"}


def download(frame, name):
    st.download_button("Download " + name.replace("_", " "), frame.to_csv(index=False).encode(),
                       file_name=name + ".csv", mime="text/csv", key="download_" + name)


def bundle():
    with st.spinner("Loading verified experiment artifacts…"):
        data = load_stage2()
    if data is None:
        st.info("Stage 2 experiments are not available. Run `python -m src.evaluation`.")
        return None
    run = data["manifest"]
    st.caption(f"Saved run {run['run_id']} · generated {run['created_at_utc']}")
    if data["stale"]:
        st.warning("Processed inputs have changed since this experiment. Charts use the saved input snapshot; rerun `python -m src.evaluation` to update.")
    st.caption("Retrospective final-vintage data. Origin = 15th of the month after the last observed month. "
               "Previous-month CPI, FX, Brent and rate are assumed available; release dates and historical revisions are not verified.")
    return data


def controls(prefix):
    left, right = st.columns(2)
    target = left.selectbox("Target", list(TARGET_NAMES), format_func=TARGET_NAMES.get, key=prefix + "_target")
    window = right.selectbox("Training window", list(WINDOWS), format_func=WINDOW_NAMES.get, key=prefix + "_window")
    return target, window


def labelled(frame):
    return frame.assign(model=frame.model.map(MODEL_NAMES))


def show_evaluation():
    page_header("Model Evaluation", "Rolling-origin results, feature experiments and uncertainty diagnostics.", eyebrow="Forecast laboratory")
    reference = st.radio("Experiment", ["Stage 2 · multiple horizons", "Stage 1 · original one-step reference"], horizontal=True)
    if reference.startswith("Stage 1"):
        st.info("Original Stage 1 experiment: different test dates and feature construction. Its scores are a separate reference, not a multi-horizon ranking.")
        from pages.predictions import show_predictions
        show_predictions()
        return
    data = bundle()
    if data is None:
        return
    target, window = controls("evaluation")
    horizon = st.select_slider("Forecast horizon (months after last observation)", options=[1, 3, 6, 12], value=1)
    p = data["predictions"].query("experiment == 'windows' and target == @target and window == @window and horizon == @horizon").copy()
    metrics = data["metrics"].query("experiment == 'windows' and target == @target and window == @window and horizon == @horizon").sort_values(["eligible", "MAE"], ascending=[False, True])
    st.caption(f"Last-observation cutoffs: {p.cutoff.min():%Y-%m} – {p.cutoff.max():%Y-%m} · "
               f"target months: {p.date.min():%Y-%m} – {p.date.max():%Y-%m} · "
               f"training observations/pairs: {p.n_train.min()}–{p.n_train.max()}.")
    st.warning(f"Only {p.cutoff.nunique()} origins per model. Multi-month forecast errors overlap and are dependent. "
               "Rankings are exploratory; selecting a window or feature set here does not establish out-of-sample superiority.")
    if p.status.ne("ok").any():
        st.warning("Some fits failed and are explicitly replaced by Naive. These runs are excluded from best-model selection; see status in the prediction download.")
    tabs = st.tabs(["Comparison", "Errors & intervals", "Windows & features", "Explainability", "Methodology"])
    with tabs[0]:
        view = labelled(metrics)[["model", "MAE", "RMSE", "sMAPE", "MASE", "gain_vs_naive_pct", "gain_vs_seasonal_naive_pct", "n", "fallbacks", "eligible"]]
        st.dataframe(view, hide_index=True, width="stretch")
        st.caption("MAE/RMSE: KZT per USD or inflation percentage points; sMAPE and baseline gains: %. "
                   "MASE uses each origin's full past scaling series, shared across windows. Positive gain = lower MAE than baseline.")
        names = metrics.model.tolist()
        chosen = st.multiselect("Models", names, default=names[:2], format_func=MODEL_NAMES.get)
        figure = go.Figure()
        actual = p.drop_duplicates("date").sort_values("date")
        figure.add_trace(go.Scatter(x=actual.date, y=actual.actual, name="Actual", line=dict(color="#172B4D", width=3)))
        for model in chosen:
            rows = p[p.model == model].sort_values("date")
            figure.add_trace(go.Scatter(x=rows.date, y=rows.predicted, name=MODEL_NAMES[model], mode="lines+markers"))
        figure.update_layout(title=f"{horizon}-month endpoint forecasts · {TARGET_NAMES[target]}", height=420, hovermode="x unified")
        st.plotly_chart(figure, width="stretch")
        st.caption("Each point comes from a different origin. This is not a single 12-month trajectory.")
        st.dataframe(p.groupby("model").agg(training_start=("train_start", "min"), latest_training_end=("train_end", "max"), min_pairs=("n_train", "min"), max_pairs=("n_train", "max")), width="stretch")
        download(metrics, "model_metrics")
        download(p, "rolling_predictions")
    with tabs[1]:
        p["error"] = p.predicted - p.actual
        p["absolute_error"] = p.error.abs()
        p["year"] = p.date.dt.year
        diagnostic_model = st.selectbox("Diagnostic model", list(MODEL_NAMES), format_func=MODEL_NAMES.get)
        residuals = p[p.model == diagnostic_model]
        left, right = st.columns(2)
        left.plotly_chart(px.bar(residuals, x="date", y="error", title="Forecast − actual"), width="stretch")
        right.plotly_chart(px.histogram(residuals, x="error", nbins=12, title="Forecast error distribution"), width="stretch")
        yearly = p.groupby(["model", "year"], as_index=False).agg(MAE=("absolute_error", "mean"), n=("error", "size"))
        st.plotly_chart(px.bar(labelled(yearly), x="year", y="MAE", color="model", barmode="group", title="Error by target year", hover_data=["n"]), width="stretch")
        section("95% model-based prediction intervals")
        intervals = data["intervals"].query("target == @target and window == @window")
        st.dataframe(intervals[["horizon", "nominal_coverage", "coverage", "coverage_gap", "mean_width", "interval_score", "n_intervals", "n_total"]], hide_index=True, width="stretch")
        st.plotly_chart(px.line(intervals, x="horizon", y="coverage", markers=True, range_y=[0, 1], title="Empirical interval coverage").add_hline(y=.95, line_dash="dash"), width="stretch")
        rows = p[p.model == "sarima"].sort_values("date")
        figure = go.Figure()
        add_interval(figure, rows)
        figure.add_trace(go.Scatter(x=rows.date, y=rows.actual, name="Actual"))
        figure.add_trace(go.Scatter(x=rows.date, y=rows.predicted, name="ARIMA / SARIMA"))
        st.plotly_chart(figure, width="stretch")
        st.caption("Coverage is the fraction of observed targets inside bounds. Width and interval score are in target units; "
                   "lower interval score is better. These are marginal model-based prediction intervals, not confidence intervals for a mean "
                   "and not empirically recalibrated intervals. They exclude parameter and data-revision uncertainty. "
                   "Overlapping errors make independent-binomial calibration tests inappropriate. Other models have no supported interval estimator.")
        download(intervals, "interval_diagnostics")
        download(yearly, "errors_by_year")
    with tabs[2]:
        windows = data["metrics"].query("experiment == 'windows' and target == @target and horizon == @horizon")
        st.plotly_chart(px.bar(labelled(windows), x="model", y="MAE", color="window", barmode="group", title="Same origins · training window comparison"), width="stretch")
        st.dataframe(windows[["window", "model", "MAE", "RMSE", "MASE", "gain_vs_naive_pct", "min_train", "max_train", "eligible"]], hide_index=True)
        st.caption("Rolling 120 months keeps ten annual cycles while downweighting older regimes by exclusion. "
                   "Post-2015 begins September 2015. Baselines can coincide across windows. Hyperparameters are fixed; AIC order selection uses only the first training origin.")
        ablation = data["metrics"].query("experiment == 'ablation' and target == @target and horizon == @horizon").copy()
        baseline = ablation[ablation.feature_set == "ar"].set_index("model").MAE
        ablation["gain_vs_ar_pct"] = 100 * (1 - ablation.MAE / ablation.model.map(baseline))
        st.plotly_chart(px.bar(labelled(ablation), x="feature_set", y="MAE", color="model", barmode="group", title="Feature ablation · identical post-2015 training pairs"), width="stretch")
        st.dataframe(ablation[["model", "feature_set", "MAE", "RMSE", "gain_vs_ar_pct", "gain_vs_naive_pct", "min_train", "max_train"]], hide_index=True)
        st.caption("ar = own lags + target calendar month; cross = inflation lags for FX, FX-change lags for inflation; "
                   "brent = oil percentage-change lags; rate = base-rate level lags. All variants use the same complete-case rows of the full feature set. "
                   "The primary window experiment retains Stage 1 predictor groups; ablations are a separate comparison.")
        download(windows, "window_experiment")
        download(ablation, "feature_ablation")
    with tabs[3]:
        explain_model = st.selectbox("ML model", ["lag_regression", "gradient_boosting"], format_func=MODEL_NAMES.get)
        weights = data["explainability"].query("target == @target and horizon == @horizon and model == @explain_model")
        sample = st.selectbox("Fit sample", ["first_origin", "last_origin", "future_fit"])
        selected = weights[weights["sample"] == sample]
        st.plotly_chart(px.bar(selected, x="value", y="feature", color="window", barmode="group", orientation="h", height=650, title="Coefficients / importance across training windows"), width="stretch")
        st.caption("Ridge: signed coefficient per one training-standard-deviation of each input, in the regression label's units "
                   "(FX cumulative % change or monthly inflation). Calendar indicators are standardized too. "
                   "Gradient Boosting: normalized impurity reduction; correlated predictors can split or distort importance.")
        st.caption("Coefficient and impurity plots use the primary predictor groups. The permutation experiment below uses all predictor groups.")
        perm = data["permutation"].query("target == @target and model == @explain_model")
        st.plotly_chart(px.bar(perm, x="feature_group", y="delta_MAE", error_y="repeat_std", title="Separate fixed-model holdout · grouped block permutation"), width="stretch")
        st.dataframe(perm, hide_index=True)
        st.caption("Permutation diagnostic: horizon 1, post-2015, all predictors, fixed training cutoff before final 24 target months, "
                   "five permutations of three-month blocks. Error bars show variation across permutations, not confidence intervals. "
                   "Negative importance is possible. This diagnostic uses a different period and fit from rolling-origin rankings. "
                   "Importance varies by window and does not prove causation.")
        download(weights, "model_explainability")
        download(perm, "permutation_diagnostic")
    with tabs[4]:
        st.markdown("""
        - **Direct ML:** separate Ridge and Gradient Boosting fit for each horizon. Inputs stop at the last completed month. FX predicts cumulative change from that month; inflation predicts the target month's m/m value.
        - **Statistical forecasts:** recursive ARIMA/SARIMA and ETS paths. Naive holds the last value; Seasonal Naive repeats the last 12 observed months.
        - **Windows:** expanding full history, expanding since September 2015, or the last 120 monthly observations. Lag inputs preceding a window are allowed; training anchors and labels are inside it.
        - **Fair evaluation:** the same origins and realized endpoints for all six models within each horizon. Endpoints differ across horizons. Labels for training anchors must be observed by the cutoff.
        - **Uncertainty:** only SARIMA has model-based 95% marginal prediction intervals. The log-FX point forecast is a transformed conditional median, not a bias-corrected mean.
        - **Limits:** historical final vintages and assumed releases; small dependent samples, structural shifts, multiple exploratory comparisons. No nested tuning or untouched final test is claimed.
        """)
        st.json({k: data["manifest"][k] for k in ("availability_policy", "strategy", "scale", "selection")})


def add_interval(figure, rows):
    supported = rows.dropna(subset=["lower", "upper"])
    if supported.empty:
        return
    figure.add_trace(go.Scatter(x=rows.date, y=rows.upper, mode="lines", line=dict(width=0), showlegend=False, connectgaps=False))
    figure.add_trace(go.Scatter(x=rows.date, y=rows.lower, mode="lines", line=dict(width=0), fill="tonexty",
                               fillcolor="rgba(0,163,217,.18)", name="SARIMA 95% model interval", connectgaps=False))


def show_future():
    page_header("Future Forecasting", "All six models · twelve target months · explicit information cutoffs.", eyebrow="Forecast outlook")
    data = bundle()
    if data is None:
        return
    target, window = controls("future")
    rows = data["future"].query("target == @target and window == @window").copy()
    cutoff, origin = rows.cutoff.iloc[0], rows.origin.iloc[0]
    st.caption(f"Last observed target month: {cutoff:%Y-%m} · assumed forecast origin: {origin:%Y-%m-%d} · "
               f"target months: {rows.date.min():%Y-%m} – {rows.date.max():%Y-%m}")
    st.info("Ridge and Gradient Boosting use twelve independent direct fits. Lines connect endpoint estimates; "
            "they are not recursive simulations or a joint predictive distribution. No future inflation, Brent, FX or rate values are supplied.")
    chosen = st.multiselect("Forecast models", list(MODEL_NAMES), default=["naive", "sarima", "lag_regression", "gradient_boosting"], format_func=MODEL_NAMES.get)
    months = st.slider("History months", 12, 120, 36)
    column = "inflation_mom" if target == "inflation_mom" else "usd_kzt_avg"
    history = data["input"].dropna(subset=[column]).tail(months)
    figure = go.Figure(go.Scatter(x=history.date, y=history[column], name="Observed", line=dict(color="#172B4D", width=3)))
    for model in chosen:
        block = rows[rows.model == model].sort_values("horizon")
        label = MODEL_NAMES[model]
        if block.status.ne("ok").any():
            st.warning(f"{label}: fit failed, explicitly displaying Naive fallback; see status below.")
            label += " (Naive fallback)"
        if model == "sarima":
            add_interval(figure, block)
        figure.add_trace(go.Scatter(x=block.date, y=block.predicted, name=label, mode="lines+markers", line=dict(dash="dash")))
    figure.update_layout(title=TARGET_NAMES[target], height=470, hovermode="x unified")
    st.plotly_chart(figure, width="stretch")
    st.caption("Only SARIMA provides supported model-based intervals. ML, ETS and baseline uncertainty bounds are unavailable. "
               "Intervals are marginal by month, not a simultaneous path envelope. Forecast months awaiting data may already be in the past.")
    selected = rows[rows.model.isin(chosen)]
    st.dataframe(labelled(selected)[["model", "horizon", "date", "predicted", "lower", "upper", "method", "status", "n_train", "train_start", "train_end"]], hide_index=True, width="stretch")
    download(selected, "future_forecasts")
