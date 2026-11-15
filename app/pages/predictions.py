"""Forecasts, backtest and model ranking from the saved results of src/models.py."""

from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_utils import load_model_results
from src.dashboard_data import show_forecast_metadata
from utils.ui import (
    GOLD, GREEN, MUTED, NAVY, RED, SKY, callout, delta_html, html, page_header, section, tiles,
)


TARGETS = {
    "inflation_mom": {
        "label": "Inflation (m/m)", "long": "monthly inflation", "column": "inflation_mom",
        "unit": "%", "digits": 2,
    },
    "usd_kzt": {
        "label": "USD/KZT exchange rate", "long": "the USD/KZT exchange rate",
        "column": "usd_kzt_avg", "unit": "₸", "digits": 1,
    },
}
MODELS = {
    # name, family, chart color
    "naive": ("Naive", "Baseline", MUTED),
    "seasonal_naive": ("Seasonal Naive", "Baseline", "#9AA7B8"),
    "arima": ("ARIMA / SARIMA", "Statistical", SKY),
    "sarima": ("ARIMA / SARIMA", "Statistical", SKY),
    "ets": ("ETS", "Statistical", GOLD),
    "lag_regression": ("Ridge", "Machine learning", "#7A5AF8"),
    "gradient_boosting": ("Gradient Boosting", "Machine learning", GREEN),
}
BADGE_CLASS = {"Baseline": "base", "Statistical": "stat", "Machine learning": "ml"}
HISTORY_WINDOWS = {"1 year": 1, "3 years": 3, "5 years": 5, "10 years": 10, "Since 2005": None}


def model_name(model):
    return MODELS.get(model, (model.replace("_", " ").title(),))[0]


def model_color(model):
    return MODELS.get(model, (None, None, MUTED))[2]


def rgba(hex_color, alpha):
    red, green, blue = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def fmt(value, info, sign=False):
    text = f"{value:+.{info['digits']}f}" if sign else f"{value:.{info['digits']}f}"
    if info["unit"] == "₸":
        return f"{text} ₸"
    return f"{text} p.p." if sign else f"{text}%"  # a change in a rate is in percentage points


def show_predictions():
    try:
        results = load_model_results()
    except (OSError, ValueError) as err:
        st.error(f"Cannot load consistent model artifacts: {err}")
        return
    if results is None:
        page_header("Forecasts", "Model results are not available yet.", eyebrow="Predictions")
        st.warning("Run `python main.py` and choose **6** (full pipeline), then reload this page.")
        return

    labels = {info["label"]: key for key, info in TARGETS.items()}
    choice = st.segmented_control(
        "Indicator", list(labels), default=list(labels)[0], label_visibility="collapsed",
    )
    target = labels[choice or list(labels)[0]]
    info = TARGETS[target]
    summary = results["summary"][target]
    metrics = results["metrics"][results["metrics"]["target"] == target]
    metrics = metrics.sort_values(["eligible", "MAE"], ascending=[False, True]) if "eligible" in metrics else metrics.sort_values("MAE")
    show_forecast_metadata(results, target)
    forecasts = results["forecasts"][results["forecasts"]["target"] == target]
    backtest = results["backtest"][results["backtest"]["target"] == target]
    history = results["monthly"].set_index("date")[info["column"]].dropna()

    test_start, test_end = (pd.Timestamp(day) for day in summary["test_period"])
    page_header(
        f"Forecast: {info['label']}",
        "Six models trained on historical data, tested month by month on data they never "
        "saw, and compared with naive baselines.",
        eyebrow="Predictions · out-of-sample evaluation",
        stamp=f"Test period {test_start:%b %Y} – {test_end:%b %Y}",
    )

    best = summary["best_model"]
    available = list(forecasts["model"].unique())
    ranked_available = [model for model in metrics["model"] if model in available]
    outlook_model = best if best in available else ranked_available[0]
    if best != outlook_model:
        st.info(f"Backtest winner: {model_name(best)}. Its future path is unavailable; "
                f"the outlook explicitly uses {model_name(outlook_model)}.")

    show_headline(metrics, forecasts, history, summary, info, best, outlook_model, target)

    outlook_tab, backtest_tab, ranking_tab, method_tab = st.tabs(
        ["Outlook", "Backtest", "Model ranking", "Methodology"]
    )
    with outlook_tab:
        show_outlook(history, forecasts, ranked_available, outlook_model, best, info, target)
    with backtest_tab:
        show_backtest(backtest, metrics, best, info, target)
    with ranking_tab:
        show_ranking(metrics, backtest, results["errors_by_year"], best, info, target)
    with method_tab:
        show_method(summary)


def show_headline(metrics, forecasts, history, summary, info, best, outlook_model, target):
    best_row = metrics.iloc[0]
    naive_mae = metrics.loc[metrics["model"] == "naive", "MAE"].iloc[0]
    gain = (1 - best_row["MAE"] / naive_mae) * 100
    path = forecasts[forecasts["model"] == outlook_model].sort_values("date")
    last_value = history.iloc[-1]
    next_month = path.iloc[0]
    horizon = path.iloc[-1]

    if target == "inflation_mom":
        projected = summary.get("projected_12m_inflation_pct", {}).get(outlook_model)
        horizon_tile = {
            "label": "Inflation, next 12 months", "value": f"{projected:.1f}", "unit": "%",
            "delta": delta_html(None, "cumulative, compounded"),
            "note": f"{model_name(outlook_model)} · to {horizon['date']:%b %Y}",
        }
    else:
        change = (horizon["forecast"] / last_value - 1) * 100
        horizon_tile = {
            "label": f"Forecast for {horizon['date']:%b %Y}", "value": f"{horizon['forecast']:.1f}",
            "unit": "₸", "delta": delta_html(change, f"{change:+.1f}% vs latest", good_when_up=False),
            "note": f"{model_name(outlook_model)} · 12 months ahead",
        }

    next_change = next_month["forecast"] - last_value
    tiles([
        {
            "label": "Lowest MAE on test", "value": model_name(best),
            "delta": delta_html(gain, f"{gain:.1f}% lower error than naive"),
            "note": f"Ranked 1st of {len(metrics)} on the test period",
        },
        {
            "label": "Average error (MAE)", "value": f"{best_row['MAE']:.{info['digits']}f}",
            "unit": info["unit"],
            "delta": delta_html(None, f"RMSE {best_row['RMSE']:.{info['digits']}f}"),
            "note": "One-month-ahead forecasts, 36 months",
        },
        {
            "label": f"Forecast for {next_month['date']:%b %Y}",
            "value": f"{next_month['forecast']:.{info['digits']}f}", "unit": info["unit"],
            "delta": delta_html(next_change, f"{fmt(next_change, info, sign=True)} vs latest actual",
                                good_when_up=False),
            "note": f"{model_name(outlook_model)} · latest actual {fmt(last_value, info)}",
        },
        horizon_tile,
    ])

    if target == "inflation_mom":
        message = (
            f"<b>{model_name(outlook_model)}</b> expects prices to rise about "
            f"<b>{summary['projected_12m_inflation_pct'][outlook_model]:.1f}%</b> over the next 12 "
            f"target months after the last observation. On the test period, <b>{model_name(best)}</b> "
            f"achieved <b>{gain:.1f}%</b> lower MAE than simply repeating last month's value."
        )
    else:
        message = (
            f"<b>{model_name(outlook_model)}</b> projects <b>{horizon['forecast']:.1f} ₸</b> per US "
            f"dollar by {horizon['date']:%B %Y}. On the test period, "
            f"<b>{model_name(best)}</b> beat the naive forecast by <b>{gain:.1f}%</b>, an average "
            f"miss of {best_row['MAE']:.1f} ₸ per month."
        )
        if best != outlook_model:
            message += (f" {model_name(best)} only forecasts one month ahead, so the 12-month "
                        f"path uses {model_name(outlook_model)}.")
    callout(message)


def show_outlook(history, forecasts, available, outlook_model, best, info, target):
    left, right = st.columns([3, 1])
    models = left.multiselect(
        "Models", available, default=[outlook_model] + (["naive"] if outlook_model != "naive" else []),
        format_func=model_name, key=f"fc_{target}",
    )
    window = right.selectbox("History", list(HISTORY_WINDOWS), index=1, key=f"hist_{target}")
    years = HISTORY_WINDOWS[window]
    shown = history if years is None else history[history.index >= history.index[-1] - pd.DateOffset(years=years)]

    last_date, last_value = history.index[-1], history.iloc[-1]
    figure = go.Figure(go.Scatter(
        x=shown.index, y=shown.values, name="Actual", line=dict(color=NAVY, width=2.4),
    ))
    if not forecasts.empty:
        figure.add_vrect(x0=last_date, x1=forecasts["date"].max(), fillcolor=rgba(GOLD, 0.07),
                         line_width=0, layer="below")
        figure.add_vline(x=last_date, line=dict(color=MUTED, width=1, dash="dot"))
        figure.add_annotation(x=last_date, y=1, yref="paper", text="Forecast →", showarrow=False,
                              xanchor="left", xshift=6, font=dict(color=MUTED, size=12))
    for model in models:
        rows = forecasts[forecasts["model"] == model].sort_values("date")
        color = model_color(model)
        if rows["lower"].notna().all():
            figure.add_trace(go.Scatter(
                x=list(rows["date"]) + list(rows["date"][::-1]),
                y=list(rows["upper"]) + list(rows["lower"][::-1]),
                fill="toself", fillcolor=rgba(color, 0.15), line=dict(width=0),
                hoverinfo="skip", name=f"{model_name(model)} · 95% interval",
            ))
        figure.add_trace(go.Scatter(
            x=[last_date] + list(rows["date"]), y=[last_value] + list(rows["forecast"]),
            name=model_name(model), mode="lines+markers", marker=dict(size=5),
            line=dict(color=color, width=2.2, dash="dash"),
        ))
    figure.update_layout(height=470, hovermode="x unified", yaxis_title=info["unit"],
                         title=f"{info['label']}: history and 12-month forecast")
    st.plotly_chart(figure, width="stretch")

    if not models:
        st.caption("Select at least one model to show a forecast.")
        return
    section("Forecast values", "Monthly point forecasts with 95% interval where the model provides one")
    table = forecasts[forecasts["model"].isin(models)].copy()
    table["Month"] = table["date"].dt.strftime("%b %Y")
    wide = table.pivot(index="Month", columns="model", values="forecast")
    wide = wide.loc[table.drop_duplicates("Month").sort_values("date")["Month"]]
    for model in models:
        rows = table[table["model"] == model].set_index("Month")
        if rows["lower"].notna().all():
            wide[f"{model}_interval"] = (rows["lower"].round(info["digits"]).astype(str) + " – "
                                         + rows["upper"].round(info["digits"]).astype(str))
    wide.columns = [model_name(col) if col in MODELS else model_name(col.removesuffix("_interval"))
                    + " · 95% interval" for col in wide.columns]
    st.dataframe(wide.round(info["digits"]), width="stretch")
    st.download_button("Download forecast (CSV)", wide.to_csv().encode("utf-8"),
                       file_name=f"forecast_{target}.csv", mime="text/csv")


def direction_accuracy(backtest):
    """Share of months where the model got the direction of change right."""
    actual = backtest.drop_duplicates("date").set_index("date")["actual"].sort_index()
    previous = actual.shift(1)
    scores = {}
    for model, rows in backtest.groupby("model"):
        rows = rows.set_index("date").sort_index()
        prev = previous.reindex(rows.index)
        valid = prev.notna()
        predicted = np.sign(rows["forecast"] - prev)[valid]
        if (predicted == 0).all():
            continue  # the naive model never predicts a change, so it has no direction score
        scores[model] = (predicted == np.sign(rows["actual"] - prev)[valid]).mean() * 100
    return scores


def direction_text(directions, model):
    return f"{directions[model]:.0f}%" if model in directions else "—"


def show_backtest(backtest, metrics, best, info, target):
    callout("Each month the model was refit on all earlier data and asked to predict only the "
            "next month, using final-vintage data and documented publication assumptions.")
    models = st.multiselect(
        "Models", list(metrics["model"]), default=[best] + (["naive"] if best != "naive" else []),
        format_func=model_name, key=f"bt_{target}",
    )
    actual = backtest.drop_duplicates("date").sort_values("date")
    figure = go.Figure(go.Scatter(x=actual["date"], y=actual["actual"], name="Actual",
                                  line=dict(color=NAVY, width=3)))
    for model in models:
        rows = backtest[backtest["model"] == model].sort_values("date")
        figure.add_trace(go.Scatter(x=rows["date"], y=rows["forecast"], name=model_name(model),
                                    line=dict(color=model_color(model), width=2, dash="dash")))
    figure.update_layout(height=420, hovermode="x unified", yaxis_title=info["unit"],
                         title="Predicted vs actual, one month ahead")
    st.plotly_chart(figure, width="stretch")

    if not models:
        return
    errors = backtest[backtest["model"].isin(models)].assign(
        error=lambda frame: frame["forecast"] - frame["actual"],
    )
    figure = go.Figure()
    for model in models:
        rows = errors[errors["model"] == model]
        figure.add_trace(go.Bar(x=rows["date"], y=rows["error"], name=model_name(model),
                                marker_color=model_color(model)))
    figure.add_hline(y=0, line=dict(color=MUTED, width=1))
    figure.update_layout(height=300, barmode="group", yaxis_title=info["unit"],
                         title="Forecast error (above zero = forecast too high)")
    st.plotly_chart(figure, width="stretch")


def show_ranking(metrics, backtest, errors_by_year, best, info, target):
    naive_mae = metrics.loc[metrics["model"] == "naive", "MAE"].iloc[0]
    directions = direction_accuracy(backtest)
    worst_gain = max(abs(1 - row.MAE / naive_mae) for row in metrics.itertuples()) or 1
    digits = info["digits"] + 1

    rows = []
    for rank, row in enumerate(metrics.itertuples(), start=1):
        family = MODELS.get(row.model, ("", "Statistical"))[1]
        badges = f'<span class="kz-badge {BADGE_CLASS.get(family, "stat")}">{family}</span>'
        if row.model == best:
            badges += ' <span class="kz-badge best">Best</span>'
        gain = (1 - row.MAE / naive_mae) * 100
        width = min(abs(gain) / (worst_gain * 100) * 100, 100)
        bar_color = GREEN if gain > 0 else RED if gain < 0 else MUTED
        rows.append(
            f'<tr class="{"best" if row.model == best else ""}">'
            f"<td><b>{rank}</b></td>"
            f"<td><b>{escape(model_name(row.model))}</b><br>{badges}</td>"
            f'<td class="r">{row.MAE:.{digits}f}</td>'
            f'<td class="r">{row.RMSE:.{digits}f}</td>'
            f'<td class="r">{row.MASE:.2f}</td>'
            f'<td class="r">{direction_text(directions, row.model)}</td>'
            f'<td><span class="kz-bar"><span style="width:{width:.0f}%;background:{bar_color}"></span></span>'
            f'<span style="color:{bar_color};font-weight:600">{gain:+.1f}%</span></td>'
            "</tr>"
        )
    html(
        '<table class="kz-table"><thead><tr><th>#</th><th>Model</th>'
        f'<th class="r">MAE, {escape(info["unit"])}</th><th class="r">RMSE</th>'
        '<th class="r">MASE</th><th class="r">Direction</th><th>vs naive</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )
    st.caption(
        "Reliable runs first, then MAE (average absolute miss). Failed-fit fallbacks are flagged above. RMSE punishes large misses more. "
        "MASE uses the training naive scale (seasonal for inflation), not test-baseline superiority. Direction = share of months where the "
        "model correctly predicted whether the value would go up or down."
    )

    section("Error by year", "MAE of each model in each year of the test period; darker = larger error")
    yearly = errors_by_year[errors_by_year["target"] == target].drop(columns="target")
    yearly = yearly.dropna(axis=1, how="all").set_index("year")
    yearly = yearly[[model for model in metrics["model"] if model in yearly.columns]]
    figure = px.imshow(
        yearly.T.values, x=[str(year) for year in yearly.index],
        y=[model_name(model) for model in yearly.columns],
        color_continuous_scale=["#F3F8FD", SKY, NAVY], text_auto=f".{digits}f", aspect="auto",
        labels=dict(color=f"MAE, {info['unit']}"),
    )
    figure.update_layout(height=360, xaxis_title="", yaxis_title="")
    st.plotly_chart(figure, width="stretch")


def show_method(summary):
    order = summary.get("arima_order")
    steps = [
        ("Prepare the series", "Daily rates are averaged to months; inflation is the "
         "month-over-month change of the consumer price index."),
        ("Handle gaps", "Stage 1 forecasts use past-only forward filling. Historical charts retain "
         "flagged interpolation; legacy forecasts are identified above."),
        ("Split by time", "The last 36 months form the test set. Models never see data from the "
         "future target observations. Publication timing is assumed; vintage revisions remain a limitation."),
        ("Train models", "Two naive baselines, ARIMA/SARIMA"
         + (f" (order {tuple(order)})" if order else "")
         + ", exponential smoothing, lag regression and gradient boosting."),
        ("Evaluate", "Rolling one-month-ahead forecasts scored with MAE, RMSE, sMAPE and MASE, "
         "and compared with the naive baselines."),
        ("Forecast", "Available models refit on the target history predict 12 target months, with model-based 95% intervals "
         "where the model provides them."),
    ]
    for start in range(0, len(steps), 3):
        for column, (number, (title, text)) in zip(
            st.columns(3), enumerate(steps[start:start + 3], start=start + 1)
        ):
            with column:
                html(f'<div class="kz-card" style="margin-bottom:14px"><div class="num">{number}</div>'
                     f"<h4>{escape(title)}</h4><p>{escape(text)}</p></div>")
    callout(
        "<b>Limitations.</b> Forecasts assume past patterns continue. Shocks such as oil-price "
        "crashes or devaluations (2009, 2014, 2015, 2022) cannot be predicted from history "
        "alone, and intervals widen quickly beyond a few months.",
        warn=True,
    )
