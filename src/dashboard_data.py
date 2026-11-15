"""Version-aware readers shared by both dashboards, including legacy snapshots."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.artifacts import ArtifactError, assert_idle, fingerprint, verify_manifest
from src.paths import PROCESSED_DIR

MODEL_FILES = ("forecasts.parquet", "backtest_predictions.parquet", "model_metrics.csv",
               "errors_by_year.csv", "forecast_summary.json")
MARKET_FILES = {"fx": "usd_kzt_daily.parquet", "base_rate": "base_rate.parquet",
                "brent": "brent_daily.parquet", "monthly": "monthly_macro.parquet"}


def _check_idle(directory):
    assert_idle(directory.parent)  # full pipeline publishes under the data lock
    assert_idle(directory)


@st.cache_data
def _read_csv(path, version):
    data = pd.read_csv(path)
    if "date" in data:
        data["date"] = pd.to_datetime(data.date, errors="raise")
    if fingerprint([path]) != version:
        raise ArtifactError("Dataset changed while loading; reload the page")
    return data


def load_csv(path):
    path = Path(path)
    _check_idle(path.parent)
    return _read_csv(str(path), fingerprint([path]))


@st.cache_data
def _read_models(directory, input_name, version):
    directory = Path(directory)
    run = verify_manifest(directory, "model_manifest.json")
    result = {
        "forecasts": pd.read_parquet(directory / "forecasts.parquet"),
        "backtest": pd.read_parquet(directory / "backtest_predictions.parquet"),
        "metrics": pd.read_csv(directory / "model_metrics.csv"),
        "errors_by_year": pd.read_csv(directory / "errors_by_year.csv"),
        "monthly": pd.read_parquet(directory / input_name),
        "summary": json.loads((directory / "forecast_summary.json").read_text(encoding="utf-8")),
        "manifest": run,
    }
    paths = [directory / name for name in (*MODEL_FILES, input_name, "model_manifest.json", "model_diagnostics.json")]
    _check_idle(directory)
    if fingerprint(paths) != version:
        raise ArtifactError("Model artifacts changed while loading; reload the page")
    return result


def load_model_results(directory=None):
    directory = Path(directory) if directory is not None else PROCESSED_DIR
    _check_idle(directory)
    input_name = "model_input.parquet" if (directory / "model_manifest.json").exists() else "monthly_macro.parquet"
    # Missing output is deliberately checked OUTSIDE the cached function.
    if not all((directory / name).exists() for name in (*MODEL_FILES, input_name)):
        return None
    paths = [directory / name for name in (*MODEL_FILES, input_name, "model_manifest.json", "model_diagnostics.json")]
    return _read_models(str(directory), input_name, fingerprint(paths))


@st.cache_data
def _read_market(directory, version):
    directory = Path(directory)
    verify_manifest(directory, "preprocessing_manifest.json")
    result = {key: pd.read_parquet(directory / name) for key, name in MARKET_FILES.items()}
    _check_idle(directory)
    paths = [directory / name for name in (*MARKET_FILES.values(), "preprocessing_manifest.json")]
    if fingerprint(paths) != version:
        raise ArtifactError("Market data changed while loading; reload the page")
    return result


def load_market_data(directory=None):
    directory = Path(directory) if directory is not None else PROCESSED_DIR
    _check_idle(directory)
    if not all((directory / name).exists() for name in MARKET_FILES.values()):
        return None
    paths = [directory / name for name in (*MARKET_FILES.values(), "preprocessing_manifest.json")]
    return _read_market(str(directory), fingerprint(paths))


@st.cache_data
def _read_stage2(directory, version):
    from src.evaluation import OUTPUTS
    directory = Path(directory)
    run = verify_manifest(directory, "manifest.json")
    if run is None:
        raise ArtifactError("Stage 2 manifest is required")
    result = {name.removesuffix(".parquet"): pd.read_parquet(directory / name)
              for name in OUTPUTS if name.endswith(".parquet")}
    result.update({name.removesuffix(".csv"): pd.read_csv(directory / name)
                   for name in OUTPUTS if name.endswith(".csv")})
    result["manifest"] = run
    _check_idle(directory.parent)
    if fingerprint([directory / name for name in (*OUTPUTS, "manifest.json")]) != version:
        raise ArtifactError("Stage 2 artifacts changed while loading; reload the page")
    return result


def load_stage2(directory=None):
    from src.evaluation import OUTPUTS
    from src.artifacts import digest
    directory = Path(directory) if directory is not None else PROCESSED_DIR / "stage2"
    _check_idle(directory.parent)
    if not all((directory / name).exists() for name in (*OUTPUTS, "manifest.json")):
        return None
    result = _read_stage2(str(directory), fingerprint([directory / name for name in (*OUTPUTS, "manifest.json")]))
    current = directory.parent / "monthly_forecasting.parquet"
    recorded = result["manifest"]["inputs"]["monthly_forecasting"]
    result["stale"] = not current.exists() or digest(current) != recorded
    return result


def load_analysis_artifacts():
    _check_idle(PROCESSED_DIR)
    verify_manifest(PROCESSED_DIR, "preprocessing_manifest.json")
    verify_manifest(PROCESSED_DIR, "analysis_manifest.json")
    paths = [PROCESSED_DIR / name for name in ("quality_report.json", "analysis_results.json")]
    before = fingerprint(paths)
    result = [json.loads(path.read_text()) for path in paths]
    _check_idle(PROCESSED_DIR)
    if fingerprint(paths) != before:
        raise ArtifactError("Analysis artifacts changed while loading; reload the page")
    return result


def show_forecast_metadata(results, target):
    """Use stored origin metadata; never invent an origin for legacy results."""
    info = results["summary"][target]
    column = "inflation_mom" if target == "inflation_mom" else "usd_kzt_avg"
    latest = results["monthly"].dropna(subset=[column]).date.max()
    forecasts = results["forecasts"].query("target == @target")
    origin = info.get("forecast_origin", "not recorded (legacy results)")
    st.caption(f"Last observed target month: {latest:%Y-%m}. "
               f"Forecast origin: {origin}. Target months: "
               f"{forecasts.date.min():%Y-%m} – {forecasts.date.max():%Y-%m}. "
               "Months awaiting publication may already be in the past.")
    if results.get("manifest"):
        run = results["manifest"]
        st.caption(f"Generated (UTC): {run['created_at_utc']} · Run: {run['run_id']} · "
                   "Origin assumes previous-month inputs are available by the 15th; "
                   "final-vintage retrospective evaluation, not verified real-time availability.")
    else:
        st.warning("Legacy results: no experiment manifest or verified convergence metadata. "
                   "Rebuild the pipeline to apply Stage 1 safeguards.")
    fallback = {k: v for k, v in info.get("backtest_fallbacks", {}).items() if v}
    if fallback:
        st.warning(f"Failed fits replaced explicitly by naive in backtest: {fallback}. "
                   "These model runs are excluded from best-model selection.")
    for model, reason in info.get("forecast_failures", {}).items():
        st.warning(f"Future forecast unavailable for {model}: {reason}")
