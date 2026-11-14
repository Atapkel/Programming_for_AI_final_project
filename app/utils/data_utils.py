"""Shared local dataset selection and cached loading."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit starts in app/; add the project root to import the existing loader.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_loader import RAW_DIR


DATASETS = {
    "USD/KZT exchange rate": "usd_kzt_daily.csv",
    "Base interest rate": "base_rate.csv",
    "Consumer price index": "cpi_monthly.csv",
    "Brent oil price": "brent_daily.csv",
    "World Bank indicators": "world_bank_annual.csv",
}




@st.cache_data
def load_dataset(name):
    """Cache local CSV data without modifying or downloading source files."""
    data = pd.read_csv(RAW_DIR / DATASETS[name])
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data


PROCESSED_DIR = RAW_DIR.parent / "processed"
MODEL_FILES = (
    "forecasts.parquet", "backtest_predictions.parquet", "model_metrics.csv",
    "errors_by_year.csv", "forecast_summary.json", "monthly_macro.parquet",
)


@st.cache_data
def load_model_results():
    """Load the saved outputs of src/models.py; return None if they are missing."""
    if not all((PROCESSED_DIR / name).exists() for name in MODEL_FILES):
        return None
    with open(PROCESSED_DIR / "forecast_summary.json", encoding="utf-8") as file:
        summary = json.load(file)
    return {
        "forecasts": pd.read_parquet(PROCESSED_DIR / "forecasts.parquet"),
        "backtest": pd.read_parquet(PROCESSED_DIR / "backtest_predictions.parquet"),
        "metrics": pd.read_csv(PROCESSED_DIR / "model_metrics.csv"),
        "errors_by_year": pd.read_csv(PROCESSED_DIR / "errors_by_year.csv"),
        "monthly": pd.read_parquet(PROCESSED_DIR / "monthly_macro.parquet"),
        "summary": summary,
    }


MARKET_FILES = {
    "fx": "usd_kzt_daily.parquet",
    "base_rate": "base_rate.parquet",
    "brent": "brent_daily.parquet",
    "monthly": "monthly_macro.parquet",
}


@st.cache_data
def load_market_data():
    """Latest processed market tables for the home page; None if not built yet."""
    if not all((PROCESSED_DIR / name).exists() for name in MARKET_FILES.values()):
        return None
    return {key: pd.read_parquet(PROCESSED_DIR / name) for key, name in MARKET_FILES.items()}


def select_dataset():
    name = st.selectbox("Dataset", list(DATASETS))
    try:
        data = load_dataset(name)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not read this dataset: {error}")
        st.stop()
    if data.empty:
        st.info("This dataset has no rows yet.")
        st.stop()
    return name, data
