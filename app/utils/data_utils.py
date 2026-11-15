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

WORLD_BANK_UNITS = {
    "exports_pct_gdp": "% of GDP", "gdp_current_usd": "current USD",
    "gdp_growth_pct": "% annual growth", "inflation_cpi_pct": "% annual CPI inflation",
    "unemployment_pct": "% of labor force", "usd_kzt_annual_avg": "KZT per USD",
}




from src.dashboard_data import (
    load_csv, load_model_results as _load_model_results,
    load_market_data as _load_market_data, MODEL_FILES, MARKET_FILES,
)
from src.paths import PROCESSED_DIR


def load_dataset(name):
    return load_csv(RAW_DIR / DATASETS[name])


def load_model_results():
    return _load_model_results(PROCESSED_DIR)


def load_market_data():
    return _load_market_data(PROCESSED_DIR)


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
