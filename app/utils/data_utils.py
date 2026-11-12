"""Shared local dataset selection and cached loading."""

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
