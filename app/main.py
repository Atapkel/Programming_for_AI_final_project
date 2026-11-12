"""Homepage for the Programming for AI final project."""

import streamlit as st

from pages.analysis import show_analysis
from pages.predictions import show_predictions
from pages.visualization import show_visualization


st.set_page_config(
    page_title="Kazakhstan Economy | Programming for AI",
    page_icon="🇰🇿",
    layout="wide",
)

with st.sidebar:
    st.title("Programming for AI")
    st.caption("Final project")
    st.divider()
    st.subheader("Navigation")
    page = st.radio(
        "Page", ["Home", "Data Analysis", "Visualization", "Predictions"],
        label_visibility="collapsed",
    )

if page == "Data Analysis":
    show_analysis()
    st.stop()
if page == "Visualization":
    show_visualization()
    st.stop()
if page == "Predictions":
    show_predictions()
    st.stop()

st.caption("PROGRAMMING FOR AI · FINAL PROJECT")
st.title("Kazakhstan Economy Explorer")
st.write(
    "A project exploring Kazakhstan's economy through USD/KZT exchange rates, "
    "inflation, interest rates, oil prices, and annual economic indicators. "
    "Explore the datasets and visualizations, with predictions planned for a future update."
)
st.divider()

st.subheader("Project overview")
cards = [
    (
        "Exchange & interest rates",
        "Daily USD/KZT exchange rates and base rate decisions from the "
        "National Bank of Kazakhstan.",
    ),
    (
        "Inflation & oil prices",
        "Monthly consumer price index from the IMF and daily Brent crude "
        "oil prices from FRED.",
    ),
    (
        "Economic indicators",
        "Annual GDP, inflation, unemployment, and export indicators from "
        "the World Bank.",
    ),
]

for column, (title, description) in zip(st.columns(3), cards):
    with column:
        with st.container(border=True):
            st.subheader(title)
            st.write(description)

st.info(
    "Choose Data Analysis or Visualization in the sidebar to explore the data. "
    "Prediction functionality is coming soon."
)
