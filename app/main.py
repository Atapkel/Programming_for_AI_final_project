"""Homepage for the Programming for AI final project."""

import streamlit as st


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
    st.button("Home", type="primary", width="stretch")
    # These sections are placeholders until their pages are implemented.
    for page in ("Data Analysis", "Visualization", "Predictions"):
        st.button(page, disabled=True, width="stretch")
    st.caption("More pages coming soon.")

st.caption("PROGRAMMING FOR AI · FINAL PROJECT")
st.title("Kazakhstan Economy Explorer")
st.write(
    "A project exploring Kazakhstan's economy through USD/KZT exchange rates, "
    "inflation, interest rates, oil prices, and annual economic indicators. "
    "Future sections will bring together data analysis, visualizations, and predictions."
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
    "You're on the project homepage. Data Analysis, Visualization, and "
    "Predictions are planned for a future update."
)
