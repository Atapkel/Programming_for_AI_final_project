"""Overview of future forecasting work."""

import streamlit as st


def show_predictions():
    st.title("Predictions")
    st.info("Coming soon")
    st.write(
        "Future work will explore forecasting Kazakhstan's economic indicators "
        "using historical data. No prediction models or saved results are available yet."
    )
    st.subheader("Planned forecasting workflow")
    st.markdown(
        "1. Select an economic indicator and prepare its time series.\n"
        "2. Handle missing values and split the data chronologically.\n"
        "3. Train a simple baseline forecasting model.\n"
        "4. Evaluate forecasts on held-out data using MAE and RMSE.\n"
        "5. Compare predicted and actual values in a chart."
    )
