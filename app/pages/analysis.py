"""Dataset summaries and downloads."""

import pandas as pd
import streamlit as st

from utils.data_utils import DATASETS, select_dataset


def show_analysis():
    st.title("Data Analysis")
    st.write("Preview the local datasets and explore their basic statistics.")
    name, data = select_dataset()
    left, right = st.columns(2)
    left.metric("Rows", f"{len(data):,}")
    right.metric("Columns", len(data.columns))
    st.subheader("Dataset information")
    st.write(f"**{name}** · Source file: `{DATASETS[name]}`")
    time_column = "date" if "date" in data.columns else "year"
    times = data[time_column].dropna()
    if not times.empty:
        if time_column == "date":
            st.write(f"Date range: {times.min():%Y-%m-%d} to {times.max():%Y-%m-%d}")
        else:
            st.write(f"Year range: {int(times.min())} to {int(times.max())}")
    st.dataframe(
        data.dtypes.astype(str).rename("Data type").rename_axis("Column").reset_index(),
        hide_index=True, width="stretch",
    )
    st.download_button(
        "Download dataset as CSV", data.to_csv(index=False).encode("utf-8"),
        file_name=DATASETS[name], mime="text/csv",
    )
    st.subheader("Data preview")
    st.caption("First 100 rows")
    st.dataframe(data.head(100), hide_index=True, width="stretch")
    st.subheader("Basic statistics")
    st.dataframe(data.describe(include="all").T, width="stretch")
    st.subheader("Missing values")
    missing = pd.DataFrame({
        "Column": data.columns,
        "Missing values": data.isna().sum().values,
        "Missing (%)": (data.isna().mean() * 100).round(2).values,
    })
    st.dataframe(missing, hide_index=True, width="stretch")

