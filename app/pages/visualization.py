"""Interactive charts with time filters."""

import plotly.express as px
import streamlit as st

from utils.data_utils import select_dataset


def show_visualization():
    st.title("Visualization")
    st.write("Explore trends with interactive charts.")
    name, data = select_dataset()
    chart_title = name
    if "indicator" in data.columns:
        indicator = st.selectbox("Indicator", sorted(data["indicator"].dropna().unique()))
        data = data[data["indicator"] == indicator]
        chart_title = f"{name}: {indicator.replace('_', ' ').title()}"
    x_columns = [column for column in ("date", "year") if column in data.columns]
    if not x_columns:
        st.info("This dataset has no date or year column to plot.")
        return
    x_column = st.selectbox("X-axis", x_columns)
    data = data.dropna(subset=[x_column]).sort_values(x_column)
    if data.empty:
        st.info("No valid dates are available for this dataset.")
        return
    if x_column == "date":
        first, last = data[x_column].min().date(), data[x_column].max().date()
        dates = st.date_input(
            "Date range", value=(first, last), min_value=first, max_value=last,
            key=f"dates_{name}",
        )
        if len(dates) != 2:
            st.info("Select both a start and an end date.")
            return
        start, end = dates
        data = data[data[x_column].dt.date.between(start, end)]
        period = f"{start} to {end}"
    else:
        first, last = int(data[x_column].min()), int(data[x_column].max())
        start, end = (first, last)
        if first < last:
            start, end = st.slider(
                "Year range", first, last, (first, last), key=f"years_{indicator}",
            )
        data = data[data[x_column].between(start, end)]
        period = f"{start}–{end}"
    chart_type = st.selectbox("Chart type", ["Line", "Bar", "Scatter"])
    numeric_columns = [
        column for column in data.select_dtypes(include="number").columns
        if column != x_column
    ]
    y_columns = st.multiselect("Y-axis columns", numeric_columns, default=numeric_columns[:1])
    if not y_columns:
        st.info("Select at least one Y-axis column to display a chart.")
        return
    chart_data = data.dropna(subset=[x_column]).sort_values(x_column)
    if chart_data[y_columns].dropna(how="all").empty:
        st.info("No values are available for the selected columns.")
        return
    make_chart = {"Line": px.line, "Bar": px.bar, "Scatter": px.scatter}[chart_type]
    figure = make_chart(
        chart_data, x=x_column, y=y_columns,
        title=f"{chart_title} · {period}", template="plotly_dark",
        labels={x_column: x_column.title(), "value": "Value", "variable": "Series"},
    )
    st.plotly_chart(figure, width="stretch")
    st.caption("Hover for values, drag to zoom, and double-click to reset. Columns may use different units.")

