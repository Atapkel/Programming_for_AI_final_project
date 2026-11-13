# Kazakhstan Macroeconomic Indicators and Exchange Rate

Final project, Programming for AI — topic 59 (Cluster 10: Forecasting & Open Data Analytics).

**Team (AAI-2604):** Yerassyl Iskakov, Atapkel Beibarys, Aitymov Altair

**Goal:** analyze the dynamics of the USD/KZT exchange rate, inflation and the National Bank
base rate, find structural breaks (2009, 2014, 2015, 2022), study the link with oil prices, and
forecast the key indicators against naive baselines.

## Data

| Series | Frequency | Period | Source |
|---|---|---|---|
| Official USD/KZT rate | daily | 2000 – today | [National Bank of Kazakhstan](https://nationalbank.kz) |
| Base rate decisions | per decision | Sep 2015 – today | [NBK decision schedule](https://nationalbank.kz/en/news/grafik-prinyatiya-resheniy-po-bazovoy-stavke) |
| Consumer price index (2020 = 100) | monthly | 2005 – today | [IMF CPI dataset](https://data.imf.org/en/datasets/IMF.STA:CPI) |
| Brent crude oil price | daily | 2000 – today | [FRED, DCOILBRENTEU](https://fred.stlouisfed.org/series/DCOILBRENTEU) |
| GDP, inflation, unemployment, exports | annual | 1990 – 2025 | [World Bank Open Data](https://data.worldbank.org/country/KZ) |

All data is downloaded by `src/data_loader.py` and is also committed in `data/raw/`, so the
project runs without internet access.

Known data limitations:
- USD/KZT has no official values for 28.07.2023 – 28.08.2023 (missing in the NBK archive).
  These days are filled by time interpolation and flagged with `is_filled = 1`.
- The base rate exists only since September 2015; before that NBK used the refinancing rate.
- Monthly CPI starts in 2005.

## Project structure

```
main.py                  console menu (runs every step)
app.py                   Streamlit dashboard
src/
  data_loader.py         download raw data (NBK, IMF, FRED, World Bank)
  preprocessing.py       cleaning, gap filling, monthly table, data audit, CSV vs Parquet
  database.py            SQLite schema, loading, 11 SQL queries
  analysis.py            stationarity tests, STL, ACF/PACF, Chow tests, figures
  models.py              backtest of 6 forecasting models, 12-month forecast
data/
  raw/                   downloaded CSV files
  processed/             Parquet tables, SQL results, metrics, forecasts
  macro.db               SQLite database
docs/er_diagram.md       ER diagram of the database
reports/figures/         charts for the report (PNG + interactive HTML)
```

## Setup

Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py                     # console menu
```

Menu options: 1 download data · 2 preprocess · 3 build database + SQL · 4 analysis ·
5 models · 6 full pipeline (2–5) · 7 show an SQL query result · 8 open dashboard.

Choose **6** once to build everything (about 1 minute), then **8** for the dashboard.
Each step can also be run directly:

```bash
python -m src.data_loader          # add --force to download again
python -m src.preprocessing
python -m src.database
python -m src.analysis
python -m src.models
streamlit run app.py
```

## Methods

- **Preprocessing:** daily series reindexed to a full calendar, gaps interpolated and flagged;
  monthly table with averages, end-of-month values, m/m and y/y inflation, base rate in force,
  real rate and exchange-rate regime. Data audit and memory/format measurements in
  `data/processed/quality_report.json`.
- **Database:** normalized schema `sources → indicators → observations` plus the derived
  `monthly_macro` table ([ER diagram](docs/er_diagram.md)). 11 queries with JOINs,
  GROUP BY/HAVING, a subquery, CTEs and window functions (`LAG`, `RANK`, rolling `AVG`).
- **Time-series analysis:** ADF and KPSS stationarity tests, STL decomposition, ACF/PACF,
  correlation of monthly changes, exchange-rate pass-through by lag, Chow tests at the
  known break dates.
- **Forecasting:** expanding-window one-step-ahead backtest on the last 36 months (models are
  refit every month on past data only). Models: naive, seasonal naive, SARIMA/ARIMA
  (order chosen by AIC), ETS, Ridge lag regression, gradient boosting. Metrics: MAE, RMSE,
  sMAPE, MASE, fit time, MAE by year.

## Main results

- Largest tenge depreciations: 2015 (+86% Dec–Dec, +35.5% on 21.08.2015), 2009 (+22.8%),
  2014 (+18.7%).
- After the 2015 float the average monthly exchange-rate move tripled (0.73% → 2.18%).
- The exchange rate passes into inflation with a 1–2 month lag (correlation 0.31 at lag 1,
  about 0 after 4 months).
- Monthly inflation shows a significant structural break in 2022 (Chow test p < 0.001).
- Inflation forecast: SARIMA has the lowest MAE (0.182 vs 0.200 for naive, MASE 0.38).
  USD/KZT forecast: gradient boosting MAE 7.8 KZT vs 8.8 for naive. Gains over the naive
  baseline are modest (9–12%), as is typical for inflation and exchange rates.
