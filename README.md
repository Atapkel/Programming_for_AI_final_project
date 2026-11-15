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
app.py                   compatibility launcher → app/main.py
app/main.py              unified Streamlit dashboard (8 sections)
src/
  data_loader.py         download raw data (NBK, IMF, FRED, World Bank)
  preprocessing.py       cleaning, gap filling, monthly table, data audit, CSV vs Parquet
  database.py            SQLite schema, loading, 11 SQL queries
  analysis.py            stationarity tests, STL, ACF/PACF, Chow tests, figures
  models.py              backtest of 6 forecasting models, 12-month forecast
  forecasting.py         direct ML and statistical forecast adapters
  evaluation.py          rolling origins, windows, ablation and uncertainty
data/
  raw/                   downloaded CSV files
  processed/             Parquet tables, SQL results, metrics, forecasts
    stage2/              isolated multi-horizon results and input snapshot
  macro.db               SQLite database
docs/er_diagram.md       ER diagram of the database
reports/figures/         charts for the report (PNG + interactive HTML)
```

## Setup

Python 3.12–3.14. Stage 1 was verified locally with Python 3.14.3; CI is configured for 3.12 and 3.14.

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
5 Stage 1 models · 6 full pipeline (2–5) · 7 show an SQL query result · 8 open dashboard ·
9 Stage 2 multi-horizon experiments.

Choose **6** to rebuild the original pipeline in an isolated staging directory, then **9** for Stage 2 and **8** for the unified dashboard. Option 6 preserves the Stage 2 snapshot; if inputs change, the dashboard marks that snapshot stale until option 9 succeeds. Both `streamlit run app.py` and `streamlit run app/main.py` launch the same application. Ordinary navigation reads saved artifacts and never trains models.
Each step can also be run directly:

```bash
python -m src.data_loader          # add --force to download again
python -m src.preprocessing
python -m src.database
python -m src.analysis
python -m src.models
python -m src.evaluation           # Stage 2; preserves original one-step artifacts
streamlit run app.py
```

## Methods

- **Preprocessing:** daily FX gaps are interpolated and flagged for retrospective EDA; forecasting uses a separate `monthly_forecasting.parquet` table with past-only forward filling. Original values are retained in raw CSV and `source_usd_kzt`. Source validation rejects duplicates, empty responses and missing CPI months;
  monthly table with averages, end-of-month values, m/m and y/y inflation, base rate in force,
  real rate and exchange-rate regime. Data audit and memory/format measurements in
  `data/processed/quality_report.json`.
- **Database:** normalized schema `sources → indicators → observations` plus the derived
  `monthly_macro` table ([ER diagram](docs/er_diagram.md)). 11 queries with JOINs,
  GROUP BY/HAVING, a subquery, CTEs and window functions (`LAG`, `RANK`, rolling `AVG`).
- **Time-series analysis:** ADF and KPSS stationarity tests, STL decomposition, ACF/PACF,
  correlation of monthly changes, exchange-rate pass-through by lag, Chow tests at the
  known break dates. The whole event month is post-event; the pipeline Chow test concerns a shift in the mean only. Its classical p-values assume iid/homoskedastic errors and are exploratory.
- **Forecasting:** expanding-window one-step-ahead backtest on the last 36 months (models are
  refit every month on earlier observation months). Final-vintage data are used with explicit, unverified publication assumptions; this is not a real-time vintage backtest. Models: naive, seasonal naive, SARIMA/ARIMA
  (order chosen by AIC), ETS, Ridge lag regression, gradient boosting. Metrics: MAE, RMSE,
  sMAPE, MASE, fit time, MAE by year. SARIMA candidates must converge; unsuccessful fits receive at most one cold retry. Explicit naive fallbacks are flagged and excluded from best-model selection.

## Main results

- Largest tenge depreciations: 2015 (+86% Dec–Dec, +35.5% on 21.08.2015), 2009 (+22.8%),
  2014 (+18.7%).
- After the 2015 float the average monthly exchange-rate move tripled (0.73% → 2.18%).
- Inflation and earlier FX changes have a descriptive lagged association (correlation 0.31 at lag 1, about 0 after 4 months). This does not establish causal pass-through.
- The exploratory mean-shift test for monthly inflation at the 2022 event gives a classical p-value below 0.001. It is not a robust test of all structural changes or causality.
- Stage 1 rebuild: inflation SARIMA MAE **0.194366** vs **0.200151** for naive (2.89% lower MAE); USD/KZT gradient boosting MAE **7.854948** vs **8.848591** (11.23% lower MAE). These are one-step retrospective errors, not validated 12-month accuracy.
- Stage 2 adds future Ridge and Gradient Boosting predictions for months 1–12 using independent direct horizon models. Inputs end at the last observed month; future exogenous values are never fabricated. The original one-step experiment remains available under Model Evaluation → Stage 1 reference.
- Reproducibility details, old/new metrics and remaining limitations: [Stage 1 report](docs/stage1_reliability.md).


## Stage 1 verification and safe updates

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pip check
python main.py                     # option 6: staged full pipeline
python -m streamlit run app.py      # unified dashboard, also console option 8
python -m streamlit run app/main.py # identical app through the direct entry point
```

`requirements.txt` is the single runtime dependency source; `pyproject.toml`
reads it through setuptools. Pytest is a development dependency only.

Each full run records `data/pipeline_manifest.json`; preprocessing, analysis and
models have their own manifests. Model results include a frozen input table,
SHA-256 hashes, dependency versions, source-code hashes, timestamps and fit diagnostics.
Dashboards invalidate caches by file content and distinguish target observation
month, assumed forecast origin, target horizon and generation time. Existing
legacy snapshots without manifests remain readable with a visible warning.

A failed calculation publishes nothing. A failed replacement rolls back the bundle.
An interrupted process leaves `.update-in-progress` and a backup directory: readers
refuse mixed outputs. Do not delete these blindly; see the recovery procedure in
the Stage 1 report. Use one pipeline writer at a time.

Tests use synthetic fixtures and saved small datasets; they do not download data
or train the full model grid. GitHub Actions is configured but has not run until
these changes are committed and pushed by the team.

## Stage 2: evaluation and unified dashboard

Navigation: **Overview · Data Explorer · Data Quality · Advanced Data Analysis ·
Interactive Visualization · SQL Explorer · Model Evaluation · Future Forecasting**.
World Bank descriptive statistics require selection of one indicator first.

The saved experiment evaluates six models on **36 common rolling origins** at
**1, 3, 6 and 12 months**, using full expanding history, expanding history since
September 2015 and rolling 120-month windows. Two ML models also run eight feature
variants on identical post-2015 training pairs. MAE, RMSE, sMAPE, per-origin MASE,
baseline gains, statuses, dates and training sample sizes are downloadable.
SARIMA intervals include observed coverage, width and interval score diagnostics.
Ridge coefficients, boosting importance and a separate chronological holdout
block-permutation diagnostic are available in Model Evaluation.

The run contains **9,792 evaluation rows**, **432 future rows**, 272 metric groups
and 24 interval groups. A full local run takes about 3.5 minutes with one BLAS
thread. Saved intervals cover all 36 observations in each tested group but are
wide; this is not evidence of exact 95% calibration. Rankings differ across
horizons, windows and predictor sets. All comparisons are exploratory, use final
vintages and assumed release timing, and have dependent overlapping errors.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.evaluation
python -m pytest -q
python -m streamlit run app.py
```

The original Stage 1 score table is not directly comparable to Stage 2: its
one-step test dates differ. Stage 2 primary feature groups are fixed, with
ablations reported separately rather than silently selecting a new model.
Full methods, actual numeric results, artifact inventory, verification and
remaining limits: [Stage 2 report (Russian)](docs/stage2_functional_upgrades.md).
