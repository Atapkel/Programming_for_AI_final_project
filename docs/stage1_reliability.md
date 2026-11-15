# Stage 1 — correctness, reliability and reproducibility

Implemented against local baseline `ab35d6e6eec92674b2b6de28446019409ecb75b0`.
The original and newer dashboards, six model families, SQL queries, notebook and
raw datasets are retained. Stage 2 model/horizon expansion is not implemented.

## Statistical interpretation

- `event_partition` converts a historical day to its month start. Default
  `transition="post"` includes the **whole event month** in the post-event group.
  `transition="exclude"` omits the mixed month; neither policy puts it in pre-event.
- The pipeline Chow test is an intercept-only **mean-shift** test. The notebook
  retains its different log-linear trend-break test and uses the shared boundary.
- Classical F p-values assume independent, homoskedastic errors. They are
  exploratory for these series; autocorrelation, multiple comparisons and causal
  identification are not solved by correcting the date. Lag correlations and
  Granger associations are not evidence of a causal economic mechanism.
- For FX changes, correcting the event boundary changes the classical p-values
  for 2009 from 0.1210 to 0.0181 and 2014 from 0.0851 to 0.0311. For the 2015
  event the value changes from 0.2016 to 0.0908. These are sensitivity results,
  not newly established causal effects.

## Data contracts and forecasting availability

Raw schemas, finite values, date alignment, duplicate keys, CPI continuity and
minimum coverage are checked before writing. Existing observation keys may not
silently disappear during a refresh. Initial full-history downloads require at
least 365 FX rows, 250 oil rows, 24 CPI months, 12 rate decisions and 60 annual
indicator rows. Daily calendar coverage must be at least 90% for FX and 70% of
business days for oil. The existing data satisfy these deliberately conservative
completeness thresholds. Legitimate source deletions need explicit review.

`monthly_macro.parquet` remains the retrospective EDA/SQL table. Its numerical
series are unchanged. `monthly_forecasting.parquet` instead uses **daily forward
fill from the past** before aggregation. Future observations cannot change earlier
filled values. Raw CSVs, observed daily values (`source_usd_kzt`) and fill flags are
preserved. Long gaps still contain estimates, not reconstructed official rates.
Missing dates and missing cells are reported separately; duplicate dates are
rejected instead of silently deduplicated. Missing internal CPI months cause a
clear error, never an incorrect month-on-month percentage.

The explicit retrospective protocol assumes previous-month FX, Brent and CPI
are available on the **15th of the target month**. This day is a declared
evaluation assumption, **not a verified release calendar**. Only completed
earlier months supply features; no current-target-month observations are used.
Because an origin can be inside the target month, this can be a monthly nowcast,
not a full month of lead time. The source snapshot does not contain publication
timestamps or historical vintages; revisions and real publication delays remain
uncontrolled. Do not describe these results as proven real-time performance.

For future paths, metadata distinguish the last observed target month, the end
of that observation period (`information_cutoff`), assumed historical
`forecast_origin`, target months and actual run creation timestamp. An old
unpublished target month can already be in the past when the app is opened.
The World Bank one-year lags and retrospective crisis dummies in the notebook
are descriptive only and are not fed to the implemented models.

## Model reliability

The original ARIMA order grid, seasonal specification, Ridge/boosting settings,
random seed, 36-month test windows and 12-month statistical paths are retained.
For each SARIMAX fit:

1. Try the same optimizer with its original 50-iteration budget, optionally using
   the preceding successful fit from this backtest.
2. If unsuccessful, retry once with a cold start and 200 iterations, preserving
   the model specification and stationarity/invertibility constraints.
3. Accept only a converged result with finite parameters and AIC. Failed grid
   candidates cannot win AIC selection.
4. If a backtest fit still fails, record `fit_status="fallback_naive"`. Its metrics
   explicitly include `fallback_count`; that model run cannot be selected best.
   A failed future statistical fit is omitted with a recorded reason and visible
   UI warning; no falsely labelled 95% interval is produced.

`model_diagnostics.json` records attempts and warnings. Warm starts reset at each
independent backtest and after a model run. ETS optimizer success is also checked.
sMAPE defines the zero/zero contribution as zero. MASE with a zero training scale
is undefined (`NaN`), rather than a fabricated success. MASE uses the training
naive scale, not the error of naive on the test set.

## Measured result changes

The following comparison is from the baseline's committed CSV and the actual
Stage 1 rebuild in Python 3.14.3 with the pinned dependencies. Fit duration varies
by machine. Values below are not a promise of improved predictive accuracy.

| Target | Model | Old MAE | Stage 1 MAE | Old RMSE | Stage 1 RMSE |
|---|---|---:|---:|---:|---:|
| Inflation m/m | SARIMA | 0.182396 | 0.194366 | 0.232338 | 0.250747 |
| Inflation m/m | Naive | 0.200151 | 0.200151 | 0.240805 | 0.240805 |
| Inflation m/m | Ridge | 0.205401 | 0.206373 | 0.265792 | 0.267880 |
| Inflation m/m | ETS | 0.205985 | 0.205985 | 0.284898 | 0.284898 |
| Inflation m/m | Gradient boosting | 0.256719 | 0.255846 | 0.321614 | 0.322528 |
| Inflation m/m | Seasonal naive | 0.314358 | 0.314358 | 0.380751 | 0.380751 |
| USD/KZT | Gradient boosting | 7.811700 | 7.854948 | 10.277830 | 10.084513 |
| USD/KZT | ARIMA | 8.178501 | 8.120726 | 10.747813 | 10.657877 |
| USD/KZT | ETS | 8.710656 | 8.721338 | 11.157190 | 11.171383 |
| USD/KZT | Naive | 8.848591 | 8.848591 | 11.112543 | 11.112543 |
| USD/KZT | Ridge | 9.061554 | 9.072174 | 11.451346 | 11.447178 |
| USD/KZT | Seasonal naive | 36.648999 | 36.770365 | 45.758814 | 45.834599 |

Explanations:

- Inflation SARIMA order changes `(2,0,2)` → `(2,0,1)` because the former candidate
  fails both bounded convergence attempts. Its grid was not tuned to the test set.
  The compounded 12-target-month point path changes from 9.50% to 7.94%.
- FX ARIMA changes `(1,1,0)` → `(1,1,1)` under the causal input and convergence
  policy. All accepted backtest and future fits in this rebuild converged;
  recorded failed attempts are excluded or successfully retried. No backtest
  naive fallback was needed in this particular run.
- Forward fill changes historical FX means around gaps, affecting training data,
  lagged FX features for inflation, and seasonal-naive test forecasts whose
  one-year lag points to an imputed month. This explains changes even though the
  current test target values themselves remain unchanged.
- CPI and the inflation univariate baselines/ETS input remain unchanged. Minor
  floating-point differences across environments should not be called improvements.
- Best MAE gains versus naive are now **2.89%** (inflation) and **11.23%** (FX).
  SARIMA inflation has worse RMSE than naive, so “best” here means lowest MAE only.
- One-step accuracy still does not validate 12-month accuracy or interval coverage.

### Rebuild of 15.11.2026 (Python 3.14.2, same pinned packages, different machine)

The committed artifacts were rebuilt after the benchmark fixes and the addition of
Diebold–Mariano tests. Inputs (`model_input.parquet`) are byte-identical to the run
above, yet the AIC-selected orders differ: inflation `(2,0,2)` instead of `(2,0,1)`
and FX `(1,1,0)` instead of `(1,1,1)`. On this machine the `(2,0,2)` inflation
candidate converges; on the 3.14.3 machine it failed both attempts. Repeated runs on
one machine are identical, with and without single-threaded BLAS, so the difference
comes from CPU-specific floating-point paths (OpenBLAS `DYNAMIC_ARCH`) acting on
borderline optimiser convergence, not from randomness in the code.

| Target | Model | MAE | RMSE | DM p vs naive |
|---|---|---:|---:|---:|
| Inflation m/m | SARIMA (2,0,2) | 0.183645 | 0.232385 | 0.40 |
| Inflation m/m | Naive | 0.200151 | 0.240805 | – |
| USD/KZT | Gradient boosting | 7.854948 | 10.084513 | 0.19 |
| USD/KZT | ARIMA (1,1,0) | 8.178540 | 10.746701 | 0.26 |
| USD/KZT | Naive | 8.848591 | 11.112543 | – |

- Best MAE gains versus naive: **8.25%** (inflation) and **11.23%** (FX). With order
  `(2,0,2)` SARIMA also has lower RMSE than naive for inflation.
- Diebold–Mariano tests (absolute loss, HLN correction, two-sided, n = 36) find no
  model significantly better than naive; the best-model label is a point ranking only.
- The compounded 12-month SARIMA inflation path is 9.50% under `(2,0,2)`.
- Practical rule: produce the final submission artifacts on one machine and present
  those; the dashboard reads saved files and never retrains.

## Safe publication and recovery

Downloads stage and validate all requested sources before replacing any source.
Preprocessing, statistics, SQL exports and model outputs are staged as bundles.
SQLite is built separately, closed, checked for integrity/foreign keys/nonempty
tables and exercised with all 11 queries before replacement.

Console option 6 executes every stage against a temporary copy of raw data using
`MACRO_DATA_DIR` and `MACRO_FIGURES_DIR`. Only a completed run is published. The raw
inputs are checked again before publication; calculations do not modify them.
Models save an immutable input snapshot (`model_input.parquet`) alongside the
predictions, so later preprocessing cannot silently change the history paired
with an older forecast.

Manifests connect outputs by SHA-256 and record run ID, UTC creation time, source
hashes, configuration, dependencies and input hashes. The full-run manifest is a
receipt for that run; running an individual stage later can legitimately make
that earlier full-run receipt stale. Per-stage manifests remain authoritative
for their own output bundle. Caches use content fingerprints, including file
absence; absent artifacts are checked outside cached functions.

Publication uses destination-filesystem temporary files, validation and atomic
replacement. A single shared data-directory marker prevents simultaneous
publication by the main stage writers. On an ordinary exception, replaced files
are restored. This is a local filesystem protocol, not a database transaction
across all files; do not run competing pipeline writers.

If the process or machine is killed during publication, `.update-in-progress`
and `.artifact-backup-*` deliberately remain. Readers fail closed. Recovery:

1. Stop writers and dashboards; confirm no update process is still running.
2. Read the marker's `backup`, ordered `files`, and `existed` lists.
3. If the backup directory contains `ready`, all old files were backed up before
   any replacement. Restore each originally existing destination from its
   numbered backup; remove only newly created destinations marked `existed=false`.
4. Without `ready`, replacement had not started; destinations remain the previous
   set. Inspect before cleanup. Never restore a partially copied backup blindly.
5. After validating the recovered files, remove the marker/backup and run option 6.

Power-loss durability, cross-machine locking and automated disaster recovery are
not claimed. A failed refresh that loses previously present source keys is rejected;
legitimate upstream deletions must be reviewed rather than accepted automatically.

## Reproduce verification

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pip check
python main.py                       # choose 6; after completion choose 0
python -m streamlit run app.py       # original interface, including SQL
python -m streamlit run app/main.py  # newer interface; run separately
```

On Windows activate with `.venv\Scripts\activate`. Python 3.14.3 was executed
locally. CI is configured for Python 3.12 and 3.14, but hosted Actions and a fresh
Python 3.12 installation were not executed during this implementation.

The pytest suite uses small synthetic cases plus the saved project artifacts for
UI/SQL smoke checks. Full ARIMA grid training is not part of routine tests.
AppTest verifies script execution and widget interactions, not browser layout.
Source APIs were not refreshed during verification; all five raw CSVs remain
byte-for-byte identical to baseline. Notebook source cells were updated; retained
rendered outputs are labelled as pre-Stage-1 results until the notebook is rerun.

## Stage 2 recommendations (not implemented)

- Rolling-origin evaluation at horizons 1/3/6/12 and empirical interval coverage.
- One-step future Ridge/boosting forecasts, then justified multi-step strategies.
- Compare post-2015 versus full-history windows and feature ablations.
- Robust statistical inference, diagnostics and uncertainty of baseline gains.
- Merge dashboards while preserving SQL and integrate the existing EDA.


## Recorded local verification — 2026-11-15

- `python -m pytest -q`: **45 passed in 3.88s** on the final tested source.
- `python -m pip check`: no broken requirements.
- AST parsing: all 25 Python files and 73 notebook code cells; valid TOML.
- Full staged pipeline: completed and published run
  `19ca2e6885744f23b834092794eb3d60`; all four manifests and backend source hashes verified.
- Both Streamlit entry points executed through AppTest, including target switches,
  page navigation, empty model selections and an absent SQLite database.
- Notebook: 72 computation cells executed with `MPLBACKEND=Agg`; its CSV write cell
  was deliberately skipped and the computed table matched the existing CSV.
  An initial non-headless macOS attempt aborted in the GUI backend; setting Agg
  resolved the verification environment issue without changing notebook logic.
- All 11 SQL queries execute; rebuilt SQLite passes integrity and foreign-key checks.
- Saved model bundle validates against its frozen input and recomputed metrics.
- All raw CSV bytes and retrospective monthly table values match the baseline.
- `python -m src.data_loader`: all five existing sources validated and skipped;
  no external downloads were needed.
- `git diff --check`: clean. No interrupted-update marker remains.

## File inventory

The working changes below were not committed, pushed or merged. Baseline HEAD is
unchanged. Generated reports are included so the dashboards open the verified
Stage 1 results immediately.

### Modified tracked files

- `.gitignore`
- `EDA/eda_usd_kzt.ipynb`
- `README.md`
- `app.py`
- `app/main.py`
- `app/pages/predictions.py`
- `app/utils/data_utils.py`
- `data/macro.db`
- `data/processed/analysis_results.json`
- `data/processed/backtest_predictions.parquet`
- `data/processed/errors_by_year.csv`
- `data/processed/forecast_summary.json`
- `data/processed/forecasts.parquet`
- `data/processed/model_metrics.csv`
- `data/processed/monthly_macro.parquet`
- `data/processed/quality_report.json`
- `data/processed/usd_kzt_daily.parquet`
- `main.py`
- `pyproject.toml`
- `reports/figures/01_usd_kzt_breaks.png`
- `reports/figures/02_inflation_stl.png`
- `reports/figures/03_inflation_acf_pacf.png`
- `reports/figures/04_corr_changes.png`
- `reports/figures/05_base_rate_vs_inflation.png`
- `reports/figures/06_backtest_inflation_mom.png`
- `reports/figures/06_backtest_usd_kzt.png`
- `reports/figures/macro_interactive.html`
- `requirements.txt`
- `src/analysis.py`
- `src/data_loader.py`
- `src/database.py`
- `src/models.py`
- `src/preprocessing.py`

### New files

- `.github/workflows/tests.yml`
- `data/pipeline_manifest.json`
- `data/processed/analysis_manifest.json`
- `data/processed/model_diagnostics.json`
- `data/processed/model_input.parquet`
- `data/processed/model_manifest.json`
- `data/processed/monthly_forecasting.parquet`
- `data/processed/preprocessing_manifest.json`
- `docs/stage1_reliability.md`
- `requirements-dev.txt`
- `src/artifacts.py`
- `src/dashboard_data.py`
- `src/paths.py`
- `src/validation.py`
- `tests/conftest.py`
- `tests/test_artifacts.py`
- `tests/test_dashboards.py`
- `tests/test_data_safety.py`
- `tests/test_model_reliability.py`

