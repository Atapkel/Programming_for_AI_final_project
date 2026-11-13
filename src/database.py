"""Load the processed data into SQLite and run the analysis SQL queries.

Schema (see docs/er_diagram.md):
    sources       1 ── N  indicators  1 ── N  observations
    monthly_macro  (derived analysis table, one row per month)

Usage:
    python -m src.database          # build data/macro.db and run all queries
"""

import sqlite3
from pathlib import Path

import pandas as pd

from src.preprocessing import PROCESSED_DIR, run_preprocessing

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "macro.db"
SQL_RESULTS_DIR = PROCESSED_DIR / "sql"

SCHEMA = """
CREATE TABLE sources (
    source_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    url         TEXT
);

CREATE TABLE indicators (
    indicator_id INTEGER PRIMARY KEY,
    code         TEXT NOT NULL UNIQUE,
    name         TEXT NOT NULL,
    unit         TEXT,
    frequency    TEXT NOT NULL,
    source_code  TEXT,
    source_id    INTEGER NOT NULL REFERENCES sources(source_id)
);

CREATE TABLE observations (
    indicator_id INTEGER NOT NULL REFERENCES indicators(indicator_id),
    date         TEXT NOT NULL,              -- ISO 'YYYY-MM-DD'; annual data use 'YYYY-01-01'
    value        REAL NOT NULL,
    is_filled    INTEGER NOT NULL DEFAULT 0, -- 1 = interpolated, not an official value
    PRIMARY KEY (indicator_id, date)
);
CREATE INDEX idx_observations_date ON observations(date);

CREATE TABLE monthly_macro (
    month              TEXT PRIMARY KEY,     -- 'YYYY-MM'
    usd_kzt_avg        REAL,
    usd_kzt_eom        REAL,
    fx_filled_days     INTEGER,
    brent_avg          REAL,
    usd_kzt_change_pct REAL,
    brent_change_pct   REAL,
    cpi_index          REAL,
    inflation_mom      REAL,
    inflation_yoy      REAL,
    base_rate          REAL,
    real_rate          REAL,
    regime             TEXT
);
"""

SOURCES = {
    "National Bank of Kazakhstan": "https://nationalbank.kz",
    "IMF": "https://data.imf.org",
    "FRED": "https://fred.stlouisfed.org",
    "World Bank": "https://data.worldbank.org",
}

# code -> (name, unit, frequency, source, source series code)
INDICATORS = {
    "usd_kzt": ("Official USD/KZT exchange rate", "KZT per USD", "daily",
                "National Bank of Kazakhstan", None),
    "base_rate": ("NBK base rate", "%", "decision",
                  "National Bank of Kazakhstan", None),
    "cpi_index": ("Consumer price index (2020 = 100)", "index", "monthly", "IMF", "CPI"),
    "brent": ("Brent crude oil price", "USD per barrel", "daily", "FRED", "DCOILBRENTEU"),
    "inflation_cpi_pct": ("Inflation, consumer prices", "% per year", "annual",
                          "World Bank", "FP.CPI.TOTL.ZG"),
    "gdp_growth_pct": ("GDP growth", "% per year", "annual", "World Bank", "NY.GDP.MKTP.KD.ZG"),
    "gdp_current_usd": ("GDP", "current USD", "annual", "World Bank", "NY.GDP.MKTP.CD"),
    "usd_kzt_annual_avg": ("Official exchange rate, annual average", "KZT per USD", "annual",
                           "World Bank", "PA.NUS.FCRF"),
    "unemployment_pct": ("Unemployment", "% of labor force", "annual",
                         "World Bank", "SL.UEM.TOTL.ZS"),
    "exports_pct_gdp": ("Exports of goods and services", "% of GDP", "annual",
                        "World Bank", "NE.EXP.GNFS.ZS"),
}

# Reusable fragment: observations of one indicator
OBS = "observations o JOIN indicators i USING (indicator_id)"

QUERIES = {
    "01_indicator_coverage": (
        "Coverage of every indicator by source (JOIN of 3 tables, GROUP BY)",
        f"""
        SELECT s.name AS source, i.code, i.frequency, COUNT(*) AS n_obs,
               MIN(o.date) AS first_date, MAX(o.date) AS last_date,
               SUM(o.is_filled) AS filled_values
        FROM observations o
        JOIN indicators i ON i.indicator_id = o.indicator_id
        JOIN sources s ON s.source_id = i.source_id
        GROUP BY s.name, i.code, i.frequency
        ORDER BY s.name, i.code
        """),
    "02_usd_kzt_yearly": (
        "Yearly USD/KZT statistics on official (not filled) days (GROUP BY + HAVING)",
        f"""
        SELECT substr(o.date, 1, 4) AS year,
               ROUND(AVG(o.value), 2) AS avg_rate,
               MIN(o.value) AS min_rate, MAX(o.value) AS max_rate,
               ROUND((MAX(o.value) - MIN(o.value)) / MIN(o.value) * 100, 1) AS range_pct,
               COUNT(*) AS n_days
        FROM {OBS}
        WHERE i.code = 'usd_kzt' AND o.is_filled = 0
        GROUP BY year
        HAVING COUNT(*) >= 200
        ORDER BY year
        """),
    "03_largest_daily_moves": (
        "Top 10 daily USD/KZT moves - the devaluation dates (CTE + LAG window)",
        f"""
        WITH fx AS (
            SELECT o.date, o.value,
                   LAG(o.value) OVER (ORDER BY o.date) AS prev_value
            FROM {OBS}
            WHERE i.code = 'usd_kzt'
        )
        SELECT date, prev_value AS rate_before, value AS rate_after,
               ROUND((value - prev_value) / prev_value * 100, 2) AS change_pct
        FROM fx
        WHERE prev_value IS NOT NULL
        ORDER BY ABS(value - prev_value) / prev_value DESC
        LIMIT 10
        """),
    "04_base_rate_changes": (
        "Every base rate hike or cut (CTE + LAG window + CASE)",
        f"""
        WITH br AS (
            SELECT o.date, o.value AS rate,
                   LAG(o.value) OVER (ORDER BY o.date) AS prev_rate
            FROM {OBS}
            WHERE i.code = 'base_rate'
        )
        SELECT date, prev_rate, rate, rate - prev_rate AS change_pp,
               CASE WHEN rate > prev_rate THEN 'hike' ELSE 'cut' END AS decision
        FROM br
        WHERE prev_rate IS NOT NULL AND rate <> prev_rate
        ORDER BY date
        """),
    "05_fx_vs_oil_monthly": (
        "Monthly USD/KZT and Brent averages joined by month (two CTEs + JOIN)",
        f"""
        WITH fx AS (
            SELECT substr(o.date, 1, 7) AS month, AVG(o.value) AS usd_kzt
            FROM {OBS} WHERE i.code = 'usd_kzt' GROUP BY month
        ),
        oil AS (
            SELECT substr(o.date, 1, 7) AS month, AVG(o.value) AS brent
            FROM {OBS} WHERE i.code = 'brent' GROUP BY month
        )
        SELECT fx.month, ROUND(fx.usd_kzt, 2) AS usd_kzt, ROUND(oil.brent, 2) AS brent
        FROM fx JOIN oil ON oil.month = fx.month
        ORDER BY fx.month
        """),
    "06_inflation_yoy_from_cpi": (
        "Year-over-year inflation computed in SQL from the CPI index (LAG 12 window)",
        f"""
        WITH cpi AS (
            SELECT o.date, o.value,
                   LAG(o.value, 12) OVER (ORDER BY o.date) AS value_12m_ago
            FROM {OBS}
            WHERE i.code = 'cpi_index'
        )
        SELECT date, ROUND(value, 2) AS cpi_index,
               ROUND((value / value_12m_ago - 1) * 100, 2) AS inflation_yoy
        FROM cpi
        WHERE value_12m_ago IS NOT NULL
        ORDER BY date
        """),
    "07_high_inflation_years": (
        "Years with inflation above the 2000+ average (subquery)",
        f"""
        SELECT CAST(substr(o.date, 1, 4) AS INTEGER) AS year,
               ROUND(o.value, 2) AS inflation_pct
        FROM {OBS}
        WHERE i.code = 'inflation_cpi_pct' AND o.date >= '2000'
          AND o.value > (
              SELECT AVG(o2.value)
              FROM observations o2 JOIN indicators i2 USING (indicator_id)
              WHERE i2.code = 'inflation_cpi_pct' AND o2.date >= '2000'
          )
        ORDER BY o.value DESC
        """),
    "08_regime_comparison": (
        "Average monthly FX move, inflation and base rate per exchange-rate regime (GROUP BY)",
        """
        SELECT regime, MIN(month) AS first_month, MAX(month) AS last_month,
               COUNT(*) AS n_months,
               ROUND(AVG(usd_kzt_change_pct), 2) AS avg_fx_change_pct,
               ROUND(AVG(ABS(usd_kzt_change_pct)), 2) AS avg_abs_fx_change_pct,
               ROUND(AVG(inflation_yoy), 2) AS avg_inflation_yoy,
               ROUND(AVG(base_rate), 2) AS avg_base_rate
        FROM monthly_macro
        GROUP BY regime
        ORDER BY first_month
        """),
    "09_rolling_12m_fx": (
        "Monthly USD/KZT with 12-month rolling average (window frame)",
        """
        SELECT month, ROUND(usd_kzt_avg, 2) AS usd_kzt,
               ROUND(AVG(usd_kzt_avg) OVER (
                   ORDER BY month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW), 2) AS rolling_12m
        FROM monthly_macro
        ORDER BY month
        """),
    "10_depreciation_ranking": (
        "Years ranked by December-to-December tenge depreciation (CTE + LAG + RANK)",
        """
        WITH dec AS (
            SELECT substr(month, 1, 4) AS year, usd_kzt_eom
            FROM monthly_macro
            WHERE substr(month, 6, 2) = '12'
        ),
        chg AS (
            SELECT year, usd_kzt_eom,
                   LAG(usd_kzt_eom) OVER (ORDER BY year) AS prev_eom
            FROM dec
        )
        SELECT year, ROUND(prev_eom, 2) AS start_rate, ROUND(usd_kzt_eom, 2) AS end_rate,
               ROUND((usd_kzt_eom / prev_eom - 1) * 100, 1) AS change_pct,
               RANK() OVER (ORDER BY usd_kzt_eom / prev_eom DESC) AS depreciation_rank
        FROM chg
        WHERE prev_eom IS NOT NULL
        ORDER BY depreciation_rank
        """),
    "11_real_rate_by_year": (
        "Average real base rate (base rate - inflation) per year since 2016 (GROUP BY + HAVING)",
        """
        SELECT substr(month, 1, 4) AS year,
               ROUND(AVG(base_rate), 2) AS avg_base_rate,
               ROUND(AVG(inflation_yoy), 2) AS avg_inflation_yoy,
               ROUND(AVG(real_rate), 2) AS avg_real_rate
        FROM monthly_macro
        WHERE real_rate IS NOT NULL
        GROUP BY year
        HAVING COUNT(*) >= 6
        ORDER BY year
        """),
}


def load_processed():
    """Read processed Parquet tables; run preprocessing first if they are missing."""
    needed = ["usd_kzt_daily", "brent_daily", "cpi_monthly", "base_rate",
              "world_bank_annual", "monthly_macro"]
    if not all((PROCESSED_DIR / f"{name}.parquet").exists() for name in needed):
        print("Processed files not found - running preprocessing first")
        tables, _ = run_preprocessing()
        return tables
    return {name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet") for name in needed}


def iso(dates):
    """Datetime column -> 'YYYY-MM-DD' strings for SQLite."""
    return pd.to_datetime(dates).dt.strftime("%Y-%m-%d")


def build_observations(tables, indicator_ids):
    """Stack every series into the long observations format."""
    parts = [
        pd.DataFrame({"code": "usd_kzt", "date": iso(tables["usd_kzt_daily"]["date"]),
                      "value": tables["usd_kzt_daily"]["usd_kzt"],
                      "is_filled": tables["usd_kzt_daily"]["is_filled"].astype(int)}),
        pd.DataFrame({"code": "base_rate", "date": iso(tables["base_rate"]["date"]),
                      "value": tables["base_rate"]["base_rate"], "is_filled": 0}),
        pd.DataFrame({"code": "cpi_index", "date": iso(tables["cpi_monthly"]["date"]),
                      "value": tables["cpi_monthly"]["cpi_index"], "is_filled": 0}),
        pd.DataFrame({"code": "brent", "date": iso(tables["brent_daily"]["date"]),
                      "value": tables["brent_daily"]["brent_usd"], "is_filled": 0}),
    ]
    wb = tables["world_bank_annual"]
    parts.append(pd.DataFrame({"code": wb["indicator"], "date": wb["year"].astype(str) + "-01-01",
                               "value": wb["value"], "is_filled": 0}))
    obs = pd.concat(parts, ignore_index=True)
    obs["indicator_id"] = obs["code"].map(indicator_ids)
    unknown = obs["indicator_id"].isna()
    if unknown.any():
        raise ValueError(f"Unknown indicator codes: {sorted(obs.loc[unknown, 'code'].unique())}")
    return obs[["indicator_id", "date", "value", "is_filled"]]


def build_database(db_path=DB_PATH):
    """Create a fresh SQLite database from the processed tables."""
    tables = load_processed()
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)

        conn.executemany("INSERT INTO sources (name, url) VALUES (?, ?)", SOURCES.items())
        source_ids = dict(conn.execute("SELECT name, source_id FROM sources").fetchall())

        rows = [(code, name, unit, freq, src_code, source_ids[source])
                for code, (name, unit, freq, source, src_code) in INDICATORS.items()]
        conn.executemany(
            "INSERT INTO indicators (code, name, unit, frequency, source_code, source_id)"
            " VALUES (?, ?, ?, ?, ?, ?)", rows)
        indicator_ids = dict(conn.execute("SELECT code, indicator_id FROM indicators").fetchall())

        obs = build_observations(tables, indicator_ids)
        obs.to_sql("observations", conn, if_exists="append", index=False)

        monthly = tables["monthly_macro"].copy()
        monthly.insert(0, "month", pd.to_datetime(monthly.pop("date")).dt.strftime("%Y-%m"))
        monthly.to_sql("monthly_macro", conn, if_exists="append", index=False)

    print(f"Database built: {db_path} ({len(obs)} observations, {len(monthly)} months)")
    return db_path


def run_query(name, db_path=DB_PATH):
    """Run one query from QUERIES and return the result as a DataFrame."""
    _, sql = QUERIES[name]
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(sql, conn)


def run_all_queries(db_path=DB_PATH, save=True):
    """Run every query, optionally saving each result to data/processed/sql/<name>.csv."""
    results = {}
    if save:
        SQL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, (description, _) in QUERIES.items():
        try:
            df = run_query(name, db_path)
        except (sqlite3.Error, pd.errors.DatabaseError) as err:
            print(f"[error] {name}: {err}")
            continue
        results[name] = df
        if save:
            df.to_csv(SQL_RESULTS_DIR / f"{name}.csv", index=False)
        print(f"[ok] {name}: {len(df)} rows - {description}")
    return results


if __name__ == "__main__":
    build_database()
    run_all_queries()
