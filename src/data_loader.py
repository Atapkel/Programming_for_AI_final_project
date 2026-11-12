"""Download raw data for the Kazakhstan macro project into data/raw/.

Sources:
    - USD/KZT daily official rate: National Bank of Kazakhstan (nationalbank.kz)
    - Brent oil price, daily: FRED (series DCOILBRENTEU)
    - Annual macro indicators: World Bank API
    - NBK base rate: entered by hand into data/raw/base_rate.csv (no machine-readable source)

Usage:
    python -m src.data_loader            # download only missing files
    python -m src.data_loader --force    # download everything again
"""

import json
import re
import sys
import time
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
START_YEAR = 2000

NBK_URL = (
    "https://nationalbank.kz/ru/exchangerates/ezhednevnye-oficialnye-rynochnye-kursy-valyut/report"
)
NBK_USD_CODE = "5"  # internal NBK id for USD in the report form
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
WB_URL = "https://api.worldbank.org/v2/country/KZ/indicator/{code}?format=json&per_page=200"

WORLD_BANK_INDICATORS = {
    "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "PA.NUS.FCRF": "usd_kzt_annual_avg",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
    "NE.EXP.GNFS.ZS": "exports_pct_gdp",
}

# One row per <tr>: date, quantity, rate
NBK_ROW_RE = re.compile(
    r"<td[^>]*>\s*(\d{4}-\d{2}-\d{2})\s*</td>\s*"
    r"<td[^>]*>\s*(\d+)\s*</td>\s*"
    r"<td[^>]*>\s*([\d.,]+)\s*</td>"
)


def fetch(url, params=None, retries=3):
    """GET a URL with a few retries; return the response text."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as err:
            print(f"  attempt {attempt}/{retries} failed: {err}")
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def download_usd_kzt(start_year=START_YEAR, end_year=None):
    """Daily official USD/KZT rate from NBK, requested one year at a time."""
    end_year = end_year or date.today().year
    rows = []
    for year in range(start_year, end_year + 1):
        params = {
            "rates[]": NBK_USD_CODE,
            "beginDate": f"01.01.{year}",
            "endDate": f"31.12.{year}",
        }
        html = fetch(NBK_URL, params=params)
        year_rows = NBK_ROW_RE.findall(html)
        print(f"  USD/KZT {year}: {len(year_rows)} days")
        rows.extend(year_rows)

    df = pd.DataFrame(rows, columns=["date", "quantity", "rate"])
    df["date"] = pd.to_datetime(df["date"])
    df["rate"] = df["rate"].str.replace(",", ".").astype(float) / df["quantity"].astype(int)
    return df[["date", "rate"]].drop_duplicates("date").sort_values("date")


def download_brent():
    """Daily Brent crude price (USD per barrel) from FRED."""
    text = fetch(FRED_URL.format(series="DCOILBRENTEU"))
    df = pd.read_csv(StringIO(text), na_values=".")
    df.columns = ["date", "brent_usd"]
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"].dt.year >= START_YEAR].dropna()


def download_world_bank(indicators=WORLD_BANK_INDICATORS):
    """Annual indicators for Kazakhstan, returned in long format."""
    records = []
    for code, name in indicators.items():
        data = json.loads(fetch(WB_URL.format(code=code)))
        if len(data) < 2 or data[1] is None:
            print(f"  World Bank {code}: no data")
            continue
        before = len(records)
        for item in data[1]:
            if item["value"] is not None:
                records.append({"indicator": name, "code": code,
                                "year": int(item["date"]), "value": item["value"]})
        print(f"  World Bank {code}: {len(records) - before} years")
    return pd.DataFrame(records).sort_values(["indicator", "year"])


def load_base_rate(path=RAW_DIR / "base_rate.csv"):
    """Read the hand-entered NBK base rate history (date, rate).

    The NBK site only shows this as a web page, so the team fills the CSV
    from https://nationalbank.kz/ru/page/bazovaya-stavka-dkp-functions .
    """
    if not path.exists():
        pd.DataFrame(columns=["date", "rate"]).to_csv(path, index=False)
        print(f"  created empty template {path.name}: fill in decision dates and rates")
        return pd.DataFrame(columns=["date", "rate"])
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        print(f"  {path.name} is still empty: fill it in by hand")
    return df


def save_if_needed(name, download_func, force=False):
    """Run download_func and save to data/raw/<name>.csv unless the file exists."""
    path = RAW_DIR / f"{name}.csv"
    if path.exists() and not force:
        print(f"[skip] {path.name} already exists")
        return
    print(f"[download] {name}")
    try:
        df = download_func()
    except (requests.RequestException, ValueError) as err:
        print(f"[error] {name}: {err}")
        return
    df.to_csv(path, index=False)
    print(f"[saved] {path.name}: {len(df)} rows")


def download_all(force=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    save_if_needed("usd_kzt_daily", download_usd_kzt, force)
    save_if_needed("brent_daily", download_brent, force)
    save_if_needed("world_bank_annual", download_world_bank, force)
    load_base_rate()


if __name__ == "__main__":
    download_all(force="--force" in sys.argv)
