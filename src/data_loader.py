"""Download raw data for the Kazakhstan macro project into data/raw/.

Sources:
    - USD/KZT daily official rate: National Bank of Kazakhstan (nationalbank.kz)
    - NBK base rate decisions (2015+): nationalbank.kz decision schedule pages
    - Monthly CPI index (2005+): IMF CPI dataset (api.imf.org)
    - Brent oil price, daily: FRED (series DCOILBRENTEU)
    - Annual macro indicators: World Bank API

Usage:
    python -m src.data_loader            # download only missing files
    python -m src.data_loader --force    # download everything again
"""

import html
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

NBK_RATES_URL = (
    "https://nationalbank.kz/ru/exchangerates/ezhednevnye-oficialnye-rynochnye-kursy-valyut/report"
)
NBK_USD_CODE = "5"  # internal NBK id for USD in the report form
NBK_BASE_RATE_URL = "https://nationalbank.kz/en/news/grafik-prinyatiya-resheniy-po-bazovoy-stavke"
IMF_CPI_URL = "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/CPI/%2B/KAZ.CPI._T.IX.M"
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


def fetch(url, params=None, headers=None, retries=3):
    """GET a URL with a few retries; return the response text."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as err:
            print(f"  attempt {attempt}/{retries} failed: {err}")
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def html_table_rows(page):
    """Return every <tr> of an HTML page as a list of clean cell strings."""
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        # remove tags without adding spaces: NBK sometimes writes 10.03<span>.2021</span>
        cells = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", c))).strip() for c in cells]
        rows.append(cells)
    return rows


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
        page = fetch(NBK_RATES_URL, params=params)
        year_rows = NBK_ROW_RE.findall(page)
        print(f"  USD/KZT {year}: {len(year_rows)} days")
        rows.extend(year_rows)

    df = pd.DataFrame(rows, columns=["date", "quantity", "rate"])
    df["date"] = pd.to_datetime(df["date"])
    df["rate"] = df["rate"].str.replace(",", ".").astype(float) / df["quantity"].astype(int)
    return df[["date", "rate"]].drop_duplicates("date").sort_values("date")


def download_base_rate():
    """NBK base rate decisions from the yearly 'decision making schedule' tables.

    Each row: effective date, base rate, corridor. Future meetings have no rate
    yet and are skipped.
    """
    index_page = fetch(NBK_BASE_RATE_URL)
    year_pages = dict(re.findall(r'/rubrics/(\d+)">\s*(\d{4})\s*<', index_page))
    records = []
    for rubric_id, year in sorted(year_pages.items(), key=lambda item: item[1]):
        page = fetch(f"{NBK_BASE_RATE_URL}/rubrics/{rubric_id}")
        count = 0
        for cells in html_table_rows(page):
            if len(cells) < 2:
                continue
            day = re.search(r"\d{2}\.\d{2}\.\d{4}", cells[0])
            rate = re.search(r"\d+(?:[.,]\d+)?", cells[1])
            if day and rate:
                records.append({"date": day.group(), "rate": float(rate.group().replace(",", "."))})
                count += 1
        print(f"  base rate {year}: {count} decisions")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    return df.drop_duplicates("date").sort_values("date")


def download_cpi_monthly():
    """Monthly consumer price index for Kazakhstan (2020 = 100) from the IMF."""
    text = fetch(IMF_CPI_URL, headers={"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"})
    raw = pd.read_csv(StringIO(text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    df = raw.dropna().rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "cpi_index"})
    df["date"] = pd.to_datetime(df["date"].str.replace("-M", "-"), format="%Y-%m")
    return df.sort_values("date")


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


def save_if_needed(name, download_func, force=False):
    """Run download_func and save to data/raw/<name>.csv unless the file exists."""
    path = RAW_DIR / f"{name}.csv"
    if path.exists() and not force:
        print(f"[skip] {path.name} already exists")
        return
    print(f"[download] {name}")
    try:
        df = download_func()
    except (requests.RequestException, ValueError, KeyError) as err:
        print(f"[error] {name}: {err}")
        return
    df.to_csv(path, index=False)
    print(f"[saved] {path.name}: {len(df)} rows")


def download_all(force=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        "usd_kzt_daily": download_usd_kzt,
        "base_rate": download_base_rate,
        "cpi_monthly": download_cpi_monthly,
        "brent_daily": download_brent,
        "world_bank_annual": download_world_bank,
    }
    for name, func in sources.items():
        save_if_needed(name, func, force)


if __name__ == "__main__":
    download_all(force="--force" in sys.argv)
