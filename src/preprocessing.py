"""Clean the raw data and build analysis-ready tables in data/processed/.

Outputs (Parquet):
    usd_kzt_daily     - every calendar day, gaps interpolated, flagged in is_filled
    brent_daily       - trading days only (weekends/holidays are not filled)
    cpi_monthly       - CPI index + month-over-month and year-over-year inflation
    base_rate         - base rate decisions + change vs previous decision
    world_bank_annual - annual indicators (long format)
    monthly_macro     - one row per month with all series joined (main analysis table)
Plus quality_report.json with the data audit and format/memory measurements.

Usage:
    python -m src.preprocessing
"""

import json
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import ROOT, RAW_DIR, PROCESSED_DIR
from src.validation import dated_series, validate_monthly_index, validate_source
from src.artifacts import assert_idle, fingerprint, manifest, publish_files, write_json

RAW_FILES = {
    "usd_kzt": "usd_kzt_daily.csv",
    "brent": "brent_daily.csv",
    "cpi": "cpi_monthly.csv",
    "base_rate": "base_rate.csv",
    "world_bank": "world_bank_annual.csv",
}
CHUNKSIZE = 2_000  # rows per chunk in the chunked-loading comparison

# Exchange-rate regimes used to compare periods (start month, label)
REGIMES = [
    ("2000-01-01", "managed_rate"),      # managed exchange rate, devaluations in 2009 and 2014
    ("2015-08-01", "free_float"),        # tenge floated on 20 Aug 2015, inflation targeting
    ("2022-02-01", "post_2022_shock"),   # sanctions on Russia, base rate jump, inflation peak
]


def load_raw(raw_dir=RAW_DIR):
    """Read all raw CSV files into a dict of DataFrames."""
    assert_idle(raw_dir)
    assert_idle(raw_dir.parent)
    data = {}
    for name, filename in RAW_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run: python -m src.data_loader")
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        validate_source(Path(filename).stem, df)
        data[name] = df
    return data


def audit_series(series, freq=None):
    """Data quality summary of one time series: gaps, duplicates, outliers (IQR rule)."""
    if series.empty or series.index.hasnans:
        raise ValueError("Cannot audit an empty series or invalid timestamps")
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
    report = {
        "rows": int(len(series)),
        "start": str(series.index.min().date()),
        "end": str(series.index.max().date()),
        "missing_values": int(series.isna().sum()),
        "duplicate_dates": int(series.index.duplicated().sum()),
        "iqr_outliers": int(outliers),
        "min": round(float(series.min()), 2),
        "median": round(float(series.median()), 2),
        "max": round(float(series.max()), 2),
    }
    if freq:
        expected = pd.date_range(series.index.min(), series.index.max(), freq=freq)
        report["missing_periods"] = int(len(expected.difference(series.index)))
    return report


def to_series(df, value_col):
    """Sorted series; ambiguous duplicate dates are rejected before any cleaning."""
    return dated_series(df, value_col)


def clean_usd_kzt(df, *, causal=False):
    """Reindex to every calendar day and fill gaps by time interpolation.

    NBK publishes a rate for every day (weekends included), so missing days are
    real gaps - the biggest one is 28.07.2023-28.08.2023, missing in the NBK archive.
    """
    s = to_series(df, "rate")
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="D", name="date")
    out = s.reindex(full_index).to_frame("usd_kzt")
    out["is_filled"] = out["usd_kzt"].isna()
    out["source_usd_kzt"] = out["usd_kzt"]  # original values, never overwritten
    # Interpolation is retrospective EDA only. Models use causal forward fill.
    out["usd_kzt"] = (out["usd_kzt"].ffill() if causal
                      else out["usd_kzt"].interpolate(method="time"))
    if out["usd_kzt"].isna().any():
        raise ValueError("USD/KZT has leading missing values with no past observation")
    return out.reset_index()


def clean_brent(df):
    """Brent trades on business days only; missing days are holidays, so no filling."""
    return to_series(df, "brent_usd").dropna().reset_index()


def clean_cpi(df):
    """Add month-over-month and year-over-year inflation (%) to the CPI index."""
    s = to_series(df, "cpi_index")
    validate_monthly_index(s.index)
    if s.isna().any() or (s <= 0).any():
        raise ValueError("CPI must contain positive observations for every month")
    out = s.to_frame()
    out["inflation_mom"] = s.pct_change(fill_method=None) * 100
    out["inflation_yoy"] = s.pct_change(12, fill_method=None) * 100
    return out.reset_index()


def clean_base_rate(df):
    """Base rate decisions with the change (percentage points) and decision type."""
    out = to_series(df, "rate").to_frame("base_rate")
    out["change_pp"] = out["base_rate"].diff()
    decisions = []
    for change in out["change_pp"]:
        if pd.isna(change):
            decisions.append("first")
        elif change > 0:
            decisions.append("hike")
        elif change < 0:
            decisions.append("cut")
        else:
            decisions.append("hold")
    out["decision"] = decisions
    return out.reset_index()


def assign_regime(dates):
    """Label each date with its exchange-rate regime from REGIMES."""
    labels = pd.Series(REGIMES[0][1], index=dates)
    for start, label in REGIMES:
        labels[dates >= pd.Timestamp(start)] = label
    return labels.values


def build_monthly(clean):
    """Join all series into one monthly table (month start as index)."""
    fx = clean["usd_kzt"].set_index("date")
    brent = clean["brent"].set_index("date")["brent_usd"]
    cpi = clean["cpi"].set_index("date")
    validate_monthly_index(cpi.index)
    base = clean["base_rate"].set_index("date")["base_rate"]

    monthly = pd.DataFrame({
        "usd_kzt_avg": fx["usd_kzt"].resample("MS").mean(),
        "usd_kzt_eom": fx["usd_kzt"].resample("MS").last(),
        "fx_filled_days": fx["is_filled"].resample("MS").sum(),
        "brent_avg": brent.resample("MS").mean(),
    })
    monthly["usd_kzt_change_pct"] = monthly["usd_kzt_avg"].pct_change() * 100
    monthly["brent_change_pct"] = monthly["brent_avg"].pct_change() * 100
    monthly = monthly.join(cpi[["cpi_index", "inflation_mom", "inflation_yoy"]])

    # base rate in force at the end of each month (NaN before Sept 2015)
    base_monthly = base.resample("MS").last().ffill()
    monthly["base_rate"] = base_monthly.reindex(monthly.index).ffill()
    monthly["real_rate"] = monthly["base_rate"] - monthly["inflation_yoy"]
    monthly["regime"] = assign_regime(monthly.index)

    # drop the current month if it is not finished yet
    last_day = fx.index.max()
    if not last_day.is_month_end:
        monthly = monthly[monthly.index < last_day.replace(day=1)]
    monthly.index.name = "date"
    validate_monthly_index(monthly.index)
    return monthly.reset_index()


def optimize_dtypes(df):
    """Downcast floats to float32, small ints to the smallest type, text to category."""
    out = df.copy()
    for col in out.columns:
        kind = out[col].dtype.kind
        if kind == "f":
            out[col] = out[col].astype("float32")
        elif kind in "iu":
            out[col] = pd.to_numeric(out[col], downcast="integer")
        elif kind in "OT" or str(out[col].dtype) in ("str", "string"):
            out[col] = out[col].astype("category")
    return out


def measure_memory(df):
    """Memory usage before/after dtype optimization (KB)."""
    before = df.memory_usage(deep=True).sum() / 1024
    after = optimize_dtypes(df).memory_usage(deep=True).sum() / 1024
    return {"before_kb": round(before, 1), "after_kb": round(after, 1),
            "saved_pct": round((1 - after / before) * 100, 1)}


def _mean_ms(func, repeats):
    """Average wall time of func() in milliseconds, after one untimed warm-up call.

    The warm-up removes one-off costs (library initialisation, OS file cache) that
    would otherwise be charged to whichever format happens to be measured first.
    """
    func()
    start = time.perf_counter()
    for _ in range(repeats):
        func()
    return (time.perf_counter() - start) / repeats * 1000


def compare_formats(df, repeats=10):
    """CSV vs Parquet on the same table: file size, write/read time, loaded memory.

    Both readers must return the same typed table, so the CSV reader parses the
    date columns (Parquet stores types natively and needs no parsing).
    """
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        paths = {"csv": Path(tmp) / "t.csv", "parquet": Path(tmp) / "t.parquet"}
        writers = {"csv": lambda p: df.to_csv(p, index=False),
                   "parquet": lambda p: df.to_parquet(p, index=False)}
        readers = {"csv": lambda p: pd.read_csv(p, parse_dates=date_cols),
                   "parquet": pd.read_parquet}
        for fmt, path in paths.items():
            write_ms = _mean_ms(lambda: writers[fmt](path), repeats)
            read_ms = _mean_ms(lambda: readers[fmt](path), repeats)
            loaded = readers[fmt](path)
            results[fmt] = {"size_kb": round(path.stat().st_size / 1024, 1),
                            "write_ms": round(write_ms, 2),
                            "read_ms": round(read_ms, 2),
                            "memory_kb": round(loaded.memory_usage(deep=True).sum() / 1024, 1),
                            "types_preserved": bool((loaded.dtypes == df.dtypes).all())}
    results["parquet_size_saving_pct"] = round(
        (1 - results["parquet"]["size_kb"] / results["csv"]["size_kb"]) * 100, 1)
    results["method"] = f"mean of {repeats} runs after one warm-up run; CSV parses dates"
    return results


def read_csv_chunked(path, chunksize=CHUNKSIZE):
    """Read a dated CSV in chunks, typing each chunk before the next is read.

    Each chunk's text dates become datetime64 and floats become float32 at once,
    so the untyped text of the whole file is never held in memory together.
    """
    parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if "date" in chunk.columns:
            chunk["date"] = pd.to_datetime(chunk["date"])
        parts.append(optimize_dtypes(chunk))
    return pd.concat(parts, ignore_index=True)


def _peak_kb(func):
    """Peak traced Python/NumPy allocation (KB) while func runs, and its result."""
    tracemalloc.start()
    try:
        result = func()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return result, round(peak / 1024, 1)


def compare_loading(path, chunksize=CHUNKSIZE):
    """Whole-file read vs chunked read with per-chunk dtype optimisation."""
    def full_read():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    full, full_peak = _peak_kb(full_read)
    chunked, chunked_peak = _peak_kb(lambda: read_csv_chunked(path, chunksize))
    same_values = (len(full) == len(chunked) and full["date"].equals(chunked["date"])
                   and bool(np.allclose(full.drop(columns="date"), chunked.drop(columns="date"), rtol=1e-6)))
    if not same_values:
        raise ValueError(f"Chunked loading of {path.name} changed the data")
    return {
        "file": path.name, "rows": len(full), "chunksize": chunksize,
        "chunks": -(-len(full) // chunksize),
        "full_read": {"peak_kb": full_peak, "time_ms": round(_mean_ms(full_read, 5), 2),
                      "result_kb": round(full.memory_usage(deep=True).sum() / 1024, 1)},
        "chunked_read": {"peak_kb": chunked_peak,
                         "time_ms": round(_mean_ms(lambda: read_csv_chunked(path, chunksize), 5), 2),
                         "result_kb": round(chunked.memory_usage(deep=True).sum() / 1024, 1)},
        "same_values": same_values,
        "note": "peak = tracemalloc peak during loading; float32 keeps ~7 significant digits",
    }


def run_preprocessing(save=True):
    """Full pipeline: load raw -> audit -> clean -> monthly table -> save."""
    input_paths = [RAW_DIR / f for f in RAW_FILES.values()]
    input_version = fingerprint(input_paths)
    raw = load_raw()

    quality = {"raw_audit": {
        "usd_kzt": audit_series(raw["usd_kzt"].set_index("date")["rate"], freq="D"),
        "brent": audit_series(raw["brent"].set_index("date")["brent_usd"], freq="B"),
        "cpi": audit_series(raw["cpi"].set_index("date")["cpi_index"], freq="MS"),
        "base_rate": audit_series(raw["base_rate"].set_index("date")["rate"]),
    }}

    clean = {
        "usd_kzt_daily": clean_usd_kzt(raw["usd_kzt"]),
        "brent_daily": clean_brent(raw["brent"]),
        "cpi_monthly": clean_cpi(raw["cpi"]),
        "base_rate": clean_base_rate(raw["base_rate"]),
        "world_bank_annual": raw["world_bank"],
    }
    clean["monthly_macro"] = build_monthly({
        "usd_kzt": clean["usd_kzt_daily"], "brent": clean["brent_daily"],
        "cpi": clean["cpi_monthly"], "base_rate": clean["base_rate"],
    })
    clean["monthly_forecasting"] = build_monthly({
        "usd_kzt": clean_usd_kzt(raw["usd_kzt"], causal=True),
        "brent": clean["brent_daily"], "cpi": clean["cpi_monthly"],
        "base_rate": clean["base_rate"],
    })
    quality["forecast_preprocessing"] = "causal forward fill; interpolation is EDA-only"

    quality["usd_kzt_filled_days"] = int(clean["usd_kzt_daily"]["is_filled"].sum())
    quality["memory_optimization"] = {name: measure_memory(df) for name, df in clean.items()}
    quality["csv_vs_parquet_usd_kzt_daily"] = compare_formats(clean["usd_kzt_daily"])
    quality["chunked_loading"] = {name: compare_loading(RAW_DIR / RAW_FILES[name])
                                  for name in ("usd_kzt", "brent")}
    if fingerprint(input_paths) != input_version:
        raise ValueError("Raw inputs changed during preprocessing; nothing was published")

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="preprocessing-") as tmp:
            staged = Path(tmp)
            for name, df in clean.items():
                path = staged / f"{name}.parquet"
                df.to_parquet(path, index=False)
                pd.testing.assert_frame_equal(df, pd.read_parquet(path))
            write_json(staged / "quality_report.json", quality)
            outputs = {p.name: p for p in staged.iterdir()}
            write_json(staged / "preprocessing_manifest.json", manifest(
                "preprocessing", {f: RAW_DIR / f for f in RAW_FILES.values()}, outputs))
            outputs["preprocessing_manifest.json"] = staged / "preprocessing_manifest.json"
            publish_files({PROCESSED_DIR / name: p for name, p in outputs.items()}, PROCESSED_DIR.parent)
    return clean, quality


if __name__ == "__main__":
    tables, report = run_preprocessing()
    for table_name, table in tables.items():
        print(f"{table_name:18s} {len(table):6d} rows  {list(table.columns)}")
    print(f"\nUSD/KZT days filled by interpolation: {report['usd_kzt_filled_days']}")
    print("CSV vs Parquet:", report["csv_vs_parquet_usd_kzt_daily"])
    for source, result in report["chunked_loading"].items():
        print(f"Chunked loading {source}: peak {result['full_read']['peak_kb']} KB -> "
              f"{result['chunked_read']['peak_kb']} KB")
    print(f"Saved to {PROCESSED_DIR}")
