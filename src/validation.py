"""Source and calendar contracts; reject ambiguous data instead of hiding it."""

import numpy as np
import pandas as pd


SOURCE_COLUMNS = {
    "usd_kzt_daily": ("date", "rate"), "brent_daily": ("date", "brent_usd"),
    "cpi_monthly": ("date", "cpi_index"), "base_rate": ("date", "rate"),
    "world_bank_annual": ("indicator", "code", "year", "value"),
}
# Full-history downloader contracts, deliberately below current coverage.
MIN_ROWS = {"usd_kzt_daily": 365, "brent_daily": 250, "cpi_monthly": 24,
            "base_rate": 12, "world_bank_annual": 60}


def validate_monthly_index(index):
    if not isinstance(index, pd.DatetimeIndex) or index.empty or index.hasnans:
        raise ValueError("Expected a nonempty DatetimeIndex without NaT")
    if index.tz is not None or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Monthly timestamps must be unique, sorted and timezone-naive")
    expected = pd.date_range(index.min(), index.max(), freq="MS")
    if not (index.is_month_start.all() and (index == index.normalize()).all()
            and index.equals(expected)):
        raise ValueError("Monthly index must contain every month, labelled at month start")


def dated_series(df, column):
    if df.empty or not {"date", column}.issubset(df):
        raise ValueError(f"Empty data or missing date/{column}")
    dates = pd.to_datetime(df["date"], errors="raise")
    if dates.isna().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Dates must be non-null, timezone-naive calendar dates")
    if dates.duplicated().any():
        raise ValueError(f"Duplicate dates before cleaning: {dates[dates.duplicated()].tolist()[:5]}")
    values = pd.to_numeric(df[column], errors="raise")
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError(f"Infinite values in {column}")
    return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates, name="date"), name=column).sort_index()


def validate_source(name, df, previous=None, *, full_history=True):
    required = SOURCE_COLUMNS[name]
    if df.empty or not set(required).issubset(df.columns):
        raise ValueError(f"{name}: empty response or missing columns {required}")
    if df[list(required)].isna().any().any():
        raise ValueError(f"{name}: missing source values")
    if name == "world_bank_annual":
        from src.data_loader import WORLD_BANK_INDICATORS
        if df.duplicated(["indicator", "year"]).any():
            raise ValueError("Duplicate indicator/year keys")
        years = pd.to_numeric(df.year, errors="raise")
        if not np.isfinite(years).all() or not (years == years.astype(int)).all():
            raise ValueError("Invalid annual timestamps")
        values = pd.to_numeric(df.value, errors="raise")
        if not np.isfinite(values).all():
            raise ValueError("Non-finite World Bank values")
        mapping = df.code.map(WORLD_BANK_INDICATORS)
        if mapping.isna().any() or not mapping.eq(df.indicator).all():
            raise ValueError("Unexpected World Bank code/indicator mapping")
        if full_history and set(df.indicator) != set(WORLD_BANK_INDICATORS.values()):
            raise ValueError("Incomplete World Bank indicator coverage")
        keys = ["indicator", "year"]
    else:
        s = dated_series(df, required[1])
        if not np.isfinite(s).all() or (s <= 0).any():
            raise ValueError(f"{name}: expected positive finite source values")
        if name == "cpi_monthly":
            validate_monthly_index(s.index)
        if full_history and name in ("usd_kzt_daily", "brent_daily"):
            freq, minimum = ("D", 0.90) if name == "usd_kzt_daily" else ("B", 0.70)
            expected = pd.date_range(s.index.min(), s.index.max(), freq=freq)
            if len(expected.intersection(s.index)) / len(expected) < minimum:
                raise ValueError(f"{name}: insufficient calendar coverage")
        keys = ["date"]
    if full_history and len(df) < MIN_ROWS[name]:
        raise ValueError(f"{name}: only {len(df)} rows; require at least {MIN_ROWS[name]}")
    if previous is not None and not previous.empty:
        old, new = previous.copy(), df.copy()
        if keys == ["date"]:
            old["date"], new["date"] = pd.to_datetime(old.date), pd.to_datetime(new.date)
        old_keys = pd.MultiIndex.from_frame(old[keys])
        new_keys = pd.MultiIndex.from_frame(new[keys])
        if len(old_keys.difference(new_keys)):
            raise ValueError(f"{name}: update loses existing observation keys; explicit review required")
    return df


def event_partition(index, event_date, transition="post"):
    """The whole event month belongs to post, or is explicitly excluded."""
    validate_monthly_index(index)
    boundary = pd.Timestamp(event_date).to_period("M").to_timestamp()
    if transition not in ("post", "exclude"):
        raise ValueError("transition must be 'post' or 'exclude'")
    before = index < boundary
    after = index >= boundary if transition == "post" else index > boundary
    return before, after, boundary
