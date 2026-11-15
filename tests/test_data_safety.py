import numpy as np
import pandas as pd
import pytest

from src.analysis import BREAKS, chow_test
from src.preprocessing import audit_series, clean_cpi, clean_usd_kzt, to_series
from src.validation import event_partition, validate_monthly_index, validate_source
from src.models import TARGETS, make_features, target_series


@pytest.mark.parametrize("event", list(BREAKS))
def test_event_month_is_never_pre_event(event):
    index = pd.date_range("2000-01-01", "2026-10-01", freq="MS")
    before, after, boundary = event_partition(index, event)
    assert not before[index == boundary].any()
    assert after[index == boundary].all()
    assert before.sum() + after.sum() == len(index)
    before, after, _ = event_partition(index, event, "exclude")
    assert not before[index == boundary].any() and not after[index == boundary].any()
    assert before.sum() + after.sum() == len(index) - 1


def test_chow_transition_policy_changes_membership_explicitly():
    s = pd.Series([1., 3., 2., 9., 8., 10.], index=pd.date_range("2020-01-01", periods=6, freq="MS"))
    result = chow_test(s, "2020-04-20")
    assert result["n_before"] == result["n_after"] == 3
    assert result["mean_before"] == 2 and result["mean_after"] == 9
    assert chow_test(s, "2020-04-20", "exclude")["n_after"] == 2
    with pytest.raises(ValueError):
        chow_test(s, "2019-01-01")
    with pytest.raises(ValueError):
        chow_test(s * 0, "2020-04-01")


@pytest.mark.parametrize("dates", [
    ["2020-01-01", "2020-03-01"], ["2020-01-01", "2020-01-01"],
    ["2020-02-01", "2020-01-01"], ["2020-01-02", "2020-02-02"],
    ["2020-01-01 12:00", "2020-02-01 12:00"], [],
])
def test_reject_invalid_monthly_index(dates):
    with pytest.raises(ValueError):
        validate_monthly_index(pd.DatetimeIndex(dates))


def test_cpi_gap_is_not_reported_as_monthly_inflation():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-03-01"]), "cpi_index": [100., 110.]})
    with pytest.raises(ValueError, match="every month"):
        clean_cpi(df)


def test_cpi_correct_monthly_and_annual_changes():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=13, freq="MS"),
                       "cpi_index": 100 * 1.01 ** np.arange(13)})
    result = clean_cpi(df)
    assert np.allclose(result.inflation_mom.iloc[1:], 1)
    assert result.inflation_yoy.iloc[-1] == pytest.approx((1.01 ** 12 - 1) * 100)


def test_duplicate_detection_precedes_any_cleaning():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-03"]),
                       "rate": [100., 101., 102.]})
    audit = audit_series(df.set_index("date").rate, "D")
    assert audit["duplicate_dates"] == 1
    assert audit["missing_periods"] == 1
    assert audit["missing_values"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        to_series(df, "rate")


def test_calendar_gaps_and_missing_cells_are_distinct():
    s = pd.Series([100., np.nan, 105.], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"]))
    result = audit_series(s, "D")
    assert result["missing_values"] == result["missing_periods"] == 1


def test_future_observation_cannot_change_causal_gap_fill():
    raw = pd.DataFrame({"date": pd.to_datetime(["2020-01-30", "2020-02-02"]), "rate": [100., 200.]})
    changed = raw.copy(); changed.loc[1, "rate"] = 1000.
    before, after = clean_usd_kzt(raw, causal=True), clean_usd_kzt(changed, causal=True)
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])
    assert before.usd_kzt.tolist() == [100., 100., 100., 200.]
    assert before.is_filled.tolist() == [False, True, True, False]
    assert before.source_usd_kzt.iloc[[0, -1]].tolist() == [100., 200.]
    assert before.source_usd_kzt.iloc[1:3].isna().all()
    assert clean_usd_kzt(raw).usd_kzt.iloc[1] != before.usd_kzt.iloc[1]


def test_future_rows_cannot_change_past_features():
    index = pd.date_range("2020-01-01", periods=30, freq="MS")
    df = pd.DataFrame({"usd_kzt_change_pct": np.arange(30.), "brent_change_pct": np.arange(30.)}, index=index)
    changed = df.copy(); changed.loc[index[20]:] = 99999
    first, second = make_features(df, TARGETS["usd_kzt"]), make_features(changed, TARGETS["usd_kzt"])
    pd.testing.assert_frame_equal(first.loc[:index[20]], second.loc[:index[20]])
    with pytest.raises(ValueError, match="positive"):
        make_features(df, TARGETS["usd_kzt"], lags=(0,))


def test_internal_target_nan_cannot_compress_the_calendar():
    df = pd.DataFrame({"inflation_mom": [1., np.nan, 2.]}, index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    with pytest.raises(ValueError, match="every month"):
        target_series(df, TARGETS["inflation_mom"])


@pytest.mark.parametrize("df", [pd.DataFrame(), pd.DataFrame({"bad": [1]}),
    pd.DataFrame({"date": ["2020-01-01"], "rate": [np.nan]}),
    pd.DataFrame({"date": ["2020-01-01"], "rate": [-1.]})])
def test_source_validation_rejects_empty_malformed_missing_and_negative(df):
    with pytest.raises(ValueError):
        validate_source("usd_kzt_daily", df, full_history=False)


def test_minimum_coverage_and_loss_of_old_keys():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "rate": [100., 101.]})
    with pytest.raises(ValueError, match="at least"):
        validate_source("usd_kzt_daily", df)
    with pytest.raises(ValueError, match="loses"):
        validate_source("usd_kzt_daily", df.iloc[1:], previous=df, full_history=False)
