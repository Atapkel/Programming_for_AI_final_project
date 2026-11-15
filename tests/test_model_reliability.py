from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import models as m


def fitted(converged=True, aic=10.):
    return SimpleNamespace(mle_retvals={"converged": converged}, aic=aic, params=np.array([0.1]))


def test_bounded_retry_is_cold_and_logs_failure(monkeypatch, caplog):
    calls = []
    class Fake:
        def __init__(self, *a, **k): pass
        def fit(self, **kwargs):
            calls.append(kwargs)
            return fitted(len(calls) == 2)
    monkeypatch.setattr(m, "SARIMAX", Fake)
    result = m._checked_sarimax(pd.Series(range(50)), m.TARGETS["inflation_mom"], (1, 0, 0), [0.3])
    assert result.mle_retvals["converged"]
    assert [call["maxiter"] for call in calls] == [50, 200]
    assert calls[0]["start_params"] == [0.3] and calls[1]["start_params"] is None
    assert "failure" in caplog.text


def test_no_unconverged_candidate_can_win_by_aic(monkeypatch):
    calls = []
    class Fake:
        def __init__(self, y, order, **kw): self.order = order
        def fit(self, **kw):
            calls.append(self.order)
            return fitted(self.order == (1, 0, 1), 100 if self.order == (1, 0, 1) else -999)
    monkeypatch.setattr(m, "SARIMAX", Fake)
    assert m.select_arima_order(pd.Series(range(50)), m.TARGETS["inflation_mom"], max_pq=1) == (1, 0, 1)
    assert len(calls) == 7


def test_exhausted_retries_do_not_populate_warm_start(monkeypatch):
    class Fake:
        def __init__(self, *a, **kw): pass
        def fit(self, **kw): return fitted(False)
    monkeypatch.setattr(m, "SARIMAX", Fake)
    m._WARM_START.clear()
    with pytest.raises(m.ModelFitError):
        m.fit_sarima(pd.Series(range(50)), m.TARGETS["inflation_mom"], (1, 0, 0), "test")
    assert not m._WARM_START
    assert m.select_arima_order(pd.Series(range(50)), m.TARGETS["inflation_mom"], max_pq=0) is None


def frame(n=60):
    return pd.DataFrame({"inflation_mom": np.linspace(.4, .8, n),
                         "usd_kzt_change_pct": np.zeros(n), "brent_change_pct": np.zeros(n)},
                        index=pd.date_range("2010-01-01", periods=n, freq="MS"))


def test_fallback_is_explicit_and_ineligible_and_state_resets(monkeypatch):
    def broken(*args): raise m.ModelFitError("failed")
    monkeypatch.setattr(m, "select_arima_order", lambda *a: None)
    monkeypatch.setattr(m, "get_models", lambda cfg: {"naive": m.model_naive, "sarima": broken})
    m._WARM_START["old_future_parameters"] = [123]
    pred, times, order, train = m.backtest(frame(), m.TARGETS["inflation_mom"], test_months=3)
    assert not m._WARM_START
    assert pred.attrs["fit_status"]["sarima"] == ["fallback_naive"] * 3
    assert pred.sarima.equals(pred.naive)
    metrics = m.compute_metrics(pred, times, train, 12).set_index("model")
    assert metrics.loc["sarima", "fallback_count"] == 3
    assert not metrics.loc["sarima", "eligible"] and metrics.loc["naive", "eligible"]


def test_missing_next_features_abort_instead_of_imputation(monkeypatch):
    df = frame(); df.iloc[-2, df.columns.get_loc("brent_change_pct")] = np.nan
    monkeypatch.setattr(m, "select_arima_order", lambda *a: (0, 0, 0))
    with pytest.raises(ValueError, match="Unavailable"):
        m.backtest(df, m.TARGETS["inflation_mom"], test_months=1)


def test_future_timestamp_alignment_and_explicit_failed_models(monkeypatch):
    def broken(*a, **kw): raise m.ModelFitError("did not converge")
    monkeypatch.setattr(m, "fit_sarima", broken)
    monkeypatch.setattr(m, "fit_ets", broken)
    df = frame()
    result = m.future_forecast(df, m.TARGETS["inflation_mom"], None, horizon=12)
    expected = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(), periods=12, freq="MS")
    assert set(result.model) == {"naive", "seasonal_naive"}
    for _, rows in result.groupby("model"):
        assert pd.DatetimeIndex(rows.date).equals(expected)
    assert set(result.attrs["failures"]) == {"sarima", "ets"}
    with pytest.raises(ValueError):
        m.future_forecast(df, m.TARGETS["inflation_mom"], None, horizon=0)


def test_zero_denominators_are_defined():
    assert m.smape([0.], [0.]) == 0
    pred = pd.DataFrame({"actual": [2., 2.], "naive": [2., 2.]})
    result = m.compute_metrics(pred, {"naive": 0}, pd.Series([2.] * 24), 12)
    assert result.MAE.iloc[0] == 0
    assert pd.isna(result.MASE.iloc[0])  # undefined scale, not a fabricated zero
