"""Tests for ARIMA forecasting and frequency inference (analytics/forecasting.py)."""

import numpy as np
import pandas as pd
import pytest

from analytics.forecasting import (
    ForecastResult,
    _create_forecast_chart,
    _infer_freq,
    auto_detect_date_column,
    run_arima_forecast,
)

pmdarima = pytest.importorskip("pmdarima", reason="forecasting requires pmdarima")


@pytest.fixture
def daily_series() -> pd.DataFrame:
    """A clean linear trend: 60 days of steadily increasing values."""
    n = 60
    return pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=n, freq="D"),
            "val": np.arange(n, dtype=float) * 2.0 + 10.0,
        }
    )


# ── date column auto-detection ──
def test_detects_native_datetime_column(daily_series):
    assert auto_detect_date_column(daily_series) == ("ds", "D")


def test_detects_string_dates_on_pandas_3(daily_series):
    """Regression: pandas 3.0 uses StringDtype, so an ``dtype == 'object'``
    guard silently skipped every string date column."""
    df = daily_series.assign(ds=daily_series["ds"].dt.strftime("%Y-%m-%d"))
    assert pd.api.types.is_string_dtype(df["ds"])
    detected = auto_detect_date_column(df)
    assert detected is not None
    assert detected[0] == "ds"


def test_returns_none_when_no_date_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert auto_detect_date_column(df) is None


def test_skips_columns_that_are_not_mostly_dates():
    """The detector requires a parse rate above 0.8."""
    df = pd.DataFrame({
        "mostly": ["2024-01-0{}".format(i % 9 + 1) for i in range(4)] + ["nope"] * 6,
        "val": range(10),
    })
    detected = auto_detect_date_column(df)
    assert detected is None or detected[0] != "mostly"


# ── frequency inference ──
@pytest.mark.parametrize(
    "freq,expected",
    [("D", "D"), ("W", "W"), ("ME", "M"), ("QE", "Q"), ("YE", "Y")],
)
def test_infers_series_frequency(freq, expected):
    series = pd.Series(pd.date_range("2024-01-01", periods=60, freq=freq))
    assert _infer_freq(series) == expected


def test_short_series_defaults_to_daily():
    assert _infer_freq(pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))) == "D"


def test_unsorted_series_is_handled():
    series = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D")).sample(
        frac=1, random_state=0
    )
    assert _infer_freq(series) == "D"


# ── forecasting ──
def test_forecast_succeeds_on_a_linear_trend(daily_series):
    result = run_arima_forecast(daily_series, "ds", "val", periods=6)
    assert isinstance(result, ForecastResult)
    assert result.success is True, result.error
    assert result.error is None


def test_forecast_returns_requested_number_of_periods(daily_series):
    result = run_arima_forecast(daily_series, "ds", "val", periods=6)
    assert result.periods == 6
    assert len(result.forecast_values) == 6
    assert len(result.forecast_dates) == 6
    assert len(result.lower_bound) == 6
    assert len(result.upper_bound) == 6


def test_forecast_dates_extend_beyond_the_training_window(daily_series):
    result = run_arima_forecast(daily_series, "ds", "val", periods=6)
    last_observed = pd.Timestamp(daily_series["ds"].iloc[-1])
    first_forecast = pd.Timestamp(result.forecast_dates[0])
    assert first_forecast > last_observed


def test_confidence_bounds_bracket_the_point_forecast(daily_series):
    result = run_arima_forecast(daily_series, "ds", "val", periods=6)
    for point, low, high in zip(
        result.forecast_values, result.lower_bound, result.upper_bound
    ):
        assert low <= point <= high


def test_wider_confidence_produces_wider_intervals(daily_series):
    narrow = run_arima_forecast(daily_series, "ds", "val", periods=4, confidence=0.80)
    wide = run_arima_forecast(daily_series, "ds", "val", periods=4, confidence=0.99)
    narrow_width = sum(h - l for l, h in zip(narrow.lower_bound, narrow.upper_bound))
    wide_width = sum(h - l for l, h in zip(wide.lower_bound, wide.upper_bound))
    assert wide_width > narrow_width


def test_forecast_reports_model_metadata(daily_series):
    result = run_arima_forecast(daily_series, "ds", "val", periods=4)
    assert set(result.metrics) == {"order", "aic", "bic", "training_points", "frequency"}
    assert result.metrics["training_points"] == 60
    assert result.metrics["frequency"] == "D"
    assert result.model_name.startswith("ARIMA")


def test_forecast_attaches_a_three_trace_chart(daily_series):
    result = run_arima_forecast(daily_series, "ds", "val", periods=4)
    assert result.fig is not None
    assert len(result.fig.data) == 3
    names = {trace.name for trace in result.fig.data}
    assert names == {"Historical", "Forecast", "95% CI"}


def test_forecast_chart_traces_are_documented(daily_series):
    result = run_arima_forecast(daily_series, "ds", "val", periods=4)
    assert result.fig.layout.xaxis.title.text == "Date"
    assert result.fig.layout.yaxis.title.text == "val"


# ── input validation ──
def test_rejects_series_shorter_than_ten_points():
    df = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=5),
        "val": np.arange(5, dtype=float),
    })
    result = run_arima_forecast(df, "ds", "val", periods=3)
    assert result.success is False
    assert "at least 10 data points" in result.error


def test_clamps_periods_to_the_configured_maximum(daily_series):
    from utils.config import FORECAST_MAX_PERIODS

    result = run_arima_forecast(daily_series, "ds", "val", periods=10_000)
    assert result.success is True
    assert result.periods == FORECAST_MAX_PERIODS
    assert len(result.forecast_values) == FORECAST_MAX_PERIODS


def test_periods_below_one_are_clamped_up(daily_series):
    result = run_arima_forecast(daily_series, "ds", "val", periods=0)
    assert result.success is True
    assert result.periods == 1


def test_missing_values_are_dropped_before_fitting():
    n = 60
    values = np.arange(n, dtype=float) * 2.0 + 10.0
    values[5] = np.nan
    values[20] = np.nan
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=n), "val": values})
    result = run_arima_forecast(df, "ds", "val", periods=4)
    assert result.success is True, result.error
    assert result.metrics["training_points"] == n - 2


def test_unsorted_input_is_sorted_before_fitting(daily_series):
    shuffled = daily_series.sample(frac=1.0, random_state=7).reset_index(drop=True)
    ordered = run_arima_forecast(daily_series, "ds", "val", periods=4)
    scrambled = run_arima_forecast(shuffled, "ds", "val", periods=4)
    assert scrambled.success is True
    assert scrambled.historical_values == ordered.historical_values


def test_string_date_column_is_parsed_automatically(daily_series):
    df = daily_series.assign(ds=daily_series["ds"].dt.strftime("%Y-%m-%d"))
    result = run_arima_forecast(df, "ds", "val", periods=4)
    assert result.success is True, result.error


def test_failure_is_reported_rather_than_raised(daily_series):
    result = run_arima_forecast(daily_series, "ds", "no_such_column", periods=4)
    assert result.success is False
    assert result.error is not None


# ── chart helper ──
def test_chart_helper_builds_expected_traces():
    fig = _create_forecast_chart(
        historical_dates=["2024-01-01", "2024-01-02"],
        historical_values=[1.0, 2.0],
        forecast_dates=["2024-01-03"],
        forecast_values=[3.0],
        lower_bound=[2.5],
        upper_bound=[3.5],
        column_name="val",
        confidence=0.9,
    )
    assert len(fig.data) == 3
    assert fig.data[1].fill == "toself"
    assert fig.data[1].name == "90% CI"


def test_result_defaults_lists_to_empty():
    result = ForecastResult(success=False, error="nope")
    assert result.historical_dates == []
    assert result.forecast_values == []
    assert result.metrics == {}
