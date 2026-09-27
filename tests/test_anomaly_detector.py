"""Tests for Z-Score / IQR outlier detection (analytics/anomaly_detector.py)."""

import numpy as np
import pandas as pd
import pytest

from analytics.anomaly_detector import (
    anomalies_to_dataframe,
    detect_all_anomalies,
    detect_iqr_anomalies,
    detect_zscore_anomalies,
)


# ── Z-Score ──
def test_zscore_finds_the_planted_outlier(numeric_df):
    result = detect_zscore_anomalies(numeric_df, "v")
    assert result is not None
    assert result.method == "zscore"
    assert result.anomaly_count == 1
    assert result.anomaly_values == [50.0]
    assert result.threshold == 3.0


def test_zscore_result_reports_consistent_percentage(numeric_df):
    result = detect_zscore_anomalies(numeric_df, "v")
    expected = round(1 / 101 * 100, 2)
    assert result.anomaly_percentage == expected
    assert 0 < result.anomaly_percentage < 100


def test_zscore_attaches_a_two_trace_plotly_figure(numeric_df):
    result = detect_zscore_anomalies(numeric_df, "v")
    assert result.fig is not None
    assert len(result.fig.data) == 2
    assert result.fig.data[0].name == "Normal"
    assert "Anomaly" in result.fig.data[1].name


def test_zscore_isolated_points_exceed_the_threshold(numeric_df):
    result = detect_zscore_anomalies(numeric_df, "v")
    assert abs((50.0 - result.mean) / result.std) > result.threshold


def test_zscore_honours_a_custom_threshold(numeric_df):
    assert detect_zscore_anomalies(numeric_df, "v", threshold=100.0) is None
    assert detect_zscore_anomalies(numeric_df, "v", threshold=0.5) is not None


def test_zscore_returns_none_when_no_outliers():
    """A large, evenly-spread sample yields no point beyond 3 sigma."""
    df = pd.DataFrame({"v": np.linspace(-3.0, 3.0, 2001)})
    assert detect_zscore_anomalies(df, "v") is None


# ── insufficient / degenerate data ──
def test_zscore_requires_a_minimum_sample_size():
    assert detect_zscore_anomalies(pd.DataFrame({"v": [1, 2, 3]}), "v") is None


def test_zscore_returns_none_for_zero_variance():
    assert detect_zscore_anomalies(pd.DataFrame({"v": [7] * 50}), "v") is None


def test_zscore_returns_none_for_non_numeric_column(mixed_df):
    assert detect_zscore_anomalies(mixed_df, "cat") is None


def test_zscore_tolerates_missing_values(rng):
    values = list(rng.normal(0, 1, 100)) + [50.0]
    values[0] = None
    result = detect_zscore_anomalies(pd.DataFrame({"v": values}), "v")
    assert result is not None
    assert result.anomaly_count == 1


# ── IQR ──
def test_iqr_finds_the_planted_outlier(numeric_df):
    result = detect_iqr_anomalies(numeric_df, "v")
    assert result is not None
    assert result.method == "iqr"
    assert result.anomaly_count >= 1
    assert 50.0 in result.anomaly_values
    assert result.threshold == 1.5


def test_iqr_figure_draws_both_boundaries(numeric_df):
    result = detect_iqr_anomalies(numeric_df, "v")
    assert result.fig is not None
    # two scatter traces + two dashed hlines
    assert len(result.fig.data) == 2
    assert len(result.fig.layout.shapes) == 2
    assert len(result.fig.layout.annotations) == 2


def test_iqr_wider_factor_flags_more_points(numeric_df):
    tight = detect_iqr_anomalies(numeric_df, "v", factor=0.5)
    loose = detect_iqr_anomalies(numeric_df, "v", factor=3.0)
    assert tight.anomaly_count > loose.anomaly_count


def test_iqr_returns_none_for_zero_iqr():
    assert detect_iqr_anomalies(pd.DataFrame({"v": [3] * 50}), "v") is None


def test_iqr_returns_none_for_small_sample():
    assert detect_iqr_anomalies(pd.DataFrame({"v": [1, 2, 3]}), "v") is None


# ── orchestration ──
def test_detect_all_runs_both_methods_on_numeric_columns(mixed_df):
    results = detect_all_anomalies(mixed_df)
    assert results, "expected at least one anomaly"
    methods = {r.method for r in results}
    assert methods <= {"zscore", "iqr"}
    columns = {r.column for r in results}
    assert "cat" not in columns  # non-numeric columns are skipped
    assert "flag" not in columns


def test_detect_all_can_be_restricted_to_one_method(numeric_df):
    results = detect_all_anomalies(numeric_df, methods=["zscore"])
    assert results
    assert {r.method for r in results} == {"zscore"}


def test_detect_all_returns_empty_for_clean_data():
    df = pd.DataFrame({"v": np.linspace(-3.0, 3.0, 2001)})
    assert detect_all_anomalies(df) == []


# ── tabular summary ──
def test_anomalies_to_dataframe_shape(mixed_df):
    summary = anomalies_to_dataframe(detect_all_anomalies(mixed_df))
    assert list(summary.columns) == [
        "Column", "Method", "Anomalies", "Percentage", "Threshold", "Mean", "Std",
    ]
    assert len(summary) == len(detect_all_anomalies(mixed_df))


def test_anomalies_to_dataframe_formats_percentage(mixed_df):
    summary = anomalies_to_dataframe(detect_all_anomalies(mixed_df))
    assert all(str(v).endswith("%") for v in summary["Percentage"])


def test_anomalies_to_dataframe_handles_empty_input():
    assert anomalies_to_dataframe([]).empty


def test_display_list_is_capped_at_100_entries():
    """The result caps displayed indices/values at 100 while keeping the true count.

    A right-skewed (exponential) sample is used because a bimodal split cannot
    produce more than a handful of points beyond 3 sigma.
    """
    values = np.random.default_rng(0).exponential(1.0, 20_000)
    result = detect_zscore_anomalies(pd.DataFrame({"v": values}), "v")
    assert result is not None
    assert result.anomaly_count > 100
    assert len(result.anomaly_indices) == 100
    assert len(result.anomaly_values) == 100
    assert result.anomaly_count == 378  # 1.89% of 20,000
    assert result.anomaly_percentage == pytest.approx(1.89, abs=0.05)
