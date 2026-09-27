"""Tests for descriptive statistics (profiling/stats_generator.py)."""

import numpy as np
import pandas as pd
import pytest

from profiling.stats_generator import (
    compute_all_stats,
    compute_categorical_stats,
    compute_correlation_matrix,
    compute_numeric_stats,
)


# ── numeric statistics ──
def test_numeric_stats_on_a_known_distribution():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    stats = compute_numeric_stats(df, "x")
    assert stats.column == "x"
    assert stats.count == 5
    assert stats.mean == 3.0
    assert stats.min == 1.0
    assert stats.max == 5.0
    assert stats.median == 3.0
    assert stats.iqr == pytest.approx(2.0)


def test_numeric_stats_quantiles_are_correct():
    df = pd.DataFrame({"x": np.arange(1, 101, dtype=float)})
    stats = compute_numeric_stats(df, "x")
    assert stats.q1 == pytest.approx(25.75)
    assert stats.q3 == pytest.approx(75.25)
    assert stats.iqr == pytest.approx(49.5)


def test_numeric_stats_counts_zeros_and_negatives():
    df = pd.DataFrame({"x": [0.0, -1.0, 2.0, 0.0, 3.0, -4.0]})
    stats = compute_numeric_stats(df, "x")
    assert stats.zeros_count == 2
    assert stats.zeros_percentage == pytest.approx(33.33, abs=0.01)
    assert stats.negative_count == 2


def test_numeric_stats_ignores_nulls():
    df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0]})
    stats = compute_numeric_stats(df, "x")
    assert stats.count == 3
    assert stats.mean == pytest.approx(7 / 3, abs=1e-3)  # rounded to 4dp


def test_numeric_stats_coerces_strings():
    df = pd.DataFrame({"x": ["1", "2", "3", "oops"]})
    stats = compute_numeric_stats(df, "x")
    assert stats.count == 3
    assert stats.max == 3.0


def test_empty_numeric_column_returns_zeroed_stats():
    df = pd.DataFrame({"x": [None, None]})
    stats = compute_numeric_stats(df, "x")
    assert stats.count == 0
    assert stats.mean == 0
    assert stats.iqr == 0


def test_numeric_stats_are_rounded_to_four_places(rng):
    df = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    stats = compute_numeric_stats(df, "x")
    for value in (stats.mean, stats.std, stats.q1, stats.median, stats.q3):
        assert value == round(value, 4)


def test_skew_and_kurtosis_are_reported(rng):
    df = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    stats = compute_numeric_stats(df, "x")
    assert abs(stats.skewness) < 0.5   # roughly symmetric
    assert stats.kurtosis < 1.0        # not heavy-tailed


# ── categorical statistics ──
def test_categorical_stats_identify_the_mode():
    df = pd.DataFrame({"c": ["a"] * 5 + ["b"] * 3 + ["c"] * 2})
    stats = compute_categorical_stats(df, "c")
    assert stats.top_value == "a"
    assert stats.top_frequency == 5
    assert stats.top_percentage == 50.0
    assert stats.unique_count == 3
    assert stats.count == 10


def test_categorical_stats_value_counts():
    df = pd.DataFrame({"c": ["a"] * 5 + ["b"] * 3 + ["c"] * 2})
    assert compute_categorical_stats(df, "c").value_counts == {"a": 5, "b": 3, "c": 2}


def test_entropy_is_zero_for_a_constant_column():
    df = pd.DataFrame({"c": ["same"] * 20})
    assert compute_categorical_stats(df, "c").entropy == 0.0


def test_entropy_is_one_for_two_equally_likely_values():
    df = pd.DataFrame({"c": ["a", "b"] * 50})
    assert compute_categorical_stats(df, "c").entropy == pytest.approx(1.0, abs=1e-3)


def test_entropy_is_two_for_four_equally_likely_values():
    df = pd.DataFrame({"c": ["a", "b", "c", "d"] * 25})
    assert compute_categorical_stats(df, "c").entropy == pytest.approx(2.0, abs=1e-3)


def test_categorical_stats_ignore_nulls():
    df = pd.DataFrame({"c": ["a", "b", None, None]})
    stats = compute_categorical_stats(df, "c")
    assert stats.count == 2
    assert stats.top_percentage == 50.0


def test_value_counts_are_capped_at_twenty():
    df = pd.DataFrame({"c": [f"v{i}" for i in range(100)]})
    assert len(compute_categorical_stats(df, "c").value_counts) == 20


# ── correlation ──
def test_correlation_matrix_shape(rng):
    df = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
    corr = compute_correlation_matrix(df)
    assert corr.shape == (2, 2)
    assert list(corr.columns) == ["a", "b"]


def test_correlation_detects_a_perfect_relationship():
    df = pd.DataFrame({"a": np.arange(50, dtype=float), "b": np.arange(50, dtype=float) * 3})
    assert compute_correlation_matrix(df).loc["a", "b"] == pytest.approx(1.0)


def test_correlation_is_symmetric_with_unit_diagonal(rng):
    df = pd.DataFrame({
        "a": rng.normal(0, 1, 100),
        "b": rng.normal(0, 1, 100),
        "c": rng.normal(0, 1, 100),
    })
    corr = compute_correlation_matrix(df)
    assert np.allclose(np.diag(corr.values), 1.0)
    assert np.allclose(corr.values, corr.values.T)


def test_correlation_needs_at_least_two_numeric_columns():
    assert compute_correlation_matrix(pd.DataFrame({"a": [1, 2, 3]})).empty


def test_correlation_ignores_non_numeric_columns(rng):
    df = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(0, 1, 50), "c": list("xy") * 25})
    assert compute_correlation_matrix(df).shape == (2, 2)


# ── whole-dataset report ──
def test_compute_all_stats_partitions_columns(mixed_df):
    stats = compute_all_stats(mixed_df)
    numeric = {s.column for s in stats.numeric_stats}
    categorical = {s.column for s in stats.categorical_stats}
    assert numeric == {"v", "w"}
    assert "cat" in categorical
    assert "flag" not in numeric
    assert numeric.isdisjoint(categorical)


def test_compute_all_stats_summary_counts(mixed_df):
    summary = compute_all_stats(mixed_df).summary
    assert summary["total_rows"] == len(mixed_df)
    assert summary["total_columns"] == len(mixed_df.columns)
    assert summary["numeric_columns"] == 2
    assert summary["categorical_columns"] >= 1
    assert summary["memory_mb"] >= 0.0
    assert set(summary) == {
        "total_rows", "total_columns", "numeric_columns",
        "categorical_columns", "memory_mb",
    }


def test_compute_all_stats_includes_correlation_when_possible(mixed_df):
    assert compute_all_stats(mixed_df).correlation_matrix is not None


def test_compute_all_stats_handles_single_numeric_column():
    df = pd.DataFrame({"only": np.arange(10, dtype=float)})
    stats = compute_all_stats(df)
    assert len(stats.numeric_stats) == 1
    assert stats.correlation_matrix is None


def test_compute_all_stats_on_empty_frame():
    stats = compute_all_stats(pd.DataFrame())
    assert stats.numeric_stats == []
    assert stats.categorical_stats == []
    assert stats.summary["total_rows"] == 0
