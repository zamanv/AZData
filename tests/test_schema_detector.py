"""Tests for schema inference and date-column detection (profiling/schema_detector.py)."""

import pandas as pd
import pytest

from profiling.schema_detector import (
    DATE_KEYWORDS,
    DATE_PATTERNS,
    DatasetSchema,
    _infer_column_type,
    _is_likely_datetime_column,
    detect_date_columns,
    detect_schema,
)


# ── column-level type inference ──
def test_infers_numeric_for_continuous_numbers(rng):
    assert _infer_column_type("x", pd.Series(rng.normal(0, 1, 100))) == "numeric"


def test_infers_categorical_for_low_cardinality_strings():
    assert _infer_column_type("cat", pd.Series(["a", "b", "c"] * 40)) == "categorical"


def test_infers_boolean_for_true_false_columns():
    assert _infer_column_type("flag", pd.Series([True] * 50 + [False] * 50)) == "boolean"
    assert _infer_column_type("flag", pd.Series(["yes"] * 50 + ["no"] * 50)) == "boolean"


def test_infers_text_for_long_unique_strings():
    long_text = [f"value {'abcdefghij' * 8}{i}" for i in range(100)]
    assert _infer_column_type("body", pd.Series(long_text)) == "text"


def test_low_cardinality_numeric_is_treated_as_categorical():
    """A numeric column with almost no unique values carries no signal."""
    assert _infer_column_type("grade", pd.Series([1] * 95 + [2] * 5)) == "categorical"


def test_all_null_column_falls_back_to_categorical():
    assert _infer_column_type("x", pd.Series([None] * 10, dtype="object")) == "categorical"


# ── datetime heuristics ──
def test_date_keyword_in_name_is_enough():
    assert _is_likely_datetime_column("created_at", pd.Series(["a", "b"])) is True
    assert _is_likely_datetime_column("order_date", pd.Series(["a", "b"])) is True


def test_date_patterns_recognise_string_dates():
    assert _is_likely_datetime_column("x", pd.Series(["2024-01-15"] * 20)) is True
    assert _is_likely_datetime_column("x", pd.Series(["01/15/2024"] * 20)) is True


def test_non_date_strings_are_not_datetimes():
    assert _is_likely_datetime_column("name", pd.Series(["alice", "bob"] * 20)) is False


def test_eight_leading_digits_trigger_the_compact_date_pattern():
    """Documents a false positive: the ``\\d{4}\\d{2}\\d{2}`` pattern is not
    anchored at the end, so ANY string whose first 8 characters are digits is
    classified as a date even when the rest is prose.

    A tightening fix would anchor the pattern (e.g. ``\\d{4}\\d{2}\\d{2}$``) and
    require the column to be homogeneous. Pinned here so the change is visible.
    """
    identifier_like = [f"{i:08d}-customer-record" for i in range(20)]
    assert _is_likely_datetime_column("ref", pd.Series(identifier_like)) is True


def test_date_keyword_set_is_populated():
    assert {"date", "created", "timestamp"} <= DATE_KEYWORDS
    assert len(DATE_PATTERNS) >= 5


# ── dataset-level report ──
def test_detect_schema_reports_shape(mixed_df):
    schema = detect_schema(mixed_df)
    assert isinstance(schema, DatasetSchema)
    assert schema.total_rows == len(mixed_df)
    assert schema.total_columns == len(mixed_df.columns)
    assert len(schema.columns) == len(mixed_df.columns)


def test_detect_schema_buckets_columns_by_type(mixed_df):
    schema = detect_schema(mixed_df)
    assert "v" in schema.numeric_columns
    assert "cat" in schema.categorical_columns
    assert "flag" in schema.boolean_columns
    assert "created_at" in schema.datetime_columns


def test_every_column_lands_in_exactly_one_bucket(mixed_df):
    schema = detect_schema(mixed_df)
    buckets = (
        schema.numeric_columns + schema.categorical_columns
        + schema.datetime_columns + schema.boolean_columns + schema.text_columns
    )
    assert sorted(buckets) == sorted(mixed_df.columns)
    assert len(buckets) == len(set(buckets))


def test_column_stats_include_numeric_summary(mixed_df):
    schema = detect_schema(mixed_df)
    col = next(c for c in schema.columns if c.name == "v")
    assert set(col.stats) == {"mean", "std", "min", "max", "median"}
    assert col.stats["max"] == pytest.approx(50.0)


def test_column_stats_include_mode_for_categoricals(mixed_df):
    schema = detect_schema(mixed_df)
    col = next(c for c in schema.columns if c.name == "cat")
    assert "mode" in col.stats
    assert "top_values" in col.stats


def test_null_accounting_is_consistent(mixed_df):
    schema = detect_schema(mixed_df)
    for col in schema.columns:
        assert col.non_null_count + col.null_count == schema.total_rows


def test_duplicate_rows_are_counted():
    df = pd.DataFrame({"a": [1, 1, 1, 2, 2, 3]})
    schema = detect_schema(df)
    assert schema.duplicate_rows == 3
    assert schema.duplicate_percentage == 50.0


def test_memory_usage_is_reported(mixed_df):
    assert detect_schema(mixed_df).memory_usage_mb >= 0.0


def test_empty_frame_yields_empty_schema():
    schema = detect_schema(pd.DataFrame())
    assert schema.total_rows == 0
    assert schema.total_columns == 0
    assert schema.columns == []


# ── date column scanning ──
def test_detect_date_columns_finds_iso_date_strings():
    df = pd.DataFrame({"d": ["2024-01-01"] * 20, "n": list(range(20))})
    found = dict(detect_date_columns(df))
    assert "d" in found
    assert found["d"] == 1.0


def test_detect_date_columns_finds_native_datetime_dtype():
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=5)})
    assert dict(detect_date_columns(df))["ts"] == 1.0


def test_detect_date_columns_ignores_numbers():
    df = pd.DataFrame({"n": list(range(20))})
    assert detect_date_columns(df) == []


def test_detect_date_columns_results_are_sorted_by_parse_rate():
    df = pd.DataFrame({
        "mostly": [None] * 5 + ["2024-01-0{}".format(i % 9 + 1) for i in range(15)],
        "always": ["2024-01-01"] * 20,
    })
    rates = [rate for _, rate in detect_date_columns(df)]
    assert rates == sorted(rates, reverse=True)


def test_native_datetime_dtype_is_always_inferred_as_datetime():
    """Regression: a column pandas already parsed to datetime64 must be typed
    as datetime regardless of its name."""
    series = pd.Series(pd.date_range("2024-01-01", periods=10))
    assert _infer_column_type("value", series) == "datetime"
    assert _infer_column_type("dt", series) == "datetime"
    assert _infer_column_type("created_at", series) == "datetime"


def test_datetime_inference_no_longer_depends_on_column_name():
    values = pd.Series(pd.date_range("2024-01-01", periods=10))
    inferred = {_infer_column_type(name, values) for name in ("created_at", "dt", "value")}
    assert inferred == {"datetime"}


def test_string_dtype_columns_are_still_scanned_for_dates():
    """Regression: pandas 3.0 infers StringDtype for pure-string columns, so
    an ``dtype == 'object'`` guard would silently skip every string date."""
    series = pd.Series(["2024-01-15"] * 20)
    assert pd.api.types.is_string_dtype(series)
    assert _is_likely_datetime_column("col", series) is True
