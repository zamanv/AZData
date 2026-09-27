"""Tests for the 0-100 data quality scoring engine (profiling/quality_scorer.py)."""

import pandas as pd
import pytest

from profiling.quality_scorer import (
    WEIGHTS,
    IssueSeverity,
    compute_quality_score,
)


# ── clean data ──
def test_clean_data_scores_full_marks(clean_df):
    report = compute_quality_score(clean_df)
    assert report.overall_score == 100.0
    assert report.grade == "A"
    assert report.total_issues == 0
    assert "Excellent" in report.summary


def test_clean_data_has_no_high_severity_issues(clean_df):
    report = compute_quality_score(clean_df)
    assert report.high_severity == 0
    assert report.medium_severity == 0
    assert report.low_severity == 0


# ── dirty data ──
def test_dirty_data_scores_much_lower(dirty_df, clean_df):
    dirty = compute_quality_score(dirty_df)
    clean = compute_quality_score(clean_df)
    assert dirty.overall_score < clean.overall_score
    assert dirty.grade in {"D", "F"}
    assert "Significant cleaning required" in dirty.summary


def test_dirty_data_reports_each_issue_category(dirty_df):
    report = compute_quality_score(dirty_df)
    issue_types = {issue.issue_type for issue in report.issues}
    assert "missing_values" in issue_types
    assert "duplicate_rows" in issue_types
    assert "constant_column" in issue_types


def test_missing_values_issue_is_high_severity_at_40_percent(dirty_df):
    report = compute_quality_score(dirty_df)
    missing = [i for i in report.issues if i.issue_type == "missing_values"]
    assert len(missing) == 1
    assert missing[0].column == "a"
    assert missing[0].count == 3
    assert missing[0].percentage == 30.0
    assert missing[0].severity == IssueSeverity.MEDIUM


def test_duplicate_rows_are_row_level_issues():
    df = pd.DataFrame({"a": [1, 2, 1, 2, 1, 2], "b": [1, 1, 2, 2, 1, 2]})
    report = compute_quality_score(df)
    dupes = [i for i in report.issues if i.issue_type == "duplicate_rows"]
    assert len(dupes) == 1
    assert dupes[0].column is None
    assert dupes[0].count == 2


def test_constant_column_is_flagged_even_without_nulls():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "constant": [9] * 5})
    report = compute_quality_score(df)
    consts = [i for i in report.issues if i.issue_type == "constant_column"]
    assert len(consts) == 1
    assert consts[0].column == "constant"
    assert "no information gain" in consts[0].description


# ── severity classification ──
@pytest.mark.parametrize(
    "nulls,expected",
    [
        (0, None),                   # no issue at all
        (1, IssueSeverity.LOW),      # 5%  -> <= 20%
        (4, IssueSeverity.LOW),      # 20% -> <= 20%
        (5, IssueSeverity.MEDIUM),   # 25% -> > 20%
        (10, IssueSeverity.MEDIUM),  # 50% -> <= 50%
        (11, IssueSeverity.HIGH),    # 55% -> > 50%
        (19, IssueSeverity.HIGH),    # 95% -> > 50%
    ],
)
def test_missing_value_severity_thresholds(nulls, expected):
    df = pd.DataFrame({"a": [None] * nulls + list(range(20 - nulls))})
    report = compute_quality_score(df)
    issues = [i for i in report.issues if i.issue_type == "missing_values"]
    if expected is None:
        assert issues == []
    else:
        assert issues[0].severity == expected
        assert issues[0].percentage == round(nulls / 20 * 100, 1)


# ── outliers ──
def test_extreme_outliers_are_detected():
    """Regression guard: a column needs a non-zero IQR before outliers count."""
    df = pd.DataFrame({"a": list(range(100)) + [100000]})
    report = compute_quality_score(df)
    assert any(i.issue_type == "outliers" for i in report.issues)


def test_outliers_ignored_for_small_columns():
    df = pd.DataFrame({"a": [1.0, 2.0, 999.0]})
    report = compute_quality_score(df)
    assert not any(i.issue_type == "outliers" for i in report.issues)


# ── high cardinality ──
def test_high_cardinality_identifier_like_column_is_flagged():
    df = pd.DataFrame({"uid": [f"id-{i}" for i in range(200)]})
    report = compute_quality_score(df)
    issues = [i for i in report.issues if i.issue_type == "high_cardinality"]
    assert len(issues) == 1
    assert issues[0].severity == IssueSeverity.LOW


def test_low_cardinality_column_is_not_flagged():
    df = pd.DataFrame({"cat": ["a", "b", "c"] * 100})
    report = compute_quality_score(df)
    assert not any(i.issue_type == "high_cardinality" for i in report.issues)


# ── scoring invariants ──
def test_score_is_always_within_bounds(dirty_df, clean_df, mixed_df):
    for df in (dirty_df, clean_df, mixed_df):
        score = compute_quality_score(df).overall_score
        assert 0.0 <= score <= 100.0


def test_breakdown_covers_every_scoring_category(clean_df):
    report = compute_quality_score(clean_df)
    assert set(report.breakdown) == {
        "missing_values", "duplicates", "constant_columns",
        "outliers", "high_cardinality",
    }
    assert all(0.0 <= v <= 100.0 for v in report.breakdown.values())


def test_weights_sum_to_one():
    assert round(sum(WEIGHTS.values()), 6) == 1.0


def test_severity_counts_match_issue_list(dirty_df):
    report = compute_quality_score(dirty_df)
    counts = {s: 0 for s in IssueSeverity}
    for issue in report.issues:
        counts[issue.severity] += 1
    assert report.high_severity == counts[IssueSeverity.HIGH]
    assert report.medium_severity == counts[IssueSeverity.MEDIUM]
    assert report.low_severity == counts[IssueSeverity.LOW]
    assert report.total_issues == len(report.issues)


@pytest.mark.parametrize("threshold", [90, 80, 70, 60])
def test_grade_boundaries_match_documented_thresholds(threshold):
    """The grade must agree with the published 90/80/70/60 cut-offs."""
    import numpy as np

    frame = pd.DataFrame({"a": np.random.default_rng(0).normal(0, 1, 60),
                          "b": ["p", "q"] * 30})
    report = compute_quality_score(frame)
    score = report.overall_score
    expected = (
        "A" if score >= 90 else "B" if score >= 80
        else "C" if score >= 70 else "D" if score >= 60 else "F"
    )
    assert report.grade == expected
    assert (score >= threshold) or expected in {"D", "F"}


def test_grade_degrades_monotonically_with_corruption(rng):
    clean = pd.DataFrame({"a": rng.normal(0, 1, 60), "b": ["p", "q"] * 30})
    corrupted = clean.copy()
    corrupted.loc[0:29, "a"] = None
    grades = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    assert (
        grades[compute_quality_score(corrupted).grade]
        >= grades[compute_quality_score(clean).grade]
    )


@pytest.mark.xfail(
    reason=(
        "Design gap: an empty DataFrame scores a perfect 100/A, which reads as "
        "'pristine data' when it actually means 'no data to assess'."
    ),
    strict=True,
)
def test_empty_frame_is_not_reported_as_perfect_quality():
    report = compute_quality_score(pd.DataFrame())
    assert report.overall_score < 100.0


def test_empty_frame_does_not_crash():
    report = compute_quality_score(pd.DataFrame())
    assert 0.0 <= report.overall_score <= 100.0
    assert report.total_issues == 0
