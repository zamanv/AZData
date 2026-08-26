"""Data quality scoring engine — computes a 0-100 health score based on multiple factors."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np


class IssueSeverity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class QualityIssue:
    """A single data quality issue."""
    column: Optional[str]  # None for row-level issues
    issue_type: str
    severity: IssueSeverity
    description: str
    count: int = 0
    percentage: float = 0.0


@dataclass
class QualityReport:
    """Complete data quality report."""
    overall_score: float  # 0-100
    grade: str  # A, B, C, D, F
    total_issues: int
    high_severity: int
    medium_severity: int
    low_severity: int
    issues: List[QualityIssue]
    breakdown: Dict[str, float]  # category -> score contribution
    summary: str


def _score_missing_values(df: pd.DataFrame) -> tuple:
    """Score based on missing value ratio. Returns (score, issues)."""
    issues: List[QualityIssue] = []
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())
    missing_ratio = total_missing / total_cells if total_cells > 0 else 0

    # Per-column analysis
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            null_pct = round(null_count / len(df) * 100, 1)
            if null_pct > 50:
                severity = IssueSeverity.HIGH
            elif null_pct > 20:
                severity = IssueSeverity.MEDIUM
            else:
                severity = IssueSeverity.LOW

            issues.append(QualityIssue(
                column=col,
                issue_type="missing_values",
                severity=severity,
                description=f"{null_count} missing values ({null_pct}%)",
                count=null_count,
                percentage=null_pct,
            ))

    # Score: 100 if no missing, down to 0 if >50% missing
    score = max(0, 100 - (missing_ratio * 200))
    return round(score, 1), issues


def _score_duplicates(df: pd.DataFrame) -> tuple:
    """Score based on duplicate rows."""
    issues: List[QualityIssue] = []
    dup_count = int(df.duplicated().sum())
    dup_pct = round(dup_count / len(df) * 100, 1) if len(df) > 0 else 0

    if dup_count > 0:
        severity = IssueSeverity.HIGH if dup_pct > 10 else (
            IssueSeverity.MEDIUM if dup_pct > 5 else IssueSeverity.LOW
        )
        issues.append(QualityIssue(
            column=None,
            issue_type="duplicate_rows",
            severity=severity,
            description=f"{dup_count} duplicate rows ({dup_pct}%)",
            count=dup_count,
            percentage=dup_pct,
        ))

    score = max(0, 100 - (dup_pct * 5))
    return round(score, 1), issues


def _score_constant_columns(df: pd.DataFrame) -> tuple:
    """Score based on columns with only one unique value."""
    issues: List[QualityIssue] = []
    constant_count = 0

    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            constant_count += 1
            issues.append(QualityIssue(
                column=col,
                issue_type="constant_column",
                severity=IssueSeverity.MEDIUM,
                description=f"Column '{col}' has only {nunique} unique value(s) — no information gain",
                count=1,
            ))

    total_cols = len(df.columns)
    ratio = constant_count / total_cols if total_cols > 0 else 0
    score = max(0, 100 - (ratio * 200))
    return round(score, 1), issues


def _score_outliers(df: pd.DataFrame) -> tuple:
    """Score based on extreme outlier ratio in numeric columns."""
    issues: List[QualityIssue] = []
    total_outliers = 0
    total_numeric_cells = 0

    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        outlier_mask = (series < lower) | (series > upper)
        outlier_count = int(outlier_mask.sum())
        total_outliers += outlier_count
        total_numeric_cells += len(series)

        if outlier_count > 0:
            outlier_pct = round(outlier_count / len(series) * 100, 1)
            if outlier_pct > 10:
                severity = IssueSeverity.HIGH
            elif outlier_pct > 5:
                severity = IssueSeverity.MEDIUM
            else:
                severity = IssueSeverity.LOW

            issues.append(QualityIssue(
                column=col,
                issue_type="outliers",
                severity=severity,
                description=f"{outlier_count} extreme outliers ({outlier_pct}%)",
                count=outlier_count,
                percentage=outlier_pct,
            ))

    ratio = total_outliers / total_numeric_cells if total_numeric_cells > 0 else 0
    score = max(0, 100 - (ratio * 300))
    return round(score, 1), issues


def _score_high_cardinality(df: pd.DataFrame) -> tuple:
    """Score based on high-cardinality categorical columns."""
    issues: List[QualityIssue] = []
    problem_cols = 0

    for col in df.select_dtypes(include=["object", "category"]).columns:
        nunique = df[col].nunique()
        if nunique > 100 and nunique / len(df) > 0.9:
            problem_cols += 1
            issues.append(QualityIssue(
                column=col,
                issue_type="high_cardinality",
                severity=IssueSeverity.LOW,
                description=f"High cardinality: {nunique} unique values — may be an identifier, not a feature",
                count=nunique,
            ))

    total_cat = len(df.select_dtypes(include=["object", "category"]).columns)
    ratio = problem_cols / total_cat if total_cat > 0 else 0
    score = max(0, 100 - (ratio * 100))
    return round(score, 1), issues


# ──────────────────────────────────────────────
# Main scoring function
# ──────────────────────────────────────────────
WEIGHTS = {
    "missing_values": 0.25,
    "duplicates": 0.20,
    "constant_columns": 0.15,
    "outliers": 0.15,
    "high_cardinality": 0.10,
    "base": 0.15,  # bonus for clean data
}


def compute_quality_score(df: pd.DataFrame) -> QualityReport:
    """Compute a comprehensive 0-100 data quality score.

    Args:
        df: The input DataFrame.

    Returns:
        QualityReport with overall score, grade, issues, and breakdown.
    """
    all_issues: List[QualityIssue] = []
    breakdown: Dict[str, float] = {}

    # Run all checks
    miss_score, miss_issues = _score_missing_values(df)
    dup_score, dup_issues = _score_duplicates(df)
    const_score, const_issues = _score_constant_columns(df)
    outlier_score, outlier_issues = _score_outliers(df)
    card_score, card_issues = _score_high_cardinality(df)

    breakdown["missing_values"] = miss_score
    breakdown["duplicates"] = dup_score
    breakdown["constant_columns"] = const_score
    breakdown["outliers"] = outlier_score
    breakdown["high_cardinality"] = card_score

    all_issues.extend(miss_issues)
    all_issues.extend(dup_issues)
    all_issues.extend(const_issues)
    all_issues.extend(outlier_issues)
    all_issues.extend(card_issues)

    # Weighted score
    weighted = (
        miss_score * WEIGHTS["missing_values"]
        + dup_score * WEIGHTS["duplicates"]
        + const_score * WEIGHTS["constant_columns"]
        + outlier_score * WEIGHTS["outliers"]
        + card_score * WEIGHTS["high_cardinality"]
        + 100 * WEIGHTS["base"]
    )
    overall_score = round(min(100, max(0, weighted)), 1)

    # Grade
    if overall_score >= 90:
        grade = "A"
    elif overall_score >= 80:
        grade = "B"
    elif overall_score >= 70:
        grade = "C"
    elif overall_score >= 60:
        grade = "D"
    else:
        grade = "F"

    high_count = sum(1 for i in all_issues if i.severity == IssueSeverity.HIGH)
    med_count = sum(1 for i in all_issues if i.severity == IssueSeverity.MEDIUM)
    low_count = sum(1 for i in all_issues if i.severity == IssueSeverity.LOW)

    # Summary
    if overall_score >= 90:
        summary = "Excellent data quality. Few or no issues detected."
    elif overall_score >= 75:
        summary = "Good data quality with some minor issues that should be addressed."
    elif overall_score >= 60:
        summary = "Moderate data quality. Several issues need attention before analysis."
    else:
        summary = "Poor data quality. Significant cleaning required before reliable analysis."

    return QualityReport(
        overall_score=overall_score,
        grade=grade,
        total_issues=len(all_issues),
        high_severity=high_count,
        medium_severity=med_count,
        low_severity=low_count,
        issues=all_issues,
        breakdown=breakdown,
        summary=summary,
    )
