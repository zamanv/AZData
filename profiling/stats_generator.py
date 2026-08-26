"""Descriptive statistics generator for numeric and categorical columns."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np


@dataclass
class NumericStats:
    """Descriptive statistics for a numeric column."""
    column: str
    count: int
    mean: float
    std: float
    min: float
    q1: float
    median: float
    q3: float
    max: float
    skewness: float
    kurtosis: float
    zeros_count: int
    zeros_percentage: float
    negative_count: int
    iqr: float


@dataclass
class CategoricalStats:
    """Descriptive statistics for a categorical column."""
    column: str
    count: int
    unique_count: int
    top_value: str
    top_frequency: int
    top_percentage: float
    value_counts: Dict[str, int]
    entropy: float


@dataclass
class DatasetStats:
    """Complete statistics report for a dataset."""
    numeric_stats: List[NumericStats]
    categorical_stats: List[CategoricalStats]
    correlation_matrix: Optional[pd.DataFrame] = None
    summary: Dict[str, Any] = field(default_factory=dict)


def compute_numeric_stats(df: pd.DataFrame, column: str) -> NumericStats:
    """Compute descriptive statistics for a single numeric column.

    Args:
        df: The input DataFrame.
        column: Column name.

    Returns:
        NumericStats with computed statistics.
    """
    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if len(series) == 0:
        return NumericStats(
            column=column, count=0, mean=0, std=0, min=0,
            q1=0, median=0, q3=0, max=0, skewness=0, kurtosis=0,
            zeros_count=0, zeros_percentage=0, negative_count=0, iqr=0,
        )

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    zeros = int((series == 0).sum())
    negatives = int((series < 0).sum())

    return NumericStats(
        column=column,
        count=int(len(series)),
        mean=round(float(series.mean()), 4),
        std=round(float(series.std()), 4),
        min=round(float(series.min()), 4),
        q1=round(q1, 4),
        median=round(float(series.median()), 4),
        q3=round(q3, 4),
        max=round(float(series.max()), 4),
        skewness=round(float(series.skew()), 4),
        kurtosis=round(float(series.kurtosis()), 4),
        zeros_count=zeros,
        zeros_percentage=round(zeros / len(series) * 100, 2),
        negative_count=negatives,
        iqr=round(q3 - q1, 4),
    )


def compute_categorical_stats(df: pd.DataFrame, column: str) -> CategoricalStats:
    """Compute descriptive statistics for a single categorical column.

    Args:
        df: The input DataFrame.
        column: Column name.

    Returns:
        CategoricalStats with computed statistics.
    """
    series = df[column].dropna()
    value_counts = series.value_counts()

    # Shannon entropy
    probs = value_counts / len(series)
    entropy = -float((probs * np.log2(probs)).sum()) if len(probs) > 0 else 0

    top_val = str(value_counts.index[0]) if len(value_counts) > 0 else ""
    top_freq = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0

    return CategoricalStats(
        column=column,
        count=int(len(series)),
        unique_count=int(series.nunique()),
        top_value=top_val,
        top_frequency=top_freq,
        top_percentage=round(top_freq / len(series) * 100, 2) if len(series) > 0 else 0,
        value_counts={str(k): int(v) for k, v in value_counts.head(20).items()},
        entropy=round(entropy, 4),
    )


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix for numeric columns.

    Args:
        df: The input DataFrame.

    Returns:
        Correlation matrix DataFrame.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr().round(4)


def compute_all_stats(df: pd.DataFrame) -> DatasetStats:
    """Compute complete statistics for all columns in a DataFrame.

    Args:
        df: The input DataFrame.

    Returns:
        DatasetStats with all computed statistics.
    """
    numeric_stats: List[NumericStats] = []
    categorical_stats: List[CategoricalStats] = []

    for col in df.select_dtypes(include=[np.number]).columns:
        numeric_stats.append(compute_numeric_stats(df, col))

    for col in df.select_dtypes(include=["object", "category"]).columns:
        categorical_stats.append(compute_categorical_stats(df, col))

    corr = compute_correlation_matrix(df)

    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": len(numeric_stats),
        "categorical_columns": len(categorical_stats),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }

    return DatasetStats(
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
        correlation_matrix=corr if not corr.empty else None,
        summary=summary,
    )
