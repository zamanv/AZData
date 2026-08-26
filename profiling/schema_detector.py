"""Schema inference engine — auto-detects column types, date columns, and data structure."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import re
from datetime import datetime


@dataclass
class ColumnSchema:
    """Schema information for a single column."""
    name: str
    inferred_type: str  # 'numeric', 'categorical', 'datetime', 'boolean', 'text'
    pandas_dtype: str
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: List[Any] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSchema:
    """Complete schema information for a dataset."""
    columns: List[ColumnSchema]
    total_rows: int
    total_columns: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    boolean_columns: List[str]
    text_columns: List[str]
    memory_usage_mb: float
    duplicate_rows: int
    duplicate_percentage: float


# ──────────────────────────────────────────────
# Date pattern detection
# ──────────────────────────────────────────────
DATE_PATTERNS: List[str] = [
    r"\d{4}-\d{2}-\d{2}",           # 2024-01-15
    r"\d{2}/\d{2}/\d{4}",           # 01/15/2024
    r"\d{2}-\d{2}-\d{4}",           # 15-01-2024
    r"\d{4}/\d{2}/\d{2}",           # 2024/01/15
    r"\d{4}\d{2}\d{2}",             # 20240115
    r"\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}",
]

DATE_KEYWORDS: frozenset = frozenset({
    "date", "time", "timestamp", "datetime", "created", "updated",
    "modified", "year", "month", "day", "hour", "minute", "second",
    "period", "quarter", "fiscal", "dob", "birth", "hire", "start",
    "end", "expire", "due", "issued", "published", "indexed",
})


def _is_likely_datetime_column(col_name: str, sample: pd.Series) -> bool:
    """Heuristic: is this column likely a datetime column?"""
    name_lower = col_name.lower().strip()

    # Check name for date keywords
    if any(kw in name_lower for kw in DATE_KEYWORDS):
        return True

    # Check if pandas can parse it
    if sample.dtype == "object" and len(sample) > 0:
        non_null = sample.dropna().head(50)
        if len(non_null) == 0:
            return False

        # Try pandas datetime conversion
        try:
            pd.to_datetime(non_null, infer_datetime_format=True, errors="raise")
            return True
        except (ValueError, TypeError):
            pass

        # Check regex patterns
        sample_strs = non_null.astype(str).head(20)
        for pattern in DATE_PATTERNS:
            matches = sample_strs.str.match(pattern, na=False)
            if matches.mean() > 0.8:
                return True

    return False


def _infer_column_type(
    col_name: str,
    series: pd.Series,
) -> str:
    """Infer the semantic type of a column.

    Returns one of: 'numeric', 'categorical', 'datetime', 'boolean', 'text'.
    """
    non_null = series.dropna()

    if len(non_null) == 0:
        return "categorical"

    # Boolean check
    unique_vals = set(non_null.unique())
    if unique_vals.issubset({True, False, "True", "False", "true", "false", "yes", "no", "Yes", "No", 0, 1}):
        return "boolean"

    # Datetime check
    if _is_likely_datetime_column(col_name, series):
        return "datetime"

    # Numeric check
    if pd.api.types.is_numeric_dtype(series):
        # But if there are very few unique values relative to row count, treat as categorical
        if len(non_null) > 0 and len(unique_vals) / len(non_null) < 0.05 and len(unique_vals) <= 15:
            return "categorical"
        return "numeric"

    # Text vs Categorical heuristic
    # If unique ratio is high and strings are long → text
    # If unique ratio is low → categorical
    unique_ratio = len(unique_vals) / len(non_null) if len(non_null) > 0 else 0
    avg_str_len = non_null.astype(str).str.len().mean()

    if unique_ratio > 0.8 and avg_str_len > 50:
        return "text"
    elif unique_ratio > 0.5 and avg_str_len > 100:
        return "text"
    else:
        return "categorical"


def detect_schema(df: pd.DataFrame) -> DatasetSchema:
    """Analyze a DataFrame and produce a complete schema report.

    Args:
        df: The input DataFrame.

    Returns:
        DatasetSchema with detailed column and dataset information.
    """
    columns: List[ColumnSchema] = []
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []
    boolean_cols: List[str] = []
    text_cols: List[str] = []

    total_rows = len(df)

    for col_name in df.columns:
        series = df[col_name]
        non_null_count = int(series.notna().sum())
        null_count = int(series.isna().sum())
        null_pct = round(null_count / total_rows * 100, 2) if total_rows > 0 else 0
        unique_count = int(series.nunique())
        unique_pct = round(unique_count / total_rows * 100, 2) if total_rows > 0 else 0

        inferred_type = _infer_column_type(col_name, series)

        # Collect sample values
        sample = series.dropna().head(5).tolist()
        sample = [str(v) if not isinstance(v, (int, float, bool)) else v for v in sample]

        # Per-type stats
        stats: Dict[str, Any] = {}
        if inferred_type == "numeric":
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric_series) > 0:
                stats = {
                    "mean": round(float(numeric_series.mean()), 4),
                    "std": round(float(numeric_series.std()), 4),
                    "min": round(float(numeric_series.min()), 4),
                    "max": round(float(numeric_series.max()), 4),
                    "median": round(float(numeric_series.median()), 4),
                }
        elif inferred_type == "categorical":
            top_values = series.value_counts().head(5)
            stats = {
                "top_values": top_values.to_dict(),
                "mode": str(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
            }

        col_schema = ColumnSchema(
            name=col_name,
            inferred_type=inferred_type,
            pandas_dtype=str(series.dtype),
            non_null_count=non_null_count,
            null_count=null_count,
            null_percentage=null_pct,
            unique_count=unique_count,
            unique_percentage=unique_pct,
            sample_values=sample,
            stats=stats,
        )
        columns.append(col_schema)

        # Categorize
        if inferred_type == "numeric":
            numeric_cols.append(col_name)
        elif inferred_type == "categorical":
            categorical_cols.append(col_name)
        elif inferred_type == "datetime":
            datetime_cols.append(col_name)
        elif inferred_type == "boolean":
            boolean_cols.append(col_name)
        elif inferred_type == "text":
            text_cols.append(col_name)

    # Memory usage
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = round(memory_bytes / (1024 * 1024), 2)

    # Duplicates
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round(duplicate_count / total_rows * 100, 2) if total_rows > 0 else 0

    return DatasetSchema(
        columns=columns,
        total_rows=total_rows,
        total_columns=len(df.columns),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        boolean_columns=boolean_cols,
        text_columns=text_cols,
        memory_usage_mb=memory_mb,
        duplicate_rows=duplicate_count,
        duplicate_percentage=duplicate_pct,
    )


def detect_date_columns(df: pd.DataFrame) -> List[Tuple[str, int]]:
    """Find columns that can be parsed as datetime.

    Returns:
        List of (column_name, parse成功率) tuples, sorted by parse rate desc.
    """
    results: List[Tuple[str, int]] = []

    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                parse_rate = parsed.notna().mean()
                if parse_rate > 0.8:
                    results.append((col, parse_rate))
            except Exception:
                pass
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            results.append((col, 1.0))

    return sorted(results, key=lambda x: x[1], reverse=True)
