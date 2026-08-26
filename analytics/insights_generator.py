"""LLM-powered insights generator — contextualizes computed statistics into natural language."""

from typing import List, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from llm.code_generator import CodeGenerator, GeneratedCode


@dataclass
class InsightResult:
    """Result of insights generation."""
    success: bool
    insights: List[str]
    source: str  # 'llm' or 'heuristic'
    error: Optional[str] = None


def _generate_heuristic_insights(df: pd.DataFrame) -> List[str]:
    """Generate basic insights without LLM as fallback.

    Args:
        df: Input DataFrame.

    Returns:
        List of insight strings.
    """
    insights = []
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    missing_pct = round(missing / (rows * cols) * 100, 1) if rows * cols > 0 else 0
    dups = int(df.duplicated().sum())

    insights.append(f"Dataset contains {rows:,} rows and {cols} columns.")

    if missing > 0:
        insights.append(f"Found {missing:,} missing values ({missing_pct}% of all cells).")
    else:
        insights.append("No missing values detected — clean dataset.")

    if dups > 0:
        dup_pct = round(dups / rows * 100, 1)
        insights.append(f"Found {dups:,} duplicate rows ({dup_pct}%).")

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:3]:
        series = df[col].dropna()
        if len(series) > 0:
            mean = series.mean()
            std = series.std()
            cv = std / mean if mean != 0 else 0
            if cv > 1:
                insights.append(f"'{col}' shows high variability (CV={cv:.2f}).")

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols[:2]:
        nunique = df[col].nunique()
        if nunique > 50:
            insights.append(f"'{col}' has {nunique} unique values — consider if it's an identifier.")

    return insights


def generate_insights(
    df: pd.DataFrame,
    code_generator: Optional[CodeGenerator] = None,
) -> InsightResult:
    """Generate natural language insights from a DataFrame.

    Tries LLM first, falls back to heuristic generation.

    Args:
        df: Input DataFrame.
        code_generator: Optional CodeGenerator for LLM insights.

    Returns:
        InsightResult with list of insight strings.
    """
    # Build stats summary for LLM
    stats_lines = []
    stats_lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    stats_lines.append(f"Missing cells: {df.isna().sum().sum()} ({round(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 1)}%)")
    stats_lines.append(f"Duplicate rows: {df.duplicated().sum()}")
    stats_lines.append("")

    # Numeric summaries
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        stats_lines.append("NUMERIC COLUMNS:")
        desc = numeric_df.describe().round(2)
        stats_lines.append(desc.to_string())
        stats_lines.append("")

    # Categorical summaries
    cat_df = df.select_dtypes(include=["object", "category"])
    if len(cat_df.columns) > 0:
        stats_lines.append("CATEGORICAL COLUMNS:")
        for col in cat_df.columns[:5]:
            vc = df[col].value_counts().head(5)
            stats_lines.append(f"  {col}: {df[col].nunique()} unique values")
            stats_lines.append(f"    Top: {vc.to_dict()}")
        stats_lines.append("")

    # Correlations
    if len(numeric_df.columns) >= 2:
        corr = numeric_df.corr()
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    high_corr.append(
                        f"{corr.columns[i]} <-> {corr.columns[j]}: {val:.3f}"
                    )
        if high_corr:
            stats_lines.append("HIGH CORRELATIONS (|r| > 0.7):")
            stats_lines.extend(high_corr)

    stats_summary = "\n".join(stats_lines)

    # Try LLM
    if code_generator:
        result = code_generator.generate_insights(stats_summary)
        if result.success:
            # Parse JSON array from response
            import json
            try:
                text = result.code.strip()
                # Try to extract JSON array
                start = text.find("[")
                end = text.rfind("]") + 1
                if start >= 0 and end > start:
                    insights = json.loads(text[start:end])
                    if isinstance(insights, list):
                        return InsightResult(
                            success=True,
                            insights=insights,
                            source="llm",
                        )
            except (json.JSONDecodeError, ValueError):
                pass

    # Fallback to heuristic
    insights = _generate_heuristic_insights(df)
    return InsightResult(
        success=True,
        insights=insights,
        source="heuristic",
    )
