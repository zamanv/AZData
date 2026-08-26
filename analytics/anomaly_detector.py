"""Outlier detection using Z-Score and IQR methods with LLM-powered explanations."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.config import ANOMALY_ZSCORE_THRESHOLD, ANOMALY_IQR_FACTOR


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single column."""
    column: str
    method: str
    anomaly_indices: List[int]
    anomaly_values: List[Any]
    anomaly_count: int
    anomaly_percentage: float
    threshold: float
    mean: float
    std: float
    fig: Optional[go.Figure] = None
    explanation: Optional[str] = None


def detect_zscore_anomalies(
    df: pd.DataFrame,
    column: str,
    threshold: float = ANOMALY_ZSCORE_THRESHOLD,
) -> Optional[AnomalyResult]:
    """Detect outliers using Z-Score method.

    Args:
        df: Input DataFrame.
        column: Column name to analyze.
        threshold: Z-Score threshold (default 3.0).

    Returns:
        AnomalyResult or None if column not numeric / no data.
    """
    series = pd.to_numeric(df[column], errors="coerce")
    non_null = series.dropna()

    if len(non_null) < 10:
        return None

    mean = float(non_null.mean())
    std = float(non_null.std())

    if std == 0:
        return None

    z_scores = (series - mean) / std
    anomaly_mask = z_scores.abs() > threshold

    anomaly_indices = series.index[anomaly_mask].tolist()
    anomaly_values = series[anomaly_mask].tolist()

    if not anomaly_indices:
        return None

    # Create visualization
    fig = go.Figure()

    # Normal points
    normal_mask = ~anomaly_mask & series.notna()
    fig.add_trace(go.Scatter(
        x=df.index[normal_mask],
        y=series[normal_mask],
        mode="markers",
        name="Normal",
        marker=dict(color="#4C78A8", size=6, opacity=0.6),
        hovertemplate="Index: %{x}<br>Value: %{y:.2f}<extra></extra>",
    ))

    # Anomaly points
    fig.add_trace(go.Scatter(
        x=df.index[anomaly_mask],
        y=series[anomaly_mask],
        mode="markers",
        name=f"Anomaly (|z|>{threshold})",
        marker=dict(color="#E45756", size=10, symbol="x", line=dict(width=2)),
        hovertemplate="Index: %{x}<br>Value: %{y:.2f}<br>Z-Score: %{customdata:.2f}<extra></extra>",
        customdata=z_scores[anomaly_mask],
    ))

    fig.update_layout(
        title=f"Z-Score Anomalies: {column} (threshold={threshold})",
        title_x=0.5,
        xaxis_title="Index",
        yaxis_title=column,
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return AnomalyResult(
        column=column,
        method="zscore",
        anomaly_indices=anomaly_indices[:100],  # cap for display
        anomaly_values=[float(v) for v in anomaly_values[:100]],
        anomaly_count=len(anomaly_indices),
        anomaly_percentage=round(len(anomaly_indices) / len(non_null) * 100, 2),
        threshold=threshold,
        mean=mean,
        std=std,
        fig=fig,
    )


def detect_iqr_anomalies(
    df: pd.DataFrame,
    column: str,
    factor: float = ANOMALY_IQR_FACTOR,
) -> Optional[AnomalyResult]:
    """Detect outliers using IQR method.

    Args:
        df: Input DataFrame.
        column: Column name to analyze.
        factor: IQR multiplier (default 1.5).

    Returns:
        AnomalyResult or None if column not numeric / no data.
    """
    series = pd.to_numeric(df[column], errors="coerce")
    non_null = series.dropna()

    if len(non_null) < 10:
        return None

    q1 = float(non_null.quantile(0.25))
    q3 = float(non_null.quantile(0.75))
    iqr = q3 - q1

    if iqr == 0:
        return None

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    anomaly_mask = (series < lower_bound) | (series > upper_bound)
    anomaly_indices = series.index[anomaly_mask].tolist()
    anomaly_values = series[anomaly_mask].tolist()

    if not anomaly_indices:
        return None

    mean = float(non_null.mean())
    std = float(non_null.std())

    # Create visualization
    fig = go.Figure()

    normal_mask = ~anomaly_mask & series.notna()
    fig.add_trace(go.Scatter(
        x=df.index[normal_mask],
        y=series[normal_mask],
        mode="markers",
        name="Normal",
        marker=dict(color="#4C78A8", size=6, opacity=0.6),
    ))

    fig.add_trace(go.Scatter(
        x=df.index[anomaly_mask],
        y=series[anomaly_mask],
        mode="markers",
        name=f"Anomaly (IQR×{factor})",
        marker=dict(color="#E45756", size=10, symbol="x", line=dict(width=2)),
    ))

    # Add boundary lines
    fig.add_hline(y=upper_bound, line_dash="dash", line_color="orange",
                  annotation_text=f"Upper: {upper_bound:.2f}")
    fig.add_hline(y=lower_bound, line_dash="dash", line_color="orange",
                  annotation_text=f"Lower: {lower_bound:.2f}")

    fig.update_layout(
        title=f"IQR Anomalies: {column} (factor={factor})",
        title_x=0.5,
        xaxis_title="Index",
        yaxis_title=column,
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return AnomalyResult(
        column=column,
        method="iqr",
        anomaly_indices=anomaly_indices[:100],
        anomaly_values=[float(v) for v in anomaly_values[:100]],
        anomaly_count=len(anomaly_indices),
        anomaly_percentage=round(len(anomaly_indices) / len(non_null) * 100, 2),
        threshold=factor,
        mean=mean,
        std=std,
        fig=fig,
    )


def detect_all_anomalies(
    df: pd.DataFrame,
    methods: List[str] = None,
) -> List[AnomalyResult]:
    """Run anomaly detection on all numeric columns.

    Args:
        df: Input DataFrame.
        methods: List of methods to use ('zscore', 'iqr'). Default: both.

    Returns:
        List of AnomalyResult for all detected anomalies.
    """
    if methods is None:
        methods = ["zscore", "iqr"]

    results: List[AnomalyResult] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if "zscore" in methods:
            result = detect_zscore_anomalies(df, col)
            if result:
                results.append(result)

        if "iqr" in methods:
            result = detect_iqr_anomalies(df, col)
            if result:
                results.append(result)

    return results


def anomalies_to_dataframe(results: List[AnomalyResult]) -> pd.DataFrame:
    """Convert anomaly results to a summary DataFrame.

    Args:
        results: List of AnomalyResult objects.

    Returns:
        Summary DataFrame with one row per anomaly detection.
    """
    rows = []
    for r in results:
        rows.append({
            "Column": r.column,
            "Method": r.method,
            "Anomalies": r.anomaly_count,
            "Percentage": f"{r.anomaly_percentage}%",
            "Threshold": r.threshold,
            "Mean": round(r.mean, 4),
            "Std": round(r.std, 4),
        })

    return pd.DataFrame(rows)
