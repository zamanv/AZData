"""Automated EDA engine — generates Plotly interactive charts for correlation, distributions, missingness, and categories."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.config import EDA_DISTRIBUTION_BINS, EDA_MAX_CATEGORIES


@dataclass
class EDAReport:
    """Complete EDA report with all generated charts."""
    correlation_heatmap: Optional[go.Figure] = None
    distribution_charts: List[go.Figure] = field(default_factory=list)
    missingness_chart: Optional[go.Figure] = None
    categorical_charts: List[go.Figure] = field(default_factory=list)
    boxplots: List[go.Figure] = field(default_factory=list)
    scatter_matrix: Optional[go.Figure] = None
    summary_stats: Dict[str, Any] = field(default_factory=dict)


def generate_correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    """Generate an interactive correlation heatmap for numeric columns.

    Args:
        df: Input DataFrame.

    Returns:
        Plotly Figure or None if <2 numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate=(
                "X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Correlation Heatmap",
        title_x=0.5,
        xaxis_title="",
        yaxis_title="",
        height=max(400, len(corr.columns) * 40),
        width=max(500, len(corr.columns) * 50),
    )

    return fig


def generate_distributions(df: pd.DataFrame) -> List[go.Figure]:
    """Generate distribution plots for all numeric columns.

    Args:
        df: Input DataFrame.

    Returns:
        List of Plotly Figures, one per numeric column.
    """
    charts = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"{col} — Distribution", f"{col} — Box Plot"),
            column_widths=[0.65, 0.35],
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=series,
                nbinsx=EDA_DISTRIBUTION_BINS,
                name="Distribution",
                marker_color="#4C78A8",
                opacity=0.8,
            ),
            row=1, col=1,
        )

        # Box plot
        fig.add_trace(
            go.Box(
                y=series,
                name="Box",
                marker_color="#4C78A8",
                boxpoints="outliers",
            ),
            row=1, col=2,
        )

        fig.update_layout(
            height=300,
            showlegend=False,
            margin=dict(t=40, b=20, l=40, r=20),
        )

        charts.append(fig)

    return charts


def generate_missingness_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Generate a bar chart showing missing values per column.

    Args:
        df: Input DataFrame.

    Returns:
        Plotly Figure or None if no missing values.
    """
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if len(missing) == 0:
        return None

    colors = []
    for val in missing:
        pct = val / len(df) * 100
        if pct > 50:
            colors.append("#E45756")
        elif pct > 20:
            colors.append("#F58518")
        else:
            colors.append("#4C78A8")

    fig = go.Figure(
        data=go.Bar(
            x=missing.values,
            y=missing.index.tolist(),
            orientation="h",
            marker_color=colors,
            text=[f"{v} ({v/len(df)*100:.1f}%)" for v in missing.values],
            textposition="outside",
            hovertemplate=(
                "Column: %{y}<br>Missing: %{x}<br>"
                "Percentage: %{text}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Missing Values by Column",
        title_x=0.5,
        xaxis_title="Count",
        yaxis_title="",
        height=max(300, len(missing) * 30),
        margin=dict(t=40, b=30, l=120, r=30),
    )

    return fig


def generate_categorical_charts(df: pd.DataFrame) -> List[go.Figure]:
    """Generate bar charts for categorical columns.

    Args:
        df: Input DataFrame.

    Returns:
        List of Plotly Figures, one per categorical column.
    """
    charts = []
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        vc = df[col].value_counts().head(EDA_MAX_CATEGORIES)
        if len(vc) == 0:
            continue

        fig = go.Figure(
            data=go.Bar(
                x=vc.index.astype(str).tolist(),
                y=vc.values.tolist(),
                marker_color="#54A24B",
                text=vc.values.tolist(),
                textposition="outside",
                hovertemplate=(
                    "Category: %{x}<br>Count: %{y}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=f"{col} — Value Counts (top {min(len(vc), EDA_MAX_CATEGORIES)})",
            title_x=0.5,
            xaxis_title="",
            yaxis_title="Count",
            height=350,
            xaxis_tickangle=-45,
            margin=dict(t=40, b=60, l=50, r=20),
        )

        charts.append(fig)

    return charts


def generate_scatter_matrix(df: pd.DataFrame, max_cols: int = 5) -> Optional[go.Figure]:
    """Generate a scatter plot matrix for up to 5 numeric columns.

    Args:
        df: Input DataFrame.
        max_cols: Maximum number of columns to include.

    Returns:
        Plotly Figure or None if <2 numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None

    cols = numeric_df.columns[:max_cols]

    fig = px.scatter_matrix(
        numeric_df[cols],
        dimensions=cols.tolist(),
        opacity=0.5,
        height=600,
        width=700,
    )

    fig.update_layout(
        title="Scatter Matrix",
        title_x=0.5,
        margin=dict(t=40, b=20, l=20, r=20),
    )

    fig.update_traces(diagonal_visible=True)

    return fig


def compute_eda(df: pd.DataFrame) -> EDAReport:
    """Compute a complete EDA report with all charts.

    Args:
        df: Input DataFrame.

    Returns:
        EDAReport with all generated charts and summary.
    """
    corr_fig = generate_correlation_heatmap(df)
    dist_figs = generate_distributions(df)
    missing_fig = generate_missingness_chart(df)
    cat_figs = generate_categorical_charts(df)
    scatter_fig = generate_scatter_matrix(df)

    # Summary stats
    numeric_df = df.select_dtypes(include=[np.number])
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": list(numeric_df.columns),
        "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns),
        "missing_total": int(df.isna().sum().sum()),
        "missing_pct": round(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
    }

    return EDAReport(
        correlation_heatmap=corr_fig,
        distribution_charts=dist_figs,
        missingness_chart=missing_fig,
        categorical_charts=cat_figs,
        scatter_matrix=scatter_fig,
        summary_stats=summary,
    )
