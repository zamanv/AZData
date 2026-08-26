"""Time series forecasting engine using ARIMA with auto-parameter selection."""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.config import FORECAST_DEFAULT_PERIODS, FORECAST_MAX_PERIODS


@dataclass
class ForecastResult:
    """Result of a time series forecast."""
    success: bool
    column: str = ""
    date_column: str = ""
    historical_dates: list = None
    historical_values: list = None
    forecast_dates: list = None
    forecast_values: list = None
    lower_bound: list = None
    upper_bound: list = None
    model_name: str = "ARIMA"
    periods: int = 0
    confidence_level: float = 0.95
    fig: Optional[go.Figure] = None
    metrics: Dict[str, Any] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.historical_dates is None:
            self.historical_dates = []
        if self.historical_values is None:
            self.historical_values = []
        if self.forecast_dates is None:
            self.forecast_dates = []
        if self.forecast_values is None:
            self.forecast_values = []
        if self.lower_bound is None:
            self.lower_bound = []
        if self.upper_bound is None:
            self.upper_bound = []
        if self.metrics is None:
            self.metrics = {}


def auto_detect_date_column(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """Auto-detect the best date column for time series analysis.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (date_column, inferred_freq) or None.
    """
    for col in df.columns:
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col, _infer_freq(df[col])

    # Try to parse object columns
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                parse_rate = parsed.notna().mean()
                if parse_rate > 0.8:
                    return col, _infer_freq(parsed.dropna())
            except Exception:
                continue

    return None


def _infer_freq(series: pd.Series) -> str:
    """Infer the frequency of a datetime series.

    Args:
        series: Datetime Series.

    Returns:
        Frequency string (e.g., 'D', 'W', 'M').
    """
    if len(series) < 3:
        return "D"

    try:
        diff = series.sort_values().diff().dropna()
        median_diff = diff.median()

        if median_diff <= pd.Timedelta(hours=25):
            return "D"
        elif median_diff <= pd.Timedelta(days=8):
            return "W"
        elif median_diff <= pd.Timedelta(days=35):
            return "M"
        elif median_diff <= pd.Timedelta(days=100):
            return "Q"
        else:
            return "Y"
    except Exception:
        return "D"


def run_arima_forecast(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    periods: int = FORECAST_DEFAULT_PERIODS,
    confidence: float = 0.95,
) -> ForecastResult:
    """Run ARIMA forecasting on a time series.

    Args:
        df: Input DataFrame.
        date_column: Name of the datetime column.
        value_column: Name of the value column to forecast.
        periods: Number of future periods to forecast.
        confidence: Confidence interval level.

    Returns:
        ForecastResult with forecast data and visualization.
    """
    try:
        import pmdarima as pm
    except ImportError:
        return ForecastResult(
            success=False,
            error="pmdarima not installed. Run: pip install pmdarima"
        )

    # Validate inputs
    periods = max(1, min(periods, FORECAST_MAX_PERIODS))

    try:
        # Prepare data
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            dates = pd.to_datetime(df[date_column], errors="coerce")
        else:
            dates = df[date_column].copy()

        values = pd.to_numeric(df[value_column], errors="coerce")

        # Drop NaN pairs
        mask = dates.notna() & values.notna()
        dates = dates[mask]
        values = values[mask]

        if len(values) < 10:
            return ForecastResult(
                success=False,
                column=value_column,
                date_column=date_column,
                error="Need at least 10 data points for forecasting"
            )

        # Sort by date
        sort_idx = dates.argsort()
        dates = dates.iloc[sort_idx].reset_index(drop=True)
        values = values.iloc[sort_idx].reset_index(drop=True)

        # Create time series
        ts = pd.Series(values.values, index=pd.DatetimeIndex(dates))

        # Auto-fit ARIMA
        model = pm.auto_arima(
            ts,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_order=8,
            trace=False,
        )

        # Forecast
        forecast, conf_int = model.predict(
            n_periods=periods,
            return_conf_int=True,
            alpha=1 - confidence,
        )

        # Generate future dates
        freq = _infer_freq(ts.index)
        last_date = ts.index[-1]
        future_dates = pd.date_range(
            start=last_date,
            periods=periods + 1,
            freq=freq,
        )[1:]

        # Model metrics
        aic = float(model.aic()) if hasattr(model, 'aic') else 0
        bic = float(model.bic()) if hasattr(model, 'bic') else 0

        metrics = {
            "order": str(model.order),
            "aic": round(aic, 2),
            "bic": round(bic, 2),
            "training_points": len(ts),
            "frequency": freq,
        }

        # Create visualization
        fig = _create_forecast_chart(
            historical_dates=ts.index.tolist(),
            historical_values=ts.values.tolist(),
            forecast_dates=future_dates.tolist(),
            forecast_values=forecast.tolist(),
            lower_bound=conf_int[:, 0].tolist(),
            upper_bound=conf_int[:, 1].tolist(),
            column_name=value_column,
            confidence=confidence,
        )

        return ForecastResult(
            success=True,
            column=value_column,
            date_column=date_column,
            historical_dates=[d.isoformat() for d in ts.index],
            historical_values=[float(v) for v in ts.values],
            forecast_dates=[d.isoformat() for d in future_dates],
            forecast_values=[float(v) for v in forecast],
            lower_bound=[float(v) for v in conf_int[:, 0]],
            upper_bound=[float(v) for v in conf_int[:, 1]],
            model_name=f"ARIMA{model.order}",
            periods=periods,
            confidence_level=confidence,
            fig=fig,
            metrics=metrics,
        )

    except Exception as e:
        return ForecastResult(
            success=False,
            column=value_column,
            date_column=date_column,
            error=f"Forecasting failed: {type(e).__name__}: {e}"
        )


def _create_forecast_chart(
    historical_dates: list,
    historical_values: list,
    forecast_dates: list,
    forecast_values: list,
    lower_bound: list,
    upper_bound: list,
    column_name: str,
    confidence: float,
) -> go.Figure:
    """Create a Plotly chart for the forecast.

    Args:
        Historical and forecast data, plus chart parameters.

    Returns:
        Plotly Figure with forecast visualization.
    """
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_values,
        mode="lines+markers",
        name="Historical",
        line=dict(color="#4C78A8", width=2),
        marker=dict(size=4),
    ))

    # Confidence interval (filled area)
    fig.add_trace(go.Scatter(
        x=forecast_dates + forecast_dates[::-1],
        y=upper_bound + lower_bound[::-1],
        fill="toself",
        fillcolor="rgba(228, 87, 86, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"{int(confidence*100)}% CI",
        showlegend=True,
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#E45756", width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
    ))

    fig.update_layout(
        title=f"Time Series Forecast: {column_name}",
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title=column_name,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )

    return fig
