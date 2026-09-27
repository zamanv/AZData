"""Tests for global configuration constants and validation (utils/config.py)."""

import dataclasses

import pytest

from utils.config import (
    ANOMALY_IQR_FACTOR,
    ANOMALY_ZSCORE_THRESHOLD,
    FORECAST_MAX_PERIODS,
    MAX_UPLOAD_SIZE_MB,
    OLLAMA_DEFAULT_MODEL,
    SANDBOX_TIMEOUT_SECONDS,
    AppConfig,
)


# ── defaults ──
def test_default_config_is_valid():
    assert AppConfig().validate() == []


def test_documented_default_values():
    assert SANDBOX_TIMEOUT_SECONDS == 10
    assert ANOMALY_ZSCORE_THRESHOLD == 3.0
    assert ANOMALY_IQR_FACTOR == 1.5
    assert OLLAMA_DEFAULT_MODEL == "llama3.2:3b"
    assert MAX_UPLOAD_SIZE_MB == 500
    assert FORECAST_MAX_PERIODS == 100


# ── validation ──
@pytest.mark.parametrize("temperature", [-0.1, 2.1, 5.0])
def test_rejects_out_of_range_temperature(temperature):
    warnings = AppConfig(ollama_temperature=temperature).validate()
    assert any("Temperature" in w for w in warnings)


@pytest.mark.parametrize("temperature", [0.0, 0.3, 1.0, 2.0])
def test_accepts_valid_temperature(temperature):
    assert AppConfig(ollama_temperature=temperature).validate() == []


@pytest.mark.parametrize("timeout", [0, -5])
def test_rejects_non_positive_sandbox_timeout(timeout):
    warnings = AppConfig(sandbox_timeout=timeout).validate()
    assert any("sandbox timeout" in w.lower() for w in warnings)


@pytest.mark.parametrize("periods", [0, -1, FORECAST_MAX_PERIODS + 1])
def test_rejects_out_of_range_forecast_periods(periods):
    warnings = AppConfig(forecast_periods=periods).validate()
    assert any("Forecast periods" in w for w in warnings)


@pytest.mark.parametrize("periods", [1, 12, FORECAST_MAX_PERIODS])
def test_accepts_valid_forecast_periods(periods):
    assert AppConfig(forecast_periods=periods).validate() == []


def test_reports_every_problem_at_once():
    warnings = AppConfig(
        ollama_temperature=9.0, sandbox_timeout=0, forecast_periods=999
    ).validate()
    assert len(warnings) == 3


# ── dataclass behaviour ──
def test_config_is_a_dataclass_with_expected_fields():
    assert dataclasses.is_dataclass(AppConfig)
    names = {f.name for f in dataclasses.fields(AppConfig)}
    assert {"ollama_url", "ollama_model", "sandbox_timeout"} <= names


def test_instances_are_independent():
    a, b = AppConfig(), AppConfig()
    a.sandbox_timeout = 99
    assert b.sandbox_timeout == SANDBOX_TIMEOUT_SECONDS


# ── documented defect ──
def test_quality_check_weights_constant_is_a_bare_dataclass_field():
    """QUALITY_CHECK_WEIGHTS is annotated ``dict`` but assigned a
    ``dataclasses.field(...)`` sentinel, so it is a Field object, not a dict.

    It is currently unused (profiling/quality_scorer.py owns its own WEIGHTS),
    so nothing breaks today — but any future ``QUALITY_CHECK_WEIGHTS['x']``
    would raise TypeError at import-time-adjacent call sites.
    """
    from utils import config

    assert not isinstance(config.QUALITY_CHECK_WEIGHTS, dict)
    assert isinstance(config.QUALITY_CHECK_WEIGHTS, dataclasses.Field)
