"""Global configuration constants, paths, and default settings for AZData."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent
CACHE_DIR: Path = BASE_DIR / ".cache"
REPORTS_DIR: Path = BASE_DIR / "generated_reports"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# Ollama Defaults
# ──────────────────────────────────────────────
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL: str = "llama3.2:3b"
OLLAMA_AVAILABLE_MODELS: List[str] = [
    "llama3.2:3b",
    "llama3.2:1b",
    "llama3.1:8b",
    "qwen2.5:7b",
    "qwen2.5:3b",
    "qwen2.5:1.5b",
    "mistral:7b",
    "phi3:mini",
    "gemma2:2b",
    "codellama:7b",
]
OLLAMA_REQUEST_TIMEOUT: int = 60
OLLAMA_TEMPERATURE: float = 0.3
OLLAMA_MAX_TOKENS: int = 2048


# ──────────────────────────────────────────────
# Sandbox / Security
# ──────────────────────────────────────────────
SANDBOX_TIMEOUT_SECONDS: int = 10
SANDBOX_MAX_MEMORY_BYTES: int = 100 * 1024 * 1024  # 100 MB


# ──────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB: int = 500
CSV_SAMPLE_ROWS: int = 5000
EXCEL_MAX_SHEETS: int = 50


# ──────────────────────────────────────────────
# Profiling
# ──────────────────────────────────────────────
QUALITY_CHECK_WEIGHTS: dict = field(default_factory=lambda: {
    "missing_values": 0.25,
    "duplicate_rows": 0.20,
    "constant_columns": 0.15,
    "outlier_ratio": 0.15,
    "type_mismatch": 0.15,
    "high_cardinality": 0.10,
})


# ──────────────────────────────────────────────
# Analytics
# ──────────────────────────────────────────────
ANOMALY_ZSCORE_THRESHOLD: float = 3.0
ANOMALY_IQR_FACTOR: float = 1.5
FORECAST_DEFAULT_PERIODS: int = 12
FORECAST_MAX_PERIODS: int = 100
EDA_MAX_CATEGORIES: int = 20
EDA_DISTRIBUTION_BINS: int = 30


# ──────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────
REPORT_TEMPLATE_DIR: Path = BASE_DIR / "reports" / "templates"
REPORT_OUTPUT_DIR: Path = REPORTS_DIR


@dataclass
class AppConfig:
    """Centralized application configuration."""

    ollama_url: str = OLLAMA_BASE_URL
    ollama_model: str = OLLAMA_DEFAULT_MODEL
    ollama_temperature: float = OLLAMA_TEMPERATURE
    ollama_max_tokens: int = OLLAMA_MAX_TOKENS
    sandbox_timeout: int = SANDBOX_TIMEOUT_SECONDS
    max_upload_mb: int = MAX_UPLOAD_SIZE_MB
    anomaly_zscore_threshold: float = ANOMALY_ZSCORE_THRESHOLD
    anomaly_iqr_factor: float = ANOMALY_IQR_FACTOR
    forecast_periods: int = FORECAST_DEFAULT_PERIODS

    def validate(self) -> List[str]:
        """Return list of configuration warnings."""
        warnings: List[str] = []
        if self.ollama_temperature < 0 or self.ollama_temperature > 2.0:
            warnings.append("Temperature should be between 0.0 and 2.0")
        if self.sandbox_timeout < 1:
            warnings.append("Sandbox timeout must be at least 1 second")
        if self.forecast_periods < 1 or self.forecast_periods > FORECAST_MAX_PERIODS:
            warnings.append(f"Forecast periods must be 1-{FORECAST_MAX_PERIODS}")
        return warnings
