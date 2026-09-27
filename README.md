# AZData

[![CI](https://github.com/zamanv/AZData/actions/workflows/ci.yml/badge.svg)](https://github.com/zamanv/AZData/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-260%20passing-brightgreen)](https://github.com/zamanv/AZData/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

AI-powered offline data analytics platform. Upload a dataset, ask questions in plain English, and get insights, anomaly detection, forecasting, and reports — all processed locally with zero API keys.

## Features

- **Natural Language Queries** — Ask questions in plain English, get pandas code and results via Ollama
- **Automated EDA** — Correlation heatmaps, distributions, missing value analysis
- **Data Quality Scoring** — 0-100 score with letter grade and issue breakdown
- **Anomaly Detection** — Z-Score and IQR outlier detection with charts
- **Time Series Forecasting** — ARIMA auto-parameter forecasting with confidence intervals
- **Executive Reports** — One-click HTML/PDF report generation
- **Multi-Source Ingestion** — CSV, Excel, SQLite, PostgreSQL, MySQL
- **100% Offline** — No data leaves your machine. No API keys. No cloud.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally (for NL queries)

## Quick Start

```bash
# Clone
git clone https://github.com/zamanv/AZData.git
cd AZData

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Docker

```bash
docker compose up --build
```

## Ollama Setup

Install Ollama from [ollama.ai](https://ollama.ai), then pull a model:

```bash
ollama pull llama3.2:3b
```

The app connects to `http://localhost:11434` by default.

## Project Structure

```
AZData/
├── app.py                  # Streamlit entry point
├── ingestion/              # CSV, Excel, database loaders
├── profiling/              # Schema detection, quality scoring, stats
├── analytics/              # EDA, anomaly detection, forecasting
├── llm/                    # Ollama client, code generation, prompts
├── security/               # Code sandbox, encryption, SQL guard
├── reports/                # HTML/PDF report builder
├── utils/                  # Config, session manager
├── tests/                  # pytest suite
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── Dockerfile
├── .github/workflows/      # CI
└── docker-compose.yml
```

## Testing

The suite covers the security, profiling, analytics, ingestion, and config
modules. It needs no Ollama server and no network access.

```bash
pip install -r requirements-dev.txt
pytest                                    # 260 tests
pytest --cov=analytics --cov=ingestion \
       --cov=profiling --cov=security \
       --cov=utils --cov-report=term-missing
```

| Module | Coverage |
| --- | --- |
| `security/sql_guard.py` | 100% |
| `security/crypto.py` | 100% |
| `profiling/stats_generator.py` | 100% |
| `analytics/anomaly_detector.py` | 100% |
| `profiling/quality_scorer.py` | 99% |
| `profiling/schema_detector.py` | 95% |
| `analytics/forecasting.py` | 95% |
| `ingestion/csv_loader.py` | 91% |
| `security/sandbox.py` | 81% |

A few tests are marked `xfail` to pin **known, deliberate limitations** rather
than hide them. Run `pytest -rx` to list them. They currently cover:

- **Sandbox timeout cannot interrupt GIL-holding code.** The limit is enforced
  with `Thread.join(timeout=...)`, which cannot pre-empt a tight C-level loop.
  Real containment needs a separate process, `RLIMIT_CPU`, or a container.
- **`FORBIDDEN_NODE_TYPES` is not enforced.** `ASTVerifier` defines
  `visit_AnyNode`, but `ast.NodeVisitor` dispatches on exact class names, so
  that policy table never runs.
- **An empty DataFrame scores 100/A** for data quality, which reads as
  "pristine data" rather than "nothing to assess".

## Security

- Code execution sandboxed via AST verification (no `os`, `subprocess`, `import`)
- SQL queries restricted to read-only operations
- Database credentials encrypted with Fernet (AES-128-CBC)
- All LLM prompts include injection defense preamble

## License

MIT
