# AZData

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
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Security

- Code execution sandboxed via AST verification (no `os`, `subprocess`, `import`)
- SQL queries restricted to read-only operations
- Database credentials encrypted with Fernet (AES-128-CBC)
- All LLM prompts include injection defense preamble

## License

MIT
