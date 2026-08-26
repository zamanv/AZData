"""AZData — AI-Powered Offline Data Analyst.

Main Streamlit application entry point.
Upload any dataset, ask questions in plain English, and get insights,
anomaly detection, forecasting, and reports — 100% offline.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from io import StringIO

from utils.config import AppConfig, OLLAMA_BASE_URL, OLLAMA_AVAILABLE_MODELS
from utils.session_manager import SessionManager

from ingestion.csv_loader import load_csv_from_bytes
from ingestion.excel_loader import load_excel_from_bytes
from ingestion.db_connector import (
    connect_sqlite, connect_postgresql, connect_mysql,
    load_table_to_dataframe,
)

from profiling.schema_detector import detect_schema, detect_date_columns
from profiling.quality_scorer import compute_quality_score

from analytics.eda_engine import compute_eda
from analytics.anomaly_detector import detect_all_anomalies, anomalies_to_dataframe
from analytics.forecasting import run_arima_forecast
from analytics.insights_generator import generate_insights

from llm.ollama_client import OllamaClient
from llm.code_generator import CodeGenerator

from security.sandbox import execute_pandas_code
from security.crypto import secure_store_credential, wipe_all_credentials

from reports.report_builder import build_report, save_report, save_report_pdf

# ══════════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="AZData — AI Data Analyst",
    page_icon="AZ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
# Custom CSS — Professional Dark Theme
# ══════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global ── */
:root {
    --bg: #0e1117;
    --bg-card: #161b22;
    --border: #21262d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #484f58;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --orange: #d29922;
    --radius: 6px;
}

.stApp {
    background: var(--bg) !important;
}

.stApp > header {
    background: transparent !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] .stMarkdown h1 {
    color: var(--text-primary) !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
    margin-bottom: 0 !important;
}

section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin-top: 1rem !important;
    margin-bottom: 0.4rem !important;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.9rem 1rem !important;
}

[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    padding: 2px !important;
    gap: 2px !important;
    border: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 8px 16px !important;
    border: none !important;
    background: transparent !important;
    border-radius: 4px !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
}

.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"],
.stButton > button:first-child {
    background: #238636 !important;
    color: white !important;
    border: 1px solid rgba(240,246,252,0.1) !important;
    border-radius: var(--radius) !important;
    font-weight: 500 !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button:first-child:hover {
    background: #2ea043 !important;
}

.stDownloadButton > button {
    background: #1f6feb !important;
    color: white !important;
    border: 1px solid rgba(240,246,252,0.1) !important;
    border-radius: var(--radius) !important;
    font-weight: 500 !important;
}

/* ── Expanders ── */
.stExpander {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

.stExpander summary {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* ── DataFrames ── */
.stDataFrame {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: var(--accent) !important;
}

/* ── Alerts ── */
.stAlert {
    border-radius: var(--radius) !important;
}

/* ── Empty state ── */
.az-empty {
    text-align: center;
    padding: 48px 20px;
    color: var(--text-muted);
}

.az-empty h3 {
    color: var(--text-secondary);
    font-weight: 500;
    margin-bottom: 6px;
    font-size: 1rem;
}

.az-empty p {
    font-size: 0.85rem;
    max-width: 360px;
    margin: 0 auto;
}

/* ── Quality Ring ── */
.az-quality-ring {
    display: flex;
    align-items: center;
    gap: 32px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 32px;
    margin: 16px 0;
}

.az-quality-ring .ring-container {
    position: relative;
    width: 120px;
    height: 120px;
    flex-shrink: 0;
}

.az-quality-ring .ring-bg {
    fill: none;
    stroke: var(--border);
    stroke-width: 8;
}

.az-quality-ring .ring-fill {
    fill: none;
    stroke-width: 8;
    stroke-linecap: round;
    transform: rotate(-90deg);
    transform-origin: center;
}

.az-quality-ring .ring-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
}

.az-quality-ring .ring-score {
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1;
}

.az-quality-ring .ring-grade {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
    color: var(--text-secondary);
}

.az-quality-ring .ring-details {
    flex: 1;
    max-width: 400px;
}

.az-quality-ring .ring-details h3 {
    color: var(--text-primary);
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 6px;
}

.az-quality-ring .ring-details p {
    color: var(--text-secondary);
    font-size: 0.85rem;
    line-height: 1.5;
}

.az-quality-ring .ring-details .stats-row {
    display: flex;
    gap: 16px;
    margin-top: 12px;
}

.az-quality-ring .ring-details .stat-item {
    text-align: center;
}

.az-quality-ring .ring-details .stat-value {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-primary);
}

.az-quality-ring .ring-details .stat-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Insight Cards ── */
.az-insight {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px 14px;
    margin: 6px 0;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}

.az-insight .insight-icon {
    color: var(--accent);
    font-size: 0.9rem;
    margin-top: 1px;
    flex-shrink: 0;
}

.az-insight .insight-text {
    color: var(--text-secondary);
    font-size: 0.85rem;
    line-height: 1.5;
}

/* ── Status Badge ── */
.az-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
}

.az-status.connected {
    background: rgba(63, 185, 80, 0.1);
    color: var(--green);
}

.az-status.offline {
    background: rgba(248, 81, 73, 0.1);
    color: var(--red);
}

.az-status .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}

.az-status.connected .dot {
    background: var(--green);
}

.az-status.offline .dot {
    background: var(--red);
}

/* ── Section Headers ── */
.az-section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 20px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

.az-section-header h2 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
}

.az-section-header .section-icon {
    display: none;
}

/* ── Footer ── */
.az-footer {
    text-align: center;
    padding: 16px;
    color: var(--text-muted);
    font-size: 0.75rem;
    border-top: 1px solid var(--border);
    margin-top: 32px;
}

.az-footer span {
    color: var(--text-secondary);
    font-weight: 500;
}

/* ── Sidebar Radio (horizontal) ── */
.stRadio > div {
    gap: 6px !important;
}

/* ── Divider ── */
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    opacity: 0.6;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# Initialize
# ══════════════════════════════════════════════
SessionManager.initialize()
config = AppConfig()


@st.cache_resource
def get_ollama_client(base_url: str, model: str) -> OllamaClient:
    return OllamaClient(base_url=base_url, model=model)


def check_ollama() -> bool:
    client = get_ollama_client(OLLAMA_BASE_URL, st.session_state.get(SessionManager.KEY_SELECTED_MODEL, "llama3.2:3b"))
    connected = client.health_check()
    SessionManager.set_ollama_status(connected)
    return connected


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## AZData")
    st.caption("Offline Data Analyst")

    ollama_connected = check_ollama()
    status_class = "connected" if ollama_connected else "offline"
    status_text = "Connected" if ollama_connected else "Offline"
    st.markdown(
        f'<div class="az-status {status_class}">'
        f'<span class="dot"></span>Ollama {status_text}'
        f'</div>',
        unsafe_allow_html=True,
    )

    available_models = []
    if ollama_connected:
        client = get_ollama_client(OLLAMA_BASE_URL, "llama3.2:3b")
        available_models = client.list_models()
        if not available_models:
            available_models = OLLAMA_AVAILABLE_MODELS

    selected_model = st.selectbox(
        "LLM Model",
        options=available_models if available_models else OLLAMA_AVAILABLE_MODELS,
        index=0,
        key=SessionManager.KEY_SELECTED_MODEL,
    )

    st.markdown("---")

    st.markdown("### Data Source")
    data_source = st.radio(
        "Source",
        options=["Upload File", "SQLite", "PostgreSQL", "MySQL"],
        horizontal=True,
    )

    if data_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Drop your file here",
            type=["csv", "xlsx", "xls"],
            help="CSV or Excel up to 500MB",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            file_name = uploaded_file.name
            with st.spinner("Loading..."):
                if file_name.endswith(".csv"):
                    result = load_csv_from_bytes(file_bytes, file_name)
                else:
                    result = load_excel_from_bytes(file_bytes, file_name)
                if result.success:
                    SessionManager.set_dataset(result.dataframe, file_name)
                    st.success(f"Loaded {result.rows:,} rows x {result.columns} cols")
                else:
                    st.error(result.error)

    elif data_source == "SQLite":
        db_path = st.text_input("File path", placeholder="/path/to/database.db")
        if db_path and st.button("Connect", use_container_width=True):
            with st.spinner("Connecting..."):
                result = connect_sqlite(db_path)
                if result.success:
                    st.session_state[SessionManager.KEY_DB_CONNECTION] = result.engine
                    st.session_state[SessionManager.KEY_DB_TABLES] = result.tables
                    st.success(f"Found {len(result.tables)} tables")
                else:
                    st.error(result.error)

        tables = st.session_state.get(SessionManager.KEY_DB_TABLES, [])
        if tables:
            selected_table = st.selectbox("Select table", tables)
            if st.button("Load Table", use_container_width=True):
                engine = st.session_state.get(SessionManager.KEY_DB_CONNECTION)
                if engine and selected_table:
                    with st.spinner("Loading..."):
                        df_load = load_table_to_dataframe(engine, selected_table)
                        SessionManager.set_dataset(df_load, selected_table)
                        st.success(f"Loaded {len(df_load):,} rows")

    elif data_source in ("PostgreSQL", "MySQL"):
        host = st.text_input("Host", value="localhost")
        port = st.number_input("Port", value=5432 if data_source == "PostgreSQL" else 3306)
        dbname = st.text_input("Database")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Connect", use_container_width=True):
            with st.spinner("Connecting..."):
                if data_source == "PostgreSQL":
                    result = connect_postgresql(host, port, dbname, username, password)
                else:
                    result = connect_mysql(host, port, dbname, username, password)
                if result.success:
                    secure_store_credential("db_password", password)
                    st.session_state[SessionManager.KEY_DB_CONNECTION] = result.engine
                    st.session_state[SessionManager.KEY_DB_TABLES] = result.tables
                    st.success(f"Found {len(result.tables)} tables")
                else:
                    st.error(result.error)

        tables = st.session_state.get(SessionManager.KEY_DB_TABLES, [])
        if tables:
            selected_table = st.selectbox("Select table", tables)
            if st.button("Load Table", use_container_width=True):
                engine = st.session_state.get(SessionManager.KEY_DB_CONNECTION)
                if engine and selected_table:
                    with st.spinner("Loading..."):
                        df_load = load_table_to_dataframe(engine, selected_table)
                        SessionManager.set_dataset(df_load, selected_table)
                        st.success(f"Loaded {len(df_load):,} rows")

    st.markdown("---")

    df = SessionManager.get_dataset()
    if df is not None:
        st.markdown("### Dataset")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Cols", len(df.columns))
        c3.metric("Missing", f"{df.isna().sum().sum():,}")

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color: #484f58; font-size: 0.75rem;">'
        'AZData | Built with Python, Streamlit & Ollama</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════
df = SessionManager.get_dataset()

if df is None:
    # ── Landing Page ──
    st.markdown("# AZData")
    st.markdown("**AI-Powered Offline Data Analyst**")
    st.markdown("")
    st.markdown(
        "Upload a dataset from the sidebar to begin. "
        "Ask questions in plain English, run automated EDA, detect anomalies, "
        "forecast time series, and generate reports — all processed locally."
    )
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Natural Language Queries**\n\nAsk questions in plain English. Get pandas code, charts, and answers.")
    c2.markdown("**Automated EDA**\n\nCorrelation heatmaps, distributions, missing value analysis — generated automatically.")
    c3.markdown("**Anomaly & Forecast**\n\nZ-Score/IQR outlier detection, ARIMA forecasting with confidence intervals.")
    st.markdown("")
    c4, c5, c6 = st.columns(3)
    c4.markdown("**Executive Reports**\n\nOne-click HTML report generation with quality scores and insights.")
    c5.markdown("**Multi-Source**\n\nCSV, Excel, SQLite, PostgreSQL, MySQL — all supported natively.")
    c6.markdown("**100% Private**\n\nRuns on your machine. No data leaves. No API keys. No cloud.")

else:
    tab_overview, tab_chat, tab_eda, tab_anomalies, tab_forecast, tab_reports = st.tabs(
        ["Overview", "Chat", "EDA", "Anomalies", "Forecasting", "Reports"]
    )

    # ══════════════════════════════════════
    # TAB: Overview
    # ══════════════════════════════════════
    with tab_overview:
        if st.session_state.get(SessionManager.KEY_SCHEMA) is None:
            with st.spinner("Analyzing schema..."):
                st.session_state[SessionManager.KEY_SCHEMA] = detect_schema(df)
        schema = st.session_state[SessionManager.KEY_SCHEMA]

        if st.session_state.get(SessionManager.KEY_QUALITY) is None:
            with st.spinner("Computing quality score..."):
                st.session_state[SessionManager.KEY_QUALITY] = compute_quality_score(df)
        quality = st.session_state[SessionManager.KEY_QUALITY]

        # Quality Ring
        grade = quality.grade
        score = quality.overall_score
        circumference = 2 * np.pi * 55
        offset = circumference - (score / 100) * circumference

        grade_colors = {"A": "#22c55e", "B": "#84cc16", "C": "#eab308", "D": "#f97316", "F": "#ef4444"}
        grade_color = grade_colors.get(grade, "#94a3b8")

        st.markdown(f"""
        <div class="az-quality-ring">
            <div class="ring-container">
                <svg width="140" height="140" viewBox="0 0 140 140">
                    <circle class="ring-bg" cx="70" cy="70" r="55"/>
                    <circle class="ring-fill" cx="70" cy="70" r="55"
                        stroke="{grade_color}"
                        stroke-dasharray="{circumference}"
                        stroke-dashoffset="{offset}"/>
                </svg>
                <div class="ring-text">
                    <div class="ring-score" style="color: {grade_color}">{score:.0f}</div>
                    <div class="ring-grade" style="color: {grade_color}">Grade {grade}</div>
                </div>
            </div>
            <div class="ring-details">
                <h3>Data Quality Assessment</h3>
                <p>{quality.summary}</p>
                <div class="stats-row">
                    <div class="stat-item">
                        <div class="stat-value" style="color: #ef4444">{quality.high_severity}</div>
                        <div class="stat-label">High</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" style="color: #f97316">{quality.medium_severity}</div>
                        <div class="stat-label">Medium</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" style="color: #22c55e">{quality.low_severity}</div>
                        <div class="stat-label">Low</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{quality.total_issues}</div>
                        <div class="stat-label">Total</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{schema.total_rows:,}")
        col2.metric("Columns", schema.total_columns)
        col3.metric("Numeric", len(schema.numeric_columns))
        col4.metric("Categorical", len(schema.categorical_columns))

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Missing Cells", f"{df.isna().sum().sum():,}")
        col6.metric("Duplicates", f"{schema.duplicate_rows:,}")
        col7.metric("Memory", f"{schema.memory_usage_mb} MB")
        col8.metric("Issues", quality.total_issues)

        # Quality Breakdown
        st.markdown("""
        <div class="az-section-header">
            <h2>Quality Breakdown</h2>
        </div>
        """, unsafe_allow_html=True)

        bd_cols = st.columns(len(quality.breakdown))
        for i, (category, score_val) in enumerate(quality.breakdown.items()):
            bd_cols[i].metric(category.replace("_", " ").title(), f"{score_val:.0f}/100")

        # Insights
        st.markdown("""
        <div class="az-section-header">
            <h2>AI-Generated Insights</h2>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.get(SessionManager.KEY_INSIGHTS):
            with st.spinner("Generating insights..."):
                ollama_client = get_ollama_client(OLLAMA_BASE_URL, selected_model)
                code_gen = CodeGenerator(ollama_client) if ollama_connected else None
                insight_result = generate_insights(df, code_gen)
                st.session_state[SessionManager.KEY_INSIGHTS] = insight_result.insights

        for insight in st.session_state.get(SessionManager.KEY_INSIGHTS, []):
            st.markdown(
                f'<div class="az-insight">'
                f'<span class="insight-icon">▸</span>'
                f'<span class="insight-text">{insight}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Column Details
        with st.expander("Column Details", expanded=False):
            col_data = []
            for c in schema.columns:
                col_data.append({
                    "Column": c.name,
                    "Type": c.inferred_type,
                    "Non-Null": c.non_null_count,
                    "Unique": c.unique_count,
                    "Null %": f"{c.null_percentage}%",
                    "Sample": str(c.sample_values[:3]),
                })
            st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════
    # TAB: Chat
    # ══════════════════════════════════════
    with tab_chat:
        if not ollama_connected:
            st.warning("Ollama is not connected. Start it with `ollama serve` to use NL queries.")

        chat_history = SessionManager.get_chat_history()
        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("code"):
                    with st.expander("View Generated Code"):
                        st.code(msg["code"], language="python")
                if msg.get("chart"):
                    st.plotly_chart(msg["chart"], use_container_width=True)

        if question := st.chat_input("Ask anything about your data..."):
            with st.chat_message("user"):
                st.markdown(question)
            SessionManager.add_chat_message("user", question)

            if not ollama_connected:
                with st.chat_message("assistant"):
                    st.error("Ollama is not connected. Please start Ollama and try again.")
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Generating code..."):
                        ollama_client = get_ollama_client(OLLAMA_BASE_URL, selected_model)
                        code_gen = CodeGenerator(ollama_client)

                        if st.session_state.get(SessionManager.KEY_SCHEMA) is None:
                            st.session_state[SessionManager.KEY_SCHEMA] = detect_schema(df)
                        schema = st.session_state[SessionManager.KEY_SCHEMA]

                        sample_str = df.head(5).to_string()
                        gen_result = code_gen.generate_pandas_code(question, schema, sample_str)

                    if not gen_result.success:
                        st.error(f"Code generation failed: {gen_result.error}")
                        SessionManager.add_chat_message("assistant", f"Error: {gen_result.error}")
                    else:
                        st.markdown("**Generated Code:**")
                        st.code(gen_result.code, language="python")

                        with st.spinner("Executing..."):
                            exec_result = execute_pandas_code(gen_result.code, df)

                        if not exec_result.success:
                            st.error(f"Execution error: {exec_result.error}")
                            SessionManager.add_chat_message(
                                "assistant", f"Error: {exec_result.error}",
                                code=gen_result.code,
                            )
                        else:
                            chart_fig = None
                            if exec_result.result_df is not None:
                                st.dataframe(exec_result.result_df, use_container_width=True)
                                st.caption(f"Executed in {exec_result.execution_time_ms}ms")

                            if exec_result.fig is not None:
                                st.plotly_chart(exec_result.fig, use_container_width=True)
                                chart_fig = exec_result.fig

                            if exec_result.output is not None:
                                st.markdown(f"**Result:** {exec_result.output}")

                            if exec_result.result_df is None and exec_result.fig is None and exec_result.output is None:
                                st.info("Code executed successfully. Set `result_df` or `fig` for output.")

                            if exec_result.result_df is not None and exec_result.fig is None:
                                rdf = exec_result.result_df
                                if rdf.shape == (1, 1):
                                    st.metric("Result", rdf.iloc[0, 0])
                                elif rdf.shape[1] == 2 and len(rdf) <= 20:
                                    import plotly.express as px
                                    try:
                                        auto_fig = px.bar(rdf, x=rdf.columns[0], y=rdf.columns[1])
                                        st.plotly_chart(auto_fig, use_container_width=True)
                                        chart_fig = auto_fig
                                    except Exception:
                                        pass

                            SessionManager.add_query(
                                question=question, code=gen_result.code,
                                code_type="pandas", success=True,
                                execution_ms=exec_result.execution_time_ms,
                            )
                            SessionManager.add_chat_message(
                                "assistant", "Query executed successfully.",
                                code=gen_result.code, chart=chart_fig,
                            )

    # ══════════════════════════════════════
    # TAB: EDA
    # ══════════════════════════════════════
    with tab_eda:
        if st.session_state.get(SessionManager.KEY_EDA_REPORT) is None:
            with st.spinner("Generating EDA..."):
                st.session_state[SessionManager.KEY_EDA_REPORT] = compute_eda(df)
        eda_report = st.session_state[SessionManager.KEY_EDA_REPORT]

        if eda_report.correlation_heatmap is not None:
            st.plotly_chart(eda_report.correlation_heatmap, use_container_width=True)

        if eda_report.missingness_chart is not None:
            st.plotly_chart(eda_report.missingness_chart, use_container_width=True)

        if eda_report.distribution_charts:
            st.markdown("### Numeric Distributions")
            for fig in eda_report.distribution_charts:
                st.plotly_chart(fig, use_container_width=True)

        if eda_report.categorical_charts:
            st.markdown("### Categorical Distributions")
            for fig in eda_report.categorical_charts:
                st.plotly_chart(fig, use_container_width=True)

        if eda_report.scatter_matrix is not None:
            st.plotly_chart(eda_report.scatter_matrix, use_container_width=True)

    # ══════════════════════════════════════
    # TAB: Anomalies
    # ══════════════════════════════════════
    with tab_anomalies:
        method = st.radio("Detection Method", ["Z-Score", "IQR", "Both"], horizontal=True)
        methods = ["zscore"] if method == "Z-Score" else ["iqr"] if method == "IQR" else ["zscore", "iqr"]

        c1, c2 = st.columns(2)
        with c1:
            zscore_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)
        with c2:
            iqr_factor = st.slider("IQR Factor", 0.5, 3.0, 1.5, 0.1)

        if st.button("Run Anomaly Detection", use_container_width=True):
            with st.spinner("Scanning for outliers..."):
                from utils import config as cfg
                cfg.ANOMALY_ZSCORE_THRESHOLD = zscore_threshold
                cfg.ANOMALY_IQR_FACTOR = iqr_factor
                results = detect_all_anomalies(df, methods)
                st.session_state[SessionManager.KEY_ANOMALIES] = results

        results = st.session_state.get(SessionManager.KEY_ANOMALIES, [])
        if results:
            summary_df = anomalies_to_dataframe(results)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            for result in results:
                with st.expander(f"{result.column} — {result.method} ({result.anomaly_count} outliers)"):
                    if result.fig:
                        st.plotly_chart(result.fig, use_container_width=True)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Anomalies", result.anomaly_count)
                    c2.metric("Percentage", f"{result.anomaly_percentage}%")
                    c3.metric("Threshold", result.threshold)

            if ollama_connected and results:
                with st.expander("AI Explanations"):
                    with st.spinner("Generating explanations..."):
                        ollama_client = get_ollama_client(OLLAMA_BASE_URL, selected_model)
                        code_gen = CodeGenerator(ollama_client)
                        gen_result = code_gen.generate_anomaly_explanation(summary_df.to_string())
                    if gen_result.success:
                        try:
                            explanations = json.loads(gen_result.code)
                            for exp in explanations:
                                st.markdown(
                                    f'<div class="az-insight">'
                                    f'<span class="insight-icon">▸</span>'
                                    f'<span class="insight-text">{exp}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        except (json.JSONDecodeError, ValueError):
                            st.markdown(gen_result.code)
        else:
            st.markdown("""
            <div class="az-empty">
                <h3>No Anomalies Detected</h3>
                <p>Configure parameters and click "Run Anomaly Detection" to find outliers.</p>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════
    # TAB: Forecasting
    # ══════════════════════════════════════
    with tab_forecast:
        date_cols = detect_date_columns(df)
        date_col_options = [c[0] for c in date_cols] if date_cols else list(df.columns)

        c1, c2 = st.columns(2)
        with c1:
            date_col = st.selectbox("Date Column", date_col_options)
        with c2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_col = st.selectbox("Value Column", numeric_cols if numeric_cols else ["(none)"])

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            periods = st.slider("Forecast Periods", 1, 100, 12)
        with c2:
            confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
        with c3:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            run_forecast = st.button("🔮 Generate", use_container_width=True)

        if run_forecast:
            with st.spinner("Fitting ARIMA model..."):
                forecast_result = run_arima_forecast(df, date_col, value_col, periods, confidence)
                st.session_state[SessionManager.KEY_FORECAST] = forecast_result

        forecast = st.session_state.get(SessionManager.KEY_FORECAST)
        if forecast and forecast.success:
            if forecast.fig:
                st.plotly_chart(forecast.fig, use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Model", forecast.model_name)
            m2.metric("Training Points", forecast.metrics.get("training_points", "N/A"))
            m3.metric("AIC", forecast.metrics.get("aic", "N/A"))
            m4.metric("Frequency", forecast.metrics.get("frequency", "N/A"))

            with st.expander("Forecast Data"):
                forecast_df = pd.DataFrame({
                    "Date": forecast.forecast_dates,
                    "Forecast": [round(v, 4) for v in forecast.forecast_values],
                    "Lower": [round(v, 4) for v in forecast.lower_bound],
                    "Upper": [round(v, 4) for v in forecast.upper_bound],
                })
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            if ollama_connected:
                with st.expander("AI Analysis"):
                    with st.spinner("Generating..."):
                        ollama_client = get_ollama_client(OLLAMA_BASE_URL, selected_model)
                        code_gen = CodeGenerator(ollama_client)
                        forecast_str = (
                            f"Column: {forecast.column}, Model: {forecast.model_name}, "
                            f"Periods: {forecast.periods}, "
                            f"Last: {forecast.historical_values[-1] if forecast.historical_values else 'N/A'}, "
                            f"Mean forecast: {np.mean(forecast.forecast_values):.4f}"
                        )
                        gen_result = code_gen.generate_forecast_explanation(forecast_str)
                    if gen_result.success:
                        try:
                            analysis = json.loads(gen_result.code)
                            for key, val in analysis.items():
                                st.markdown(f"**{key.title()}:** {val}")
                        except (json.JSONDecodeError, ValueError):
                            st.markdown(gen_result.code)

        elif forecast and forecast.error:
            st.error(forecast.error)
        else:
            st.markdown("""
            <div class="az-empty">
                <h3>Time Series Forecasting</h3>
                <p>Select a date and value column, configure parameters, and click Generate.</p>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════
    # TAB: Reports
    # ══════════════════════════════════════
    with tab_reports:
        file_name = st.session_state.get(SessionManager.KEY_FILE_NAME, "dataset")

        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Generate Report", use_container_width=True):
                with st.spinner("Building..."):
                    quality = st.session_state.get(SessionManager.KEY_QUALITY)
                    insights = st.session_state.get(SessionManager.KEY_INSIGHTS, [])
                    anomalies = st.session_state.get(SessionManager.KEY_ANOMALIES, [])
                    anomaly_summary = anomalies_to_dataframe(anomalies) if anomalies else None

                    report_html = build_report(
                        filename=file_name, df=df,
                        quality_score=quality.overall_score if quality else 0,
                        quality_grade=quality.grade if quality else "N/A",
                        quality_summary=quality.summary if quality else "",
                        total_issues=quality.total_issues if quality else 0,
                        insights=insights,
                        anomaly_summary=anomaly_summary,
                    )
                    filepath = save_report(report_html, file_name)

                    st.download_button(
                        label="Download HTML",
                        data=report_html,
                        file_name=f"AZData_Report_{file_name}.html",
                        mime="text/html",
                        use_container_width=True,
                    )

                    pdf_path = save_report_pdf(report_html, file_name)
                    if pdf_path:
                        pdf_bytes = pdf_path.read_bytes()
                        st.download_button(
                            label="Download PDF",
                            data=pdf_bytes,
                            file_name=f"AZData_Report_{file_name}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    else:
                        st.caption("PDF export requires weasyprint. Install with: pip install weasyprint")

                    st.success(f"Saved to {filepath.name}")

        st.markdown("""
        <div class="az-section-header">
            <h2>Query History</h2>
        </div>
        """, unsafe_allow_html=True)

        query_history = st.session_state.get(SessionManager.KEY_QUERY_HISTORY, [])
        if query_history:
            history_df = pd.DataFrame(query_history)
            st.dataframe(
                history_df[["timestamp", "question", "code_type", "success", "execution_ms"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.markdown("""
            <div class="az-empty">
                <h3>No Queries Yet</h3>
                <p>Your chat queries will appear here after execution.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("Clear All Data", type="secondary"):
            SessionManager.clear_dataset()
            SessionManager.clear_chat()
            wipe_all_credentials()
            st.rerun()

# ── Footer ──
st.markdown("""
<div class="az-footer">
    <span>AZData</span> — AI-Powered Offline Data Analyst | Built with Python, Streamlit & Ollama
</div>
""", unsafe_allow_html=True)
