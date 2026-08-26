"""Streamlit session state manager — centralized state initialization and access."""

import json
from pathlib import Path
from typing import Any, Optional, List, Dict
import streamlit as st
import pandas as pd
from datetime import datetime

QUERY_HISTORY_FILE = Path(__file__).resolve().parent.parent / "query_history.json"


class SessionManager:
    """Manages all Streamlit session state keys with typed accessors."""

    # ── Session Keys ──
    KEY_DATASET: str = "az_dataset"
    KEY_RAW_DATASET: str = "az_raw_dataset"
    KEY_FILE_NAME: str = "az_file_name"
    KEY_SCHEMA: str = "az_schema"
    KEY_QUALITY: str = "az_quality"
    KEY_EDA_REPORT: str = "az_eda_report"
    KEY_CHAT_HISTORY: str = "az_chat_history"
    KEY_ANOMALIES: str = "az_anomalies"
    KEY_FORECAST: str = "az_forecast"
    KEY_INSIGHTS: str = "az_insights"
    KEY_DB_CONNECTION: str = "az_db_connection"
    KEY_DB_TABLES: str = "az_db_tables"
    KEY_OLLAMA_STATUS: str = "az_ollama_status"
    KEY_SELECTED_MODEL: str = "az_selected_model"
    KEY_QUERY_HISTORY: str = "az_query_history"
    KEY_REPORT_DATA: str = "az_report_data"

    @classmethod
    def initialize(cls) -> None:
        """Initialize all session state keys with default values."""
        defaults: Dict[str, Any] = {
            cls.KEY_DATASET: None,
            cls.KEY_RAW_DATASET: None,
            cls.KEY_FILE_NAME: None,
            cls.KEY_SCHEMA: None,
            cls.KEY_QUALITY: None,
            cls.KEY_EDA_REPORT: None,
            cls.KEY_CHAT_HISTORY: [],
            cls.KEY_ANOMALIES: None,
            cls.KEY_FORECAST: None,
            cls.KEY_INSIGHTS: [],
            cls.KEY_DB_CONNECTION: None,
            cls.KEY_DB_TABLES: [],
            cls.KEY_OLLAMA_STATUS: False,
            cls.KEY_SELECTED_MODEL: "llama3.2:3b",
            cls.KEY_QUERY_HISTORY: cls._load_query_history(),
            cls.KEY_REPORT_DATA: {},
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @classmethod
    def _load_query_history(cls) -> List[Dict[str, Any]]:
        """Load query history from disk."""
        if QUERY_HISTORY_FILE.exists():
            try:
                return json.loads(QUERY_HISTORY_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    @classmethod
    def _save_query_history(cls) -> None:
        """Persist query history to disk."""
        history = st.session_state.get(cls.KEY_QUERY_HISTORY, [])
        serializable = []
        for entry in history:
            serializable.append({k: v for k, v in entry.items() if k != "chart"})
        try:
            QUERY_HISTORY_FILE.write_text(
                json.dumps(serializable, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError:
            pass

    # ── Dataset Accessors ──
    @classmethod
    def get_dataset(cls) -> Optional[pd.DataFrame]:
        """Return the currently loaded DataFrame."""
        return st.session_state.get(cls.KEY_DATASET)

    @classmethod
    def set_dataset(cls, df: pd.DataFrame, file_name: str = "uploaded") -> None:
        """Store a DataFrame and metadata in session state."""
        st.session_state[cls.KEY_DATASET] = df.copy()
        st.session_state[cls.KEY_RAW_DATASET] = df.copy()
        st.session_state[cls.KEY_FILE_NAME] = file_name
        # Reset downstream state when new data arrives
        st.session_state[cls.KEY_SCHEMA] = None
        st.session_state[cls.KEY_QUALITY] = None
        st.session_state[cls.KEY_EDA_REPORT] = None
        st.session_state[cls.KEY_ANOMALIES] = None
        st.session_state[cls.KEY_FORECAST] = None
        st.session_state[cls.KEY_INSIGHTS] = []

    @classmethod
    def clear_dataset(cls) -> None:
        """Remove all dataset-related state."""
        st.session_state[cls.KEY_DATASET] = None
        st.session_state[cls.KEY_RAW_DATASET] = None
        st.session_state[cls.KEY_FILE_NAME] = None
        st.session_state[cls.KEY_SCHEMA] = None
        st.session_state[cls.KEY_QUALITY] = None
        st.session_state[cls.KEY_EDA_REPORT] = None
        st.session_state[cls.KEY_ANOMALIES] = None
        st.session_state[cls.KEY_FORECAST] = None
        st.session_state[cls.KEY_INSIGHTS] = []

    # ── Chat Accessors ──
    @classmethod
    def add_chat_message(cls, role: str, content: str,
                         code: Optional[str] = None,
                         chart: Optional[Any] = None) -> None:
        """Append a message to chat history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "code": code,
            "chart": chart,
        }
        st.session_state[cls.KEY_CHAT_HISTORY].append(message)

    @classmethod
    def get_chat_history(cls) -> List[Dict[str, Any]]:
        """Return the full chat history."""
        return st.session_state.get(cls.KEY_CHAT_HISTORY, [])

    @classmethod
    def clear_chat(cls) -> None:
        """Clear chat history."""
        st.session_state[cls.KEY_CHAT_HISTORY] = []

    # ── Query History ──
    @classmethod
    def add_query(cls, question: str, code: str, code_type: str,
                  success: bool, execution_ms: int = 0) -> None:
        """Log an executed query and persist to disk."""
        entry = {
            "question": question,
            "code": code,
            "code_type": code_type,
            "success": success,
            "execution_ms": execution_ms,
            "timestamp": datetime.now().isoformat(),
        }
        st.session_state[cls.KEY_QUERY_HISTORY].append(entry)
        cls._save_query_history()

    @classmethod
    def clear_query_history(cls) -> None:
        """Clear query history from session and disk."""
        st.session_state[cls.KEY_QUERY_HISTORY] = []
        cls._save_query_history()

    # ── Model Status ──
    @classmethod
    def set_ollama_status(cls, connected: bool) -> None:
        """Update Ollama connection status."""
        st.session_state[cls.KEY_OLLAMA_STATUS] = connected

    @classmethod
    def is_ollama_connected(cls) -> bool:
        """Check if Ollama is connected."""
        return st.session_state.get(cls.KEY_OLLAMA_STATUS, False)

    # ── Generic ──
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Generic getter for any session key."""
        return st.session_state.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Generic setter for any session key."""
        st.session_state[key] = value
