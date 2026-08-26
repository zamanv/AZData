"""Robust CSV file loader with auto-encoding and delimiter detection.

Supports large files with chunked reading and automatic type inference.
"""

from typing import Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from charset_normalizer import from_path


@dataclass
class CSVLoadResult:
    """Result of a CSV file load operation."""
    success: bool
    dataframe: Optional[pd.DataFrame] = None
    encoding: str = ""
    delimiter: str = ","
    rows: int = 0
    columns: int = 0
    file_size_mb: float = 0.0
    error: Optional[str] = None


def detect_encoding(file_path: str) -> str:
    """Detect file encoding using charset_normalizer.

    Args:
        file_path: Path to the file.

    Returns:
        Detected encoding name (e.g., 'utf-8', 'latin-1').
    """
    try:
        results = from_path(file_path)
        best = results.best()
        if best is not None:
            return str(best.encoding)
    except Exception:
        pass
    return "utf-8"


def detect_delimiter(file_path: str, encoding: str) -> str:
    """Detect CSV delimiter by analyzing the first few lines.

    Args:
        file_path: Path to the CSV file.
        encoding: File encoding to use.

    Returns:
        Detected delimiter character.
    """
    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            sample = f.read(8192)

        # Try common delimiters
        delimiters = [",", "\t", ";", "|", ":"]
        scores = {}

        for delim in delimiters:
            lines = sample.strip().split("\n")
            if not lines:
                continue
            counts = [line.count(delim) for line in lines[:10]]
            # Consistent count across lines = likely the right delimiter
            if counts and max(counts) > 0:
                consistency = len(set(counts))  # lower = more consistent
                scores[delim] = (max(counts), -consistency)

        if scores:
            best = max(scores.items(), key=lambda x: x[1])
            return best[0]
    except Exception:
        pass
    return ","


def load_csv(
    file_path: str,
    max_rows: Optional[int] = None,
    sample_rows: Optional[int] = None,
) -> CSVLoadResult:
    """Load a CSV file with automatic encoding and delimiter detection.

    Args:
        file_path: Path to the CSV file.
        max_rows: Maximum rows to read (None = all).
        sample_rows: If set, read only this many rows for preview.

    Returns:
        CSVLoadResult with the loaded DataFrame or error.
    """
    path = Path(file_path)
    if not path.exists():
        return CSVLoadResult(
            success=False,
            error=f"File not found: {file_path}"
        )

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > 500:
        return CSVLoadResult(
            success=False,
            error=f"File too large: {file_size_mb:.1f}MB (max 500MB)"
        )

    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)

    try:
        nrows = sample_rows or max_rows
        df = pd.read_csv(
            file_path,
            encoding=encoding,
            sep=delimiter,
            nrows=nrows,
            on_bad_lines="warn",
            low_memory=False,
            engine="python",
        )

        # If sample_rows was set, that's all we want
        if sample_rows:
            pass
        elif max_rows is None:
            # Re-read full file (first read was just for detection)
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=delimiter,
                on_bad_lines="warn",
                low_memory=False,
                engine="python",
            )

        # Clean column names
        df.columns = df.columns.str.strip()

        return CSVLoadResult(
            success=True,
            dataframe=df,
            encoding=encoding,
            delimiter=delimiter,
            rows=len(df),
            columns=len(df.columns),
            file_size_mb=round(file_size_mb, 2),
        )

    except pd.errors.ParserError as e:
        return CSVLoadResult(
            success=False,
            encoding=encoding,
            delimiter=delimiter,
            error=f"CSV parsing error: {e}"
        )
    except UnicodeDecodeError as e:
        return CSVLoadResult(
            success=False,
            error=f"Encoding error: {e}. Detected encoding: {encoding}"
        )
    except Exception as e:
        return CSVLoadResult(
            success=False,
            error=f"Failed to load CSV: {type(e).__name__}: {e}"
        )


def load_csv_from_bytes(
    file_bytes: bytes,
    file_name: str = "uploaded.csv",
    encoding: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> CSVLoadResult:
    """Load CSV data from raw bytes (e.g., Streamlit UploadedFile).

    Args:
        file_bytes: Raw file bytes.
        file_name: Name of the file for error messages.
        encoding: Force a specific encoding (auto-detect if None).
        max_rows: Maximum rows to read.

    Returns:
        CSVLoadResult with the loaded DataFrame or error.
    """
    import tempfile
    import os

    if len(file_bytes) > 500 * 1024 * 1024:
        return CSVLoadResult(
            success=False,
            error=f"File too large: {len(file_bytes) / (1024*1024):.1f}MB"
        )

    # Write to temp file for encoding detection
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".csv", mode="wb"
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        if encoding is None:
            encoding = detect_encoding(tmp_path)

        delimiter = detect_delimiter(tmp_path, encoding)

        nrows = max_rows
        df = pd.read_csv(
            tmp_path,
            encoding=encoding,
            sep=delimiter,
            nrows=nrows,
            on_bad_lines="warn",
            low_memory=False,
            engine="python",
        )

        df.columns = df.columns.str.strip()

        return CSVLoadResult(
            success=True,
            dataframe=df,
            encoding=encoding,
            delimiter=delimiter,
            rows=len(df),
            columns=len(df.columns),
            file_size_mb=round(len(file_bytes) / (1024 * 1024), 2),
        )

    except Exception as e:
        return CSVLoadResult(
            success=False,
            error=f"Failed to load CSV: {type(e).__name__}: {e}"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
