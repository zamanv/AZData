"""Multi-sheet Excel file parser with type inference."""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ExcelLoadResult:
    """Result of an Excel file load operation."""
    success: bool
    dataframe: Optional[pd.DataFrame] = None
    sheet_names: Optional[List[str]] = None
    selected_sheet: str = ""
    rows: int = 0
    columns: int = 0
    file_size_mb: float = 0.0
    error: Optional[str] = None


def get_excel_sheets(file_bytes: bytes, file_name: str) -> List[str]:
    """Extract sheet names from an Excel file.

    Args:
        file_bytes: Raw file bytes.
        file_name: Name of the file for format detection.

    Returns:
        List of sheet names.
    """
    import io
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        return xls.sheet_names
    except Exception:
        return []


def load_excel(
    file_path: str,
    sheet_name: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> ExcelLoadResult:
    """Load an Excel file with optional sheet selection.

    Args:
        file_path: Path to the Excel file (.xlsx, .xls).
        sheet_name: Specific sheet to load (None = first sheet).
        max_rows: Maximum rows to read.

    Returns:
        ExcelLoadResult with the loaded DataFrame or error.
    """
    path = Path(file_path)
    if not path.exists():
        return ExcelLoadResult(
            success=False,
            error=f"File not found: {file_path}"
        )

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > 500:
        return ExcelLoadResult(
            success=False,
            error=f"File too large: {file_size_mb:.1f}MB (max 500MB)"
        )

    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        if sheet_name is None:
            sheet_name = sheet_names[0]

        df = pd.read_excel(
            xls,
            sheet_name=sheet_name,
            nrows=max_rows,
        )

        # Clean column names
        df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

        return ExcelLoadResult(
            success=True,
            dataframe=df,
            sheet_names=sheet_names,
            selected_sheet=sheet_name,
            rows=len(df),
            columns=len(df.columns),
            file_size_mb=round(file_size_mb, 2),
        )

    except Exception as e:
        return ExcelLoadResult(
            success=False,
            error=f"Failed to load Excel: {type(e).__name__}: {e}"
        )


def load_excel_from_bytes(
    file_bytes: bytes,
    file_name: str = "uploaded.xlsx",
    sheet_name: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> ExcelLoadResult:
    """Load Excel data from raw bytes.

    Args:
        file_bytes: Raw file bytes.
        file_name: Original filename for format detection.
        sheet_name: Specific sheet to load.
        max_rows: Maximum rows to read.

    Returns:
        ExcelLoadResult with the loaded DataFrame or error.
    """
    import io

    if len(file_bytes) > 500 * 1024 * 1024:
        return ExcelLoadResult(
            success=False,
            error=f"File too large: {len(file_bytes) / (1024*1024):.1f}MB"
        )

    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        sheet_names = xls.sheet_names

        if sheet_name is None:
            sheet_name = sheet_names[0]

        df = pd.read_excel(
            xls,
            sheet_name=sheet_name,
            nrows=max_rows,
        )

        df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

        return ExcelLoadResult(
            success=True,
            dataframe=df,
            sheet_names=sheet_names,
            selected_sheet=sheet_name,
            rows=len(df),
            columns=len(df.columns),
            file_size_mb=round(len(file_bytes) / (1024 * 1024), 2),
        )

    except Exception as e:
        return ExcelLoadResult(
            success=False,
            error=f"Failed to load Excel: {type(e).__name__}: {e}"
        )


def get_sheet_preview(
    file_bytes: bytes,
    sheet_name: str,
    max_rows: int = 5,
) -> Optional[pd.DataFrame]:
    """Get a preview of a specific sheet.

    Args:
        file_bytes: Raw file bytes.
        sheet_name: Name of the sheet to preview.
        max_rows: Number of preview rows.

    Returns:
        Preview DataFrame or None on error.
    """
    import io
    try:
        df = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=sheet_name,
            nrows=max_rows,
        )
        return df
    except Exception:
        return None
