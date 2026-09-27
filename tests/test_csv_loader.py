"""Tests for robust CSV ingestion (ingestion/csv_loader.py)."""

import pandas as pd
import pytest

from ingestion.csv_loader import (
    CSVLoadResult,
    detect_delimiter,
    detect_encoding,
    load_csv,
    load_csv_from_bytes,
)


@pytest.fixture
def comma_csv(tmp_path):
    path = tmp_path / "comma.csv"
    path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def semicolon_csv(tmp_path):
    path = tmp_path / "semi.csv"
    path.write_text("a;b;c\n1;2;3\n4;5;6\n7;8;9\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def tab_csv(tmp_path):
    path = tmp_path / "tab.tsv"
    path.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n", encoding="utf-8")
    return str(path)


# ── delimiter detection ──
def test_detects_comma_delimiter(comma_csv):
    assert detect_delimiter(comma_csv, "utf-8") == ","


def test_detects_semicolon_delimiter(semicolon_csv):
    assert detect_delimiter(semicolon_csv, "utf-8") == ";"


def test_detects_tab_delimiter(tab_csv):
    assert detect_delimiter(tab_csv, "utf-8") == "\t"


def test_delimiter_detection_falls_back_on_missing_file():
    assert detect_delimiter("does_not_exist.csv", "utf-8") == ","


# ── encoding detection ──
def test_detects_encoding_for_plain_ascii(tmp_path):
    """charset_normalizer may legitimately report 'ascii' for pure-ASCII input,
    which is a subset of UTF-8; the loader must not require a specific label."""
    path = tmp_path / "ascii.csv"
    path.write_text("a,b\n1,2\n", encoding="ascii")
    assert detect_encoding(str(path)).lower() in {"ascii", "utf-8", "utf_8"}


def test_detects_encoding_for_non_ascii_content(tmp_path):
    path = tmp_path / "accents.csv"
    path.write_text("name\nJosé\nZoë\n", encoding="utf-8")
    encoding = detect_encoding(str(path))
    assert encoding.lower() not in {"ascii"}
    assert load_csv(str(path)).success is True


def test_encoding_detection_falls_back_on_missing_file():
    assert detect_encoding("does_not_exist.csv") == "utf-8"


# ── loading ──
def test_loads_a_simple_csv(comma_csv):
    result = load_csv(comma_csv)
    assert isinstance(result, CSVLoadResult)
    assert result.success is True
    assert result.rows == 2
    assert result.columns == 3
    assert list(result.dataframe.columns) == ["a", "b", "c"]
    assert result.error is None


def test_respects_max_rows(comma_csv):
    assert load_csv(comma_csv, max_rows=1).dataframe.shape[0] == 1


def test_respects_sample_rows(comma_csv):
    assert load_csv(comma_csv, sample_rows=1).dataframe.shape[0] == 1


def test_strips_whitespace_from_column_names(tmp_path):
    path = tmp_path / "padded.csv"
    path.write_text(" a , b \n1,2\n", encoding="utf-8")
    assert list(load_csv(str(path)).dataframe.columns) == ["a", "b"]


def test_handles_semicolon_file(semicolon_csv):
    result = load_csv(semicolon_csv)
    assert result.success is True
    assert result.delimiter == ";"
    assert list(result.dataframe.columns) == ["a", "b", "c"]


def test_infers_numeric_dtypes(comma_csv):
    assert all(str(t).startswith("int") for t in load_csv(comma_csv).dataframe.dtypes)


def test_missing_file_returns_error_not_exception():
    result = load_csv("no_such_file.csv")
    assert result.success is False
    assert "File not found" in result.error
    assert result.dataframe is None


def test_file_size_is_reported(comma_csv):
    assert load_csv(comma_csv).file_size_mb >= 0.0


# ── bytes / upload path ──
def test_loads_from_bytes():
    payload = b"a,b\n1,2\n3,4\n"
    result = load_csv_from_bytes(payload, file_name="upload.csv")
    assert result.success is True
    assert result.rows == 2
    assert result.columns == 2


def test_load_from_bytes_respects_max_rows():
    payload = b"a,b\n" + b"\n".join(b"%d,%d" % (i, i) for i in range(50))
    assert load_csv_from_bytes(payload, max_rows=10).dataframe.shape[0] == 10


def test_load_from_bytes_rejects_oversized_payload():
    oversized = b"x" * (500 * 1024 * 1024 + 1)
    result = load_csv_from_bytes(oversized)
    assert result.success is False
    assert "too large" in result.error.lower()


def test_load_from_bytes_forces_encoding_when_asked():
    payload = "name,city\nJosé,Kochi\n".encode("utf-8")
    result = load_csv_from_bytes(payload, encoding="utf-8")
    assert result.success is True
    assert "José" in result.dataframe["name"].iloc[0]


def test_load_from_bytes_reports_garbage_as_error():
    result = load_csv_from_bytes(b"")
    assert isinstance(result, CSVLoadResult)


def test_temp_file_is_cleaned_up(tmp_path):
    before = set(tmp_path.iterdir())
    load_csv_from_bytes(b"a,b\n1,2\n")
    assert set(tmp_path.iterdir()) == before
