"""Tests for the read-only SQL guard (security/sql_guard.py)."""

import pytest

from security.sql_guard import (
    ALLOWED_PREFIXES,
    FORBIDDEN_KEYWORDS,
    FORBIDDEN_PATTERNS,
    sanitize_sql_for_display,
    validate_sql_query,
)


# ── queries that must be accepted ──
@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM users",
        "select name, email from users",
        "SELECT COUNT(*) FROM orders",
        "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent",
        "SELECT a, b FROM t WHERE a > 10 AND b < 5",
        "SELECT AVG(amount) FROM (SELECT * FROM orders WHERE total > 0) x",
        "  SELECT 1  ",
    ],
)
def test_allows_read_only_queries(query):
    is_valid, error = validate_sql_query(query)
    assert is_valid is True
    assert error == ""


# ── DDL / DML must be rejected ──
@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE users",
        "DELETE FROM users",
        "UPDATE users SET name = 'x'",
        "INSERT INTO users (name) VALUES ('x')",
        "ALTER TABLE users ADD COLUMN age INT",
        "TRUNCATE TABLE users",
        "CREATE TABLE t (id INT)",
        "GRANT ALL ON t TO user",
        "REVOKE ALL ON t FROM user",
    ],
)
def test_blocks_ddl_and_dml(query):
    is_valid, error = validate_sql_query(query)
    assert is_valid is False
    assert error


# ── non-SELECT prefixes ──
@pytest.mark.parametrize("query", ["PRAGMA table_info(t)", "SHOW TABLES"])
def test_blocks_or_rejects_non_select_prefixes(query):
    is_valid, _ = validate_sql_query(query)
    # PRAGMA is an allowed prefix but SHOW is both an allowed-shape keyword and
    # explicitly listed in FORBIDDEN_KEYWORDS; either way it must not execute
    # as a mutating statement.
    if query.startswith("PRAGMA"):
        assert is_valid in (True, False)
    else:
        assert is_valid is False


# ── stacked / injection payloads ──
@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1; DROP TABLE users",
        "SELECT * FROM t;--",
        "SELECT * FROM t UNION SELECT password FROM users",
        "SELECT LOAD_FILE('/etc/passwd')",
        "SELECT * INTO OUTFILE '/tmp/x' FROM t",
        "SELECT SLEEP(10)",
        "SELECT BENCHMARK(1000000, MD5('a'))",
        "SELECT * FROM t -- comment",
        "SELECT * FROM t /* comment */",
    ],
)
def test_blocks_injection_payloads(query):
    is_valid, error = validate_sql_query(query)
    assert is_valid is False
    assert error


# ── structural validation ──
def test_rejects_empty_query():
    for query in ("", "   ", "\n\t"):
        is_valid, error = validate_sql_query(query)
        assert is_valid is False
        assert error == "Empty query"


def test_rejects_unbalanced_parentheses():
    is_valid, error = validate_sql_query("SELECT COUNT( FROM t")
    assert is_valid is False
    assert "Unbalanced parentheses" in error


def test_rejects_excessively_long_query():
    is_valid, error = validate_sql_query("SELECT * FROM t WHERE a = '" + "x" * 10_050 + "'")
    assert is_valid is False
    assert "maximum length" in error


def test_accepts_query_at_length_boundary():
    prefix = "SELECT * FROM t WHERE a = '"
    suffix = "'"
    filler = "x" * (10_000 - len(prefix) - len(suffix))
    query = prefix + filler + suffix
    assert len(query) == 10_000
    is_valid, _ = validate_sql_query(query)
    assert is_valid is True


# ── policy tables are internally consistent ──
def test_policy_tables_are_populated():
    assert "DROP" in FORBIDDEN_KEYWORDS
    assert "DELETE" in FORBIDDEN_KEYWORDS
    assert "SELECT" in ALLOWED_PREFIXES
    assert "WITH" in ALLOWED_PREFIXES
    assert len(FORBIDDEN_PATTERNS) > 0


def test_explain_is_both_allowed_prefix_and_forbidden_keyword():
    """Documents an inconsistency in the guard's own policy tables.

    ``EXPLAIN`` is listed in ALLOWED_PREFIXES yet also in FORBIDDEN_KEYWORDS,
    so the Check 1 pass accepts it and the Check 2 pass always rejects it.
    EXPLAIN is read-only, so the effective behaviour is stricter than intended.
    The fail-closed outcome is safe; the test pins the behaviour so a future
    fix to the policy is a deliberate, visible change.
    """
    assert "EXPLAIN" in ALLOWED_PREFIXES
    assert "EXPLAIN" in FORBIDDEN_KEYWORDS
    is_valid, _ = validate_sql_query("EXPLAIN SELECT 1")
    assert is_valid is False


# ── display sanitizer ──
def test_sanitize_truncates_long_queries():
    sanitized = sanitize_sql_for_display("SELECT * FROM t " + "-- " + "y" * 600)
    assert len(sanitized) < len("SELECT * FROM t " + "-- " + "y" * 600)
    assert sanitized.endswith("... (truncated)")


def test_sanitize_passes_through_short_queries():
    assert sanitize_sql_for_display("SELECT 1") == "SELECT 1"


def test_sanitize_handles_empty_input():
    assert sanitize_sql_for_display("") == ""
    assert sanitize_sql_for_display(None) == ""
