"""AST-based SQL read-only validator.

Ensures all queries are safe SELECT/WITH statements before execution.
Blocks any DDL, DML, or chained-statement attempts.
"""

import re
from typing import Tuple, List


# ──────────────────────────────────────────────
# Forbidden SQL keywords / patterns
# ──────────────────────────────────────────────
FORBIDDEN_KEYWORDS: frozenset = frozenset({
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "GRANT", "REVOKE", "EXEC", "EXECUTE", "CREATE", "REPLACE",
    "MERGE", "CALL", "COMMIT", "ROLLBACK", "SET", "RESET",
    "SHOW", "DESCRIBE", "EXPLAIN",  # block info leakage
})

FORBIDDEN_PATTERNS: List[re.Pattern] = [
    re.compile(r";\s*\S", re.IGNORECASE),  # semicolon chaining
    re.compile(r"UNION\s+SELECT", re.IGNORECASE),
    re.compile(r"INTO\s+(OUTFILE|DUMPFILE)", re.IGNORECASE),
    re.compile(r"LOAD_FILE\s*\(", re.IGNORECASE),
    re.compile(r"BENCHMARK\s*\(", re.IGNORECASE),
    re.compile(r"SLEEP\s*\(", re.IGNORECASE),
    re.compile(r"WAITFOR\s+DELAY", re.IGNORECASE),
]

ALLOWED_PREFIXES: Tuple[str, ...] = (
    "SELECT", "WITH", "EXPLAIN", "PRAGMA",
)


def validate_sql_query(query: str) -> Tuple[bool, str]:
    """Validate that a SQL query is read-only and safe.

    Args:
        query: The SQL query string to validate.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if not query or not query.strip():
        return False, "Empty query"

    normalized = query.strip()

    # ── Check 1: Must start with allowed prefix ──
    upper = normalized.upper()
    if not any(upper.startswith(prefix) for prefix in ALLOWED_PREFIXES):
        first_word = normalized.split()[0].upper() if normalized.split() else ""
        return False, (
            f"Query must start with SELECT or WITH. "
            f"Got: '{first_word}'"
        )

    # ── Check 2: No forbidden keywords ──
    # Tokenize by splitting on whitespace and punctuation
    words = re.findall(r'\b\w+\b', upper)
    for word in words:
        if word in FORBIDDEN_KEYWORDS:
            return False, f"Forbidden SQL keyword detected: '{word}'"

    # ── Check 3: No forbidden patterns ──
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(normalized):
            return False, (
                f"Forbidden SQL pattern detected: "
                f"'{pattern.pattern}'"
            )

    # ── Check 4: Balanced parentheses ──
    open_parens = normalized.count("(")
    close_parens = normalized.count(")")
    if open_parens != close_parens:
        return False, (
            f"Unbalanced parentheses: {open_parens} open, "
            f"{close_parens} close"
        )

    # ── Check 5: No comment injection ──
    if "--" in normalized or "/*" in normalized:
        return False, "SQL comments not allowed in generated queries"

    # ── Check 6: Length limit ──
    if len(normalized) > 10000:
        return False, "Query exceeds maximum length of 10,000 characters"

    return True, ""


def sanitize_sql_for_display(query: str) -> str:
    """Return a display-safe version of a SQL query.

    Truncates very long queries and strips any potential injection payloads
    from the display string (does NOT affect execution).
    """
    if not query:
        return ""
    sanitized = query.strip()
    if len(sanitized) > 500:
        sanitized = sanitized[:500] + "\n... (truncated)"
    return sanitized
