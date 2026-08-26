"""Two-pass sandboxed code execution engine for LLM-generated pandas code.

Pass 1 — AST Verification: walks the syntax tree and rejects dangerous constructs.
Pass 2 — Environment Isolation: exec() inside a stripped context with timeout.
"""

import ast
import threading
import signal
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

from utils.config import SANDBOX_TIMEOUT_SECONDS


class SecurityError(Exception):
    """Raised when code violates sandbox security policy."""


class SandboxTimeoutError(Exception):
    """Raised when code execution exceeds time limit."""


class SandboxExecutionError(Exception):
    """Raised when sandboxed code raises a runtime error."""


# ──────────────────────────────────────────────
# AST Policy — forbidden constructs
# ──────────────────────────────────────────────
FORBIDDEN_NODE_TYPES: Tuple[str, ...] = (
    "Import",
    "ImportFrom",
    "Exec",
    "Eval",
    "Global",
    "Nonlocal",
    "AsyncFunctionDef",
    "Await",
    "AsyncFor",
    "AsyncWith",
)

FORBIDDEN_NAMES: frozenset = frozenset({
    "__class__", "__subclasses__", "__mro__", "__builtins__",
    "__import__", "__loader__", "__spec__",
    "open", "eval", "exec", "compile",
    "getattr", "setattr", "delattr", "hasattr",
    "system", "popen", "os", "subprocess",
    "requests", "socket", "urllib", "http",
    "shutil", "glob", "pathlib",
    "builtins", "importlib",
    "input", "print",  # allow print via builtins, block via node check
    "exit", "quit",
})

FORBIDDEN_MODULES: frozenset = frozenset({
    "os", "sys", "subprocess", "shutil", "glob",
    "socket", "http", "urllib", "requests",
    "importlib", "builtins", "ctypes",
    "signal", "threading", "multiprocessing",
    "io", "code", "codeop",
})

FORBIDDEN_ATTRIBUTES: frozenset = frozenset({
    "__class__", "__subclasses__", "__mro__", "__builtins__",
    "__import__", "__loader__", "__spec__",
    "__globals__", "__code__", "__closure__",
    "__dict__", "__setattr__", "__delattr__",
    "system", "popen", "exec", "eval",
})


@dataclass
class SandboxResult:
    """Result of a sandboxed code execution."""
    success: bool
    output: Any = None
    result_df: Optional[pd.DataFrame] = None
    fig: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: int = 0


class ASTVerifier(ast.NodeVisitor):
    """Two-pass AST walker that rejects dangerous code patterns."""

    def __init__(self) -> None:
        self.violations: list = []

    def verify(self, code: str) -> None:
        """Parse and verify code. Raises SecurityError on violations."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SecurityError(f"Syntax error in generated code: {e}")

        self.violations = []
        self.visit(tree)

        if self.violations:
            msg = "Security violations detected:\n" + "\n".join(
                f"  - {v}" for v in self.violations
            )
            raise SecurityError(msg)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name.split(".")[0]
            self.violations.append(
                f"Import statement blocked: 'import {alias.name}'"
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            module = node.module.split(".")[0]
            self.violations.append(
                f"Import statement blocked: 'from {node.module} import ...'"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = self._get_call_name(node)
        if func_name:
            base = func_name.split(".")[0]
            if base in FORBIDDEN_NAMES or func_name in FORBIDDEN_NAMES:
                self.violations.append(
                    f"Function call blocked: '{func_name}'"
                )
            for part in func_name.split("."):
                if part in FORBIDDEN_ATTRIBUTES:
                    self.violations.append(
                        f"Forbidden attribute access: '{func_name}'"
                    )
                    break
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.violations.append(
                f"Forbidden attribute access: '.{node.attr}'"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self.violations.append(
                f"Forbidden name reference: '{node.id}'"
            )
        self.generic_visit(node)

    def visit_AnyNode(self, node: ast.AST) -> None:
        node_type = type(node).__name__
        if node_type in FORBIDDEN_NODE_TYPES:
            self.violations.append(
                f"Forbidden AST node type: '{node_type}'"
            )
        self.generic_visit(node)

    @staticmethod
    def _get_call_name(node: ast.Call) -> Optional[str]:
        """Extract the full dotted name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return None


# ──────────────────────────────────────────────
# Restricted builtins
# ──────────────────────────────────────────────
RESTRICTED_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "__build_class__": __build_class__,
    "__name__": "__main__",
}


def _execute_with_timeout(
    code: str,
    context: Dict[str, Any],
    timeout: int,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Execute code in a thread with timeout. Returns (locals, error)."""
    result_container: Dict[str, Any] = {"locals": {}, "error": None}

    def _target() -> None:
        try:
            local_ns: Dict[str, Any] = {}
            exec(code, context, local_ns)
            result_container["locals"] = local_ns
        except Exception as e:
            result_container["error"] = f"{type(e).__name__}: {e}"

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        result_container["error"] = (
            f"SandboxTimeoutError: Code execution exceeded {timeout}s limit"
        )

    return result_container["locals"], result_container["error"]


def execute_pandas_code(
    code: str,
    df: pd.DataFrame,
    timeout: int = SANDBOX_TIMEOUT_SECONDS,
) -> SandboxResult:
    """Execute LLM-generated pandas code in a sandboxed environment.

    Args:
        code: The pandas/Python code to execute.
        df: The target DataFrame (available as ``df`` in the sandbox).
        timeout: Maximum execution time in seconds.

    Returns:
        SandboxResult with output, optional DataFrame, optional figure, or error.
    """
    import time
    start = time.time()

    # ── Pass 1: AST Verification ──
    verifier = ASTVerifier()
    try:
        verifier.verify(code)
    except SecurityError as e:
        return SandboxResult(
            success=False,
            error=str(e),
            execution_time_ms=int((time.time() - start) * 1000),
        )

    # ── Pass 2: Sandboxed Execution ──
    allowed_globals: Dict[str, Any] = {
        "__builtins__": RESTRICTED_BUILTINS.copy(),
        "pd": pd,
        "np": np,
        "df": df.copy(),
    }

    local_ns, error = _execute_with_timeout(code, allowed_globals, timeout)
    elapsed_ms = int((time.time() - start) * 1000)

    if error:
        return SandboxResult(
            success=False,
            error=error,
            execution_time_ms=elapsed_ms,
        )

    # Extract results
    result_df = local_ns.get("result_df")
    fig = local_ns.get("fig")
    output = local_ns.get("output")

    # If no explicit result_df, try to capture the last expression
    if result_df is None and "result" in local_ns:
        candidate = local_ns["result"]
        if isinstance(candidate, pd.DataFrame):
            result_df = candidate

    return SandboxResult(
        success=True,
        output=output,
        result_df=result_df if isinstance(result_df, pd.DataFrame) else None,
        fig=fig,
        execution_time_ms=elapsed_ms,
    )


def execute_sql_query(
    query: str,
    engine: Any,
    timeout: int = SANDBOX_TIMEOUT_SECONDS,
) -> SandboxResult:
    """Execute a read-only SQL query against a database engine.

    Args:
        query: SQL query string (must be SELECT or WITH).
        engine: SQLAlchemy engine or connection.
        timeout: Maximum execution time in seconds.

    Returns:
        SandboxResult with a DataFrame of results or error.
    """
    import time
    from security.sql_guard import validate_sql_query

    start = time.time()

    # Validate SQL before execution
    is_valid, error_msg = validate_sql_query(query)
    if not is_valid:
        return SandboxResult(
            success=False,
            error=f"SQL Guard: {error_msg}",
            execution_time_ms=int((time.time() - start) * 1000),
        )

    # Execute with timeout
    result_container: Dict[str, Any] = {"df": None, "error": None}

    def _target() -> None:
        try:
            import pandas as pd
            result_container["df"] = pd.read_sql_query(query, engine)
        except Exception as e:
            result_container["error"] = f"{type(e).__name__}: {e}"

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    elapsed_ms = int((time.time() - start) * 1000)

    if thread.is_alive():
        return SandboxResult(
            success=False,
            error=f"Query timed out after {timeout}s",
            execution_time_ms=elapsed_ms,
        )

    if result_container["error"]:
        return SandboxResult(
            success=False,
            error=result_container["error"],
            execution_time_ms=elapsed_ms,
        )

    return SandboxResult(
        success=True,
        result_df=result_container["df"],
        execution_time_ms=elapsed_ms,
    )
