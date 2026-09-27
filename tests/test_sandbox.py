"""Tests for the two-pass sandboxed code-execution engine (security/sandbox.py)."""

import pandas as pd
import pytest

from security.sandbox import (
    FORBIDDEN_ATTRIBUTES,
    FORBIDDEN_MODULES,
    FORBIDDEN_NAMES,
    FORBIDDEN_NODE_TYPES,
    ASTVerifier,
    SecurityError,
    _execute_with_timeout,
    execute_pandas_code,
)


# ── AST policy: code that must be allowed ──
@pytest.mark.parametrize(
    "code",
    [
        "result_df = df.head(5)",
        "output = df['a'].sum()",
        "result = df.describe()",
        "result_df = df[df['a'] > 2].copy()",
        "output = round(float(df['a'].mean()), 2)",
        "result_df = pd.DataFrame({'s': df['a'].sum()})",
        "x = 1\ny = 2\noutput = x + y",
    ],
)
def test_allows_benign_pandas_code(code):
    ASTVerifier().verify(code)


# ── AST policy: code that must be blocked ──
@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "import subprocess",
        "from os import system",
        "from subprocess import Popen",
        "eval('1+1')",
        "exec('x=1')",
        "compile('1', 'f', 'eval')",
        "__import__('os').system('ls')",
        "getattr(df, 'shape')",
        "setattr(df, 'x', 1)",
    ],
)
def test_blocks_dangerous_calls_and_imports(code):
    with pytest.raises(SecurityError):
        ASTVerifier().verify(code)


@pytest.mark.parametrize(
    "code",
    [
        "x = ().__class__",
        "x = df.__class__",
        "x = df.iloc.__globals__",
        "x = df.__dict__",
        "x = type(df).__mro__",
    ],
)
def test_blocks_dunder_attribute_access(code):
    with pytest.raises(SecurityError):
        ASTVerifier().verify(code)


def test_blocks_syntax_errors_as_security_error():
    with pytest.raises(SecurityError, match="Syntax error"):
        ASTVerifier().verify("x = 1 +")


def test_error_message_enumerates_every_violation():
    """Each blocked identifier is reported twice — once by the Call check and
    once by the Name check — so three constructs yield six violation lines."""
    with pytest.raises(SecurityError) as exc_info:
        ASTVerifier().verify("import os\neval('1')\ngetattr(x,'y')")
    message = str(exc_info.value)
    assert message.startswith("Security violations detected:")
    assert message.count("\n  - ") == 6
    for fragment in ("import os", "eval", "getattr"):
        assert fragment in message
    assert "Function call blocked" in message
    assert "Forbidden name reference" in message


def test_verifier_state_is_reset_between_runs():
    """A reused verifier must not leak violations from a previous call."""
    verifier = ASTVerifier()
    with pytest.raises(SecurityError):
        verifier.verify("import os")
    verifier.verify("result_df = df.head(1)")
    assert verifier.violations == []


# ── policy tables ──
def test_policy_tables_cover_expected_threats():
    assert {"os", "sys", "subprocess", "socket"} <= FORBIDDEN_MODULES
    assert {"eval", "exec", "__import__", "open"} <= FORBIDDEN_NAMES
    assert "__class__" in FORBIDDEN_ATTRIBUTES
    assert "Import" in FORBIDDEN_NODE_TYPES


@pytest.mark.xfail(
    reason=(
        "Known gap: ASTVerifier defines visit_AnyNode, but ast.NodeVisitor "
        "dispatches on exact class names, so the FORBIDDEN_NODE_TYPES policy "
        "never runs. 'async def' currently passes verification."
    ),
    strict=True,
)
def test_forbidden_node_types_are_actually_enforced():
    with pytest.raises(SecurityError):
        ASTVerifier().verify("async def f():\n    pass")


# ── sandboxed execution ──
def test_execution_captures_output_variable(sample_df):
    result = execute_pandas_code("output = df['a'].sum()", sample_df, timeout=5)
    assert result.success is True
    assert result.error is None
    assert float(result.output) == 15.0
    assert result.execution_time_ms >= 0


def test_execution_captures_result_df(sample_df):
    result = execute_pandas_code("result_df = df.head(2)", sample_df, timeout=5)
    assert result.success is True
    assert isinstance(result.result_df, pd.DataFrame)
    assert result.result_df.shape == (2, 2)


def test_execution_promotes_bare_result_dataframe(sample_df):
    """Code assigning `result` to a DataFrame is surfaced as result_df."""
    result = execute_pandas_code("result = df.head(1)", sample_df, timeout=5)
    assert result.success is True
    assert isinstance(result.result_df, pd.DataFrame)
    assert result.result_df.shape == (1, 2)


def test_execution_reports_runtime_errors(sample_df):
    result = execute_pandas_code("output = 1 / 0", sample_df, timeout=5)
    assert result.success is False
    assert "ZeroDivisionError" in result.error


def test_execution_rejects_blocked_code_before_running(sample_df, tmp_path):
    """A blocked payload must never reach the OS."""
    marker = tmp_path / "should_not_exist.txt"
    code = "import os\nos.system('echo pwned > marker.txt')"
    result = execute_pandas_code(code, sample_df, timeout=5)
    assert result.success is False
    assert "Security violations detected" in result.error
    assert not marker.exists()


def test_execution_gives_sandbox_a_copy_of_the_frame(sample_df):
    """Mutating df inside the sandbox must not affect the caller's frame."""
    original = sample_df.copy()
    result = execute_pandas_code(
        "df.loc[0, 'a'] = 99999\noutput = int(df['a'].sum())", sample_df, timeout=5
    )
    assert result.success is True
    pd.testing.assert_frame_equal(sample_df, original)
    assert float(result.output) == 100013.0


def test_timeout_mechanism_triggers_on_a_slow_call():
    """Unit test of the timeout mechanism itself.

    A GIL-releasing call is used deliberately: a tight C-level loop such as
    ``sum(range(10**9))`` holds the GIL, so ``Thread.join(timeout=...)`` cannot
    even wake up until the loop finishes. See
    test_timeout_cannot_interrupt_gil_holding_code for that limitation.
    """
    import time as _time

    locals_, error = _execute_with_timeout(
        "sleeper(3)", {"sleeper": _time.sleep}, timeout=1
    )
    assert error is not None
    assert "exceeded" in error


def test_timeout_mechanism_allows_fast_code_to_finish():
    import time as _time

    locals_, error = _execute_with_timeout(
        "sleeper(0)\nresult = 6 * 7", {"sleeper": _time.sleep}, timeout=5
    )
    assert error is None
    assert locals_["result"] == 42


def test_timeout_mechanism_surfaces_runtime_errors():
    locals_, error = _execute_with_timeout("1 / 0", {}, timeout=5)
    assert error is not None
    assert "ZeroDivisionError" in error


@pytest.mark.xfail(
    reason=(
        "Known limitation: the timeout is enforced with Thread.join(timeout), "
        "which cannot pre-empt C-level code that holds the GIL, so a pure-CPU "
        "loop in generated code can run unbounded. Real containment needs a "
        "separate process, a resource limit (RLIMIT_CPU), or a container."
    ),
    strict=True,
)
def test_timeout_cannot_interrupt_gil_holding_code():
    import time as _time

    start = _time.time()
    locals_, error = _execute_with_timeout("sum(range(300000000))", {}, timeout=1)
    elapsed = _time.time() - start
    assert error is not None
    assert elapsed < 2.0


def test_print_is_blocked_despite_being_an_allowed_builtin(sample_df):
    """Pins an inconsistency: print is in RESTRICTED_BUILTINS but visit_Name
    rejects the name, so generated code can never print."""
    result = execute_pandas_code("print(df.shape)", sample_df, timeout=5)
    assert result.success is False
    assert "print" in result.error
