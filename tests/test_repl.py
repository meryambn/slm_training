import pytest
from tools.python_repl import PythonREPLTool

def test_repl_basic_arithmetic():
    tool = PythonREPLTool(timeout=5)
    result = tool.execute("print(2 + 2)")
    assert result.success is True
    assert result.output == "4"
    assert result.error is None
    assert result.execution_time > 0

def test_repl_complex_computation():
    tool = PythonREPLTool(timeout=5)
    code = """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(fib(10))
"""
    result = tool.execute(code)
    assert result.success is True
    assert result.output == "55"

def test_repl_runtime_error():
    tool = PythonREPLTool(timeout=5)
    result = tool.execute("print(1 / 0)")
    assert result.success is False
    assert "ZeroDivisionError" in result.error

def test_repl_timeout_handling():
    # Strict 1-second timeout
    tool = PythonREPLTool(timeout=1)
    result = tool.execute("import time; time.sleep(5)")
    assert result.success is False
    assert "TimeoutError" in result.error
    assert result.metadata.get("timed_out") is True

def test_repl_empty_code():
    tool = PythonREPLTool(timeout=5)
    result = tool.execute("   ")
    assert result.success is False
    assert "Empty code" in result.error
