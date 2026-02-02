import sys
import subprocess
import time
from typing import Dict, Any
from tools.base import BaseTool, ToolResult

class PythonREPLTool(BaseTool):
    name = "python_repl"
    description = (
        "Execute a snippet of Python code in an isolated subprocess. "
        "Use this for arithmetic calculations, data processing, statistical computing, "
        "and algorithmic logic. Always print your final result to stdout using print()."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python source code to execute. Must use print() to output results."
            }
        },
        "required": ["code"]
    }

    def __init__(self, timeout: int = 10, max_output_chars: int = 4000):
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def execute(self, code: str, **kwargs) -> ToolResult:
        start_time = time.perf_counter()
        if not code or not code.strip():
            return ToolResult(
                success=False,
                output="",
                error="Error: Empty code provided.",
                execution_time=0.0
            )

        # If code ends with an unprinted expression, wrap it in print() using AST unparse
        processed_code = code
        try:
            import ast
            parsed = ast.parse(code)
            if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                last_expr = parsed.body[-1]
                val = last_expr.value
                # Do not wrap if it's already a print()
                if not (isinstance(val, ast.Call) and getattr(val.func, "id", None) == "print"):
                    # Wrap the expression node with print()
                    print_node = ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id="print", ctx=ast.Load()),
                            args=[val],
                            keywords=[]
                        )
                    )
                    parsed.body[-1] = print_node
                    ast.fix_missing_locations(parsed)
                    processed_code = ast.unparse(parsed)
        except Exception:
            processed_code = code

        try:
            process = subprocess.run(
                [sys.executable, "-c", processed_code],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            elapsed = time.perf_counter() - start_time
            stdout = process.stdout.strip()
            stderr = process.stderr.strip()

            if process.returncode == 0:
                output = stdout if stdout else "(Execution completed with no stdout output)"
                if len(output) > self.max_output_chars:
                    output = output[:self.max_output_chars] + "\n... [Output truncated]"
                return ToolResult(
                    success=True,
                    output=output,
                    error=None,
                    execution_time=elapsed,
                    metadata={"returncode": 0}
                )
            else:
                error_msg = stderr if stderr else f"Process exited with code {process.returncode}"
                return ToolResult(
                    success=False,
                    output=stdout,
                    error=error_msg,
                    execution_time=elapsed,
                    metadata={"returncode": process.returncode}
                )

        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start_time
            return ToolResult(
                success=False,
                output="",
                error=f"TimeoutError: Execution exceeded time limit of {self.timeout}s.",
                execution_time=elapsed,
                metadata={"timed_out": True}
            )
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return ToolResult(
                success=False,
                output="",
                error=f"SystemExecutionError: {str(e)}",
                execution_time=elapsed
            )
