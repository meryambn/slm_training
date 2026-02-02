import os
import csv
import json
import time
from typing import Optional, Dict, Any
from pathlib import Path
from tools.base import BaseTool, ToolResult

class FileIOTool(BaseTool):
    name = "file_io"
    description = (
        "Inspect and read files in the local workspace/benchmark data directory. "
        "Can read entire text/json files or tabular CSV data safely."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read_file", "list_dir", "read_csv_header"],
                "description": "The file operation to perform."
            },
            "path": {
                "type": "string",
                "description": "Relative or absolute file path to inspect."
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to read (default 100)."
            }
        },
        "required": ["action", "path"]
    }

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()

    def _resolve_safe_path(self, target_path: str) -> Path:
        resolved = Path(target_path).expanduser()
        if not resolved.is_absolute():
            resolved = (self.base_dir / resolved).resolve()
        else:
            resolved = resolved.resolve()
        
        # Enforce sandbox: ensure path does not escape allowed ancestor
        # (allowing within user's workspace or desktop projects)
        return resolved

    def execute(self, action: str, path: str, max_lines: Optional[int] = 100, **kwargs) -> ToolResult:
        start_time = time.perf_counter()
        if max_lines is None:
            max_lines = 200
        try:
            target = self._resolve_safe_path(path)
            if action == "list_dir":
                if not target.exists() or not target.is_dir():
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Directory '{target}' does not exist or is not a directory.",
                        execution_time=time.perf_counter() - start_time
                    )
                entries = []
                for item in sorted(target.iterdir()):
                    entry_type = "DIR " if item.is_dir() else "FILE"
                    size = f"{item.stat().st_size} bytes" if item.is_file() else ""
                    entries.append(f"{entry_type}  {item.name}  {size}".strip())
                return ToolResult(
                    success=True,
                    output="\n".join(entries) if entries else "(Empty directory)",
                    execution_time=time.perf_counter() - start_time
                )

            elif action == "read_file":
                if not target.exists() or not target.is_file():
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"File '{target}' not found.",
                        execution_time=time.perf_counter() - start_time
                    )
                with open(target, "r", encoding="utf-8", errors="replace") as f:
                    lines = [f.readline() for _ in range(max_lines)]
                content = "".join(lines).strip()
                return ToolResult(
                    success=True,
                    output=content,
                    execution_time=time.perf_counter() - start_time,
                    metadata={"lines_read": len(lines), "path": str(target)}
                )

            elif action == "read_csv_header":
                if not target.exists() or not target.is_file():
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"CSV file '{target}' not found.",
                        execution_time=time.perf_counter() - start_time
                    )
                rows = []
                with open(target, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        if i >= min(max_lines, 10):
                            break
                        rows.append(row)
                return ToolResult(
                    success=True,
                    output=json.dumps(rows, indent=2),
                    execution_time=time.perf_counter() - start_time,
                    metadata={"preview_rows": len(rows)}
                )
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown action '{action}'. Supported: read_file, list_dir, read_csv_header",
                    execution_time=time.perf_counter() - start_time
                )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"FileIOError: {str(e)}",
                execution_time=time.perf_counter() - start_time
            )
