"""Sandboxed Python code execution tool."""

from __future__ import annotations

import subprocess
from typing import Any, Dict

TOOL_DEFINITION: Dict[str, Any] = {
    "name": "code_exec",
    "description": "Execute Python code and return stdout and stderr",
    "input_schema": {
        "type": "object",
        "properties": {
            "language": {
                "type": "string",
                "enum": ["python"],
                "description": "Programming language",
            },
            "code": {
                "type": "string",
                "description": "Code to execute",
            },
        },
        "required": ["language", "code"],
    },
}


def execute(language: str, code: str) -> Dict[str, Any]:
    """Run *code* in a subprocess and return stdout/stderr/exit_code."""
    if language != "python":
        return {
            "output": "",
            "error": f"Unsupported language: {language}",
            "success": False,
            "exit_code": -1,
        }

    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "output": result.stdout,
            "error": result.stderr or None,
            "success": result.returncode == 0,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "output": "",
            "error": "Execution timed out after 10s",
            "success": False,
            "exit_code": -1,
        }
