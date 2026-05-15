"""Code execution tool backed by a `Sandbox`."""

from __future__ import annotations

from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.sandbox.base import Sandbox


class RunPythonTool:
    def __init__(self, sandbox: Sandbox, *, timeout_s: int = 30) -> None:
        self.sandbox = sandbox
        self.timeout_s = timeout_s

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="run_python",
            description=(
                "Execute a Python snippet inside an isolated sandbox and return "
                "captured stdout/stderr. The sandbox has no network access and "
                "no persistent filesystem; use stdin/stdout to pass data."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source to execute. Print results you want to see.",
                    },
                    "timeout_s": {
                        "type": "integer",
                        "description": "Optional wall-clock timeout in seconds (default 30).",
                    },
                },
                "required": ["code"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        code = kwargs.get("code")
        if not code:
            raise ToolError("run_python requires a 'code' argument.")
        timeout = int(kwargs.get("timeout_s") or self.timeout_s)
        result = self.sandbox.run_python(code, timeout_s=timeout)
        return result.render()
