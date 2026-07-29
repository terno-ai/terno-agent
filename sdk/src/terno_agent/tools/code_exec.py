"""Code execution tool backed by a `Sandbox`."""

from __future__ import annotations

import inspect
from typing import Any

from terno_agent.core.cancel import CancelToken
from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.sandbox.base import Sandbox


class RunPythonTool:
    def __init__(
        self,
        sandbox: Sandbox,
        *,
        timeout_s: int = 30,
        cancel_token: CancelToken | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.timeout_s = timeout_s
        self.cancel_token = cancel_token

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="run_python",
            description=(
                "Run any Python code inside an isolated sandbox and return the "
                "captured stdout/stderr. Use this tool whenever you need to "
                "execute Python — arbitrary statements, expressions, scripts, "
                "multi-line programs, imports of the standard library, data "
                "processing, calculations, or quick experiments. Anything you "
                "could run in a Python interpreter can be run here. Print the "
                "values you want to see in the output, since only stdout/stderr "
                "is returned."
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
        # Only pass cancel_token if the sandbox supports it (custom sandboxes
        # implementing the older Sandbox protocol shouldn't break).
        params = inspect.signature(self.sandbox.run_python).parameters
        kwargs_extra: dict[str, Any] = {}
        if "cancel_token" in params and self.cancel_token is not None:
            kwargs_extra["cancel_token"] = self.cancel_token
        result = self.sandbox.run_python(code, timeout_s=timeout, **kwargs_extra)
        return result.render()
