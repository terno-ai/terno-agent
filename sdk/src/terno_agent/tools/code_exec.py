"""Code execution tool backed by a `Sandbox`."""

from __future__ import annotations

import inspect
from typing import Any

from terno_agent.core.cancel import CancelToken
from terno_agent.background import PYTHON, BackgroundTasks
from terno_agent.tools.background import BackgroundOption
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
        max_output_chars: int = 8000,
        background: BackgroundTasks | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.timeout_s = timeout_s
        self.cancel_token = cancel_token
        # Per-stream cap applied when rendering the result. The agent raises
        # it when a ToolOutputLimiter is shortening results (head + tail, full
        # text saved), so the limiter sees the whole output, not its head.
        self.max_output_chars = max_output_chars
        self.background = BackgroundOption(
            background,
            kind=PYTHON,
            note=(
                "A background run is a fresh Python process, not this tool's "
                "session: it must import and load everything it needs."
            ),
        )
        self.timeouts = 0

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="run_python",
            description=self.background.describe(
                "Run any Python code inside an isolated sandbox and return the "
                "captured stdout/stderr. Use this tool whenever you need to "
                "execute Python — arbitrary statements, expressions, scripts, "
                "multi-line programs, imports of the standard library, data "
                "processing, calculations, or quick experiments. Anything you "
                "could run in a Python interpreter can be run here. Print the "
                "values you want to see in the output, since only stdout/stderr "
                "is returned."
                "You have root access in the sandbox."
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
                        "description": f"Optional wall-clock timeout in seconds (default {self.timeout_s}).",
                    },
                    **self.background.properties(),
                },
                "required": ["code"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        code = kwargs.get("code")
        if not code:
            raise ToolError("run_python requires a 'code' argument.")
        if self.background.requested(kwargs):
            return self.background.start(code, kwargs.get("description"))
        timeout = int(kwargs.get("timeout_s") or self.timeout_s)
        # Only pass cancel_token if the sandbox supports it (custom sandboxes
        # implementing the older Sandbox protocol shouldn't break).
        params = inspect.signature(self.sandbox.run_python).parameters
        kwargs_extra: dict[str, Any] = {}
        if "cancel_token" in params and self.cancel_token is not None:
            kwargs_extra["cancel_token"] = self.cancel_token
        result = self.sandbox.run_python(code, timeout_s=timeout, **kwargs_extra)
        rendered = result.render(self.max_output_chars)
        if result.timed_out:
            self.timeouts += 1
            note = self.background.timeout_note(self.timeouts)
            if note:
                rendered += f"\n{note}"
        return rendered
