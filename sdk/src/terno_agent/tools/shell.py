"""Shell-execution tool.

Runs commands through a `Sandbox`'s ``run_shell``. The default backend
(`LocalSandbox`) executes the command directly on the host in the agent's
working directory; a Docker-backed sandbox runs it inside the container
instead. Either way the tool goes through the same ``run_shell`` entry
point. Output is combined stdout+stderr, truncated to protect the LLM's
context window; the sandbox enforces the wall-clock timeout and honours
an optional `CancelToken`.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from terno_agent.core.cancel import CancelToken
from terno_agent.core.exceptions import AgentCancelled, ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.sandbox.base import Sandbox
from terno_agent.sandbox.local import LocalSandbox

_MAX_OUTPUT_BYTES = 64_000


class BashTool:
    def __init__(
        self,
        *,
        workdir: Path,
        sandbox: Sandbox | None = None,
        default_timeout_s: int = 120,
        cancel_token: CancelToken | None = None,
    ) -> None:
        self.workdir = workdir
        # Bash is always sandbox-backed. With no sandbox configured we run
        # on the host via LocalSandbox (in `workdir`); a Docker sandbox runs
        # the command inside its container.
        self.sandbox: Sandbox = sandbox or LocalSandbox()
        self.default_timeout_s = default_timeout_s
        self.cancel_token = cancel_token

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="bash",
            description=(
                "Run a shell command (POSIX `sh -c`) and return combined "
                "stdout+stderr and the exit code. Output is truncated if very "
                "large. Be careful with destructive commands (rm -rf, force "
                "pushes, etc.)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "timeout_s": {
                        "type": "integer",
                        "description": (
                            "Optional wall-clock timeout in seconds "
                            "(default 120)."
                        ),
                    },
                },
                "required": ["command"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        command = (kwargs.get("command") or "").strip()
        if not command:
            raise ToolError("bash requires a 'command' argument.")
        timeout = int(kwargs.get("timeout_s") or self.default_timeout_s)
        if timeout <= 0:
            raise ToolError("timeout_s must be positive.")

        token = self.cancel_token
        if token is not None and token.is_cancelled:
            raise AgentCancelled("cancelled before bash started")

        run_shell = getattr(self.sandbox, "run_shell", None)
        if run_shell is None:
            raise ToolError("configured sandbox does not support shell commands.")

        # Forward `cwd`/`cancel_token` only if this sandbox's run_shell accepts
        # them, so third-party sandboxes on an older signature don't break.
        params = inspect.signature(run_shell).parameters
        extra: dict[str, Any] = {}
        if "cwd" in params:
            extra["cwd"] = str(self.workdir)
        if "cancel_token" in params and token is not None:
            extra["cancel_token"] = token

        result = run_shell(command, timeout_s=timeout, **extra)
        suffix = f"\n[timed out after {timeout}s]" if result.timed_out else ""
        return _format_output(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr + suffix,
        )


def _format_output(*, exit_code: int, stdout: str, stderr: str) -> str:
    combined = stdout
    if stderr:
        combined += ("\n" if combined and not combined.endswith("\n") else "") + stderr
    if len(combined) > _MAX_OUTPUT_BYTES:
        keep = _MAX_OUTPUT_BYTES // 2
        combined = (
            combined[:keep]
            + f"\n... [truncated {len(combined) - _MAX_OUTPUT_BYTES} chars] ...\n"
            + combined[-keep:]
        )
    header = f"exit_code={exit_code}\n"
    if not combined.strip():
        return header + "(no output)"
    return header + combined


__all__ = ["BashTool"]
