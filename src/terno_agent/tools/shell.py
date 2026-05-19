"""Shell-execution tool.

Runs commands via the user's default shell so chains, redirection, and
expansions work naturally. Output is combined stdout+stderr, truncated
to a reasonable size to protect the LLM's context window. The tool
runs in a fixed working directory and enforces a wall-clock timeout.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema

_MAX_OUTPUT_BYTES = 64_000


@dataclass
class BashTool:
    workdir: Path
    default_timeout_s: int = 120

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="bash",
            description=(
                "Run a shell command (POSIX `sh -c`) in the agent's working "
                "directory. Returns combined stdout+stderr and the exit "
                "code. Output is truncated if very large. Be careful with "
                "destructive commands (rm -rf, force pushes, etc.)."
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

        try:
            completed = subprocess.run(
                ["sh", "-c", command],
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return _format_output(
                exit_code=124,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + f"\n[timed out after {timeout}s]",
            )
        except OSError as exc:
            raise ToolError(f"Failed to launch shell: {exc}") from exc

        return _format_output(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
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
