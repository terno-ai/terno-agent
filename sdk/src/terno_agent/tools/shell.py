"""Shell-execution tool.

Runs commands via the user's default shell so chains, redirection, and
expansions work naturally. Output is combined stdout+stderr, truncated
to a reasonable size to protect the LLM's context window. The tool
runs in a fixed working directory, enforces a wall-clock timeout, and
honours an optional `CancelToken` — when cancellation is signalled the
whole process group is terminated (SIGTERM then SIGKILL).
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from terno_agent.core.cancel import CancelToken
from terno_agent.core.exceptions import AgentCancelled, ToolError
from terno_agent.core.tool import ToolSchema

_MAX_OUTPUT_BYTES = 64_000
_POLL_INTERVAL_S = 0.1
_TERM_GRACE_S = 0.5


@dataclass
class BashTool:
    workdir: Path
    default_timeout_s: int = 120
    cancel_token: CancelToken | None = field(default=None)

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

        token = self.cancel_token
        if token is not None and token.is_cancelled:
            raise AgentCancelled("cancelled before bash started")

        try:
            proc = subprocess.Popen(
                ["sh", "-c", command],
                cwd=str(self.workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise ToolError(f"Failed to launch shell: {exc}") from exc

        deadline = time.monotonic() + timeout
        timed_out = False
        cancelled = False
        try:
            while True:
                try:
                    stdout, stderr = proc.communicate(timeout=_POLL_INTERVAL_S)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if token is not None and token.is_cancelled:
                    cancelled = True
                    _terminate_group(proc)
                    stdout, stderr = _drain(proc)
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    _terminate_group(proc)
                    stdout, stderr = _drain(proc)
                    break
        finally:
            if proc.poll() is None:
                _terminate_group(proc)
                _drain(proc)

        if cancelled:
            # Surface as a clean cancellation up to the agent loop.
            raise AgentCancelled("bash cancelled by user")

        exit_code = proc.returncode if not timed_out else 124
        suffix = ""
        if timed_out:
            suffix = f"\n[timed out after {timeout}s]"
        return _format_output(
            exit_code=exit_code,
            stdout=stdout or "",
            stderr=(stderr or "") + suffix,
        )


def _terminate_group(proc: subprocess.Popen) -> None:
    """Send SIGTERM to the whole process group, then SIGKILL after a grace."""
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
    try:
        proc.wait(timeout=_TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
        try:
            proc.wait(timeout=_TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            pass


def _drain(proc: subprocess.Popen) -> tuple[str, str]:
    """Drain stdout/stderr after termination without blocking forever."""
    try:
        return proc.communicate(timeout=_TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        return ("", "")


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
