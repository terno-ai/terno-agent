"""Monitor tool — run a command and react to its output line-by-line.

Use cases: wait for a dev server to print "listening on …", watch a
build until the first error line, tail a long-running command up to a
timeout. The tool always kills the subprocess (and its process group)
before returning so nothing leaks past the call.

Return semantics:
- ``status=matched``  the first line matching ``until_regex`` arrived.
- ``status=exited``   the process exited on its own.
- ``status=timeout``  ``timeout_s`` elapsed before either of the above.
"""

from __future__ import annotations

import fcntl
import os
import re
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
_POLL_INTERVAL_S = 0.05
_TERM_GRACE_S = 0.5
_DEFAULT_TIMEOUT_S = 60
_DEFAULT_MAX_LINES = 200


@dataclass
class MonitorTool:
    workdir: Path
    default_timeout_s: int = _DEFAULT_TIMEOUT_S
    cancel_token: CancelToken | None = field(default=None)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="monitor",
            description=(
                "Run a shell command and watch its output line-by-line, "
                "returning as soon as one of: a line matches `until_regex`, "
                "the command exits on its own, or `timeout_s` elapses. The "
                "process is killed when the tool returns, so this is for "
                "watching, not for running long-lived servers. Returns a "
                "status header followed by the captured output (latest "
                "lines, capped by `max_lines`)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute (POSIX `sh -c`).",
                    },
                    "until_regex": {
                        "type": "string",
                        "description": (
                            "Stop as soon as an output line matches this "
                            "Python regex. Omit to simply collect output "
                            "until the command exits or `timeout_s`."
                        ),
                    },
                    "timeout_s": {
                        "type": "integer",
                        "description": (
                            "Wall-clock timeout in seconds "
                            f"(default {_DEFAULT_TIMEOUT_S})."
                        ),
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": (
                            "Cap on retained output lines (default "
                            f"{_DEFAULT_MAX_LINES}; oldest lines drop)."
                        ),
                    },
                },
                "required": ["command"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        command = (kwargs.get("command") or "").strip()
        if not command:
            raise ToolError("monitor requires a 'command' argument.")
        timeout = int(kwargs.get("timeout_s") or self.default_timeout_s)
        if timeout <= 0:
            raise ToolError("timeout_s must be positive.")
        max_lines = int(kwargs.get("max_lines") or _DEFAULT_MAX_LINES)
        if max_lines <= 0:
            raise ToolError("max_lines must be positive.")
        regex = _compile_regex(kwargs.get("until_regex"))

        token = self.cancel_token
        if token is not None and token.is_cancelled:
            raise AgentCancelled("cancelled before monitor started")

        try:
            proc = subprocess.Popen(
                ["sh", "-c", command],
                cwd=str(self.workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            raise ToolError(f"Failed to launch shell: {exc}") from exc

        assert proc.stdout is not None
        _set_nonblocking(proc.stdout.fileno())

        deadline = time.monotonic() + timeout
        retained: list[str] = []
        partial = ""
        outcome = "exited"
        matched_line: str | None = None
        cancelled = False

        try:
            while True:
                if token is not None and token.is_cancelled:
                    cancelled = True
                    break
                if time.monotonic() >= deadline:
                    outcome = "timeout"
                    break

                try:
                    chunk = proc.stdout.read(4096)
                except (BlockingIOError, OSError):
                    chunk = None

                if chunk:
                    partial += chunk
                    while "\n" in partial:
                        line, partial = partial.split("\n", 1)
                        _append_line(retained, line, max_lines)
                        if regex is not None and regex.search(line):
                            matched_line = line
                            outcome = "matched"
                            break
                    if outcome == "matched":
                        break
                    continue

                if proc.poll() is not None:
                    try:
                        tail = proc.stdout.read() or ""
                    except (BlockingIOError, OSError):
                        tail = ""
                    if tail:
                        partial += tail
                    chunks = partial.split("\n")
                    for line in chunks[:-1]:
                        _append_line(retained, line, max_lines)
                    if chunks[-1]:
                        _append_line(retained, chunks[-1], max_lines)
                    partial = ""
                    outcome = "exited"
                    break

                time.sleep(_POLL_INTERVAL_S)
        finally:
            if proc.poll() is None:
                _terminate_group(proc)

        if partial and outcome != "exited":
            _append_line(retained, partial, max_lines)

        if cancelled:
            raise AgentCancelled("monitor cancelled by user")

        return _format_output(
            outcome=outcome,
            exit_code=proc.returncode if outcome == "exited" else None,
            matched_line=matched_line,
            lines=retained,
        )


def _compile_regex(raw: Any) -> re.Pattern[str] | None:
    if raw is None or raw == "":
        return None
    try:
        return re.compile(str(raw))
    except re.error as exc:
        raise ToolError(f"invalid until_regex: {exc}") from exc


def _append_line(buffer: list[str], line: str, max_lines: int) -> None:
    buffer.append(line)
    if len(buffer) > max_lines:
        del buffer[: len(buffer) - max_lines]


def _set_nonblocking(fd: int) -> None:
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def _terminate_group(proc: subprocess.Popen[str]) -> None:
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


def _format_output(
    *,
    outcome: str,
    exit_code: int | None,
    matched_line: str | None,
    lines: list[str],
) -> str:
    header_parts = [f"status={outcome}"]
    if outcome == "exited" and exit_code is not None:
        header_parts.append(f"exit_code={exit_code}")
    if outcome == "matched" and matched_line is not None:
        header_parts.append(f"matched_line={matched_line!r}")
    header = " ".join(header_parts)

    body = "\n".join(lines).rstrip("\n")
    if len(body) > _MAX_OUTPUT_BYTES:
        keep = _MAX_OUTPUT_BYTES // 2
        body = (
            body[:keep]
            + f"\n... [truncated {len(body) - _MAX_OUTPUT_BYTES} chars] ...\n"
            + body[-keep:]
        )
    return header + "\n" + (body or "(no output)")


__all__ = ["MonitorTool"]
