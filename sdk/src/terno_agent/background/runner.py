"""Runners: what actually executes a background task.

:class:`BackgroundRunner` is the contract. :class:`SandboxBackgroundRunner`
works on any :class:`~terno_agent.sandbox.base.Sandbox`: it sends the
operations in :mod:`~terno_agent.background._sandbox_ops` through
``run_python`` to start a detached process inside the sandbox and to check
on it, so the work runs in the same environment as ``run_python`` itself.
A host with a better channel (a container API, a job queue) registers its
own runner instead.
"""

from __future__ import annotations

import inspect
import json
import os
import uuid
from dataclasses import dataclass
from functools import cache
from typing import Protocol, runtime_checkable

from terno_agent.background import _sandbox_ops
from terno_agent.background.store import BackgroundTask
from terno_agent.sandbox.base import Sandbox
from terno_agent.sandbox.local import LocalSandbox


@dataclass(frozen=True, slots=True)
class RunnerState:

    running: bool
    exit_code: int | None = None
    # The work outlived its time limit and was stopped.
    timed_out: bool = False
    error: str = ""
    ended_at: float | None = None


class BackgroundRunnerError(Exception):
    """The runner could not do what was asked (start, poll, read or stop)."""


@runtime_checkable
class BackgroundRunner(Protocol):
    def start(self, task: BackgroundTask, payload: str, *, max_runtime_s: float) -> str:
        """Start ``payload`` (code or a command, per ``task.kind``) without
        waiting for it; return a handle for the other calls. The runner should
        stop the work itself after ``max_runtime_s``."""
        ...

    def poll(self, handle: str) -> RunnerState: ...

    def read_output(self, handle: str, *, max_chars: int) -> str:
        """Return up to the last ``max_chars`` characters of output."""
        ...

    def stop(self, handle: str) -> None: ...


@runtime_checkable
class OutputFileRunner(Protocol):
    """Optional: a runner whose task output is a file the agent can open."""

    def output_path(self, handle: str) -> str: ...


# --------------------------------------------------------------------------- #
# Sandbox runner
# --------------------------------------------------------------------------- #

# Runs one operation of ``_sandbox_ops`` in a private namespace, so nothing
# leaks into a sandbox whose globals persist between calls (a Jupyter kernel).
_CALL = """
def __terno_bg_call(source, operation, args_json):
    namespace = {"__name__": "terno_bg_ops", "__terno_source__": source}
    exec(compile(source, "_sandbox_ops.py", "exec"), namespace)
    namespace["dispatch"](operation, args_json)
try:
    __terno_bg_call(%(source)r, %(operation)r, %(args)r)
finally:
    del __terno_bg_call
"""


def build_call(operation: str, **args: object) -> str:
    """The code that runs ``operation`` of ``_sandbox_ops`` in the sandbox."""
    if operation not in _sandbox_ops.OPERATIONS:
        raise ValueError(f"Unknown sandbox operation: {operation!r}")
    return _CALL % {"source": _ops_source(), "operation": operation, "args": json.dumps(args)}


@cache
def _ops_source() -> str:
    try:
        return inspect.getsource(_sandbox_ops)
    except OSError as exc:
        raise BackgroundRunnerError(
            "The background task helper source is not available in this installation."
        ) from exc


class SandboxBackgroundRunner:
    """Runs background work as a detached process inside a sandbox.

    The work runs in the same environment as ``run_python``. Each task keeps
    its files in a directory under ``root_dir`` (default: the sandbox's temp
    dir); see :mod:`terno_agent.background._sandbox_ops` for the layout.
    Output beyond ``max_log_bytes`` is trimmed to its most recent part.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        root_dir: str = "",
        cwd: str = "",
        call_timeout_s: int = 30,
        max_log_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        if max_log_bytes <= 0:
            raise ValueError("max_log_bytes must be positive.")
        self.sandbox = sandbox
        self.root_dir = root_dir
        if not cwd and isinstance(sandbox, LocalSandbox):
            cwd = os.getcwd()
        self.cwd = cwd
        self.call_timeout_s = call_timeout_s
        self.max_log_bytes = max_log_bytes

    def start(self, task: BackgroundTask, payload: str, *, max_runtime_s: float) -> str:
        if task.kind not in ("python", "shell"):
            raise BackgroundRunnerError(f"Unsupported background task kind: {task.kind!r}")
        # Named here, so work whose start reply is lost can still be stopped.
        name = f"{_safe_name(task.id)}-{uuid.uuid4().hex[:12]}"
        try:
            reply = self._call(
                "start",
                root=self.root_dir,
                name=name,
                kind=task.kind,
                payload=payload,
                cwd=self.cwd,
                max_log_bytes=self.max_log_bytes,
                max_runtime_s=max_runtime_s,
            )
            handle = reply.get("handle")
            if not handle:
                raise BackgroundRunnerError("The sandbox did not return a task handle.")
            return str(handle)
        except BackgroundRunnerError:
            try:
                self._call("stop", root=self.root_dir, name=name)
            except BackgroundRunnerError:
                pass
            raise

    def poll(self, handle: str) -> RunnerState:
        reply = self._call("poll", handle=handle)
        exit_code = reply.get("exit_code")
        return RunnerState(
            running=bool(reply.get("running")),
            exit_code=int(exit_code) if exit_code is not None else None,
            timed_out=bool(reply.get("timed_out")),
            error=str(reply.get("error") or ""),
            ended_at=float(reply["ended_at"]) if reply.get("ended_at") is not None else None,
        )

    def read_output(self, handle: str, *, max_chars: int) -> str:
        reply = self._call("read_output", handle=handle, max_chars=max_chars)
        text = str(reply.get("output") or "")
        if reply.get("truncated"):
            text = f"[earlier output omitted]\n{text}"
        return text

    def stop(self, handle: str) -> None:
        self._call("stop", handle=handle)

    def output_path(self, handle: str) -> str:
        # The handle is a path inside the sandbox, in the sandbox's own style.
        separator = "\\" if "\\" in handle and "/" not in handle else "/"
        return handle.rstrip(separator) + separator + _sandbox_ops.LOG_FILE

    def _call(self, operation: str, **args: object) -> dict:
        """Run one operation in the sandbox. Every failure, whatever the
        sandbox raises, comes out as :class:`BackgroundRunnerError`."""
        try:
            result = self.sandbox.run_python(
                build_call(operation, **args), timeout_s=self.call_timeout_s
            )
        except Exception as exc:
            raise BackgroundRunnerError(f"Sandbox {operation} failed: {exc}") from exc
        marker = _sandbox_ops.MARKER
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(marker):
                try:
                    return json.loads(line[len(marker):])
                except ValueError as exc:
                    raise BackgroundRunnerError(f"Sandbox {operation} sent a bad reply") from exc
        detail = (result.stderr or result.stdout or "no output").strip()[-500:]
        raise BackgroundRunnerError(f"Sandbox {operation} failed: {detail}")


def _safe_name(task_id: str) -> str:
    cleaned = "".join(c for c in str(task_id) if c.isalnum() or c in "-_")
    if not cleaned:
        raise BackgroundRunnerError(f"Unusable task id: {task_id!r}")
    return cleaned


__all__ = [
    "BackgroundRunner",
    "BackgroundRunnerError",
    "OutputFileRunner",
    "RunnerState",
    "SandboxBackgroundRunner",
]
