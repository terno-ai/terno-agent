"""Sandbox protocol.

Implementations run untrusted code in some isolated environment and return
captured stdout/stderr plus an exit code. Implementations should enforce a
wall-clock timeout and an output cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def render(self, max_chars: int = 8000) -> str:
        parts = [f"exit_code={self.exit_code}"]
        if self.timed_out:
            parts.append("timed_out=True")
        if self.stdout:
            parts.append("--- stdout ---\n" + _truncate(self.stdout, max_chars))
        if self.stderr:
            parts.append("--- stderr ---\n" + _truncate(self.stderr, max_chars))
        return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


@runtime_checkable
class Sandbox(Protocol):
    """Runs a snippet of code and returns the result."""

    def run_python(
        self,
        code: str,
        *,
        timeout_s: int = 30,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult: ...

    def run_shell(
        self,
        command: str,
        *,
        timeout_s: int = 30,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult: ...


@runtime_checkable
class FileOpSandbox(Protocol):
    """Optional extension a :class:`Sandbox` may also implement to perform
    file operations directly, instead of the file tools round-tripping a
    generated snippet through ``run_python`` and parsing its stdout.

    The default ``run_python`` path is only as reliable as the sandbox's
    stdout capture: a backend whose code runs but whose printed output is
    lost (e.g. a Jupyter kernel that dropped an IOPub stream message) turns
    a *successful* write/edit into an "unparsable output" error. A sandbox
    with a dedicated request/reply file channel avoids that class of failure
    entirely by exposing these methods; the ``read_file``/``write_file``/
    ``edit_file`` tools detect them (duck-typed) and prefer them.

    Each returns a result dict: on success ``{"ok": True, "output": <str>}``
    where ``output`` is the exact string the tool should return to the model
    (``read_file`` therefore returns already-formatted, line-numbered text);
    on failure ``{"ok": False, "error": <message>}``, which the tool raises
    as a ``ToolError``. Implementations should treat these as an all-or-
    nothing extension — a sandbox either provides all three or none.
    """

    def read_file(
        self, path: str, *, offset: int = 1, limit: int = 2000
    ) -> dict: ...

    def write_file(
        self, path: str, content: str, *, overwrite: bool = False
    ) -> dict: ...

    def edit_file(
        self, path: str, old: str, new: str, *, replace_all: bool = False
    ) -> dict: ...
