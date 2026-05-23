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
