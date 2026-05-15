"""Local subprocess sandbox.

Not a real security boundary — useful only for trusted local development.
Runs the snippet in a fresh subprocess with a separate working directory and
a timeout. Network and filesystem access are NOT restricted.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from terno_agent.sandbox.base import ExecutionResult


class LocalSandbox:
    def __init__(self, *, python: str | None = None) -> None:
        self.python = python or sys.executable

    def run_python(
        self,
        code: str,
        *,
        timeout_s: int = 30,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        with tempfile.TemporaryDirectory(prefix="terno_local_") as workdir:
            script = os.path.join(workdir, "snippet.py")
            with open(script, "w", encoding="utf-8") as f:
                f.write(code)
            child_env = {**os.environ, **(env or {})}
            try:
                proc = subprocess.run(
                    [self.python, script],
                    cwd=workdir,
                    env=child_env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                return ExecutionResult(
                    stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                    stderr=exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                    exit_code=124,
                    timed_out=True,
                )
            return ExecutionResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
            )
