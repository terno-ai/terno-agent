"""Local subprocess sandbox.

Not a real security boundary — useful only for trusted local development.
Runs the snippet in a fresh subprocess with a separate working directory,
a wall-clock timeout, and optional cooperative cancellation. Network and
filesystem access are NOT restricted.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time

from terno_agent.core.cancel import CancelToken
from terno_agent.core.exceptions import AgentCancelled
from terno_agent.sandbox.base import ExecutionResult

_POLL_INTERVAL_S = 0.1
_TERM_GRACE_S = 0.5


class LocalSandbox:
    def __init__(self, *, python: str | None = None, **_unused) -> None:
        # Accepts (and silently ignores) `persist` / `container_name` so it
        # can stand in for the Docker backend in fallback paths without the
        # caller having to special-case option dicts.
        self.python = python or sys.executable

    def close(self) -> None:
        """No-op — LocalSandbox spawns fresh subprocesses per call.

        Provided for protocol symmetry with `DockerSandbox.close()` so the
        agent's shutdown path can call it unconditionally.
        """
        return None

    def run_python(
        self,
        code: str,
        *,
        timeout_s: int = 30,
        env: dict[str, str] | None = None,
        cancel_token: CancelToken | None = None,
    ) -> ExecutionResult:
        with tempfile.TemporaryDirectory(prefix="terno_local_") as workdir:
            script = os.path.join(workdir, "snippet.py")
            with open(script, "w", encoding="utf-8") as f:
                f.write(code)
            child_env = {**os.environ, **(env or {})}

            if cancel_token is not None and cancel_token.is_cancelled:
                raise AgentCancelled("cancelled before run_python started")

            try:
                proc = subprocess.Popen(
                    [self.python, script],
                    cwd=workdir,
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
            except OSError as exc:
                return ExecutionResult(
                    stdout="",
                    stderr=f"failed to launch python: {exc}",
                    exit_code=127,
                )

            deadline = time.monotonic() + timeout_s
            timed_out = False
            cancelled = False
            stdout, stderr = "", ""
            try:
                while True:
                    try:
                        stdout, stderr = proc.communicate(timeout=_POLL_INTERVAL_S)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if cancel_token is not None and cancel_token.is_cancelled:
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
                raise AgentCancelled("run_python cancelled by user")

            if timed_out:
                return ExecutionResult(
                    stdout=stdout or "",
                    stderr=stderr or "",
                    exit_code=124,
                    timed_out=True,
                )
            return ExecutionResult(
                stdout=stdout or "",
                stderr=stderr or "",
                exit_code=proc.returncode,
            )


def _terminate_group(proc: subprocess.Popen) -> None:
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
    try:
        return proc.communicate(timeout=_TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        return ("", "")
