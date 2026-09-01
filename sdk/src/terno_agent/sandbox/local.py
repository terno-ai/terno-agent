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
_IS_WINDOWS = sys.platform == "win32"


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
            return self._exec(
                [self.python, script],
                workdir=workdir,
                env=env,
                timeout_s=timeout_s,
                cancel_token=cancel_token,
                what="run_python",
                launch_error="failed to launch python",
            )

    def run_shell(
        self,
        command: str,
        *,
        timeout_s: int = 30,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        cancel_token: CancelToken | None = None,
    ) -> ExecutionResult:
        # Unlike run_python (isolated temp dir), shell commands run in the
        # caller's directory on the host so they can see the real project.
        # Windows: shell=True with the raw string (not ["cmd", "/c", command]) -
        # list2cmdline's generic argv-quoting doesn't match cmd.exe's own quote
        # parsing, so nested double quotes in `command` come out mangled.
        argv: list[str] | str = command if _IS_WINDOWS else ["sh", "-c", command]
        return self._exec(
            argv,
            workdir=cwd or os.getcwd(),
            env=env,
            timeout_s=timeout_s,
            cancel_token=cancel_token,
            what="run_shell",
            launch_error="failed to launch shell",
            shell=_IS_WINDOWS,
        )

    def _exec(
        self,
        argv: list[str] | str,
        *,
        workdir: str,
        env: dict[str, str] | None,
        timeout_s: int,
        cancel_token: CancelToken | None,
        what: str,
        launch_error: str,
        shell: bool = False,
    ) -> ExecutionResult:
        child_env = {**os.environ, **(env or {})}

        if cancel_token is not None and cancel_token.is_cancelled:
            raise AgentCancelled(f"cancelled before {what} started")

        try:
            if _IS_WINDOWS:
                proc = subprocess.Popen(
                    argv,
                    cwd=workdir,
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=shell,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                proc = subprocess.Popen(
                    argv,
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
                stderr=f"{launch_error}: {exc}",
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
            raise AgentCancelled(f"{what} cancelled by user")

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
    if _IS_WINDOWS:
        _terminate_group_windows(proc)
        return
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


def _terminate_group_windows(proc: subprocess.Popen) -> None:
    # CTRL_BREAK_EVENT reaches the whole process group we created via
    # CREATE_NEW_PROCESS_GROUP, giving well-behaved children a chance to exit
    # cleanly before the hard taskkill /T fallback below.
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    except (OSError, ValueError):
        pass
    try:
        proc.wait(timeout=_TERM_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        proc.wait(timeout=_TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        pass


def _drain(proc: subprocess.Popen) -> tuple[str, str]:
    try:
        return proc.communicate(timeout=_TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        return ("", "")
