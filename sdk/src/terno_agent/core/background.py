"""Background tasks: long-running work that outlives a single tool call.

The reference harness's shape, read off the `TaskStop`/`TaskOutput`/`Bash`
descriptions: a background task gets an id, its combined stdout/stderr streams to
an **output file**, and the launching tool returns that path. The model is then
expected to `Read` the file rather than poll a dedicated tool — `TaskOutput`'s own
description is marked `DEPRECATED` and says exactly that.

This registry covers background shell commands. It is deliberately generic about
`kind` so async agents could register here later, but Terno's subagents are still
synchronous, so `local_agent` tasks don't exist yet.

Output files live under `<workdir>/.terno/tasks/` and are NOT cleaned up: a task's
output has to outlive the process that produced it, or `Read` after completion
would find nothing.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

TASKS_DIRNAME = ".terno/tasks"

# How long to wait for a terminated process group to die before SIGKILL.
_KILL_GRACE_S = 2.0
_POLL_INTERVAL_S = 0.05


@dataclass
class BackgroundTask:
    id: str
    kind: str
    description: str
    command: str
    output_path: Path
    _proc: subprocess.Popen[bytes] | None = None
    _stop_requested: bool = False

    @property
    def status(self) -> str:
        if self._proc is None:
            return "unknown"
        code = self._proc.poll()
        if code is None:
            return "running"
        if self._stop_requested:
            return "stopped"
        return "completed" if code == 0 else "failed"

    @property
    def exit_code(self) -> int | None:
        return None if self._proc is None else self._proc.poll()

    def read_output(self) -> str:
        try:
            return self.output_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def wait(self, timeout_s: float) -> bool:
        """Wait up to `timeout_s` for completion. True if the task finished."""
        if self._proc is None:
            return True
        deadline = time.monotonic() + max(0.0, timeout_s)
        while self._proc.poll() is None:
            if time.monotonic() >= deadline:
                return False
            time.sleep(_POLL_INTERVAL_S)
        return True

    def stop(self) -> bool:
        """Terminate the task's process group. False if it was already done."""
        if self._proc is None or self._proc.poll() is not None:
            return False
        self._stop_requested = True
        # The task was launched in its own session, so signal the whole group —
        # a shell pipeline leaves orphans otherwise.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(self._proc.pid), sig)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    self._proc.kill()
                except OSError:
                    pass
            if self.wait(_KILL_GRACE_S if sig is signal.SIGTERM else 0.5):
                break
        return True

    def summary(self) -> str:
        parts = [f"{self.id} [{self.status}] {self.kind}"]
        if self.description:
            parts.append(f"— {self.description}")
        return " ".join(parts)


def _describe(command: str) -> str:
    """A short label for a command, when the caller gave none."""
    try:
        parts = shlex.split(command)
    except ValueError:  # unbalanced quotes — fall back to the raw text
        parts = command.split()
    return parts[0] if parts else ""


class BackgroundTaskRegistry:
    """Process-local registry of background tasks, keyed by a short id."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = Path(workdir)
        self._tasks: dict[str, BackgroundTask] = {}
        self._next = 1
        self._lock = Lock()

    @property
    def tasks_dir(self) -> Path:
        return self.workdir / TASKS_DIRNAME

    def _new_id(self, kind: str) -> str:
        with self._lock:
            n = self._next
            self._next += 1
        return f"{kind}_{n}"

    def launch_shell(self, command: str, *, description: str = "") -> BackgroundTask:
        """Start `command` detached, streaming combined output to a file."""
        task_id = self._new_id("bash")
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.tasks_dir / f"{task_id}.output"

        # Opened unbuffered and handed to the child, so the file grows as the
        # task runs and a Read mid-flight sees partial output.
        handle = output_path.open("wb", buffering=0)
        try:
            proc = subprocess.Popen(
                ["sh", "-c", command],
                cwd=str(self.workdir),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            handle.close()
            raise RuntimeError(f"Failed to launch background shell: {exc}") from exc
        finally:
            # The child holds its own descriptor; ours would otherwise keep the
            # file open for the life of the agent.
            handle.close()

        task = BackgroundTask(
            id=task_id,
            kind="bash",
            description=description or _describe(command),
            command=command,
            output_path=output_path,
            _proc=proc,
        )
        with self._lock:
            self._tasks[task_id] = task
        return task

    def get(self, task_id: str) -> BackgroundTask | None:
        return self._tasks.get(task_id)

    def list(self) -> list[BackgroundTask]:
        return list(self._tasks.values())

    def running(self) -> list[BackgroundTask]:
        return [t for t in self._tasks.values() if t.status == "running"]

    def stop_all(self) -> int:
        """Stop every running task. Called on agent shutdown."""
        return sum(1 for t in self._tasks.values() if t.stop())


__all__ = ["TASKS_DIRNAME", "BackgroundTask", "BackgroundTaskRegistry"]
