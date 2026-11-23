"""The background task service the agent and its tools talk to.

:class:`BackgroundTasks` ties a :class:`BackgroundTaskStore` (the record of
every task) to the registered :class:`BackgroundRunner`\\ s (what runs them).
It starts, checks, reads and stops tasks, enforces the limits, and turns
finished tasks into notices for the agent.

The store is the source of truth. A host whose runner reports completion on
its own (a job queue, a container callback) just updates the store; polling
here is then only a safety net.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping

from terno_agent.background.runner import BackgroundRunner, OutputFileRunner
from terno_agent.background.store import (
    COMPLETED,
    FAILED,
    RUNNING,
    STOPPED,
    TIMED_OUT,
    BackgroundTask,
    BackgroundTaskStore,
    InMemoryBackgroundTaskStore,
)
from terno_agent.core.exceptions import ToolError

logger = logging.getLogger(__name__)

SANDBOX_RUNNER = "sandbox"
# How much of the end of a finished task's output its notice carries.
NOTICE_TAIL_CHARS = 1500


class BackgroundTasks:
    def __init__(
        self,
        runners: Mapping[str, BackgroundRunner],
        *,
        store: BackgroundTaskStore | None = None,
        max_running: int = 3,
        max_runtime_s: int = 3 * 3600,
        poll_interval_s: float = 15.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not runners:
            raise ValueError("BackgroundTasks needs at least one runner.")
        if max_running <= 0 or max_runtime_s <= 0:
            raise ValueError("Background task limits must be positive.")
        self.runners = dict(runners)
        self.store = store or InMemoryBackgroundTaskStore()
        self.max_running = max_running
        self.max_runtime_s = max_runtime_s
        self.poll_interval_s = poll_interval_s
        self._clock = clock
        self._last_refresh = float("-inf")

    # ----- lifecycle ---------------------------------------------------- #

    def start(
        self,
        *,
        kind: str,
        payload: str,
        description: str,
        runner: str = SANDBOX_RUNNER,
        metadata: dict | None = None,
    ) -> BackgroundTask:
        backend = self.runners.get(runner)
        if backend is None:
            raise ToolError(f"No background runner named {runner!r}.")
        self.refresh(force=True)
        running = self.store.running()
        if len(running) >= self.max_running:
            names = ", ".join(f"{t.id} ({t.description})" for t in running)
            raise ToolError(
                f"Already running {len(running)} background tasks, the limit: {names}. "
                "Wait for one to finish or stop one first."
            )
        task = self.store.create(description, kind=kind, runner=runner, metadata=metadata)
        try:
            handle = backend.start(task, payload, max_runtime_s=self.max_runtime_s)
        except Exception as exc:
            self.store.finish(task.id, FAILED, error=str(exc))
            self.store.update(task.id, notified=True)
            raise ToolError(f"Could not start the background task: {exc}") from exc
        return self.store.update(task.id, handle=handle)

    def stop(self, task_id: str) -> BackgroundTask:
        task = self.store.get(task_id)
        if task.finished:
            return task
        self._stop_quietly(task)
        return self.store.update(task.id, status=STOPPED, notified=True)

    def output(self, task_id: str, *, max_chars: int = 8000) -> str:
        self.refresh(force=True)
        task = self.store.get(task_id)
        try:
            text = self._runner(task).read_output(task.handle, max_chars=max_chars)
        except Exception as exc:
            text = f"[could not read output: {exc}]"
        return f"{self.describe(task)}\n--- output ---\n{text or '(no output yet)'}"

    def refresh(self, *, force: bool = False) -> None:
        """Bring running tasks up to date: record finished ones and stop any
        that ran past the time limit. Throttled unless ``force``."""
        now = self._clock()
        if not force and now - self._last_refresh < self.poll_interval_s:
            return
        self._last_refresh = now
        limit = f"stopped after the {_duration(self.max_runtime_s)} limit"
        for task in self.store.running():
            state = None
            if task.handle:
                try:
                    state = self._runner(task).poll(task.handle)
                except Exception as exc:
                    # Treat as transient: the sandbox may be busy or restarting.
                    logger.warning("Could not check background task %s: %s", task.id, exc)
            if state is None or state.running:
                if task.runtime_s(now) > self.max_runtime_s:
                    self._stop_quietly(task)
                    self.store.finish(task.id, TIMED_OUT, error=limit)
                continue
            if state.timed_out:
                self.store.finish(task.id, TIMED_OUT, error=limit)
            elif state.exit_code is None:
                self.store.finish(task.id, FAILED, error=state.error or "ended unexpectedly")
            else:
                status = COMPLETED if state.exit_code == 0 else FAILED
                self.store.finish(task.id, status, exit_code=state.exit_code)

    # ----- what the agent sees ------------------------------------------ #

    def take_notices(self) -> list[str]:
        """One notice per task that finished since the agent was last told:
        how it ended and the end of its output, so the agent rarely needs to
        ask for more."""
        self.refresh()
        return [self._notice(t) for t in self.store.claim_unnotified()]

    def output_location(self, task: BackgroundTask) -> str | None:
        """Where the task's full output is, when it is a file the agent can
        open with its file tools."""
        backend = self.runners.get(task.runner)
        if not task.handle or not isinstance(backend, OutputFileRunner):
            return None
        try:
            return backend.output_path(task.handle)
        except Exception:
            return None

    def context_block(self) -> str:
        """The tasks still running, for the start of each turn."""
        self.refresh()
        running = self.store.running()
        if not running:
            return ""
        lines = [self.describe(t) for t in running]
        return (
            "Background tasks still running (you will be told when each "
            "finishes; do not start them again or wait for them):\n- "
            + "\n- ".join(lines)
        )

    def describe(self, task: BackgroundTask) -> str:
        ran = _duration(task.runtime_s(self._clock()))
        head = f'Background task {task.id} ("{task.description}")'
        if task.status == RUNNING:
            return f"{head}: running for {ran}"
        detail = {
            COMPLETED: "completed",
            FAILED: "failed",
            STOPPED: "was stopped",
            TIMED_OUT: "timed out",
        }[task.status]
        if task.exit_code is not None:
            detail += f" (exit code {task.exit_code})"
        if task.error:
            detail += f": {task.error}"
        return f"{head} {detail} after {ran}"

    # ----- internals ---------------------------------------------------- #

    def _notice(self, task: BackgroundTask) -> str:
        lines = [f"{self.describe(task)}."]
        tail = self._tail(task)
        if tail:
            lines.append(f"End of its output:\n{tail}")
        more = f'background_output(task_id="{task.id}")'
        location = self.output_location(task)
        if location:
            more += f" or open {location}"
        lines.append(f"Full output: {more}.")
        return "\n".join(lines)

    def _tail(self, task: BackgroundTask) -> str:
        if not task.handle:
            return ""
        try:
            text = self._runner(task).read_output(task.handle, max_chars=NOTICE_TAIL_CHARS)
        except Exception as exc:
            logger.warning("Could not read background task %s output: %s", task.id, exc)
            return ""
        return text.strip()

    def _runner(self, task: BackgroundTask) -> BackgroundRunner:
        backend = self.runners.get(task.runner)
        if backend is None:
            raise ToolError(f"No background runner named {task.runner!r}.")
        return backend

    def _stop_quietly(self, task: BackgroundTask) -> None:
        if not task.handle:
            return
        try:
            self._runner(task).stop(task.handle)
        except Exception as exc:
            logger.warning("Could not stop background task %s: %s", task.id, exc)


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s" if seconds else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


__all__ = ["SANDBOX_RUNNER", "BackgroundTasks"]
