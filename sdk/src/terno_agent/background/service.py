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
from collections import defaultdict
from collections.abc import Callable, Mapping

from terno_agent.background.runner import (
    BackgroundRunner,
    BatchPollRunner,
    OutputFileRunner,
    RunnerState,
)
from terno_agent.background.store import (
    COMPLETED,
    FAILED,
    REPORTED_STATUSES,
    RUNNING,
    STOP_REQUESTED,
    STOPPED,
    TIMED_OUT,
    BackgroundTask,
    BackgroundTaskStore,
    InMemoryBackgroundTaskStore,
)
from terno_agent.core.exceptions import ToolError

logger = logging.getLogger(__name__)

SANDBOX_RUNNER = "sandbox"
# A task still without a handle this long after it was created never got
# started (whoever was starting it died). Far above any start call's timeout.
START_GRACE_S = 600
# A task out of reach this long past its time limit has been stopped by its
# own wrapper (which enforces the limit in the sandbox), so it is over.
DEADLINE_GRACE_S = 600
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
        can_start: bool = True,
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
        # False where tasks may be checked and stopped but not started, e.g.
        # a turn nobody asked for, or a watcher outside any turn.
        self.can_start = can_start

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
        if not self.can_start:
            raise ToolError("Background tasks cannot be started here.")
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
            handle = backend.prepare(task)
        except Exception as exc:
            # Nothing can run without a handle. The agent gets the error right
            # here: nothing more to tell it.
            self.store.update(task.id, status=FAILED, error=str(exc), notified=True)
            raise ToolError(f"Could not start the background task: {exc}") from exc
        task = self.store.update(task.id, handle=handle)
        try:
            backend.start(handle, task, payload, max_runtime_s=self.max_runtime_s)
        except Exception as exc:
            # The start may still happen (e.g. its reply was lost): stop it
            # like any other stop, retried until it is confirmed.
            task = self.store.update(task.id, error=f"could not start: {exc}")
            self._stop(task, FAILED)
            raise ToolError(
                f"Could not start the background task: {exc}. If any of it ran, it is being stopped."
            ) from exc
        return task

    def stop(self, task_id: str) -> BackgroundTask:
        """Stop a task. It is recorded as stopped only once the stop reached
        the work; otherwise the request is kept and :meth:`refresh` retries it."""
        task = self.store.get(task_id)
        if task.finished:
            return task
        return self._stop(task, STOPPED)

    def output(self, task_id: str, *, max_chars: int = 8000) -> str:
        self.refresh(force=True)
        task = self.store.get(task_id)
        try:
            text = self._runner(task).read_output(task.handle, max_chars=max_chars)
        except Exception as exc:
            text = f"[could not read output: {exc}]"
        return f"{self.describe(task)}\n--- output ---\n{text or '(no output yet)'}"

    def refresh(self, *, force: bool = False) -> None:
        """Bring running tasks up to date: record finished ones, retry stops
        that did not get through, and stop any past the time limit. Throttled
        unless ``force``."""
        now = self._clock()
        if not force and now - self._last_refresh < self.poll_interval_s:
            return
        self._last_refresh = now
        to_check = []
        for task in self.store.running():
            if not task.handle:
                if task.runtime_s(now) > START_GRACE_S:
                    self._record(task, _requested(task) or FAILED, error="it did not finish starting")
                continue
            requested = _requested(task)
            if requested and self._stop(task, requested).finished:
                continue
            to_check.append(task)
        states = self._poll(to_check)
        for task in to_check:
            state = states.get(task.id)
            if state is None or state.running:
                runtime = task.runtime_s(now)
                if state is None and runtime > self.max_runtime_s + DEADLINE_GRACE_S:
                    self._record(task, _requested(task) or TIMED_OUT)
                elif runtime > self.max_runtime_s:
                    self._stop(task, TIMED_OUT)
                continue
            self._record_end(task, state, now)

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
            if task.stop_requested:
                return f"{head}: stopping (running for {ran}); the stop is retried until it gets through"
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

    def _poll(self, tasks: list[BackgroundTask]) -> dict[str, RunnerState]:
        """Check tasks, one call per runner when it can check several at once."""
        by_runner = defaultdict(list)
        for task in tasks:
            by_runner[task.runner].append(task)
        states = {}
        for group in by_runner.values():
            try:
                runner = self._runner(group[0])
                if isinstance(runner, BatchPollRunner):
                    found = runner.poll_many([t.handle for t in group])
                    if not isinstance(found, dict):
                        raise TypeError(f"poll_many returned {type(found).__name__}")
                    states.update({t.id: found[t.handle] for t in group if t.handle in found})
                    continue
            except Exception as exc:
                logger.warning("Could not check background tasks %s: %s", [t.id for t in group], exc)
                continue
            for task in group:
                try:
                    states[task.id] = runner.poll(task.handle)
                except Exception as exc:
                    logger.warning("Could not check background task %s: %s", task.id, exc)
        return states

    def _stop(self, task: BackgroundTask, status: str) -> BackgroundTask:
        """Stop the work and record it ended with ``status`` (STOPPED,
        TIMED_OUT, or FAILED for a start that failed). The request is saved
        first, so a check running meanwhile records the same outcome. Until a
        check confirms the work is not running (no handle yet, the sandbox
        busy, a start still arriving) it stays requested and the next refresh
        retries it."""
        requested = _requested(task)
        if requested:
            status = requested
        else:
            task = self.store.update(task.id, metadata={**task.metadata, STOP_REQUESTED: status})
        if not task.handle:
            return task
        runner = self._runner(task)
        try:
            runner.stop(task.handle)
        except Exception as exc:
            logger.warning("Could not stop background task %s yet: %s", task.id, exc)
            return task
        try:
            state = runner.poll(task.handle)
        except Exception:
            state = None  # the stop itself got through
        if state is not None and state.running:
            return task  # e.g. it started just after the stop: try again
        if state is not None and _ended_by_itself(state):
            self._record_end(task, state, self._clock())
            return self.store.get(task.id)
        return self._record(task, status)

    def _record_end(self, task: BackgroundTask, state, now: float) -> None:
        task = self.store.get(task.id)
        ended = _clamp(state.ended_at, task.started_at, now) if state.ended_at is not None else None
        if state.stopped:
            self._record(task, _requested(task) or STOPPED, finished_at=ended)
        elif state.timed_out:
            self._record(task, TIMED_OUT, finished_at=ended, natural=True)
        elif state.exit_code is None:
            self._record(
                task, FAILED, error=state.error or "ended unexpectedly", finished_at=ended, natural=True
            )
        else:
            status = COMPLETED if state.exit_code == 0 else FAILED
            self._record(task, status, exit_code=state.exit_code, finished_at=ended, natural=True)

    def _record(
        self,
        task: BackgroundTask,
        status: str,
        *,
        error: str = "",
        exit_code: int | None = None,
        finished_at: float | None = None,
        natural: bool = False,
    ) -> BackgroundTask:
        """Record how the task ended; the first outcome recorded wins.

        ``natural`` means the work ended by itself rather than by a stop.
        Nothing is told about a task someone asked to stop, however it ended
        (it may have finished just before the stop): its result is no longer
        wanted, though it can still be read. Nor about a start that failed,
        which the agent heard about at once, unless the work did run after
        all. A time-limit stop is told.
        """
        requested = _requested(task)
        told = (
            status not in REPORTED_STATUSES
            or requested == STOPPED
            or (requested == FAILED and not natural)
        )
        fields = {"status": status, "finished_at": finished_at, "notified": told}
        if exit_code is not None:
            fields["exit_code"] = exit_code
        if status == TIMED_OUT and not error:
            error = f"stopped after the {_duration(self.max_runtime_s)} limit"
        if error:
            fields["error"] = error  # otherwise keep any recorded already
        return self.store.update(task.id, **fields)


def _ended_by_itself(state) -> bool:
    """The work ended with an outcome of its own, not because it was stopped."""
    return not state.running and not state.stopped and (
        state.exit_code is not None or state.timed_out
    )


def _requested(task: BackgroundTask) -> str | None:
    """The outcome a pending stop request will record, if there is one."""
    value = task.metadata.get(STOP_REQUESTED)
    if not value:
        return None
    return value if value in (STOPPED, TIMED_OUT, FAILED) else STOPPED  # once a plain True


def _clamp(value: float, low: float, high: float) -> float:
    """``value`` kept within [low, high] (another machine's clock can be off)."""
    return min(max(value, low), high)


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
