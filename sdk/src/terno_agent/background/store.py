"""Background task records and the store contract.

A :class:`BackgroundTask` is work that keeps running after the tool call that
started it has returned. :class:`BackgroundTaskStore` owns the rules every
implementation must share (validation, status transitions, change
notification); subclasses only persist. :class:`InMemoryBackgroundTaskStore`
is the standalone default. A host that runs the agent across processes (e.g.
a web app) subclasses the store with a database-backed one.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any

from terno_agent.core.exceptions import ToolError

RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
STOPPED = "stopped"
TIMED_OUT = "timed_out"

STATUSES = (RUNNING, COMPLETED, FAILED, STOPPED, TIMED_OUT)
FINISHED_STATUSES = frozenset(STATUSES) - {RUNNING}

_UPDATABLE = frozenset(
    {"handle", "status", "exit_code", "error", "finished_at", "notified", "metadata"}
)

BackgroundTaskCallback = Callable[["BackgroundTask"], None]


@dataclass
class BackgroundTask:
    id: str
    description: str
    # What to run ("python", "shell", or a host-defined kind) and which
    # registered runner runs it.
    kind: str
    runner: str
    status: str = RUNNING
    # Runner-specific reference to the running work (a directory, a job id).
    handle: str = ""
    exit_code: int | None = None
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    # Whether the agent has been told this task finished.
    notified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.status in FINISHED_STATUSES

    def runtime_s(self, now: float | None = None) -> float:
        end = self.finished_at if self.finished_at is not None else (now or time.time())
        return max(0.0, end - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BackgroundTaskStore(ABC):
    """Shared contract for background task persistence.

    Register an observer with :meth:`set_on_change` to mirror tasks in a UI;
    it receives the changed task after every create/update.
    """

    def __init__(self) -> None:
        self._on_change: BackgroundTaskCallback | None = None

    def set_on_change(self, callback: BackgroundTaskCallback | None) -> None:
        self._on_change = callback

    def _notify(self, task: BackgroundTask) -> None:
        cb = self._on_change
        if cb is None:
            return
        try:
            cb(task)
        except Exception:
            # A UI failure must never break a task update.
            pass

    # ----- public API ---------------------------------------------------- #

    def create(
        self,
        description: str,
        *,
        kind: str,
        runner: str,
        metadata: dict[str, Any] | None = None,
    ) -> BackgroundTask:
        description = (description or "").strip()
        if not description:
            raise ToolError("A background task needs a short description.")
        task = self._create(description, kind=kind, runner=runner, metadata=metadata or {})
        self._notify(task)
        return task

    def update(self, task_id: str, **fields: Any) -> BackgroundTask:
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"Cannot update background task fields: {sorted(unknown)}")
        status = fields.get("status")
        if status is not None and status not in STATUSES:
            raise ValueError(f"Invalid background task status {status!r}.")
        if status in FINISHED_STATUSES and fields.get("finished_at") is None:
            fields["finished_at"] = time.time()
        task = self._apply_update(str(task_id), fields)
        self._notify(task)
        return task

    def finish(
        self,
        task_id: str,
        status: str,
        *,
        exit_code: int | None = None,
        error: str = "",
    ) -> BackgroundTask:
        """Mark a task finished unless it already is (first outcome wins)."""
        if status not in FINISHED_STATUSES:
            raise ValueError(f"{status!r} is not a finished status.")
        task = self.get(task_id)
        if task.finished:
            return task
        return self.update(task_id, status=status, exit_code=exit_code, error=error)

    def running(self) -> list[BackgroundTask]:
        return [t for t in self.list() if t.status == RUNNING]

    def claim_unnotified(self) -> list[BackgroundTask]:
        """Return finished tasks the agent has not been told about, and mark
        them told. A store shared between processes should override this with
        an atomic claim so a finish is reported exactly once."""
        claimed = []
        for task in self.list():
            if task.finished and not task.notified:
                claimed.append(self.update(task.id, notified=True))
        return claimed

    # ----- persistence primitives --------------------------------------- #

    @abstractmethod
    def _create(
        self, description: str, *, kind: str, runner: str, metadata: dict[str, Any]
    ) -> BackgroundTask:
        """Persist a new running task and return it."""

    @abstractmethod
    def _apply_update(self, task_id: str, fields: dict[str, Any]) -> BackgroundTask:
        """Apply ``fields`` to the stored task and return it.

        When ``fields`` finishes the task (a finished ``status``), apply them
        only if the task is still running, atomically: the first outcome wins
        even when several threads or processes report one at once.
        """

    @abstractmethod
    def get(self, task_id: str) -> BackgroundTask:
        """Return one task, or raise :class:`ToolError` if unknown."""

    @abstractmethod
    def list(self) -> list[BackgroundTask]:
        """Return all tasks, oldest first."""


class InMemoryBackgroundTaskStore(BackgroundTaskStore):
    """Thread-safe, process-local store."""

    def __init__(self) -> None:
        super().__init__()
        self._tasks: dict[str, BackgroundTask] = {}
        self._next_id = 1
        self._lock = Lock()

    def _create(
        self, description: str, *, kind: str, runner: str, metadata: dict[str, Any]
    ) -> BackgroundTask:
        with self._lock:
            task = BackgroundTask(
                id=str(self._next_id),
                description=description,
                kind=kind,
                runner=runner,
                metadata=metadata,
            )
            self._next_id += 1
            self._tasks[task.id] = task
            return task

    def _apply_update(self, task_id: str, fields: dict[str, Any]) -> BackgroundTask:
        with self._lock:
            task = self.get(task_id)
            if fields.get("status") in FINISHED_STATUSES and task.finished:
                return task
            for key, value in fields.items():
                setattr(task, key, value)
            return task

    def get(self, task_id: str) -> BackgroundTask:
        try:
            return self._tasks[str(task_id)]
        except KeyError as exc:
            raise ToolError(f"Unknown background task id: {task_id}") from exc

    def list(self) -> list[BackgroundTask]:
        return list(self._tasks.values())
