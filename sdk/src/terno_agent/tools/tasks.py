"""Task-tracking tools and the canonical task-store contract.

This module is the single source of truth for the agent's task/todo model.
It defines:

* :class:`Task` — the neutral task record shared everywhere.
* :class:`TaskStore` — an abstract base that owns the lifecycle/status
  validation and the change-notification plumbing. Concrete stores only
  implement four persistence primitives (:meth:`~TaskStore._create`,
  :meth:`~TaskStore._apply_update`, :meth:`~TaskStore.get`,
  :meth:`~TaskStore.list`), so integrations never re-implement the rules.
* :class:`InMemoryTaskStore` — the default, process-local implementation
  used when the SDK runs standalone (CLI, benchmarks, no backend).
* The four LLM-facing tools: ``task_create``, ``task_list``, ``task_get``,
  ``task_update``.

Backends that need persistence (e.g. terno-ai's database-backed store)
subclass :class:`TaskStore`, reuse the exact same contract, and register an
``on_change`` callback to stream the current list to a UI. Tasks follow a
``pending → in_progress → completed`` lifecycle, with ``deleted`` for
removal.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema

_STATUSES = ("pending", "in_progress", "completed", "deleted")

# Fired with the full current task list after every successful mutation.
TaskListCallback = Callable[[list["Task"]], None]


@dataclass
class Task:
    id: str
    subject: str
    description: str = ""
    active_form: str = ""
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskStore(ABC):
    """Canonical task-store contract shared by terno-agent and terno-ai.

    Subclasses implement only the persistence primitives; this base owns the
    parts that must not diverge between implementations:

    * ``subject`` normalization + non-empty validation on create,
    * status validation on update,
    * the ``on_change`` notification that lets a UI mirror the list live.

    Register an observer with :meth:`set_on_change`; it receives the full
    current list (via :meth:`list`) after each create/update.
    """

    def __init__(self) -> None:
        self._on_change: TaskListCallback | None = None

    # ----- change notification (shared logic) --------------------------- #

    def set_on_change(self, callback: TaskListCallback | None) -> None:
        """Register (or clear) the observer notified after every mutation."""
        self._on_change = callback

    def _notify(self) -> None:
        cb = self._on_change
        if cb is None:
            return
        try:
            cb(self.list())
        except Exception:
            # An observer/UI failure must never break a task mutation.
            pass

    @staticmethod
    def _validate_status(status: str) -> None:
        if status not in _STATUSES:
            raise ToolError(
                f"Invalid status {status!r}. Must be one of: "
                f"{', '.join(_STATUSES)}."
            )

    # ----- public API (shared; validates + notifies) ------------------- #

    def create(
        self,
        subject: str,
        *,
        description: str = "",
        active_form: str = "",
    ) -> Task:
        subject = (subject or "").strip()
        if not subject:
            raise ToolError("task subject must be non-empty.")
        task = self._create(
            subject,
            description=(description or "").strip(),
            active_form=(active_form or "").strip(),
        )
        self._notify()
        return task

    def update(self, task_id: str, **fields: Any) -> Task:
        status = fields.get("status")
        if status is not None:
            self._validate_status(status)
        task = self._apply_update(str(task_id), fields)
        self._notify()
        return task

    # ----- persistence primitives (subclass implements) ---------------- #

    @abstractmethod
    def _create(
        self, subject: str, *, description: str, active_form: str
    ) -> Task:
        """Persist a new task (subject already normalized) and return it."""

    @abstractmethod
    def _apply_update(self, task_id: str, fields: dict[str, Any]) -> Task:
        """Apply non-None ``subject``/``description``/``active_form``/``status``
        from ``fields`` to the stored task and return it."""

    @abstractmethod
    def get(self, task_id: str) -> Task:
        """Return a single task by id, or raise :class:`ToolError`."""

    @abstractmethod
    def list(self) -> list[Task]:
        """Return all non-deleted tasks in creation order."""


class InMemoryTaskStore(TaskStore):
    """Thread-safe, process-local task list — the SDK's standalone default."""

    def __init__(self) -> None:
        super().__init__()
        self._tasks: dict[str, Task] = {}
        self._next_id = 1
        self._lock = Lock()

    def _create(
        self, subject: str, *, description: str, active_form: str
    ) -> Task:
        with self._lock:
            task_id = str(self._next_id)
            self._next_id += 1
            task = Task(
                id=task_id,
                subject=subject,
                description=description,
                active_form=active_form,
            )
            self._tasks[task_id] = task
            return task

    def get(self, task_id: str) -> Task:
        try:
            return self._tasks[str(task_id)]
        except KeyError as exc:
            raise ToolError(f"Unknown task id: {task_id}") from exc

    def list(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.status != "deleted"]

    def _apply_update(self, task_id: str, fields: dict[str, Any]) -> Task:
        with self._lock:
            task = self.get(task_id)
            if fields.get("status") is not None:
                task.status = fields["status"]
            for key in ("subject", "description", "active_form"):
                value = fields.get(key)
                if value is not None:
                    setattr(task, key, value)
            return task


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


@dataclass
class TaskCreateTool:
    store: TaskStore

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="task_create",
            description=(
                "Create a new task in the agent's task list. Use this to plan "
                "non-trivial work (3+ steps, multi-file changes). Returns the "
                "created task as JSON."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Imperative title, e.g. 'Add login endpoint'.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional longer-form description.",
                    },
                    "active_form": {
                        "type": "string",
                        "description": (
                            "Optional present-continuous form, e.g. 'Adding login endpoint'."
                        ),
                    },
                },
                "required": ["subject"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        subject = (kwargs.get("subject") or "").strip()
        if not subject:
            raise ToolError("task_create requires a non-empty 'subject'.")
        task = self.store.create(
            subject,
            description=(kwargs.get("description") or "").strip(),
            active_form=(kwargs.get("active_form") or "").strip(),
        )
        return json.dumps(task.to_dict())


@dataclass
class TaskListTool:
    store: TaskStore

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="task_list",
            description=(
                "List all tracked tasks (excluding deleted ones) with their "
                "status. Returns a JSON array."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def run(self, **_kwargs: Any) -> str:
        return json.dumps([t.to_dict() for t in self.store.list()])


@dataclass
class TaskGetTool:
    store: TaskStore

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="task_get",
            description="Fetch a single task by id. Returns the task as JSON.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task's id (as returned by task_create).",
                    }
                },
                "required": ["task_id"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        task_id = kwargs.get("task_id")
        if not task_id:
            raise ToolError("task_get requires a 'task_id'.")
        return json.dumps(self.store.get(str(task_id)).to_dict())


@dataclass
class TaskUpdateTool:
    store: TaskStore

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="task_update",
            description=(
                "Update a task's status or fields. Use this to mark a task "
                "in_progress when you start it, completed when finished, or "
                "deleted to remove it. Returns the updated task as JSON."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "The task id to update."},
                    "status": {
                        "type": "string",
                        "enum": list(_STATUSES),
                        "description": "New status.",
                    },
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                    "active_form": {"type": "string"},
                },
                "required": ["task_id"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        task_id = kwargs.get("task_id")
        if not task_id:
            raise ToolError("task_update requires a 'task_id'.")
        task = self.store.update(
            str(task_id),
            status=kwargs.get("status"),
            subject=kwargs.get("subject"),
            description=kwargs.get("description"),
            active_form=kwargs.get("active_form"),
        )
        return json.dumps(task.to_dict())


__all__ = [
    "InMemoryTaskStore",
    "Task",
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListCallback",
    "TaskListTool",
    "TaskStore",
    "TaskUpdateTool",
]
