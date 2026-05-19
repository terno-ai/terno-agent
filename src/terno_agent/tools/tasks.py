"""Task-tracking tools.

Provides a small in-memory `TaskStore` shared across a `TernoAgent`
(including its subagents) plus four tools the LLM can call:
``task_create``, ``task_list``, ``task_get``, ``task_update``.

Tasks follow a simple ``pending → in_progress → completed`` lifecycle,
with ``deleted`` available for removal. State is process-local; nothing
is persisted to disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema

_STATUSES = ("pending", "in_progress", "completed", "deleted")


@dataclass
class Task:
    id: str
    subject: str
    description: str = ""
    active_form: str = ""
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskStore:
    """Thread-safe in-memory task list."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._next_id = 1
        self._lock = Lock()

    def create(
        self,
        subject: str,
        *,
        description: str = "",
        active_form: str = "",
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
            return self._tasks[task_id]
        except KeyError as exc:
            raise ToolError(f"Unknown task id: {task_id}") from exc

    def list(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.status != "deleted"]

    def update(self, task_id: str, **fields: Any) -> Task:
        task = self.get(task_id)
        if "status" in fields and fields["status"] is not None:
            status = fields["status"]
            if status not in _STATUSES:
                raise ToolError(
                    f"Invalid status {status!r}. Must be one of: {', '.join(_STATUSES)}."
                )
            task.status = status
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
    "Task",
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskStore",
    "TaskUpdateTool",
]
