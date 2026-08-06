"""Task-tracking tools.

Provides a small in-memory `TaskStore` shared across a `TernoAgent`
(including its subagents) plus four tools the LLM can call:
``TaskCreate``, ``TaskList``, ``TaskGet``, ``TaskUpdate``.

Names, parameters and descriptions are ported from the reference harness. Its
ownership (``owner``), dependency (``addBlocks``/``addBlockedBy``) and
``metadata`` fields are omitted — this store models none of them, and a
description promising them would be a lie.

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
            name="TaskCreate",
            description=(
                "Use this tool to create a structured task list for your current"
                " coding session. This helps you track progress, organize complex"
                " tasks, and demonstrate thoroughness to the user.\n"
                "It also helps the user understand the progress of the task and"
                " overall progress of their requests.\n"
                "\n"
                "## When to Use This Tool\n"
                "\n"
                "Use this tool proactively in these scenarios:\n"
                "\n"
                "- Complex multi-step tasks - When a task requires 3 or more"
                " distinct steps or actions\n"
                "- Non-trivial and complex tasks - Tasks that require careful"
                " planning or multiple operations\n"
                "- User explicitly requests todo list - When the user directly"
                " asks you to use the todo list\n"
                "- User provides multiple tasks - When users provide a list of"
                " things to be done (numbered or comma-separated)\n"
                "- After receiving new instructions - Immediately capture user"
                " requirements as tasks\n"
                "- When you start working on a task - Mark it as in_progress"
                " BEFORE beginning work\n"
                "- After completing a task - Mark it as completed and add any new"
                " follow-up tasks discovered during implementation\n"
                "\n"
                "## When NOT to Use This Tool\n"
                "\n"
                "Skip using this tool when:\n"
                "- There is only a single, straightforward task\n"
                "- The task is trivial and tracking it provides no organizational"
                " benefit\n"
                "- The task can be completed in less than 3 trivial steps\n"
                "- The task is purely conversational or informational\n"
                "\n"
                "NOTE that you should not use this tool if there is only one"
                " trivial task to do. In this case you are better off just doing"
                " the task directly.\n"
                "\n"
                "## Task Fields\n"
                "\n"
                "- **subject**: A brief, actionable title in imperative form"
                ' (e.g., "Fix authentication bug in login flow")\n'
                "- **description**: What needs to be done\n"
                "- **activeForm** (optional): Present continuous form shown in the"
                ' spinner when the task is in_progress (e.g., "Fixing'
                ' authentication bug"). If omitted, the spinner shows the subject'
                " instead.\n"
                "\n"
                "All tasks are created with status `pending`.\n"
                "\n"
                "## Tips\n"
                "\n"
                "- Create tasks with clear, specific subjects that describe the"
                " outcome\n"
                "- Check TaskList first to avoid creating duplicate tasks"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "A brief title for the task",
                    },
                    "description": {
                        "type": "string",
                        "description": "What needs to be done",
                    },
                    "activeForm": {
                        "type": "string",
                        "description": (
                            "Present continuous form shown in spinner when"
                            ' in_progress (e.g., "Running tests")'
                        ),
                    },
                },
                "required": ["subject"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        subject = (kwargs.get("subject") or "").strip()
        if not subject:
            raise ToolError("TaskCreate requires a non-empty 'subject'.")
        task = self.store.create(
            subject,
            description=(kwargs.get("description") or "").strip(),
            active_form=(kwargs.get("activeForm") or "").strip(),
        )
        return json.dumps(task.to_dict())


@dataclass
class TaskListTool:
    store: TaskStore

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="TaskList",
            # The captured description also documents owner/blockedBy and
            # claiming; Terno's store has no ownership or dependency model, so
            # those parts are dropped rather than promised.
            description=(
                "Use this tool to list all tasks in the task list.\n"
                "\n"
                "## When to Use This Tool\n"
                "\n"
                "- To see what tasks are available to work on (status:"
                " 'pending')\n"
                "- To check overall progress on the project\n"
                "- After completing a task, to pick the next available task\n"
                "- **Prefer working on tasks in ID order** (lowest ID first) when"
                " multiple tasks are available, as earlier tasks often set up"
                " context for later ones\n"
                "\n"
                "## Output\n"
                "\n"
                "Returns a summary of each task:\n"
                "- **id**: Task identifier (use with TaskGet, TaskUpdate)\n"
                "- **subject**: Brief description of the task\n"
                "- **status**: 'pending', 'in_progress', or 'completed'"
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
            name="TaskGet",
            description=(
                "Use this tool to retrieve a task by its ID from the task"
                " list.\n"
                "\n"
                "## When to Use This Tool\n"
                "\n"
                "- When you need the full description and context before starting"
                " work on a task\n"
                "\n"
                "## Output\n"
                "\n"
                "Returns full task details:\n"
                "- **subject**: Task title\n"
                "- **description**: Detailed requirements and context\n"
                "- **status**: 'pending', 'in_progress', or 'completed'\n"
                "\n"
                "## Tips\n"
                "\n"
                "- Use TaskList to see all tasks in summary form."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "taskId": {
                        "type": "string",
                        "description": "The ID of the task to retrieve",
                    }
                },
                "required": ["taskId"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        task_id = kwargs.get("taskId")
        if not task_id:
            raise ToolError("TaskGet requires a 'taskId'.")
        return json.dumps(self.store.get(str(task_id)).to_dict())


@dataclass
class TaskUpdateTool:
    store: TaskStore

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="TaskUpdate",
            # Drops the captured owner/metadata/addBlocks/addBlockedBy fields:
            # Terno's store models neither ownership nor dependencies.
            description=(
                "Use this tool to update a task in the task list.\n"
                "\n"
                "## When to Use This Tool\n"
                "\n"
                "**Mark tasks as resolved:**\n"
                "- When you have completed the work described in a task\n"
                "- When a task is no longer needed or has been superseded\n"
                "- IMPORTANT: Always mark your assigned tasks as resolved when you"
                " finish them\n"
                "- After resolving, call TaskList to find your next task\n"
                "\n"
                "- ONLY mark a task as completed when you have FULLY accomplished"
                " it\n"
                "- If you encounter errors, blockers, or cannot finish, keep the"
                " task as in_progress\n"
                "- When blocked, create a new task describing what needs to be"
                " resolved\n"
                "- Never mark a task as completed if tests are failing\n"
                "\n"
                "- Setting status to `deleted` permanently removes the task\n"
                "\n"
                "**Update task details:**\n"
                "- When requirements change or become clearer\n"
                "\n"
                "## Fields You Can Update\n"
                "\n"
                "- **status**: The task status (see Status Workflow below)\n"
                '- **subject**: Change the task title (imperative form, e.g., "Run'
                ' tests")\n'
                "- **description**: Change the task description\n"
                "- **activeForm**: Present continuous form shown in spinner when"
                ' in_progress (e.g., "Running tests")\n'
                "\n"
                "## Status Workflow\n"
                "\n"
                "Status progresses: `pending` -> `in_progress` -> `completed`\n"
                "\n"
                "Use `deleted` to permanently remove a task.\n"
                "\n"
                "## Staleness\n"
                "\n"
                "Make sure to read a task's latest state using `TaskGet` before"
                " updating it.\n"
                "\n"
                "## Examples\n"
                "\n"
                "Mark task as in progress when starting work:\n"
                "```json\n"
                '{"taskId": "1", "status": "in_progress"}\n'
                "```\n"
                "\n"
                "Mark task as completed after finishing work:\n"
                "```json\n"
                '{"taskId": "1", "status": "completed"}\n'
                "```"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "taskId": {
                        "type": "string",
                        "description": "The ID of the task to update",
                    },
                    "status": {
                        "type": "string",
                        "enum": list(_STATUSES),
                        "description": "New status for the task",
                    },
                    "subject": {
                        "type": "string",
                        "description": "New subject for the task",
                    },
                    "description": {
                        "type": "string",
                        "description": "New description for the task",
                    },
                    "activeForm": {
                        "type": "string",
                        "description": (
                            "Present continuous form shown in spinner when"
                            ' in_progress (e.g., "Running tests")'
                        ),
                    },
                },
                "required": ["taskId"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        task_id = kwargs.get("taskId")
        if not task_id:
            raise ToolError("TaskUpdate requires a 'taskId'.")
        task = self.store.update(
            str(task_id),
            status=kwargs.get("status"),
            subject=kwargs.get("subject"),
            description=kwargs.get("description"),
            active_form=kwargs.get("activeForm"),
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
