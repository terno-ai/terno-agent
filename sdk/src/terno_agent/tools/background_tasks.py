"""`TaskStop` and `TaskOutput` — control over background tasks.

Both descriptions are ported from the capture, including the fact that
`TaskOutput` announces itself as `DEPRECATED` and redirects the model to `Read`
the output file. That is kept deliberately: the deprecation notice is what steers
the model toward the cheaper path, so removing it would make the tool *more*
attractive than the reference harness intends.

Trimmed, because Terno has none of it: agent-team teammates ("name@team"),
named background agents, and remote sessions. Terno's background tasks are
shell commands — its subagents run synchronously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from terno_agent.core.background import BackgroundTaskRegistry
from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema

# The captured schema caps the wait at 10 minutes.
_MAX_TIMEOUT_MS = 600_000
_DEFAULT_TIMEOUT_MS = 30_000


def _resolve_id(kwargs: dict[str, Any]) -> str:
    # `shell_id` is marked "Deprecated: use task_id instead" in the capture, but
    # is still accepted.
    task_id = kwargs.get("task_id") or kwargs.get("shell_id")
    if not task_id:
        raise ToolError("A 'task_id' is required.")
    return str(task_id)


def _unknown(registry: BackgroundTaskRegistry, task_id: str) -> ToolError:
    known = ", ".join(t.id for t in registry.list()) or "(none)"
    return ToolError(f"No background task with id {task_id!r}. Known tasks: {known}")


@dataclass
class TaskStopTool:
    registry: BackgroundTaskRegistry

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="TaskStop",
            description=(
                "\n"
                "- Stops a running background task by its ID\n"
                "- Takes a task_id parameter identifying the task to stop\n"
                "- Returns a success or failure status\n"
                "- Use this tool when you need to terminate a long-running task\n"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The ID of the background task to stop.",
                    },
                    "shell_id": {
                        "type": "string",
                        "description": "Deprecated: use task_id instead",
                    },
                },
            },
        )

    def run(self, **kwargs: Any) -> str:
        task_id = _resolve_id(kwargs)
        task = self.registry.get(task_id)
        if task is None:
            raise _unknown(self.registry, task_id)
        if not task.stop():
            # Not an error: the model asked for it to be over, and it is.
            return f"{task_id} was already {task.status} (exit code {task.exit_code})."
        return f"Stopped {task_id}. Output so far is in {task.output_path}"


@dataclass
class TaskOutputTool:
    registry: BackgroundTaskRegistry

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="TaskOutput",
            description=(
                "DEPRECATED: Background tasks return their output file path in"
                " the tool result, and you receive a notification with the same"
                " path when the task completes.\n"
                "- For bash tasks: prefer using the Read tool on that output file"
                " path — it contains stdout/stderr.\n"
                "\n"
                "- Retrieves output from a running or completed background"
                " task\n"
                "- Takes a task_id parameter identifying the task\n"
                "- Returns the task output along with status information\n"
                "- Use block=true (default) to wait for task completion\n"
                "- Use block=false for non-blocking check of current status"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to get output from",
                    },
                    "block": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to wait for completion",
                    },
                    "timeout": {
                        "type": "number",
                        "default": _DEFAULT_TIMEOUT_MS,
                        "minimum": 0,
                        "maximum": _MAX_TIMEOUT_MS,
                        "description": "Max wait time in ms",
                    },
                },
                "required": ["task_id", "block", "timeout"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        task_id = _resolve_id(kwargs)
        task = self.registry.get(task_id)
        if task is None:
            raise _unknown(self.registry, task_id)

        block = kwargs.get("block")
        block = True if block is None else bool(block)
        if block:
            timeout_ms = kwargs.get("timeout")
            timeout_ms = _DEFAULT_TIMEOUT_MS if timeout_ms is None else float(timeout_ms)
            timeout_ms = max(0.0, min(timeout_ms, _MAX_TIMEOUT_MS))
            task.wait(timeout_ms / 1000.0)

        status = task.status
        header = f"{task_id} [{status}]"
        if task.exit_code is not None:
            header += f" exit code {task.exit_code}"
        header += f"\noutput file: {task.output_path}"
        output = task.read_output()
        if not output.strip():
            return f"{header}\n(no output yet)"
        return f"{header}\n\n{output}"


__all__ = ["TaskOutputTool", "TaskStopTool"]
