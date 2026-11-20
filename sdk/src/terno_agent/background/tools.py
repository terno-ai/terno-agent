"""Agent tools for background tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from terno_agent.background.service import BackgroundTasks
from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema

_MAX_OUTPUT_CHARS = 100_000

_TASK_ID = {
    "task_id": {
        "type": "string",
        "description": "The id returned by run_in_background.",
    }
}


@dataclass
class RunInBackgroundTool:
    tasks: BackgroundTasks

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="run_in_background",
            description=(
                "Start Python code or a shell command in the sandbox and return "
                "at once with a task id, while it keeps running. You are told "
                "automatically when it finishes, even after this turn ends, and "
                "can then read its output with background_output.\n\n"
                "## When to use\n"
                "- Work that will clearly take more than a couple of minutes: "
                "training a model, processing many files or rows, bulk "
                "downloads, waiting on a slow external job.\n"
                "- A run_python or bash call that already hit its time limit.\n"
                "- The user asks for it to run in the background.\n\n"
                "## When NOT to use\n"
                "- Quick work, or when your very next step needs the result: "
                "use run_python or bash.\n\n"
                "After starting it, tell the user what is running and carry on "
                "or end your turn. Never wait for it with sleep or a polling "
                "loop. The code runs as a fresh process (not in run_python's "
                "session), so it must import and load what it needs, and "
                "should save results to files and print a short summary."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A few words saying what this does, shown to the user.",
                    },
                    "code": {
                        "type": "string",
                        "description": "Python source to run. Give either this or command.",
                    },
                    "command": {
                        "type": "string",
                        "description": "Shell command to run. Give either this or code.",
                    },
                },
                "required": ["description"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        code = kwargs.get("code") or ""
        command = kwargs.get("command") or ""
        if bool(code.strip()) == bool(command.strip()):
            raise ToolError("Give exactly one of 'code' or 'command'.")
        kind, payload = ("python", code) if code.strip() else ("shell", command)
        task = self.tasks.start(
            kind=kind, payload=payload, description=kwargs.get("description") or ""
        )
        return (
            f'Started background task {task.id} ("{task.description}"). You will '
            "be told when it finishes; do not wait or poll for it."
        )


@dataclass
class BackgroundOutputTool:
    tasks: BackgroundTasks

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="background_output",
            description=(
                "Show a background task's status and the latest part of its "
                "output. Use it when a task has finished, or when the user asks "
                "how one is going; not in a loop to wait for one."
            ),
            parameters={
                "type": "object",
                "properties": {
                    **_TASK_ID,
                    "max_chars": {
                        "type": "integer",
                        "description": "How much of the end of the output to show (default 8000).",
                    },
                },
                "required": ["task_id"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        task_id = kwargs.get("task_id")
        if not task_id:
            raise ToolError("background_output requires a 'task_id'.")
        max_chars = int(kwargs.get("max_chars") or 8000)
        if not 0 < max_chars <= _MAX_OUTPUT_CHARS:
            raise ToolError(f"max_chars must be between 1 and {_MAX_OUTPUT_CHARS}.")
        return self.tasks.output(str(task_id), max_chars=max_chars)


@dataclass
class StopBackgroundTaskTool:
    tasks: BackgroundTasks

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="stop_background_task",
            description="Stop a running background task and the processes it started.",
            parameters={"type": "object", "properties": _TASK_ID, "required": ["task_id"]},
        )

    def run(self, **kwargs: Any) -> str:
        task_id = kwargs.get("task_id")
        if not task_id:
            raise ToolError("stop_background_task requires a 'task_id'.")
        return self.tasks.describe(self.tasks.stop(str(task_id))) + "."


def background_tools(tasks: BackgroundTasks) -> list:
    return [
        RunInBackgroundTool(tasks),
        BackgroundOutputTool(tasks),
        StopBackgroundTaskTool(tasks),
    ]


__all__ = [
    "BackgroundOutputTool",
    "RunInBackgroundTool",
    "StopBackgroundTaskTool",
    "background_tools",
]
