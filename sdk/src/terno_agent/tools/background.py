"""Background tasks as the agent sees them.

The tools that do work (``bash``, ``run_python``) start it in the background
through a :class:`BackgroundOption`, their ``run_in_background`` flag. Tasks,
whatever started them, are then read and stopped with the tools here. The
tasks themselves are kept and run by :mod:`terno_agent.background`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from terno_agent.background import RUNNING
from terno_agent.background.service import BackgroundTasks
from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema

_MAX_OUTPUT_CHARS = 100_000

_TASK_ID = {
    "task_id": {
        "type": "string",
        "description": "The id of the background task.",
    }
}

_GUIDE = (
    "\n\n## Running in the background\n"
    "Set run_in_background to start the work and return at once with a task "
    "id while it keeps running. You are told when it finishes. Work that runs "
    "past the time limit is stopped.\n"
    "Use it for work that will take more than a couple of minutes, or that is "
    "not ready yet and has to be tried again later (a slow job, something "
    "still being prepared, an error that says to retry later). Not for quick "
    "work, or when your very next step needs the result.\n"
    "The background code must:\n"
    "- Stand on its own: load everything it needs.\n"
    "- Keep going until it has a result: retry until it succeeds or fails in a "
    "way retrying cannot fix. Don't cap the attempts; the time limit bounds it.\n"
    "- Save results to files and print a short summary, or a clear error.\n"
    "After starting it:\n"
    "- The work belongs to the background run: don't do, retry or test it "
    "again in the foreground.\n"
    "- Don't check on it or wait for it (no sleep, no polling, no "
    "background_output while it runs): you will be told when it finishes.\n"    "- Don't guess or describe its result before you are told it finished.\n"
    "- Tell the user what is running and end your turn, unless there is other, "
    "independent work they asked for: do that, then end your turn."
)

_STILL_RUNNING = (
    "It is still running: you will be told when it finishes. Don't check on it "
    "again; tell the user it is running, or carry on with other work."
)

_TIMEOUT_HINT = (
    "If this work needs longer, run it again with run_in_background instead "
    "of retrying with a longer timeout."
)


@dataclass
class BackgroundOption:
    """The ``run_in_background`` flag for a tool that runs work.

    ``kind`` is the task kind the runner starts (``PYTHON`` or ``SHELL``);
    ``note`` says anything that differs from a normal run of that tool.
    """

    tasks: BackgroundTasks | None
    kind: str
    note: str = ""

    @property
    def offered(self) -> bool:
        return self.tasks is not None and self.tasks.can_start

    def describe(self, description: str) -> str:
        """The tool's description, with how to run it in the background."""
        if not self.offered:
            return description
        note = f"\n{self.note}" if self.note else ""
        return description + _GUIDE + note

    def properties(self) -> dict:
        """The flag's schema properties, when background runs are offered."""
        if not self.offered:
            return {}
        return {
            "run_in_background": {
                "type": "boolean",
                "description": "Start it in the background and return at once (see above).",
            },
            "description": {
                "type": "string",
                "description": "Required with run_in_background: a few words saying what it does, shown to the user.",
            },
        }

    def requested(self, kwargs: dict) -> bool:
        return bool(kwargs.get("run_in_background"))

    def start(self, payload: str, description: str | None) -> str:
        if not self.offered:
            raise ToolError("Background runs are not available here.")
        task = self.tasks.start(kind=self.kind, payload=payload, description=description or "")
        started = (
            f'Started background task {task.id} ("{task.description}"). You will '
            "be told when it finishes; do not wait or poll for it, and don't run "
            "this work again in the foreground. Tell the user what is running and "
            "end your turn unless you have other, independent work."
        )
        location = self.tasks.output_location(task)
        if location:
            started += f" Its output is written to {location}."
        return started

    def timeout_note(self, timeouts: int) -> str:
        """What a run that hit its time limit adds, firmer once runs keep
        timing out; nothing when background runs are not offered."""
        if not self.offered:
            return ""
        if timeouts < 2:
            return _TIMEOUT_HINT
        return (
            f"{_TIMEOUT_HINT} Runs have now hit the time limit {timeouts} times: "
            "don't run this work in the foreground again."
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
                "output. Use it after you were told a task finished, or when "
                "the user asks how one is going; not to check on or wait for one."
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
        try:
            max_chars = int(kwargs.get("max_chars") or 8000)
        except (TypeError, ValueError) as exc:
            raise ToolError("max_chars must be a whole number.") from exc
        if not 0 < max_chars <= _MAX_OUTPUT_CHARS:
            raise ToolError(f"max_chars must be between 1 and {_MAX_OUTPUT_CHARS}.")
        text = self.tasks.output(str(task_id), max_chars=max_chars)
        if self.tasks.store.get(str(task_id)).status == RUNNING:
            text += f"\n{_STILL_RUNNING}"
        return text


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
    """The tools for reading and stopping background tasks."""
    return [BackgroundOutputTool(tasks), StopBackgroundTaskTool(tasks)]


__all__ = [
    "BackgroundOption",
    "BackgroundOutputTool",
    "StopBackgroundTaskTool",
    "background_tools",
]
