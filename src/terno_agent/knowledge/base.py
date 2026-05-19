"""Task and Phase abstractions.

A `Task` is one unit of work. A `Phase` is a named group of tasks
that runs them honoring `depends_on` and scheduling independent
tasks concurrently. Tasks may suspend by `await ctx.ask(...)`.

Concrete tasks live alongside the phases that own them in
`terno_agent.knowledge.phases.*`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from terno_agent.knowledge.context import PhaseContext, TaskContext


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class TaskResult:
    task: str
    status: TaskStatus
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class PhaseResult:
    phase: str
    tasks: list[TaskResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(t.status is not TaskStatus.FAILED for t in self.tasks)


class Task(ABC):
    """One unit of work in a phase.

    Set `depends_on` to names of in-phase tasks that must finish first.
    Cross-phase ordering is signaled via `PhaseContext` events instead.
    """

    name: str = "task"
    description: str = ""
    depends_on: tuple[str, ...] = ()

    @abstractmethod
    async def run(self, ctx: TaskContext) -> TaskResult: ...


class Phase:
    """Concrete container that runs its tasks with intra-phase parallelism.

    Subclasses set `name`, `description`, and either pass tasks to
    `super().__init__` or override `build_tasks`.
    """

    name: str = "phase"
    description: str = ""

    def __init__(self, tasks: Iterable[Task] | None = None) -> None:
        self.tasks: list[Task] = list(tasks) if tasks is not None else self.build_tasks()

    def build_tasks(self) -> list[Task]:
        return []

    async def run(self, ctx: PhaseContext) -> PhaseResult:
        results: dict[str, TaskResult] = {}
        pending: dict[str, Task] = {t.name: t for t in self.tasks}
        running: dict[str, asyncio.Task[TaskResult]] = {}

        while pending or running:
            for name, task in list(pending.items()):
                deps = task.depends_on
                if any(d in results and results[d].status is TaskStatus.FAILED for d in deps):
                    results[name] = TaskResult(
                        task=name,
                        status=TaskStatus.SKIPPED,
                        error=f"upstream task failed: {deps}",
                    )
                    pending.pop(name)
                    continue
                ready = all(
                    d in results and results[d].status is TaskStatus.COMPLETED for d in deps
                )
                if ready:
                    tctx = TaskContext(phase=ctx, phase_name=self.name, task_name=name)
                    running[name] = asyncio.create_task(
                        self._run_one(task, tctx), name=f"{self.name}:{name}"
                    )
                    pending.pop(name)

            if not running:
                break

            done, _ = await asyncio.wait(running.values(), return_when=asyncio.FIRST_COMPLETED)
            for fut in done:
                name = next(n for n, f in running.items() if f is fut)
                results[name] = fut.result()
                del running[name]

        ordered = [results[t.name] for t in self.tasks if t.name in results]
        return PhaseResult(phase=self.name, tasks=ordered)

    async def _run_one(self, task: Task, ctx: TaskContext) -> TaskResult:
        try:
            return await task.run(ctx)
        except Exception as exc:
            return TaskResult(task=task.name, status=TaskStatus.FAILED, error=repr(exc))


__all__ = ["Phase", "PhaseResult", "Task", "TaskResult", "TaskStatus"]
