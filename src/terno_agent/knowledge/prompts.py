"""User-input channel for knowledge-extraction tasks.

Mirrors Claude Code's `AskUserQuestion` shape: a question with 2-4
options, single- or multi-select, plus an optional free-text field.
Tasks await `PromptChannel.ask(...)`; the host UI drains pending
prompts and posts answers back via `PromptChannel.submit(...)`.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptOption:
    label: str
    value: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class UserPrompt:
    id: str
    phase: str
    task: str
    question: str
    options: tuple[PromptOption, ...]
    multi_select: bool = False
    allow_text: bool = True
    text_label: str | None = None

    @classmethod
    def new(
        cls,
        *,
        phase: str,
        task: str,
        question: str,
        options: Iterable[PromptOption],
        multi_select: bool = False,
        allow_text: bool = True,
        text_label: str | None = None,
    ) -> UserPrompt:
        return cls(
            id=uuid.uuid4().hex,
            phase=phase,
            task=task,
            question=question,
            options=tuple(options),
            multi_select=multi_select,
            allow_text=allow_text,
            text_label=text_label,
        )


@dataclass(frozen=True, slots=True)
class UserResponse:
    prompt_id: str
    selected: tuple[str, ...] = ()
    text: str | None = None


class PromptChannel:
    """Bidirectional async channel between tasks and the UI.

    A single channel is shared across all phases so questions from any
    phase interleave through one surface. Tasks suspend on `ask`;
    the UI consumes `next_prompt()` and resolves them with `submit`.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[UserPrompt] = asyncio.Queue()
        self._waiters: dict[str, asyncio.Future[UserResponse]] = {}

    async def ask(self, prompt: UserPrompt) -> UserResponse:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[UserResponse] = loop.create_future()
        self._waiters[prompt.id] = fut
        await self._queue.put(prompt)
        try:
            return await fut
        finally:
            self._waiters.pop(prompt.id, None)

    async def next_prompt(self) -> UserPrompt:
        return await self._queue.get()

    def submit(self, response: UserResponse) -> None:
        fut = self._waiters.get(response.prompt_id)
        if fut is not None and not fut.done():
            fut.set_result(response)

    @property
    def pending_count(self) -> int:
        return len(self._waiters)


__all__ = [
    "PromptChannel",
    "PromptOption",
    "UserPrompt",
    "UserResponse",
]
