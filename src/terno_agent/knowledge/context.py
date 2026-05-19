"""Execution context handed to phases and tasks.

`PhaseContext` is the long-lived bag of shared resources (DB, LLM,
store, prompt channel) plus cross-phase coordination signals.
`TaskContext` is a thin per-task view that exposes those resources
and a convenience `ask` wrapper around the channel.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from terno_agent.knowledge.prompts import PromptChannel, UserPrompt, UserResponse
from terno_agent.knowledge.store import KnowledgeStore

if TYPE_CHECKING:
    from terno_agent.db.connection import Database
    from terno_agent.llm.base import LLMClient


@dataclass(slots=True)
class PhaseContext:
    """Shared state for one knowledge-extraction run.

    Cross-phase coordination: phases publish completion via the
    `*_ready` events and stash artifacts in `artifacts` so dependent
    phases can pick them up without re-querying the store.
    """

    db: Database | None
    llm: LLMClient | None
    store: KnowledgeStore
    channel: PromptChannel
    org_prompt_ready: asyncio.Event = field(default_factory=asyncio.Event)
    schema_ready: asyncio.Event = field(default_factory=asyncio.Event)
    descriptions_ready: asyncio.Event = field(default_factory=asyncio.Event)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskContext:
    phase: PhaseContext
    phase_name: str
    task_name: str

    @property
    def db(self) -> Database | None:
        return self.phase.db

    @property
    def llm(self) -> LLMClient | None:
        return self.phase.llm

    @property
    def store(self) -> KnowledgeStore:
        return self.phase.store

    async def ask(self, prompt: UserPrompt) -> UserResponse:
        return await self.phase.channel.ask(prompt)


__all__ = ["PhaseContext", "TaskContext"]
