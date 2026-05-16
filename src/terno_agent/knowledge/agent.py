"""Facade for the knowledge-extraction pipeline.

`KnowledgeExtractionAgent` wires the four phases to the runner and
exposes the prompt channel so a host (CLI, web UI, test harness)
can drain user prompts and submit answers while phases run.

    agent = KnowledgeExtractionAgent(db=db, llm=llm, store=store)
    # In one task: consume agent.channel.next_prompt() in a loop
    #              and call agent.channel.submit(response).
    # In another:  report = await agent.run()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from terno_agent.knowledge.base import Phase
from terno_agent.knowledge.context import PhaseContext
from terno_agent.knowledge.phases.annotation import AnnotationPhase
from terno_agent.knowledge.phases.org_context import OrgContextPhase
from terno_agent.knowledge.phases.schema_crawl import SchemaCrawlPhase
from terno_agent.knowledge.phases.validation import ValidationPhase
from terno_agent.knowledge.prompts import PromptChannel
from terno_agent.knowledge.runner import KnowledgeReport, KnowledgeRunner
from terno_agent.knowledge.store import InMemoryStore, KnowledgeStore

if TYPE_CHECKING:
    from terno_agent.db.connection import Database
    from terno_agent.llm.base import LLMClient


class KnowledgeExtractionAgent:
    def __init__(
        self,
        *,
        db: "Database | None" = None,
        llm: "LLMClient | None" = None,
        store: KnowledgeStore | None = None,
        channel: PromptChannel | None = None,
        phases: list[Phase] | None = None,
    ) -> None:
        self.store = store if store is not None else InMemoryStore()
        self.channel = channel if channel is not None else PromptChannel()
        self.context = PhaseContext(
            db=db, llm=llm, store=self.store, channel=self.channel
        )
        self._runner = KnowledgeRunner(
            phases=phases
            if phases is not None
            else [
                OrgContextPhase(),
                SchemaCrawlPhase(),
                AnnotationPhase(),
                ValidationPhase(),
            ]
        )

    async def run(self) -> KnowledgeReport:
        return await self._runner.run(self.context)


__all__ = ["KnowledgeExtractionAgent"]
