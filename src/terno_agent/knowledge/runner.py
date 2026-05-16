"""Top-level phase coordinator.

Schedules every phase concurrently and returns a `KnowledgeReport`
once all phases settle. Cross-phase data flow happens through the
shared `PhaseContext` (events + `artifacts` dict), not through this
runner — keeping the runner free of per-phase coupling.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from terno_agent.knowledge.base import Phase, PhaseResult
from terno_agent.knowledge.context import PhaseContext


@dataclass(slots=True)
class KnowledgeReport:
    phases: list[PhaseResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(p.ok for p in self.phases)


class KnowledgeRunner:
    def __init__(self, phases: list[Phase]) -> None:
        self.phases = phases

    async def run(self, ctx: PhaseContext) -> KnowledgeReport:
        coros = [phase.run(ctx) for phase in self.phases]
        results = await asyncio.gather(*coros)
        return KnowledgeReport(phases=list(results))


__all__ = ["KnowledgeReport", "KnowledgeRunner"]
