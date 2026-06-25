"""The datasource knowledge agent — a per-turn background curator.

`KnowledgeAgent` is a small, self-contained agent loop (its own system
prompt, its own tools, its own history) that the main assistant runs once
per turn. Given the user's message and the current state of the OKF
bundle, it decides whether to build, read, or edit knowledge files — and
does so without touching the main assistant's conversation or loop.

It is deliberately built on `BaseAgent` (not `TernoAgent`) so it gets only
its curated toolset and never recurses into another knowledge agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from terno_agent.agents.base import AgentRun, BaseAgent
from terno_agent.wiki.bundle import KnowledgeBundle
from terno_agent.wiki.paths import bundle_dir
from terno_agent.wiki.prompts import KNOWLEDGE_AGENT_PROMPT
from terno_agent.wiki.tools import knowledge_agent_tools

if TYPE_CHECKING:
    from terno_agent.core.cancel import CancelToken
    from terno_agent.db.connection import Database
    from terno_agent.llm.base import LLMClient


class KnowledgeAgent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        workdir: Path,
        datasource: str,
        db: Database | None = None,
        database_url: str = "",
        enrich: bool = True,
        max_tables: int = 50,
        max_iterations: int = 8,
        cancel_token: CancelToken | None = None,
    ) -> None:
        self.workdir = Path(workdir).resolve()
        self.datasource = datasource
        # The bulk builder enriches table concepts with the LLM by default;
        # pass enrich=False for introspection-only builds (cheaper / tests).
        build_llm = llm if enrich else None
        tools = knowledge_agent_tools(
            self.workdir,
            db=db,
            database_url=database_url,
            llm=build_llm,
            datasource=datasource,
            max_tables=max_tables,
        )
        self._agent = BaseAgent(
            llm,
            KNOWLEDGE_AGENT_PROMPT,
            tools,
            cancel_token=cancel_token,
        )
        self._agent.max_iterations = max_iterations

    # ----- per-turn entry point ----------------------------------------- #

    def run_turn(self, user_task: str) -> AgentRun:
        """Curate the bundle for one user turn. Isolated from the main loop."""
        bundle = KnowledgeBundle(
            bundle_dir(self.workdir, self.datasource), name=self.datasource
        )
        exists = bundle.exists()
        state = bundle.index_text().strip() if exists else "(no bundle yet)"
        task = (
            f"The user just asked the main assistant:\n\"{user_task}\"\n\n"
            f"Datasource: {self.datasource}\n"
            f"Bundle exists: {exists}\n"
            f"Current bundle index:\n{state}\n\n"
            "Decide what knowledge work (if any) would help answer this and "
            "similar questions, then act using your tools. If the bundle is "
            "missing, build it. If it already suffices, make no changes."
        )
        # Fresh history each turn keeps this bounded and focused.
        self._agent.clear_history()
        return self._agent.run(task)

    # ----- introspection ------------------------------------------------ #

    @property
    def usage(self):
        return self._agent.usage


__all__ = ["KnowledgeAgent"]
