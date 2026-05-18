"""Public SDK for terno-agent.

The `Agent` class is the high-level entry point for using terno-agent
from your own code. Every constructor argument is optional — missing
values fall back to environment variables and `.env` files.

    from terno import Agent

    agent = Agent(api_key="sk-ant-...", database_url="sqlite:///./demo.db")
    result = agent.run("Top 10 customers by revenue last quarter")
    print(result.answer)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from terno_agent.agents.base import AgentRun
from terno_agent.agents.orchestrator import Orchestrator
from terno_agent.config import Config
from terno_agent.core.events import EventHook

if TYPE_CHECKING:
    from terno_agent.knowledge.runner import KnowledgeReport
    from terno_agent.knowledge.store import KnowledgeStore


class Agent:
    """High-level facade around the orchestrator.

    All ``__init__`` keyword arguments are optional. Unspecified fields
    are read from the environment (and a ``.env`` file in CWD or any
    parent directory).

    Examples
    --------
    Minimal — read everything from env:

        agent = Agent()
        agent.run("describe the users table").answer

    Programmatic config:

        agent = Agent(
            api_key="sk-ant-...",
            database_url="postgresql+psycopg://u:p@host/db",
            provider="anthropic",
            model="claude-opus-4-7",
            sandbox="local",
        )

    Stream events to your own UI:

        from terno_agent.core.events import TextDelta
        agent = Agent(on_event=lambda e: ... )
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        database_url: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        sandbox: str | None = None,
        sandbox_image: str | None = None,
        max_rows: int | None = None,
        read_only_sql: bool | None = None,
        config: Config | None = None,
        on_event: EventHook | None = None,
    ) -> None:
        self.config = config or _build_config(
            api_key=api_key,
            database_url=database_url,
            provider=provider,
            model=model,
            sandbox=sandbox,
            sandbox_image=sandbox_image,
            max_rows=max_rows,
            read_only_sql=read_only_sql,
        )
        self.on_event = on_event
        self._orchestrator = Orchestrator.from_config(self.config, on_event=on_event)

    # ----- Alternate constructors -------------------------------------------- #

    @classmethod
    def from_env(cls, *, on_event: EventHook | None = None) -> "Agent":
        """Build an `Agent` from environment variables and `.env`."""
        return cls(config=Config.from_env(), on_event=on_event)

    @classmethod
    def from_config(cls, config: Config, *, on_event: EventHook | None = None) -> "Agent":
        """Build an `Agent` from an explicit `Config`."""
        return cls(config=config, on_event=on_event)

    # ----- Inference --------------------------------------------------------- #

    def run(self, question: str) -> AgentRun:
        """Run one turn of the multi-agent loop and return the result."""
        return self._orchestrator.run(question)

    def ask(self, question: str) -> AgentRun:
        """Alias for `run` — kept for symmetry with the orchestrator."""
        return self._orchestrator.ask(question)

    # ----- Knowledge extraction --------------------------------------------- #

    def deep_research(
        self,
        *,
        store: "KnowledgeStore | None" = None,
        console: Any = None,
    ) -> "KnowledgeReport":
        """Run the four-phase knowledge-extraction pipeline.

        This is the same flow as ``terno deep_research`` on the command line:
        organization context, schema crawl, semantic annotation, and
        validation. Prompts are rendered to the supplied (or default)
        rich `Console`. Use the lower-level `KnowledgeExtractionAgent`
        directly if you need to drive prompts programmatically.
        """
        from terno_agent.knowledge.cli import run_knowledge_extraction

        return asyncio.run(
            run_knowledge_extraction(
                config=self.config,
                store=store,
                console=console,
            )
        )


def _build_config(
    *,
    api_key: str | None,
    database_url: str | None,
    provider: str | None,
    model: str | None,
    sandbox: str | None,
    sandbox_image: str | None,
    max_rows: int | None,
    read_only_sql: bool | None,
) -> Config:
    base = Config.from_env()
    return Config(
        llm_provider=provider or base.llm_provider,
        llm_model=model or base.llm_model,
        llm_api_key=api_key or base.llm_api_key,
        database_url=database_url if database_url is not None else base.database_url,
        sandbox=sandbox or base.sandbox,
        sandbox_image=sandbox_image or base.sandbox_image,
        max_rows=max_rows if max_rows is not None else base.max_rows,
        read_only_sql=read_only_sql if read_only_sql is not None else base.read_only_sql,
    )


__all__ = ["Agent"]
