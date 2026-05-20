"""Public SDK for terno-agent.

The `Agent` class is the high-level entry point for using terno-agent
from your own code. Every constructor argument is optional — missing
values fall back to environment variables and `.env` files.

    from terno import Agent

    agent = Agent(api_key="sk-ant-...")
    result = agent.run("Refactor utils.py into smaller modules")
    print(result.answer)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from terno_agent.agents.base import AgentRun
from terno_agent.agents.terno import TernoAgent
from terno_agent.config import Config
from terno_agent.core.events import EventHook

if TYPE_CHECKING:
    from terno_agent.knowledge.runner import KnowledgeReport
    from terno_agent.knowledge.store import KnowledgeStore


class Agent:
    """High-level facade around the single Terno agent.

    All ``__init__`` keyword arguments are optional. Unspecified fields
    are read from the environment (and a ``.env`` file in CWD or any
    parent directory).

    If you configure MCP servers (via ``.mcp.json``), use the agent as a
    context manager so MCP subprocesses are shut down cleanly::

        with Agent(api_key=...) as agent:
            agent.run("...")
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        database_url: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        config: Config | None = None,
        on_event: EventHook | None = None,
    ) -> None:
        self.config = config or _build_config(
            api_key=api_key,
            database_url=database_url,
            provider=provider,
            model=model,
        )
        self.on_event = on_event
        self._agent = TernoAgent.from_config(self.config, on_event=on_event)
        self._closed = False

    # ----- Alternate constructors -------------------------------------------- #

    @classmethod
    def from_env(cls, *, on_event: EventHook | None = None) -> Agent:
        """Build an `Agent` from environment variables and `.env`."""
        return cls(config=Config.from_env(), on_event=on_event)

    @classmethod
    def from_config(cls, config: Config, *, on_event: EventHook | None = None) -> Agent:
        """Build an `Agent` from an explicit `Config`."""
        return cls(config=config, on_event=on_event)

    # ----- Inference --------------------------------------------------------- #

    def run(self, task: str) -> AgentRun:
        """Run the agent on a task and return the result."""
        return self._agent.run(task)

    def ask(self, task: str) -> AgentRun:
        """Alias for `run`."""
        return self._agent.ask(task)

    # ----- Cancellation ----------------------------------------------------- #

    def cancel(self) -> None:
        """Ask the agent to abort its current turn. Safe across threads."""
        self._agent.cancel()

    def reset_cancel(self) -> None:
        """Clear a previous cancel signal so the next call runs normally."""
        self._agent.reset_cancel()

    # ----- Lifecycle --------------------------------------------------------- #

    def close(self) -> None:
        """Shut down owned resources (MCP servers, background loops).

        Safe to call multiple times; safe to skip if no MCP servers were
        configured. Also runs via ``atexit`` as a defensive net.
        """
        if self._closed:
            return
        self._closed = True
        manager = getattr(self._agent, "mcp_manager", None)
        if manager is not None:
            manager.shutdown()

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # ----- Knowledge extraction --------------------------------------------- #

    def deep_research(
        self,
        *,
        store: KnowledgeStore | None = None,
        console: Any = None,
    ) -> KnowledgeReport:
        """Run the four-phase knowledge-extraction pipeline.

        This is the same flow as ``terno deep_research`` on the command line:
        organization context, schema crawl, semantic annotation, and
        validation. Requires `TERNO_DATABASE_URL` to be set.
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
) -> Config:
    base = Config.from_env()
    return Config(
        llm_provider=provider or base.llm_provider,
        llm_model=model or base.llm_model,
        llm_api_key=api_key or base.llm_api_key,
        database_url=database_url if database_url is not None else base.database_url,
        sandbox=base.sandbox,
        sandbox_image=base.sandbox_image,
        max_rows=base.max_rows,
        read_only_sql=base.read_only_sql,
        mcp_enabled=base.mcp_enabled,
        mcp_config_path=base.mcp_config_path,
        memory_enabled=base.memory_enabled,
        memory_top_k=base.memory_top_k,
        embedding_provider=base.embedding_provider,
        embedding_model=base.embedding_model,
        embedding_api_key=base.embedding_api_key,
    )


__all__ = ["Agent"]
