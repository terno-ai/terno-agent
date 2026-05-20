"""The single Terno agent.

`TernoAgent` is a `BaseAgent` wired with the standard Terno toolset
(file ops, bash, task list, subagent spawner) and the canonical
`SYSTEM_PROMPT`. Subagents spawned via the `spawn_agent` tool are also
`TernoAgent` instances — they share the same LLM, task store, and
working directory, but get a caller-supplied system prompt.
"""

from __future__ import annotations

import atexit
import sys
from pathlib import Path

from terno_agent.agents.base import AgentRun, BaseAgent
from terno_agent.config import Config
from terno_agent.core.cancel import CancelToken
from terno_agent.core.exceptions import ConfigError, SandboxError
from terno_agent.llm.base import LLMClient
from terno_agent.llm.factory import create_llm_client
from terno_agent.mcp.manager import McpManager
from terno_agent.memory.extractor import MemoryExtractor
from terno_agent.memory.retriever import MemoryRetriever
from terno_agent.memory.store import MemoryStore
from terno_agent.memory.tools import SearchMemoryTool
from terno_agent.prompts.prompt import SYSTEM_PROMPT
from terno_agent.rag.embeddings import create_embedding_client
from terno_agent.sandbox.base import Sandbox
from terno_agent.sandbox.factory import create_sandbox
from terno_agent.tools.code_exec import RunPythonTool
from terno_agent.tools.files import EditFileTool, ReadFileTool, WriteFileTool
from terno_agent.tools.shell import BashTool
from terno_agent.tools.subagent import SpawnAgentTool
from terno_agent.tools.tasks import (
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskStore,
    TaskUpdateTool,
)


class TernoAgent(BaseAgent):
    name = "terno"
    max_iterations = 32

    def __init__(
        self,
        llm: LLMClient,
        *,
        system_prompt: str | None = None,
        workdir: Path | None = None,
        task_store: TaskStore | None = None,
        sandbox: Sandbox | None = None,
        mcp_manager: McpManager | None = None,
        bash_timeout_s: int = 120,
        run_python_timeout_s: int = 30,
        on_event=None,
        memory_store: MemoryStore | None = None,
        memory_retriever: MemoryRetriever | None = None,
        memory_extractor: MemoryExtractor | None = None,
        cancel_token: CancelToken | None = None,
    ) -> None:
        self.workdir = (workdir or Path.cwd()).resolve()
        self.task_store = task_store or TaskStore()
        self.sandbox = sandbox
        self.mcp_manager = mcp_manager
        self.memory_store = memory_store
        self.memory_retriever = memory_retriever
        self.memory_extractor = memory_extractor

        token = cancel_token or CancelToken()

        tools: list = [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            BashTool(
                workdir=self.workdir,
                default_timeout_s=bash_timeout_s,
                cancel_token=token,
            ),
            TaskCreateTool(self.task_store),
            TaskListTool(self.task_store),
            TaskGetTool(self.task_store),
            TaskUpdateTool(self.task_store),
            SpawnAgentTool(
                llm=llm,
                workdir=self.workdir,
                task_store=self.task_store,
                sandbox=sandbox,
                mcp_manager=mcp_manager,
                bash_timeout_s=bash_timeout_s,
                run_python_timeout_s=run_python_timeout_s,
                on_event=on_event,
                cancel_token=token,
            ),
        ]
        if sandbox is not None:
            tools.append(
                RunPythonTool(
                    sandbox,
                    timeout_s=run_python_timeout_s,
                    cancel_token=token,
                )
            )
        if mcp_manager is not None:
            tools.extend(mcp_manager.tools())
        if memory_store is not None:
            tools.append(SearchMemoryTool(memory_store))

        super().__init__(
            llm,
            system_prompt or SYSTEM_PROMPT,
            tools,
            on_event=on_event,
            post_turn_hook=(
                memory_extractor.extract if memory_extractor is not None else None
            ),
            cancel_token=token,
        )

    # ----- Cancellation -------------------------------------------------- #

    def cancel(self) -> None:
        """Request the agent to abort its current turn ASAP.

        Safe to call from any thread or from a signal handler. The agent
        will return a partial `AgentRun` with ``cancelled=True``.
        """
        self.cancel_token.cancel()
        if self.mcp_manager is not None:
            # Free any future the agent is currently blocked on.
            bridge = getattr(self.mcp_manager, "_bridge", None)
            if bridge is not None:
                cancel_inflight = getattr(bridge, "cancel_inflight", None)
                if callable(cancel_inflight):
                    cancel_inflight()

    def reset_cancel(self) -> None:
        """Clear the cancel signal so the next ``run`` starts fresh."""
        self.cancel_token.clear()

    # ----- run with memory recall --------------------------------------- #

    def run(self, task: str, *, extra_context: str | None = None) -> AgentRun:
        if self.memory_retriever is not None:
            recalled = self.memory_retriever.fetch_relevant(task)
            if recalled:
                extra_context = (
                    recalled if not extra_context else f"{recalled}\n\n---\n{extra_context}"
                )
        return super().run(task, extra_context=extra_context)

    # ----- Construction helpers ----------------------------------------- #

    @classmethod
    def from_env(cls, *, on_event=None) -> TernoAgent:
        return cls.from_config(Config.from_env(), on_event=on_event)

    @classmethod
    def from_config(cls, config: Config, *, on_event=None) -> TernoAgent:
        if not config.llm_api_key:
            raise ConfigError(
                "No LLM API key configured. Set ANTHROPIC_API_KEY or "
                "OPENAI_API_KEY (or TERNO_LLM_API_KEY)."
            )
        llm = create_llm_client(
            provider=config.llm_provider,
            model=config.llm_model,
            api_key=config.llm_api_key,
        )
        sandbox: Sandbox | None = None
        if config.sandbox != "none":
            try:
                sandbox = create_sandbox(config.sandbox)
            except SandboxError as exc:
                print(
                    f"warning: sandbox {config.sandbox!r} unavailable ({exc}); "
                    "run_python tool will be disabled. Set TERNO_SANDBOX=none "
                    "to silence this warning, or TERNO_SANDBOX=local to use "
                    "the local subprocess sandbox.",
                    file=sys.stderr,
                )

        mcp_manager: McpManager | None = None
        if config.mcp_enabled:
            mcp_manager = McpManager.start_from_path(config.mcp_config_path or None)
            if mcp_manager is not None:
                atexit.register(mcp_manager.shutdown)

        memory_store, memory_retriever, memory_extractor = _build_memory(
            config, llm, workdir=Path.cwd(), on_event=on_event
        )

        return cls(
            llm,
            sandbox=sandbox,
            mcp_manager=mcp_manager,
            on_event=on_event,
            memory_store=memory_store,
            memory_retriever=memory_retriever,
            memory_extractor=memory_extractor,
        )

    # ----- Convenience --------------------------------------------------- #

    def ask(self, task: str) -> AgentRun:
        return self.run(task)


def _build_memory(
    config: Config,
    llm: LLMClient,
    *,
    workdir: Path,
    on_event=None,
) -> tuple[MemoryStore | None, MemoryRetriever | None, MemoryExtractor | None]:
    """Construct the memory pipeline if enabled in config.

    Returns ``(None, None, None)`` if memory is disabled or if the
    embedding client can't be constructed (e.g. ``openai`` not installed).
    Failure here must never block the agent — we just log to stderr and
    proceed without memory.
    """
    if not config.memory_enabled:
        return (None, None, None)
    embedding_key = config.embedding_api_key
    if not embedding_key and config.llm_provider == "openai":
        embedding_key = config.llm_api_key
    try:
        embedder = create_embedding_client(
            provider=config.embedding_provider,
            api_key=embedding_key,
            model=config.embedding_model,
        )
    except Exception as exc:
        print(
            f"warning: memory disabled — could not build embedding client: {exc}",
            file=sys.stderr,
        )
        return (None, None, None)
    store = MemoryStore(workdir=workdir, embedder=embedder)
    retriever = MemoryRetriever(store=store, k=config.memory_top_k)
    extractor = MemoryExtractor(
        llm=llm,
        store=store,
        workdir=workdir,
        on_event=on_event,
    )
    return (store, retriever, extractor)
