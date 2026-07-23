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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from terno_agent.agents.base import AgentRun
from terno_agent.agents.terno import TernoAgent
from terno_agent.config import Config
from terno_agent.core.events import EventHook
from terno_agent.core.hooks import Hook, HookEvent, UsageMeter
from terno_agent.core.messages import Message
from terno_agent.core.permissions import (
    PermissionCallback,
    PermissionDecision,
    PermissionMode,
    PermissionPolicy,
    PermissionRequest,
)
from terno_agent.tools.ask_user import AskCallback

if TYPE_CHECKING:
    from terno_agent.knowledge.runner import KnowledgeReport
    from terno_agent.knowledge.store import KnowledgeStore
    from terno_agent.sandbox.base import Sandbox
    from terno_agent.tools.tasks import TaskStore


class Agent:
    """High-level facade around the single Terno agent.

    All ``__init__`` keyword arguments are optional. Unspecified fields
    are read from the environment (and a ``.env`` file in CWD or any
    parent directory).

    If you configure MCP servers (via ``.terno/mcp.json``), use the agent as a
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
        workdir: str | Path | None = None,
        org_workdir: str | Path | None = None,
        max_iterations: int | None = None,
        bash_timeout_s: int = 120,
        run_python_timeout_s: int = 30,
        permission_mode: PermissionMode | str | None = None,
        allow_rules: list | tuple | None = None,
        on_permission_request: PermissionCallback | None = None,
        permission_policy: PermissionPolicy | None = None,
        ask_callback: AskCallback | None = None,
        sandbox: "Sandbox | None" = None,
        task_store: "TaskStore | None" = None,
        user_memory_root: str | Path | None = None,
        org_memory_root: str | Path | None = None,
        is_org_admin: bool | None = None,
        session_id: str | None = None,
    ) -> None:
        self.config = config or _build_config(
            api_key=api_key,
            database_url=database_url,
            provider=provider,
            model=model,
        )
        _apply_memory_identity(
            self.config,
            user_memory_root=user_memory_root,
            org_memory_root=org_memory_root,
            is_org_admin=is_org_admin,
            session_id=session_id,
        )
        self.on_event = on_event
        self._agent = TernoAgent.from_config(
            self.config,
            on_event=on_event,
            workdir=workdir,
            org_workdir=org_workdir,
            max_iterations=max_iterations,
            bash_timeout_s=bash_timeout_s,
            run_python_timeout_s=run_python_timeout_s,
            permission_mode=permission_mode,
            allow_rules=allow_rules,
            on_permission_request=on_permission_request,
            ask_callback=ask_callback,
            sandbox=sandbox,
            task_store=task_store,
            permission_policy=permission_policy,
        )
        self._closed = False

    # ----- Alternate constructors -------------------------------------------- #

    @classmethod
    def from_env(
        cls,
        *,
        on_event: EventHook | None = None,
        workdir: str | Path | None = None,
        max_iterations: int | None = None,
        bash_timeout_s: int = 120,
        run_python_timeout_s: int = 30,
    ) -> Agent:
        """Build an `Agent` from environment variables and `.env`."""
        return cls(
            config=Config.from_env(),
            on_event=on_event,
            workdir=workdir,
            max_iterations=max_iterations,
            bash_timeout_s=bash_timeout_s,
            run_python_timeout_s=run_python_timeout_s,
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        on_event: EventHook | None = None,
        workdir: str | Path | None = None,
        org_workdir: str | Path | None = None,
        max_iterations: int | None = None,
        bash_timeout_s: int = 120,
        run_python_timeout_s: int = 30,
        ask_callback: AskCallback | None = None,
        sandbox: "Sandbox | None" = None,
        task_store: "TaskStore | None" = None,
    ) -> Agent:
        """Build an `Agent` from an explicit `Config`.

        Pass ``sandbox`` to inject a pre-built :class:`Sandbox` (e.g. a host
        that proxies ``run_python`` into an externally-managed container). When
        given it overrides the sandbox the SDK would build from ``config``.

        Pass ``task_store`` to inject a persistence-backed task store (e.g.
        terno-ai's database store). When omitted the agent uses a
        process-local in-memory store, so the SDK works fully standalone.

        Pass ``ask_callback`` to enable the ``ask_user`` tool: a callable
        receiving the pending ``list[Question]`` and returning one
        ``Answer`` per question. Without it, ``ask_user`` is not offered
        to the model at all (matching the SDK's standalone default).
        """
        return cls(
            config=config,
            on_event=on_event,
            workdir=workdir,
            org_workdir=org_workdir,
            max_iterations=max_iterations,
            bash_timeout_s=bash_timeout_s,
            run_python_timeout_s=run_python_timeout_s,
            ask_callback=ask_callback,
            sandbox=sandbox,
            task_store=task_store,
        )

    # ----- Inference --------------------------------------------------------- #

    def run(
        self,
        task: str,
        *,
        extra_context: str | None = None,
    ) -> AgentRun:
        """Run the agent on a task and return the result.

        ``extra_context`` is injected ahead of ``task`` as a ``<context>``
        block for this run only (e.g. tool/usage guidance the host wants the
        model to see without baking it into the persistent system prompt).
        """
        return self._agent.run(task, extra_context=extra_context)

    def ask(self, task: str) -> AgentRun:
        """Alias for `run`."""
        return self._agent.ask(task)

    # ----- Conversation state ----------------------------------------------- #

    @property
    def history(self) -> list[Message]:
        """The agent's persistent conversation (mutated in place across runs)."""
        return self._agent.history

    @property
    def usage(self) -> UsageMeter:
        """Cumulative LLM token usage reported by the provider."""
        return self._agent.usage

    def clear_history(self) -> None:
        """Reset the conversation to just the system message and zero usage."""
        self._agent.clear_history()

    def set_history(self, messages: list[Message]) -> None:
        """Seed the conversation with prior turns before the next ``run``.

        Keeps the agent's system prompt and replaces any existing
        conversation with ``messages`` — SDK
        :class:`~terno_agent.core.messages.Message` objects such as
        ``UserMessage`` / ``AssistantMessage``. A host that persists chat
        history externally (e.g. terno-ai's ``ChatMessage`` rows) uses this
        to give a freshly constructed ``Agent`` its multi-turn context, so
        ``run`` continues the conversation instead of starting cold.
        """
        self._agent.set_history(messages)

    # ----- Permissions ------------------------------------------------------ #

    @property
    def permissions(self) -> PermissionPolicy | None:
        """The active permission policy, mutable at runtime.

        Returns ``None`` if the agent was constructed with a raw
        ``permission_hook`` (back-compat path); in that case the caller
        is managing their own state.
        """
        return self._agent.permissions

    # ----- Hooks ------------------------------------------------------------ #

    def add_hook(self, event: str, hook: Hook) -> None:
        """Register a lifecycle hook (e.g. ``HookEvent.CHAT_END``).

        See :class:`terno_agent.core.hooks.HookEvent` for the supported
        event names. Hooks receive a ``HookContext`` and may mutate
        ``ctx.history`` in place (the built-in compaction hook does this).
        """
        self._agent.add_hook(event, hook)

    # ----- Cancellation ----------------------------------------------------- #

    def cancel(self) -> None:
        """Ask the agent to abort its current turn. Safe across threads."""
        self._agent.cancel()

    def reset_cancel(self) -> None:
        """Clear a previous cancel signal so the next call runs normally."""
        self._agent.reset_cancel()

    # ----- Lifecycle --------------------------------------------------------- #

    def close(self) -> None:
        """Shut down owned resources (sandbox, MCP servers, background loops).

        The sandbox container is torn down here unless `sandbox_persist`
        is set, in which case it's left running so the next session can
        attach to it. Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        sandbox = getattr(self._agent, "sandbox", None)
        if sandbox is not None:
            closer = getattr(sandbox, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
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


def _apply_memory_identity(
    config: Config,
    *,
    user_memory_root: str | Path | None,
    org_memory_root: str | Path | None,
    is_org_admin: bool | None,
    session_id: str | None,
) -> None:
    """Override memory-location fields on ``config`` from explicit kwargs.

    A host passes the already-authorized memory folders (and the org-admin
    flag) it resolved for the authenticated user. Only non-``None`` values
    override what the config/env already carries, so callers can set just the
    fields they know.
    """
    if user_memory_root is not None:
        config.user_memory_root = str(user_memory_root)
    if org_memory_root is not None:
        config.org_memory_root = str(org_memory_root)
    if is_org_admin is not None:
        config.is_org_admin = is_org_admin
    if session_id is not None:
        config.session_id = session_id


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
        sandbox_options=dict(base.sandbox_options),
        sandbox_fallback=base.sandbox_fallback,
        sandbox_persist=base.sandbox_persist,
        sandbox_container_name=base.sandbox_container_name,
        max_rows=base.max_rows,
        read_only_sql=base.read_only_sql,
        mcp_enabled=base.mcp_enabled,
        mcp_config_path=base.mcp_config_path,
        mcp_servers=base.mcp_servers,
        skills_enabled=base.skills_enabled,
        skill_paths=list(base.skill_paths),
        file_memory_enabled=base.file_memory_enabled,
        user_memory_root=base.user_memory_root,
        org_memory_root=base.org_memory_root,
        is_org_admin=base.is_org_admin,
        session_id=base.session_id,
        memory_enabled=base.memory_enabled,
        memory_top_k=base.memory_top_k,
        embedding_provider=base.embedding_provider,
        embedding_model=base.embedding_model,
        embedding_api_key=base.embedding_api_key,
        compaction_enabled=base.compaction_enabled,
        compaction_threshold_tokens=base.compaction_threshold_tokens,
        compaction_keep_last_turns=base.compaction_keep_last_turns,
        attachments_enabled=base.attachments_enabled,
        attachments_dir=base.attachments_dir,
        max_attachment_bytes=base.max_attachment_bytes,
        max_attachments_per_turn=base.max_attachments_per_turn,
        attachment_image_mode=base.attachment_image_mode,
        extra=dict(base.extra),
    )


__all__ = [
    "Agent",
    "HookEvent",
    "PermissionDecision",
    "PermissionMode",
    "PermissionPolicy",
    "PermissionRequest",
]
