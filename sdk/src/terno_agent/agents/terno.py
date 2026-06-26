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
from terno_agent.attachments import (
    AttachmentInput,
    AttachmentManager,
    AttachmentPolicy,
    AttachmentStore,
)
from terno_agent.config import Config
from terno_agent.core.cancel import CancelToken
from terno_agent.core.compaction import CompactionHook
from terno_agent.core.exceptions import ConfigError, SandboxError
from terno_agent.core.hooks import (
    HookContext,
    HookEvent,
    HookManager,
    PreToolUseHook,
)
from terno_agent.core.messages import ContentPart
from terno_agent.core.permissions import (
    PermissionCallback,
    PermissionMode,
    PermissionPolicy,
)
from terno_agent.llm.base import LLMClient
from terno_agent.llm.factory import create_llm_client
from terno_agent.mcp.manager import McpManager
from terno_agent.memory.extractor import ExtractionCallback, MemoryExtractor
from terno_agent.memory.retriever import MemoryRetriever
from terno_agent.memory.store import MemoryStore
from terno_agent.memory.tools import SearchMemoryTool
from terno_agent.prompts.prompt import SYSTEM_PROMPT
from terno_agent.rag.embeddings import create_embedding_client
from terno_agent.rag.vector_store import create_vector_store
from terno_agent.sandbox.base import Sandbox
from terno_agent.sandbox.factory import create_sandbox
from terno_agent.skills import ActivateSkillTool, SkillCatalog, discover_skills
from terno_agent.tools.ask_user import AskCallback, AskUserTool
from terno_agent.tools.code_exec import RunPythonTool
from terno_agent.tools.files import EditFileTool, ReadFileTool, WriteFileTool
from terno_agent.tools.monitor import MonitorTool
from terno_agent.tools.search import GlobTool, GrepTool
from terno_agent.tools.shell import BashTool
from terno_agent.tools.subagent import SpawnAgentTool
from terno_agent.tools.tasks import (
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskStore,
    TaskUpdateTool,
)
from terno_agent.tools.web import WebFetchTool, WebSearchTool


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
        skill_catalog: SkillCatalog | None = None,
        bash_timeout_s: int = 120,
        run_python_timeout_s: int = 30,
        max_iterations: int | None = None,
        on_event=None,
        memory_store: MemoryStore | None = None,
        memory_retriever: MemoryRetriever | None = None,
        memory_extractor: MemoryExtractor | None = None,
        attachment_manager: AttachmentManager | None = None,
        ask_callback: AskCallback | None = None,
        permission_hook: PreToolUseHook | None = None,
        permission_policy: PermissionPolicy | None = None,
        permission_mode: PermissionMode | str | None = None,
        allow_rules: list | tuple | None = None,
        on_permission_request: PermissionCallback | None = None,
        cancel_token: CancelToken | None = None,
        hook_manager: HookManager | None = None,
        compaction_hook: CompactionHook | None = None,
    ) -> None:
        self.workdir = (workdir or Path.cwd()).resolve()
        if max_iterations is not None:
            if max_iterations <= 0:
                raise ConfigError("max_iterations must be positive.")
            self.max_iterations = max_iterations
        self.task_store = task_store or TaskStore()
        self.sandbox = sandbox
        self.mcp_manager = mcp_manager
        self.skill_catalog = skill_catalog or SkillCatalog()
        self.memory_store = memory_store
        self.memory_retriever = memory_retriever
        self.memory_extractor = memory_extractor
        self.attachment_manager = attachment_manager
        self.ask_callback = ask_callback

        resolved = _resolve_permissions(
            permission_policy=permission_policy,
            permission_hook=permission_hook,
            permission_mode=permission_mode,
            allow_rules=allow_rules,
            on_permission_request=on_permission_request,
        )
        # ``permissions`` is the mutable PermissionPolicy when one is
        # in play, or None if the caller wired a raw PreToolUseHook
        # (back-compat path — they manage their own state).
        self.permissions: PermissionPolicy | None = (
            resolved if isinstance(resolved, PermissionPolicy) else None
        )
        # Subagents inherit the same callable so "allow always"
        # decisions propagate across spawn boundaries.
        self.permission_hook: PreToolUseHook | None = resolved

        token = cancel_token or CancelToken()

        tools: list = [
            ReadFileTool(workdir=self.workdir),
            WriteFileTool(workdir=self.workdir),
            EditFileTool(workdir=self.workdir),
            GlobTool(workdir=self.workdir),
            GrepTool(workdir=self.workdir),
            BashTool(
                workdir=self.workdir,
                default_timeout_s=bash_timeout_s,
                cancel_token=token,
            ),
            MonitorTool(
                workdir=self.workdir,
                cancel_token=token,
            ),
            WebFetchTool(),
            WebSearchTool(),
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
                max_iterations=max_iterations,
                on_event=on_event,
                cancel_token=token,
                ask_callback=ask_callback,
                permission_hook=self.permission_hook,
            ),
        ]
        if ask_callback is not None:
            tools.append(AskUserTool(ask_callback=ask_callback))
        if sandbox is not None:
            tools.append(
                RunPythonTool(
                    sandbox,
                    timeout_s=run_python_timeout_s,
                    cancel_token=token,
                )
            )
        if self.skill_catalog.skills:
            tools.append(ActivateSkillTool(self.skill_catalog))
        if mcp_manager is not None:
            tools.extend(mcp_manager.tools())
        if memory_store is not None:
            tools.append(SearchMemoryTool(memory_store))

        hooks = hook_manager or HookManager()
        if memory_extractor is not None:
            hooks.register(
                HookEvent.CHAT_END,
                _wrap_memory_extractor(memory_extractor),
            )
        if compaction_hook is not None:
            hooks.register(HookEvent.CHAT_END, compaction_hook)
        if self.permission_hook is not None:
            hooks.register(HookEvent.PRE_TOOL_USE, self.permission_hook)

        super().__init__(
            llm,
            _with_skill_catalog(system_prompt or SYSTEM_PROMPT, self.skill_catalog),
            tools,
            on_event=on_event,
            hook_manager=hooks,
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

    def run(
        self,
        task: str,
        *,
        extra_context: str | None = None,
        content_parts: list[ContentPart] | None = None,
        attachments: list[AttachmentInput] | None = None,
    ) -> AgentRun:
        if self.memory_retriever is not None:
            recalled = self.memory_retriever.fetch_relevant(task)
            if recalled:
                extra_context = (
                    recalled if not extra_context else f"{recalled}\n\n---\n{extra_context}"
                )
        if content_parts is not None:
            return super().run(task, extra_context=extra_context, content_parts=content_parts)
        if attachments:
            if self.attachment_manager is None:
                raise ConfigError("Attachments are disabled for this agent.")
            content_parts = self.attachment_manager.build_parts(task, list(attachments))
            return super().run(
                task,
                extra_context=extra_context,
                content_parts=content_parts,
            )
        return super().run(task, extra_context=extra_context)

    # ----- Construction helpers ----------------------------------------- #

    @classmethod
    def from_env(
        cls,
        *,
        on_event=None,
        ask_callback: AskCallback | None = None,
        on_memory_event: ExtractionCallback | None = None,
        permission_hook: PreToolUseHook | None = None,
        permission_policy: PermissionPolicy | None = None,
        permission_mode: PermissionMode | str | None = None,
        allow_rules: list | tuple | None = None,
        on_permission_request: PermissionCallback | None = None,
        workdir: Path | str | None = None,
        max_iterations: int | None = None,
        bash_timeout_s: int = 120,
        run_python_timeout_s: int = 30,
    ) -> TernoAgent:
        return cls.from_config(
            Config.from_env(),
            on_event=on_event,
            ask_callback=ask_callback,
            on_memory_event=on_memory_event,
            permission_hook=permission_hook,
            permission_policy=permission_policy,
            permission_mode=permission_mode,
            allow_rules=allow_rules,
            on_permission_request=on_permission_request,
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
        on_event=None,
        ask_callback: AskCallback | None = None,
        on_memory_event: ExtractionCallback | None = None,
        permission_hook: PreToolUseHook | None = None,
        permission_policy: PermissionPolicy | None = None,
        permission_mode: PermissionMode | str | None = None,
        allow_rules: list | tuple | None = None,
        on_permission_request: PermissionCallback | None = None,
        workdir: Path | str | None = None,
        max_iterations: int | None = None,
        bash_timeout_s: int = 120,
        run_python_timeout_s: int = 30,
    ) -> TernoAgent:
        if not config.llm_api_key:
            raise ConfigError(
                "No LLM API key configured. Set ANTHROPIC_API_KEY or "
                "OPENAI_API_KEY (or TERNO_LLM_API_KEY); for the provisioner "
                "proxy (TERNO_LLM_PROVIDER=terno) set TERNO_API_KEY."
            )
        llm = create_llm_client(
            provider=config.llm_provider,
            model=config.llm_model,
            api_key=config.llm_api_key,
            provisioner_url=config.provisioner_url or None,
            app_version=config.app_version or None,
            request_source=config.request_source or None,
        )
        sandbox = _init_sandbox(config)
        resolved_workdir = (Path(workdir) if workdir is not None else Path.cwd()).resolve()

        mcp_manager: McpManager | None = None
        if config.mcp_enabled:
            if config.mcp_servers:
                mcp_manager = McpManager.start_from_dict(config.mcp_servers)
            else:
                mcp_manager = McpManager.start_from_path(
                    config.mcp_config_path or None
                )
            if mcp_manager is not None:
                atexit.register(mcp_manager.shutdown)

        memory_store, memory_retriever, memory_extractor = _build_memory(
            config,
            llm,
            workdir=resolved_workdir,
            on_memory_event=on_memory_event,
        )
        skill_catalog = (
            discover_skills(
                resolved_workdir,
                extra_roots=[Path(p) for p in config.skill_paths],
            )
            if config.skills_enabled
            else SkillCatalog()
        )

        compaction_hook: CompactionHook | None = None
        if config.compaction_enabled:
            compaction_hook = CompactionHook(
                llm=llm,
                threshold_input_tokens=config.compaction_threshold_tokens,
                keep_last_turns=config.compaction_keep_last_turns,
            )
        attachment_manager = _build_attachments(config, resolved_workdir)

        return cls(
            llm,
            workdir=resolved_workdir,
            sandbox=sandbox,
            mcp_manager=mcp_manager,
            skill_catalog=skill_catalog,
            bash_timeout_s=bash_timeout_s,
            run_python_timeout_s=run_python_timeout_s,
            max_iterations=max_iterations,
            on_event=on_event,
            memory_store=memory_store,
            memory_retriever=memory_retriever,
            memory_extractor=memory_extractor,
            attachment_manager=attachment_manager,
            ask_callback=ask_callback,
            permission_hook=permission_hook,
            permission_policy=permission_policy,
            permission_mode=permission_mode,
            allow_rules=allow_rules,
            on_permission_request=on_permission_request,
            compaction_hook=compaction_hook,
        )

    # ----- Convenience --------------------------------------------------- #

    def ask(
        self,
        task: str,
        *,
        attachments: list[AttachmentInput] | None = None,
    ) -> AgentRun:
        return self.run(task, attachments=attachments)


def _build_memory(
    config: Config,
    llm: LLMClient,
    *,
    workdir: Path,
    on_memory_event: ExtractionCallback | None = None,
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
    vector_store = None
    if config.vector_backend != "file":
        try:
            vector_store = create_vector_store(
                config.vector_backend,
                dimensions=embedder.dimensions,
                uri=config.milvus_uri,
                token=config.milvus_token,
                collection=config.milvus_collection,
            )
        except Exception as exc:
            print(
                f"warning: falling back to file vector store — could not build "
                f"{config.vector_backend!r} backend: {exc}",
                file=sys.stderr,
            )
    store = MemoryStore(workdir=workdir, embedder=embedder, vector_store=vector_store)
    retriever = MemoryRetriever(store=store, k=config.memory_top_k)
    extractor = MemoryExtractor(
        llm=llm,
        store=store,
        workdir=workdir,
        on_complete=on_memory_event,
    )
    return (store, retriever, extractor)


def _build_attachments(config: Config, workdir: Path) -> AttachmentManager | None:
    if not config.attachments_enabled:
        return None
    root = Path(config.attachments_dir).expanduser()
    if not root.is_absolute():
        root = workdir / root
    policy = AttachmentPolicy(
        max_attachment_bytes=config.max_attachment_bytes,
        max_attachments_per_turn=config.max_attachments_per_turn,
        image_mode=config.attachment_image_mode,
    )
    return AttachmentManager(AttachmentStore(root), policy)


def _with_skill_catalog(system_prompt: str, catalog: SkillCatalog) -> str:
    section = catalog.prompt_section()
    if not section:
        return system_prompt
    return f"{system_prompt}\n\n---\n{section}"


def _resolve_permissions(
    *,
    permission_policy: PermissionPolicy | None,
    permission_hook: PreToolUseHook | None,
    permission_mode: PermissionMode | str | None,
    allow_rules: list | tuple | None,
    on_permission_request: PermissionCallback | None,
) -> PermissionPolicy | PreToolUseHook | None:
    """Reconcile the policy-related kwargs.

    Three input shapes are accepted; mixing them is an error so the
    caller can't silently override one with another:

      * Raw ``permission_hook`` — kept for back-compat with code that
        passed a hand-rolled ``PreToolUseHook``.
      * Explicit ``permission_policy`` — full control.
      * Convenience kwargs (``permission_mode`` / ``allow_rules`` /
        ``on_permission_request``) — built into a policy here.

    Returns the policy when a policy is in play, otherwise the raw hook
    (or ``None``). Callers use ``isinstance(result, PermissionPolicy)``
    to expose the policy on the agent for runtime mutation.
    """
    convenience_used = (
        permission_mode is not None
        or allow_rules is not None
        or on_permission_request is not None
    )
    sources = [permission_hook is not None, permission_policy is not None, convenience_used]
    if sum(sources) > 1:
        raise ConfigError(
            "Pass at most one of permission_hook=, permission_policy=, or the "
            "convenience kwargs (permission_mode/allow_rules/on_permission_request)."
        )
    if permission_policy is not None:
        return permission_policy
    if convenience_used:
        return PermissionPolicy.build(
            mode=permission_mode or PermissionMode.ALLOW_ALL,
            allow_rules=allow_rules or (),
            on_request=on_permission_request,
        )
    if permission_hook is not None:
        return permission_hook
    # No caller input: provide a default ALLOW_ALL policy so
    # ``agent.permissions`` is always available for runtime mutation.
    return PermissionPolicy()


def _wrap_memory_extractor(extractor: MemoryExtractor):
    """Adapt the trace-based MemoryExtractor.extract() to the hook signature."""

    def hook(ctx: HookContext) -> None:
        if ctx.run is not None and ctx.run.cancelled:
            return
        extractor.extract(ctx.history)

    return hook


def _sandbox_options(config: Config, kind: str) -> dict[str, object]:
    """Compose kwargs for ``create_sandbox`` for ``kind``.

    Free-form ``sandbox_options`` win on conflict so users can override
    the legacy ``sandbox_image`` field per-invocation. ``image`` is added
    only for the docker backend; plugin backends declare their own knobs
    via ``sandbox_options``.
    """
    opts: dict[str, object] = dict(config.sandbox_options)
    if kind == "docker":
        opts.setdefault("image", config.sandbox_image)
        opts.setdefault("persist", config.sandbox_persist)
        if config.sandbox_container_name:
            opts.setdefault("container_name", config.sandbox_container_name)
    return opts


def _init_sandbox(config: Config) -> Sandbox | None:
    """Build the primary sandbox, falling back per ``config.sandbox_fallback``.

    Returns the working `Sandbox`, or ``None`` if neither the primary nor
    the fallback could initialize. Emits a single concise notice on
    successful fallback; the louder warning fires only when no sandbox is
    usable.
    """
    primary = config.sandbox
    if primary == "none":
        return None

    try:
        return create_sandbox(primary, **_sandbox_options(config, primary))
    except SandboxError as primary_exc:
        fallback = config.sandbox_fallback
        if not fallback or fallback in {"none", primary}:
            _warn_no_sandbox(primary, primary_exc, fallback_tried=None)
            return None
        try:
            sb = create_sandbox(fallback, **_sandbox_options(config, fallback))
        except SandboxError as fallback_exc:
            _warn_no_sandbox(primary, primary_exc, fallback_tried=(fallback, fallback_exc))
            return None
        print(
            f"notice: sandbox {primary!r} unavailable — falling back to "
            f"{fallback!r}. Set TERNO_SANDBOX={fallback} (or "
            "TERNO_SANDBOX_FALLBACK=none) to silence this notice.",
            file=sys.stderr,
        )
        return sb


def _warn_no_sandbox(
    primary: str,
    primary_exc: SandboxError,
    *,
    fallback_tried: tuple[str, SandboxError] | None,
) -> None:
    detail = f"sandbox {primary!r} unavailable ({primary_exc})"
    if fallback_tried is not None:
        fb_name, fb_exc = fallback_tried
        detail += f"; fallback {fb_name!r} also failed ({fb_exc})"
    print(
        f"warning: {detail}. run_python tool will be disabled. "
        "Set TERNO_SANDBOX=none to silence this warning.",
        file=sys.stderr,
    )
