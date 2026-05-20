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
from terno_agent.core.hooks import HookContext, HookEvent, HookManager
from terno_agent.core.messages import ContentPart
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
from terno_agent.skills import ActivateSkillTool, SkillCatalog, discover_skills
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
        skill_catalog: SkillCatalog | None = None,
        bash_timeout_s: int = 120,
        run_python_timeout_s: int = 30,
        on_event=None,
        memory_store: MemoryStore | None = None,
        memory_retriever: MemoryRetriever | None = None,
        memory_extractor: MemoryExtractor | None = None,
        attachment_manager: AttachmentManager | None = None,
        cancel_token: CancelToken | None = None,
        hook_manager: HookManager | None = None,
        compaction_hook: CompactionHook | None = None,
    ) -> None:
        self.workdir = (workdir or Path.cwd()).resolve()
        self.task_store = task_store or TaskStore()
        self.sandbox = sandbox
        self.mcp_manager = mcp_manager
        self.skill_catalog = skill_catalog or SkillCatalog()
        self.memory_store = memory_store
        self.memory_retriever = memory_retriever
        self.memory_extractor = memory_extractor
        self.attachment_manager = attachment_manager

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
        sandbox = _init_sandbox(config)

        mcp_manager: McpManager | None = None
        if config.mcp_enabled:
            mcp_manager = McpManager.start_from_path(config.mcp_config_path or None)
            if mcp_manager is not None:
                atexit.register(mcp_manager.shutdown)

        memory_store, memory_retriever, memory_extractor = _build_memory(
            config, llm, workdir=Path.cwd(), on_event=on_event
        )
        skill_catalog = (
            discover_skills(
                Path.cwd(),
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
        attachment_manager = _build_attachments(config, Path.cwd())

        return cls(
            llm,
            sandbox=sandbox,
            mcp_manager=mcp_manager,
            skill_catalog=skill_catalog,
            on_event=on_event,
            memory_store=memory_store,
            memory_retriever=memory_retriever,
            memory_extractor=memory_extractor,
            attachment_manager=attachment_manager,
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
