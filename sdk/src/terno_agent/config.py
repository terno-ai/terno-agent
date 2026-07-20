"""Runtime configuration.

Precedence (highest first):
1. Explicit kwargs to `Config(...)`.
2. Process environment variables (TERNO_*, plus ANTHROPIC_API_KEY / OPENAI_API_KEY).
3. Variables loaded from a `.env` file in the current working directory or any
   parent directory (via python-dotenv). Existing process env wins over .env.
4. Defaults below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from terno_agent.core.exceptions import ConfigError


@cache
def _default_env_path() -> Path | None:
    """Search CWD and parents for a ``.env`` file (cached per process)."""
    found = find_dotenv(usecwd=True)
    return Path(found) if found else None


def load_env(
    path: str | os.PathLike[str] | None = None,
    *,
    override: bool = False,
) -> Path | None:
    """Load environment variables from a ``.env`` file.

    If ``path`` is omitted, search CWD and its parents. Returns the path that
    was loaded, or ``None`` if no file was found.

    Calling this repeatedly is safe: ``python-dotenv`` does not overwrite
    existing environment variables unless ``override=True`` is passed, and the
    parent-directory search is cached.
    """
    resolved = Path(path) if path is not None else _default_env_path()
    if resolved is None or not resolved.exists():
        return None
    load_dotenv(resolved, override=override)
    return resolved


DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-7",
    "openai": "gpt-4o",
    "terno": "gpt-5.5",
}


@dataclass(slots=True)
class Config:
    llm_provider: str = "anthropic"
    llm_model: str = ""
    llm_api_key: str | None = None
    # ----- provisioner proxy (llm_provider="terno") ------------------------ #
    provisioner_url: str = ""
    app_version: str = ""
    request_source: str = ""
    database_url: str = ""
    sandbox: str = "docker"  # built-ins: docker | local | none; plus any plugin name
    sandbox_image: str = "python:3.12-slim"
    sandbox_options: dict[str, str] = field(default_factory=dict)
    # Backend to try when the primary sandbox fails to initialize. Empty
    # string or "none" disables fallback (the old strict behavior).
    sandbox_fallback: str = "local"
    # When True, the sandbox container is given a stable name and left
    # running after this session ends so the next session can attach to
    # it. Otherwise the container is killed + removed on Agent.close().
    sandbox_persist: bool = False
    # Override the auto-derived per-cwd container name. Only honored when
    # sandbox_persist is True.
    sandbox_container_name: str = ""
    max_rows: int = 200
    read_only_sql: bool = True
    mcp_enabled: bool = True
    mcp_config_path: str = ""
    # In-memory MCP server definitions, same shape as a `.mcp.json`
    # file (``{"mcpServers": {...}}``). When set, this takes precedence
    # over ``mcp_config_path`` so callers can inject server configs
    # without writing credentials to disk.
    mcp_servers: dict | None = None
    # ----- agent skills ---------------------------------------------------- #
    skills_enabled: bool = True
    skill_paths: list[str] = field(default_factory=list)
    # Whether discovery includes the SDK's own bundled builtin skills
    # (skills/builtin/*) and the user-home roots (~/.terno/skills etc).
    # A host embedding the SDK with its own curated skill set (via
    # skill_paths) can set these False to get *only* its own skills, without
    # the SDK's generic builtins mixed in.
    skill_include_builtin: bool = True
    skill_include_user: bool = True
    # ----- memory ---------------------------------------------------------- #
    memory_enabled: bool = True
    memory_top_k: int = 5
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str | None = None
    vector_backend: str = "file"
    milvus_uri: str = ""
    milvus_token: str | None = None
    milvus_collection: str = "terno_memory"
    # File-based (terno-ai style) memory: the agent reads/writes MEMORY.md +
    # per-fact files with the ordinary file tools. Independent of the RAG
    # ``memory_enabled`` subsystem above and of the shelved wiki curator.
    file_memory_enabled: bool = True
    # ----- workspace memory location + org-admin gate ---------------------- #
    # Already-resolved, already-authorized memory folders. The caller (the app
    # server, which authenticated the user) computes these — e.g. via
    # ``wiki.paths.user_memory_dir`` / ``org_memory_dir`` — and passes them in.
    # The SDK never handles a raw username, so it can't be tricked into
    # touching another user's folder. When unset the SDK falls back to the
    # local workdir's ``memory`` folder.
    user_memory_root: str = ""
    org_memory_root: str = ""
    # Only org admins may write to the org-shared memory folder. The caller
    # sets this from the authenticated user's group membership.
    is_org_admin: bool = False
    # Stamped into memory frontmatter as ``originSessionId``.
    session_id: str = ""
    # ----- compaction ------------------------------------------------------ #
    compaction_enabled: bool = True
    compaction_threshold_tokens: int = 80_000
    compaction_keep_last_turns: int = 4
    # ----- attachments ------------------------------------------------------ #
    attachments_enabled: bool = True
    attachments_dir: str = ".terno/attachments"
    max_attachment_bytes: int = 512 * 1024 * 1024
    max_attachments_per_turn: int = 8
    attachment_image_mode: str = "auto"
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.llm_model:
            self.llm_model = DEFAULT_MODELS.get(self.llm_provider, "")
        # Sandbox kind is validated by the sandbox registry at construction
        # time so plugin names work without core-side allowlisting.
        if self.attachment_image_mode not in {"auto", "native", "metadata"}:
            raise ConfigError(
                "Invalid attachment_image_mode "
                f"{self.attachment_image_mode!r}. Must be one of: auto, native, metadata."
            )

    @classmethod
    def from_env(cls) -> Config:
        load_env()
        provider = os.getenv("TERNO_LLM_PROVIDER", "anthropic").lower()
        model = os.getenv("TERNO_LLM_MODEL") or DEFAULT_MODELS.get(provider, "")
        api_key = os.getenv("TERNO_LLM_API_KEY")
        if not api_key:
            if provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
        embedding_api_key = os.getenv("TERNO_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        memory_enabled_raw = os.getenv("TERNO_MEMORY_ENABLED", "true").lower()
        file_memory_enabled_raw = os.getenv("TERNO_FILE_MEMORY_ENABLED", "true").lower()
        skills_enabled_raw = os.getenv("TERNO_SKILLS_ENABLED", "true").lower()
        skill_include_builtin_raw = os.getenv("TERNO_SKILL_INCLUDE_BUILTIN", "true").lower()
        skill_include_user_raw = os.getenv("TERNO_SKILL_INCLUDE_USER", "true").lower()
        attachments_enabled_raw = os.getenv("TERNO_ATTACHMENTS_ENABLED", "true").lower()
        skill_paths_raw = os.getenv("TERNO_SKILL_PATHS", "")
        return cls(
            llm_provider=provider,
            llm_model=model,
            llm_api_key=api_key,
            provisioner_url=os.getenv("PROVISIONER_URL", "").strip(),
            app_version=os.getenv("APP_VERSION", "").strip(),
            request_source=os.getenv("REQUEST_SOURCE", "").strip(),
            database_url=os.getenv("TERNO_DATABASE_URL", ""),
            sandbox=_normalize_sandbox(os.getenv("TERNO_SANDBOX", "docker")),
            sandbox_image=os.getenv("TERNO_SANDBOX_IMAGE", "python:3.12-slim"),
            sandbox_options=_parse_sandbox_options(os.getenv("TERNO_SANDBOX_OPTIONS", "")),
            sandbox_fallback=_normalize_sandbox(os.getenv("TERNO_SANDBOX_FALLBACK", "local")),
            sandbox_persist=os.getenv("TERNO_SANDBOX_PERSIST", "false").lower()
            not in {"false", "0", "no", "off", ""},
            sandbox_container_name=os.getenv("TERNO_SANDBOX_CONTAINER_NAME", "").strip(),
            max_rows=int(os.getenv("TERNO_MAX_ROWS", "200")),
            read_only_sql=os.getenv("TERNO_READ_ONLY_SQL", "true").lower() != "false",
            mcp_enabled=os.getenv("TERNO_MCP_ENABLED", "true").lower() != "false",
            mcp_config_path=os.getenv("TERNO_MCP_CONFIG", ""),
            skills_enabled=skills_enabled_raw not in {"false", "0", "no", "off"},
            skill_paths=[
                path.strip()
                for path in skill_paths_raw.split(os.pathsep)
                if path.strip()
            ],
            skill_include_builtin=skill_include_builtin_raw
            not in {"false", "0", "no", "off"},
            skill_include_user=skill_include_user_raw
            not in {"false", "0", "no", "off"},
            memory_enabled=memory_enabled_raw not in {"false", "0", "no", "off"},
            file_memory_enabled=file_memory_enabled_raw
            not in {"false", "0", "no", "off"},
            user_memory_root=os.getenv("TERNO_USER_MEMORY_ROOT", "").strip(),
            org_memory_root=os.getenv("TERNO_ORG_MEMORY_ROOT", "").strip(),
            is_org_admin=os.getenv("TERNO_IS_ORG_ADMIN", "false").lower()
            not in {"false", "0", "no", "off", ""},
            session_id=os.getenv("TERNO_SESSION_ID", "").strip(),
            memory_top_k=int(os.getenv("TERNO_MEMORY_TOP_K", "5")),
            embedding_provider=os.getenv("TERNO_EMBEDDING_PROVIDER", "openai").lower(),
            embedding_model=os.getenv("TERNO_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_api_key=embedding_api_key,
            vector_backend=os.getenv("TERNO_VECTOR_BACKEND", "file").lower(),
            milvus_uri=os.getenv("TERNO_MILVUS_URI", ""),
            milvus_token=os.getenv("TERNO_MILVUS_TOKEN") or None,
            milvus_collection=os.getenv("TERNO_MILVUS_COLLECTION", "terno_memory"),
            compaction_enabled=os.getenv("TERNO_COMPACTION_ENABLED", "true").lower()
            not in {"false", "0", "no", "off"},
            compaction_threshold_tokens=int(
                os.getenv("TERNO_COMPACTION_THRESHOLD_TOKENS", "80000")
            ),
            compaction_keep_last_turns=int(
                os.getenv("TERNO_COMPACTION_KEEP_LAST_TURNS", "4")
            ),
            attachments_enabled=attachments_enabled_raw not in {"false", "0", "no", "off"},
            attachments_dir=os.getenv("TERNO_ATTACHMENTS_DIR", ".terno/attachments"),
            max_attachment_bytes=int(
                os.getenv("TERNO_MAX_ATTACHMENT_BYTES", str(512 * 1024 * 1024))
            ),
            max_attachments_per_turn=int(os.getenv("TERNO_MAX_ATTACHMENTS_PER_TURN", "8")),
            attachment_image_mode=os.getenv("TERNO_ATTACHMENT_IMAGE_MODE", "auto").lower(),
        )

    @classmethod
    def for_benchmark(
        cls,
        base: Config | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        sandbox: str = "none",
        sandbox_fallback: str = "none",
        memory_enabled: bool = False,
        mcp_enabled: bool = False,
        attachments_enabled: bool = False,
        skills_enabled: bool = False,
    ) -> Config:
        """Return a config with benchmark-host side effects disabled.

        Benchmark harnesses usually provide their own task isolation and
        lifecycle. This helper keeps provider credentials and model settings
        from ``base``/env, but disables persistent memory, MCP servers,
        attachment storage, project/user skills, and Terno's internal sandbox
        unless the caller explicitly opts back in.
        """
        cfg = replace(base) if base is not None else cls.from_env()
        previous_provider = cfg.llm_provider
        if provider is not None:
            cfg.llm_provider = provider
        if model is not None:
            cfg.llm_model = model
        elif provider is not None and cfg.llm_model == DEFAULT_MODELS.get(previous_provider, ""):
            cfg.llm_model = DEFAULT_MODELS.get(cfg.llm_provider, "")
        elif not cfg.llm_model:
            cfg.llm_model = DEFAULT_MODELS.get(cfg.llm_provider, "")
        if api_key is not None:
            cfg.llm_api_key = api_key

        cfg.sandbox = _normalize_sandbox(sandbox)
        cfg.sandbox_fallback = _normalize_sandbox(sandbox_fallback)
        cfg.memory_enabled = memory_enabled
        cfg.mcp_enabled = mcp_enabled
        cfg.attachments_enabled = attachments_enabled
        cfg.skills_enabled = skills_enabled
        return cfg

    def display(self) -> str:
        masked = "***" if self.llm_api_key else "(unset)"
        embedding_masked = "***" if self.embedding_api_key else "(unset)"
        return (
            f"llm_provider       = {self.llm_provider}\n"
            f"llm_model          = {self.llm_model}\n"
            f"llm_api_key        = {masked}\n"
            f"provisioner_url    = {self.provisioner_url or '(unset)'}\n"
            f"database_url       = {self.database_url or '(unset)'}\n"
            f"sandbox            = {self.sandbox}\n"
            f"sandbox_image      = {self.sandbox_image}\n"
            f"sandbox_options    = "
            f"{', '.join(f'{k}={v}' for k, v in self.sandbox_options.items()) or '(none)'}\n"
            f"sandbox_fallback   = {self.sandbox_fallback or '(disabled)'}\n"
            f"sandbox_persist    = {self.sandbox_persist}\n"
            f"sandbox_container_name = {self.sandbox_container_name or '(auto-derived per cwd)'}\n"
            f"max_rows           = {self.max_rows}\n"
            f"read_only_sql      = {self.read_only_sql}\n"
            f"mcp_enabled        = {self.mcp_enabled}\n"
            f"mcp_config_path    = {self.mcp_config_path or '(auto-discover)'}\n"
            f"skills_enabled     = {self.skills_enabled}\n"
            f"skill_paths        = {os.pathsep.join(self.skill_paths) or '(auto-discover)'}\n"
            f"skill_include_builtin = {self.skill_include_builtin}\n"
            f"skill_include_user = {self.skill_include_user}\n"
            f"memory_enabled     = {self.memory_enabled}\n"
            f"memory_top_k       = {self.memory_top_k}\n"
            f"embedding_provider = {self.embedding_provider}\n"
            f"embedding_model    = {self.embedding_model}\n"
            f"embedding_api_key  = {embedding_masked}\n"
            f"vector_backend     = {self.vector_backend}\n"
            f"milvus_uri         = {self.milvus_uri or '(unset)'}\n"
            f"milvus_collection  = {self.milvus_collection}\n"
            f"compaction_enabled = {self.compaction_enabled}\n"
            f"compaction_threshold_tokens = {self.compaction_threshold_tokens}\n"
            f"compaction_keep_last_turns  = {self.compaction_keep_last_turns}\n"
            f"attachments_enabled = {self.attachments_enabled}\n"
            f"attachments_dir     = {self.attachments_dir}\n"
            f"max_attachment_bytes = {self.max_attachment_bytes}\n"
            f"max_attachments_per_turn = {self.max_attachments_per_turn}\n"
            f"attachment_image_mode = {self.attachment_image_mode}\n"
        )


def _normalize_sandbox(raw: str) -> str:
    """Lowercase a registry-style sandbox name but keep import strings intact.

    ``"module.Path:Class"`` is case-sensitive (Python attribute lookup), so
    only registered short names go through ``str.lower()``.
    """
    raw = raw.strip()
    if ":" in raw:
        return raw
    return raw.lower()


def _parse_sandbox_options(raw: str) -> dict[str, str]:
    """Parse ``key=val,key=val`` env strings into a dict.

    Whitespace around keys/values is stripped. Empty entries are
    skipped. Entries without ``=`` raise `ConfigError`.
    """
    out: dict[str, str] = {}
    if not raw:
        return out
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ConfigError(
                f"TERNO_SANDBOX_OPTIONS entry {piece!r} must be 'key=value'"
            )
        key, _, value = piece.partition("=")
        out[key.strip()] = value.strip()
    return out
