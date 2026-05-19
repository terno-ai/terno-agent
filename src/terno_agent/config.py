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
from dataclasses import dataclass, field
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
}


@dataclass(slots=True)
class Config:
    llm_provider: str = "anthropic"
    llm_model: str = ""
    llm_api_key: str | None = None
    database_url: str = ""
    sandbox: str = "docker"  # docker | local | none
    sandbox_image: str = "python:3.12-slim"
    max_rows: int = 200
    read_only_sql: bool = True
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.llm_model:
            self.llm_model = DEFAULT_MODELS.get(self.llm_provider, "")
        if self.sandbox not in {"docker", "local", "none"}:
            raise ConfigError(
                f"Invalid sandbox {self.sandbox!r}. Must be one of: docker, local, none."
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
        return cls(
            llm_provider=provider,
            llm_model=model,
            llm_api_key=api_key,
            database_url=os.getenv("TERNO_DATABASE_URL", ""),
            sandbox=os.getenv("TERNO_SANDBOX", "docker").lower(),
            sandbox_image=os.getenv("TERNO_SANDBOX_IMAGE", "python:3.12-slim"),
            max_rows=int(os.getenv("TERNO_MAX_ROWS", "200")),
            read_only_sql=os.getenv("TERNO_READ_ONLY_SQL", "true").lower() != "false",
        )

    def display(self) -> str:
        masked = "***" if self.llm_api_key else "(unset)"
        return (
            f"llm_provider     = {self.llm_provider}\n"
            f"llm_model        = {self.llm_model}\n"
            f"llm_api_key      = {masked}\n"
            f"database_url     = {self.database_url or '(unset)'}\n"
            f"sandbox          = {self.sandbox}\n"
            f"sandbox_image    = {self.sandbox_image}\n"
            f"max_rows         = {self.max_rows}\n"
            f"read_only_sql    = {self.read_only_sql}\n"
        )
