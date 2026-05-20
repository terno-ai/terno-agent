"""Resolve global vs. workdir memory directories."""

from __future__ import annotations

import os
from pathlib import Path

from terno_agent.memory.types import MemoryScope, MemoryType, scope_for_type

GLOBAL_ENV_VAR = "TERNO_MEMORY_HOME"


def global_memory_dir() -> Path:
    """Per-user memory dir.

    Overridable via ``TERNO_MEMORY_HOME`` (useful in tests). Defaults to
    ``~/.terno_agent/memory``.
    """
    override = os.getenv(GLOBAL_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".terno_agent" / "memory").resolve()


def workdir_memory_dir(workdir: Path) -> Path:
    """Per-project memory dir: ``<workdir>/.terno/memory``."""
    return (Path(workdir).resolve() / ".terno" / "memory").resolve()


def resolve_dir_for_type(type_: MemoryType, workdir: Path) -> Path:
    if scope_for_type(type_) is MemoryScope.GLOBAL:
        return global_memory_dir()
    return workdir_memory_dir(workdir)


__all__ = [
    "GLOBAL_ENV_VAR",
    "global_memory_dir",
    "resolve_dir_for_type",
    "workdir_memory_dir",
]
