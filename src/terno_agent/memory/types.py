"""Memory entry types and scope rules (mirrors Claude Code)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MemoryType(StrEnum):
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


# Memories about the human follow the user across projects.
GLOBAL_TYPES = frozenset({MemoryType.USER, MemoryType.FEEDBACK})
# Memories about a specific codebase stay with that codebase.
WORKDIR_TYPES = frozenset({MemoryType.PROJECT, MemoryType.REFERENCE})


class MemoryScope(StrEnum):
    GLOBAL = "global"
    WORKDIR = "workdir"


def scope_for_type(type_: MemoryType) -> MemoryScope:
    if type_ in GLOBAL_TYPES:
        return MemoryScope.GLOBAL
    return MemoryScope.WORKDIR


@dataclass(slots=True)
class MemoryEntry:
    """A single memory: one markdown file on disk."""

    name: str
    description: str
    type: MemoryType
    body: str

    @property
    def scope(self) -> MemoryScope:
        return scope_for_type(self.type)

    def embedding_text(self) -> str:
        """Text used as the embedding payload — title + body."""
        return f"{self.name}\n{self.description}\n\n{self.body}".strip()


__all__ = [
    "GLOBAL_TYPES",
    "MemoryEntry",
    "MemoryScope",
    "MemoryType",
    "WORKDIR_TYPES",
    "scope_for_type",
]
