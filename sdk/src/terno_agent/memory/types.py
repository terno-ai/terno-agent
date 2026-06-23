"""Memory entry types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryType(str, Enum):
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"
    INSIGHT = "insight"


@dataclass(slots=True)
class MemoryEntry:
    """A single memory: one markdown file on disk."""

    name: str
    description: str
    type: MemoryType
    body: str

    def embedding_text(self) -> str:
        """Text used as the embedding payload — title + body."""
        return f"{self.name}\n{self.description}\n\n{self.body}".strip()


__all__ = [
    "MemoryEntry",
    "MemoryType",
]
