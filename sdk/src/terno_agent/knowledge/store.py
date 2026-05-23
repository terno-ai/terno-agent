"""Persistence sinks for knowledge artifacts.

The spec routes each task's output to one of a small set of stores:

    org_prompt          set/append the organization prompt text
    dbi_models          upsert table/column/relationship metadata
    column_descriptions LLM-drafted or user-edited descriptions
    embeddings          vectors for low-cardinality column values
    examples            validated (question, sql, output) pairs

`KnowledgeStore` is a `Protocol`; concrete backends (Postgres, vector
DBs, etc.) plug in here. `InMemoryStore` is a reference impl used by
tests and dev runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class KnowledgeStore(Protocol):
    def write_org_prompt(self, text: str) -> None: ...
    def append_org_prompt(self, text: str) -> None: ...
    def upsert_dbi_model(self, key: str, payload: dict[str, Any]) -> None: ...
    def write_description(self, target: str, description: str) -> None: ...
    def write_embedding(self, key: str, value: str, vector: list[float]) -> None: ...
    def write_example(self, question: str, sql: str, output: str) -> None: ...


@dataclass(slots=True)
class InMemoryStore:
    """Reference store backed by dicts/lists. Useful for dev and tests."""

    org_prompt_chunks: list[str] = field(default_factory=list)
    dbi_models: dict[str, dict[str, Any]] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
    embeddings: list[dict[str, Any]] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def write_org_prompt(self, text: str) -> None:
        self.org_prompt_chunks = [text]

    def append_org_prompt(self, text: str) -> None:
        self.org_prompt_chunks.append(text)

    def upsert_dbi_model(self, key: str, payload: dict[str, Any]) -> None:
        existing = self.dbi_models.get(key, {})
        self.dbi_models[key] = {**existing, **payload}

    def write_description(self, target: str, description: str) -> None:
        self.descriptions[target] = description

    def write_embedding(self, key: str, value: str, vector: list[float]) -> None:
        self.embeddings.append({"key": key, "value": value, "vector": vector})

    def write_example(self, question: str, sql: str, output: str) -> None:
        self.examples.append({"question": question, "sql": sql, "output": output})

    @property
    def org_prompt(self) -> str:
        return "\n\n".join(self.org_prompt_chunks)


__all__ = ["InMemoryStore", "KnowledgeStore"]
