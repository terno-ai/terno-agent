"""Memory tools exposed to agents.

* The **main** agent gets ``search_memory`` (read-only RAG).
* The **extractor** subagent gets the full CRUD set
  (``list_memories``, ``read_memory``, ``save_memory``, ``delete_memory``)
  so it can curate the store after each turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.memory.store import MemoryStore
from terno_agent.memory.types import MemoryEntry, MemoryType


@dataclass
class SearchMemoryTool:
    store: MemoryStore
    default_k: int = 5

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_memory",
            description=(
                "Search persistent memory for entries relevant to a query. "
                "Returns up to k hits with their type, description, and a "
                "snippet, ranked by semantic similarity. Use this when the "
                "user references prior conversations or you want to recall "
                "their preferences/role."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language query.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Max number of hits to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        query = (kwargs.get("query") or "").strip()
        if not query:
            raise ToolError("search_memory requires a non-empty 'query'.")
        k = int(kwargs.get("k") or self.default_k)
        hits = self.store.search(query, k=k)
        out = [
            {
                "name": h.key,
                "type": h.metadata.get("type"),
                "scope": h.metadata.get("scope"),
                "description": h.metadata.get("description"),
                "score": round(h.score, 4),
                "snippet": h.text[:400],
            }
            for h in hits
        ]
        return json.dumps(out)


@dataclass
class ListMemoriesTool:
    store: MemoryStore

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_memories",
            description=(
                "List all saved memory entries. Optionally filter by type "
                "(user|feedback|project|reference). Returns a JSON array of "
                "{name, type, description}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [t.value for t in MemoryType],
                        "description": "Optional filter by memory type.",
                    }
                },
                "required": [],
            },
        )

    def run(self, **kwargs: Any) -> str:
        type_str = kwargs.get("type")
        type_: MemoryType | None = None
        if type_str:
            try:
                type_ = MemoryType(type_str)
            except ValueError as exc:
                raise ToolError(
                    f"Invalid type {type_str!r}. Must be one of: "
                    f"{', '.join(t.value for t in MemoryType)}."
                ) from exc
        entries = self.store.list_all(type_)
        return json.dumps(
            [
                {
                    "name": e.name,
                    "type": e.type.value,
                    "description": e.description,
                }
                for e in entries
            ]
        )


@dataclass
class ReadMemoryTool:
    store: MemoryStore

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_memory",
            description=(
                "Read a single memory entry by name. Returns "
                "{name, type, description, body} as JSON, or an error if "
                "not found."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Memory name (kebab-case slug).",
                    }
                },
                "required": ["name"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        name = (kwargs.get("name") or "").strip()
        if not name:
            raise ToolError("read_memory requires a 'name'.")
        entry = self.store.read(name)
        if entry is None:
            raise ToolError(f"No memory named {name!r}.")
        return json.dumps(
            {
                "name": entry.name,
                "type": entry.type.value,
                "description": entry.description,
                "body": entry.body,
            }
        )


@dataclass
class SaveMemoryTool:
    store: MemoryStore

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="save_memory",
            description=(
                "Create or update a memory entry. Calling with an existing "
                "'name' overwrites that entry (use this to update). Returns "
                "the path written."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Kebab-case slug under 64 chars.",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-line summary for the index.",
                    },
                    "type": {
                        "type": "string",
                        "enum": [t.value for t in MemoryType],
                        "description": "Memory type.",
                    },
                    "body": {
                        "type": "string",
                        "description": (
                            "Body content. For feedback/project, include "
                            "**Why:** and **How to apply:** lines."
                        ),
                    },
                },
                "required": ["name", "description", "type", "body"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        name = (kwargs.get("name") or "").strip()
        description = (kwargs.get("description") or "").strip()
        type_str = (kwargs.get("type") or "").strip()
        body = (kwargs.get("body") or "").strip()
        if not (name and description and type_str and body):
            raise ToolError(
                "save_memory requires non-empty 'name', 'description', 'type', and 'body'."
            )
        try:
            type_ = MemoryType(type_str)
        except ValueError as exc:
            raise ToolError(
                f"Invalid type {type_str!r}. Must be one of: "
                f"{', '.join(t.value for t in MemoryType)}."
            ) from exc
        entry = MemoryEntry(name=name, description=description, type=type_, body=body)
        path = self.store.save(entry)
        return json.dumps({"name": entry.name, "path": str(path)})


@dataclass
class DeleteMemoryTool:
    store: MemoryStore

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="delete_memory",
            description=(
                "Delete a memory entry by name. Returns 'deleted' or 'not_found'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Memory name."}
                },
                "required": ["name"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        name = (kwargs.get("name") or "").strip()
        if not name:
            raise ToolError("delete_memory requires a 'name'.")
        deleted = self.store.delete(name)
        return "deleted" if deleted else "not_found"


def extractor_tools(store: MemoryStore) -> list:
    """Toolset given to the extractor subagent."""
    return [
        ListMemoriesTool(store),
        ReadMemoryTool(store),
        SaveMemoryTool(store),
        DeleteMemoryTool(store),
    ]


__all__ = [
    "DeleteMemoryTool",
    "ListMemoriesTool",
    "ReadMemoryTool",
    "SaveMemoryTool",
    "SearchMemoryTool",
    "extractor_tools",
]
