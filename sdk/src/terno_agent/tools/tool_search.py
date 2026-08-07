"""Deferred tools: advertise names, load schemas on demand.

The reference harness sends a small eager tool set plus a roster of *deferred*
tool names, and `ToolSearch` fetches a deferred tool's real schema when it's
needed. That keeps the per-request tool payload small when the surface is large.

Two behaviours verified against the capture, both easy to get wrong:

* **The roster is static.** It is written once, immediately after the first user
  message, and never rewritten. A loaded tool appears in BOTH the roster and the
  top-level `tools` array from then on — the roster is not filtered as tools load.
* **Loading is additive and permanent** for the session: once a schema has been
  returned it stays in `tools` for every later request.

`DeferredToolPlaceholder` exists because a tools array that would otherwise be
empty of deferred machinery has to keep it "active"; it is never callable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import Tool, ToolSchema

ROSTER_HEADER = (
    "The following deferred tools are now available via ToolSearch. Their "
    "schemas are NOT loaded — calling them directly will fail with "
    'InputValidationError. Use ToolSearch with query "select:<name>[,<name>...]" '
    "to load tool schemas before calling them:"
)


def roster_text(names: list[str]) -> str:
    """The mid-conversation roster: header, then one name per line, sorted."""
    return "\n".join([ROSTER_HEADER, *sorted(names)])


def _schema_json(schema: ToolSchema) -> str:
    """One `<function>` payload, matching the encoding of the eager tool list."""
    return json.dumps(
        {
            "description": schema.description,
            "name": schema.name,
            "parameters": schema.parameters,
        }
    )


def render_functions_block(schemas: list[ToolSchema]) -> str:
    lines = "\n".join(f"<function>{_schema_json(s)}</function>" for s in schemas)
    return f"<functions>\n{lines}\n</functions>"


@dataclass
class DeferredToolRegistry:
    """Holds deferred tools and tracks which have been loaded this session."""

    deferred: dict[str, Tool] = field(default_factory=dict)
    # The live tool map the agent dispatches on; loading inserts into it so the
    # tool is callable on the very next iteration.
    active: dict[str, Tool] | None = None
    loaded: set[str] = field(default_factory=set)

    @property
    def names(self) -> list[str]:
        """Every deferred name — including loaded ones, since the roster is static."""
        return sorted(self.deferred)

    def load(self, name: str) -> ToolSchema | None:
        tool = self.deferred.get(name)
        if tool is None:
            return None
        self.loaded.add(name)
        if self.active is not None:
            self.active[name] = tool
        return tool.schema

    def search(self, query: str, max_results: int) -> list[ToolSchema]:
        """Resolve a query to schemas, marking each match loaded.

        Query forms, per the captured description:
          `select:A,B`  exact names
          `+kw rest`    require `kw` in the name, rank by the remaining terms
          `kw1 kw2`     keyword search over name and description
        """
        query = (query or "").strip()
        if not query:
            raise ToolError("ToolSearch requires a non-empty 'query'.")
        limit = max(1, int(max_results or 5))

        if query.lower().startswith("select:"):
            # Exact selection ignores max_results: the caller named what it wants.
            wanted = [n.strip() for n in query[len("select:") :].split(",")]
            return [s for s in (self.load(n) for n in wanted if n) if s is not None]

        terms = query.lower().split()
        required = [t[1:] for t in terms if t.startswith("+") and len(t) > 1]
        rest = [t for t in terms if not t.startswith("+")]

        scored: list[tuple[int, str]] = []
        for name in self.names:
            lname = name.lower()
            if any(req not in lname for req in required):
                continue
            haystack = f"{lname} {self.deferred[name].schema.description.lower()}"
            score = sum(3 if t in lname else 1 for t in rest if t in haystack)
            if required and not rest:
                score = 1  # a bare `+kw` query matches on the requirement alone
            if score:
                scored.append((score, name))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        out = [self.load(name) for _score, name in scored[:limit]]
        return [s for s in out if s is not None]


@dataclass
class ToolSearchTool:
    registry: DeferredToolRegistry

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="ToolSearch",
            description=(
                "Fetches full schema definitions for deferred tools so they can"
                " be called.\n"
                "\n"
                "Deferred tools appear by name in a mid-conversation system"
                " message. Until fetched, only the name is known — there is no"
                " parameter schema, so the tool cannot be invoked. This tool"
                " takes a query, matches it against the deferred tool list, and"
                " returns the matched tools' complete JSONSchema definitions"
                " inside a <functions> block. Once a tool's schema appears in"
                " that result, it is callable exactly like any tool defined at"
                " the top of the prompt.\n"
                "\n"
                "Result format: each matched tool appears as one"
                ' <function>{"description": "...", "name": "...", "parameters":'
                " {...}}</function> line inside the <functions> block — the same"
                " encoding as the tool list at the top of this prompt.\n"
                "\n"
                "Query forms:\n"
                '- "select:Read,Edit,Grep" — fetch these exact tools by name\n'
                '- "notebook jupyter" — keyword search, up to max_results best'
                " matches\n"
                '- "+slack send" — require "slack" in the name, rank by remaining'
                " terms"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            'Query to find deferred tools. Use "select:<tool_name>"'
                            " for direct selection, or keywords to search."
                        ),
                    },
                    "max_results": {
                        "type": "number",
                        "default": 5,
                        "description": "Maximum number of results to return (default: 5)",
                    },
                },
                "required": ["query", "max_results"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        schemas = self.registry.search(
            str(kwargs.get("query") or ""), int(kwargs.get("max_results") or 5)
        )
        if not schemas:
            available = ", ".join(self.registry.names) or "(none)"
            return (
                "No deferred tools matched. Available deferred tools: "
                f"{available}"
            )
        return render_functions_block(schemas)


@dataclass
class DeferredToolPlaceholderTool:
    """Never callable; its presence is what keeps deferred loading active."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="DeferredToolPlaceholder",
            description=(
                "Reserved placeholder that keeps deferred tool loading active;"
                " never call this tool."
            ),
            parameters={"type": "object", "properties": {}},
        )

    def run(self, **_kwargs: Any) -> str:
        raise ToolError(
            "DeferredToolPlaceholder is not callable. Use ToolSearch to load a "
            "deferred tool's schema first."
        )


__all__ = [
    "ROSTER_HEADER",
    "DeferredToolPlaceholderTool",
    "DeferredToolRegistry",
    "ToolSearchTool",
    "render_functions_block",
    "roster_text",
]
