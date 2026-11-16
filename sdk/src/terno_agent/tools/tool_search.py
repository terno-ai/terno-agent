"""Load deferred tools on demand.

Large tool sets (mostly MCP servers) cost thousands of input tokens per LLM
call because every schema is re-sent on every iteration. When deferral is on,
those tools start in ``BaseAgent.deferred_tools``: the model only sees their
names (in the system prompt) and calls ``tool_search`` to load the ones it
needs. Loading moves a tool into ``agent.tools``, so its full schema is sent
from the next LLM call onward.

Loaded tools are appended (never reordered) so the tool list only grows,
which keeps the request prefix as stable as possible for prompt caching.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import Tool, ToolSchema

if TYPE_CHECKING:
    from terno_agent.agents.base import BaseAgent

TOOL_SEARCH_NAME = "tool_search"
_DEFAULT_MAX_RESULTS = 5
_MAX_RESULTS_CAP = 20
_WORD = re.compile(r"[a-z0-9]{2,}")


def estimate_schema_tokens(tools: Iterable[Tool]) -> int:
    """Rough token count of the schemas, same estimate as the token log."""
    payload = [[t.schema.name, t.schema.description, t.schema.parameters] for t in tools]
    return len(json.dumps(payload, default=str)) // 4


def server_of(tool: Tool) -> str:
    """MCP server name for grouping; ``"other"`` for non-MCP tools."""
    server = getattr(tool, "server", None)
    if isinstance(server, str) and server:
        return server
    parts = tool.schema.name.split("__")
    if len(parts) >= 3 and parts[0] == "mcp":
        return parts[1]
    return "other"


def deferred_index(tools: Iterable[Tool]) -> str:
    """Names-only listing of deferred tools, grouped by server, for the
    system prompt. Empty string when there is nothing deferred."""
    groups: dict[str, list[str]] = {}
    for tool in tools:
        groups.setdefault(server_of(tool), []).append(tool.schema.name)
    if not groups:
        return ""
    lines = [
        "## Deferred tools",
        f"These tools exist but their schemas are not loaded. Before calling one, "
        f"load it with `{TOOL_SEARCH_NAME}` — either by exact name "
        f'(`query="select:name1,name2"`), a whole server (`query="server:github"`), '
        f"or by keywords describing the task. Load only what you need.",
    ]
    for server in sorted(groups):
        lines.append(f"- {server}: {', '.join(sorted(groups[server]))}")
    return "\n".join(lines)


def rank_tools(tools: Iterable[Tool], query: str) -> list[Tool]:
    """Keyword ranking: a term in the tool name counts 3, in the server 2,
    in the description 1. Tools with no matching term are dropped."""
    terms = set(_WORD.findall(query.lower()))
    if not terms:
        return []
    scored: list[tuple[int, str, Tool]] = []
    for tool in tools:
        name = tool.schema.name.lower()
        server = server_of(tool).lower()
        desc = (tool.schema.description or "").lower()
        score = 0
        for term in terms:
            if term in name:
                score += 3
            if term in server:
                score += 2
            if term in desc:
                score += 1
        if score:
            scored.append((score, name, tool))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [tool for _, _, tool in scored]


@dataclass
class ToolSearchTool:
    """Find deferred tools and load their schemas into the agent."""

    agent: BaseAgent | None = None

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=TOOL_SEARCH_NAME,
            description=(
                "Load deferred tools (listed by name in the system prompt under "
                "'Deferred tools') so they can be called. Query forms: "
                "'select:name1,name2' loads exact tool names; 'server:<name>' "
                "loads every tool of one MCP server; anything else is a keyword "
                "search over tool names and descriptions. Loaded tools become "
                "callable from your next step."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "select:<names>, server:<name>, or keywords.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            f"Max tools to load for a keyword search "
                            f"(default {_DEFAULT_MAX_RESULTS})."
                        ),
                    },
                },
                "required": ["query"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        if self.agent is None:
            raise ToolError("tool_search is not attached to an agent.")
        query = str(kwargs.get("query") or "").strip()
        if not query:
            raise ToolError("tool_search requires a 'query'.")
        deferred = self.agent.deferred_tools
        lowered = query.lower()

        if lowered.startswith("select:"):
            wanted = [n.strip() for n in query[len("select:"):].split(",") if n.strip()]
            already = [n for n in wanted if n in self.agent.tools]
            missing = [n for n in wanted if n not in deferred and n not in self.agent.tools]
            names = [n for n in wanted if n in deferred]
        elif lowered.startswith("server:"):
            server = query[len("server:"):].strip().lower()
            names = [n for n, t in deferred.items() if server_of(t).lower() == server]
            already, missing = [], ([] if names else [f"server:{server}"])
        else:
            limit = _clamp(kwargs.get("max_results"))
            names = [t.schema.name for t in rank_tools(deferred.values(), query)[:limit]]
            already, missing = [], []

        loaded = self.agent.load_tools(names)
        over_limit = [n for n in names if n in deferred]
        return _render(loaded, already, missing, over_limit, query)


def _clamp(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_RESULTS
    return max(1, min(n, _MAX_RESULTS_CAP))


def _render(
    loaded: list[Tool],
    already: list[str],
    missing: list[str],
    over_limit: list[str],
    query: str,
) -> str:
    lines: list[str] = []
    if loaded:
        lines.append(f"Loaded {len(loaded)} tool(s); callable from your next step:")
        for tool in loaded:
            desc = (tool.schema.description or "").strip().splitlines()
            first = desc[0][:160] if desc else ""
            lines.append(f"- {tool.schema.name}: {first}")
    if already:
        lines.append(f"Already loaded: {', '.join(already)}")
    if missing:
        lines.append(f"Not found: {', '.join(missing)}")
    if over_limit:
        lines.append(
            f"Not loaded, tool limit reached: {', '.join(over_limit)}. "
            "Load fewer tools (exact names instead of a whole server)."
        )
    if not lines:
        lines.append(
            f"No deferred tools matched {query!r}. Try other keywords, "
            "'server:<name>', or exact names from the Deferred tools list."
        )
    return "\n".join(lines)


__all__ = [
    "TOOL_SEARCH_NAME",
    "ToolSearchTool",
    "deferred_index",
    "estimate_schema_tokens",
    "rank_tools",
    "server_of",
]
