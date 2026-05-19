"""Sync `Tool` adapter for one MCP-exposed tool.

`McpTool` satisfies terno's existing synchronous `Tool` protocol. Its
`run` method does **not** open the MCP session — it submits a
coroutine to an externally-owned `AsyncBridge` (passed in by the
manager) that drives the long-lived `McpSession`.

The exposed tool name is `mcp__{server}__{tool}` so the LLM can tell
remote tools apart and we avoid colliding with terno's built-ins.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema

if TYPE_CHECKING:
    from terno_agent.mcp.bridge import AsyncBridge
    from terno_agent.mcp.session import McpSession

_SANITIZE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_NAME_LEN = 64


def format_tool_name(server: str, tool: str) -> str:
    """Build the agent-facing tool name with size guard."""
    safe_server = _SANITIZE.sub("_", server) or "server"
    safe_tool = _SANITIZE.sub("_", tool) or "tool"
    name = f"mcp__{safe_server}__{safe_tool}"
    if len(name) <= _MAX_NAME_LEN:
        return name
    # Reserve room for a short hash so collisions across truncation are
    # extremely unlikely. Format: mcp__{server}__{tool[:N]}_{hash}.
    h = hashlib.sha256(tool.encode("utf-8")).hexdigest()[:6]
    prefix = f"mcp__{safe_server}__"
    keep = _MAX_NAME_LEN - len(prefix) - len(h) - 1
    if keep <= 0:
        # Server name itself is too long; truncate it too.
        safe_server = safe_server[: max(1, _MAX_NAME_LEN // 3)]
        prefix = f"mcp__{safe_server}__"
        keep = _MAX_NAME_LEN - len(prefix) - len(h) - 1
    return f"{prefix}{safe_tool[:keep]}_{h}"


@dataclass
class McpTool:
    server: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]
    session: McpSession
    bridge: AsyncBridge
    timeout_s: float = 120.0

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=format_tool_name(self.server, self.tool_name),
            description=f"[mcp:{self.server}] {self.description}".strip(),
            parameters=self.input_schema or {"type": "object", "properties": {}},
        )

    def run(self, **kwargs: Any) -> str:
        try:
            result = self.bridge.submit(
                self.session.call_tool(self.tool_name, dict(kwargs)),
                timeout=self.timeout_s,
            )
        except TimeoutError as exc:
            raise ToolError(
                f"mcp {self.server}.{self.tool_name} timed out after {self.timeout_s}s"
            ) from exc
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(
                f"mcp {self.server}.{self.tool_name} failed: {exc}"
            ) from exc

        rendered = format_content(getattr(result, "content", []))
        if getattr(result, "isError", False):
            raise ToolError(rendered or f"mcp {self.server}.{self.tool_name} returned isError")
        return rendered


def format_content(blocks: Any) -> str:
    """Flatten MCP content blocks into a single string for the LLM."""
    if not blocks:
        return ""
    parts: list[str] = []
    for block in blocks:
        text = _extract_text(block)
        if text is not None:
            parts.append(text)
            continue
        mime = _get(block, "mimeType")
        if mime is not None and _get(block, "data") is not None:
            parts.append(f"[image: {mime}]")
            continue
        uri = _resource_uri(block)
        if uri is not None:
            parts.append(f"[resource: {uri}]")
            continue
        parts.append(str(block))
    return "\n".join(parts).strip()


def _extract_text(block: Any) -> str | None:
    btype = _get(block, "type")
    if btype == "text":
        text = _get(block, "text")
        return text if isinstance(text, str) else None
    return None


def _resource_uri(block: Any) -> str | None:
    btype = _get(block, "type")
    if btype != "resource":
        return None
    resource = _get(block, "resource")
    if resource is None:
        return None
    uri = _get(resource, "uri")
    return uri if isinstance(uri, str) else None


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


__all__ = ["McpTool", "format_content", "format_tool_name"]
