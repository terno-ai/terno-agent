"""Shared MCP test fixtures.

`FakeSession` and `FakeTool` let us exercise the manager + tool adapter
without touching the real `mcp` SDK or spawning subprocesses.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

# Field names below intentionally mirror the real `mcp` SDK shape
# (camelCase per the MCP wire format). The agent reads them by name.

@dataclass
class FakeTool:
    name: str
    description: str = ""
    inputSchema: dict[str, Any] = field(  # noqa: N815
        default_factory=lambda: {"type": "object"}
    )


@dataclass
class FakeContentBlock:
    type: str
    text: str | None = None


@dataclass
class FakeCallResult:
    content: list[Any]
    isError: bool = False  # noqa: N815


@dataclass
class FakeListToolsResult:
    tools: list[FakeTool]


class FakeSession:
    """Stand-in for `McpSession` in unit tests."""

    def __init__(
        self,
        tools: list[FakeTool] | None = None,
        *,
        call_impl=None,
        fail_connect: bool = False,
        fail_list_tools: bool = False,
    ) -> None:
        self.tools = tools or []
        self.call_impl = call_impl
        self.fail_connect = fail_connect
        self.fail_list_tools = fail_list_tools
        self.connected = False
        self.closed = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def connect(self) -> None:
        if self.fail_connect:
            raise RuntimeError("connect failed")
        # Yield to the loop so this is a real coroutine pause.
        await asyncio.sleep(0)
        self.connected = True

    async def list_tools(self) -> list[FakeTool]:
        if self.fail_list_tools:
            raise RuntimeError("list_tools failed")
        await asyncio.sleep(0)
        return self.tools

    async def call_tool(self, name: str, args: dict[str, Any]) -> FakeCallResult:
        await asyncio.sleep(0)
        self.calls.append((name, args))
        if self.call_impl is None:
            return FakeCallResult(content=[FakeContentBlock(type="text", text=f"{name}:{args}")])
        return self.call_impl(name, args)

    async def aclose(self) -> None:
        await asyncio.sleep(0)
        self.closed = True
