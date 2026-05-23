"""One connection to one MCP server.

Wraps the official `mcp` Python SDK's `ClientSession` plus whichever
transport (stdio / SSE / streamable HTTP) the server config asks for,
all owned by a single `AsyncExitStack` so cleanup is atomic.

Methods are async and meant to be driven through `AsyncBridge`. A
per-session `asyncio.Lock` serializes `call_tool` because the SDK's
underlying anyio streams are not safe for interleaved use within one
session.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

from terno_agent.core.exceptions import ConfigError, ToolError
from terno_agent.mcp.config import HttpServerConfig, McpServerConfig, StdioServerConfig
from terno_agent.mcp.runner import RunnerSpec
from terno_agent.mcp.runner import resolve as resolve_runner


class McpSession:
    """One live MCP server connection."""

    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self._stack: AsyncExitStack | None = None
        self._session: Any = None  # mcp.ClientSession
        self._lock = asyncio.Lock()
        self._tools: list[Any] = []

    # ----- lifecycle ----------------------------------------------------- #

    async def connect(self) -> None:
        try:
            from mcp import ClientSession  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                "mcp package not installed. Install with: "
                "pip install 'terno-agent[mcp]'"
            ) from exc

        stack = AsyncExitStack()
        try:
            read, write = await self._open_transport(stack)
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise

        self._stack = stack
        self._session = session

    async def list_tools(self) -> list[Any]:
        if self._session is None:
            raise ToolError(f"mcp session for '{self.config.name}' is not connected")
        result = await self._session.list_tools()
        self._tools = list(result.tools)
        return self._tools

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        if self._session is None:
            raise ToolError(f"mcp session for '{self.config.name}' is not connected")
        async with self._lock:
            return await self._session.call_tool(name, args)

    async def aclose(self) -> None:
        stack = self._stack
        self._stack = None
        self._session = None
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:
                # Best-effort cleanup. Transport teardown is allowed to be noisy.
                pass

    # ----- transports ---------------------------------------------------- #

    async def _open_transport(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        if isinstance(self.config, StdioServerConfig):
            return await self._open_stdio(stack, self.config)
        if isinstance(self.config, HttpServerConfig):
            return await self._open_http(stack, self.config)
        raise ConfigError(  # pragma: no cover - exhaustive
            f"mcp server '{self.config.name}': unknown transport type"
        )

    async def _open_stdio(
        self, stack: AsyncExitStack, cfg: StdioServerConfig
    ) -> tuple[Any, Any]:
        from mcp import StdioServerParameters  # type: ignore[import-not-found]
        from mcp.client.stdio import stdio_client  # type: ignore[import-not-found]

        spec: RunnerSpec = resolve_runner(cfg)
        params = StdioServerParameters(
            command=spec.argv[0],
            args=list(spec.argv[1:]),
            env=spec.env or None,
            cwd=spec.cwd,
        )
        return await stack.enter_async_context(stdio_client(params))

    async def _open_http(
        self, stack: AsyncExitStack, cfg: HttpServerConfig
    ) -> tuple[Any, Any]:
        if cfg.transport == "sse":
            from mcp.client.sse import sse_client  # type: ignore[import-not-found]

            return await stack.enter_async_context(
                sse_client(url=cfg.url, headers=cfg.headers or None)
            )
        from mcp.client.streamable_http import (  # type: ignore[import-not-found]
            streamablehttp_client,
        )

        # streamablehttp_client yields (read, write, session_id_cb). Drop the
        # third value — we don't surface session ids today.
        read, write, _session_id = await stack.enter_async_context(
            streamablehttp_client(url=cfg.url, headers=cfg.headers or None)
        )
        return read, write


__all__ = ["McpSession"]
