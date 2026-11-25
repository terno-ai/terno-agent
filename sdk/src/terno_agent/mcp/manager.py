"""Top-level coordinator for MCP servers.

`McpManager` owns the background asyncio loop (`AsyncBridge`), opens
one `McpSession` per configured server, and aggregates every server's
tools into a list of sync `McpTool`s the agent can register.

Failures are isolated: one server failing to start or list tools
prints a warning and is skipped. The rest of the manager — and the
rest of terno — keeps running.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from terno_agent.core.exceptions import ConfigError, TernoError
from terno_agent.mcp.bridge import AsyncBridge
from terno_agent.mcp.config import (
    McpServerConfig,
    load_mcp_config,
    load_mcp_config_from_dict,
)
from terno_agent.mcp.session import McpSession
from terno_agent.mcp.tool import McpTool

SessionFactory = Callable[[McpServerConfig], "McpSessionLike"]

# Host-side startup: servers connecting at once (same limit as the sandbox's
# in-container client), and per-server time limits in seconds.
MAX_CONCURRENT_CONNECTS = 5
CONNECT_TIMEOUT_S = 30
LIST_TIMEOUT_S = 15
CLOSE_TIMEOUT_S = 5


class McpSessionLike:
    """Structural protocol — anything with async connect / list_tools /
    call_tool / aclose. `McpSession` satisfies this; tests pass a fake."""

    async def connect(self) -> None: ...  # pragma: no cover
    async def list_tools(self) -> list[Any]: ...  # pragma: no cover
    async def call_tool(self, name: str, args: dict[str, Any]) -> Any: ...  # pragma: no cover
    async def aclose(self) -> None: ...  # pragma: no cover


@dataclass(slots=True)
class _ServerEntry:
    config: McpServerConfig
    session: Any  # McpSessionLike
    tools: list[McpTool]


class McpManager:
    def __init__(self, *, bridge: AsyncBridge | None = None) -> None:
        self._bridge = bridge or AsyncBridge()
        self._bridge_owned = bridge is None
        self._servers: dict[str, _ServerEntry] = {}
        self._tools: list[McpTool] = []
        self._started = False
        self._shut_down = False
        self._on_tools_discovered: Callable[[str, list[dict[str, Any]]], None] | None = None

    # ----- factories ----------------------------------------------------- #

    @classmethod
    def start_from_path(
        cls,
        path: Path | str | None = None,
        *,
        session_factory: SessionFactory | None = None,
        on_tools_discovered: Callable[[str, list[dict[str, Any]]], None] | None = None,
    ) -> McpManager:
        try:
            configs = load_mcp_config(path)
        except ConfigError as exc:
            _warn(str(exc))
            configs = []
        return cls.start_from_configs(configs, session_factory=session_factory,
                                      on_tools_discovered=on_tools_discovered)

    @classmethod
    def start_from_dict(
        cls,
        raw: Any,
        *,
        session_factory: SessionFactory | None = None,
        on_tools_discovered: Callable[[str, list[dict[str, Any]]], None] | None = None,
    ) -> McpManager:
        """Start from an in-memory ``{"mcpServers": {...}}`` mapping."""
        try:
            configs = load_mcp_config_from_dict(raw)
        except ConfigError as exc:
            _warn(str(exc))
            configs = []
        return cls.start_from_configs(configs, session_factory=session_factory,
                                      on_tools_discovered=on_tools_discovered)

    @classmethod
    def start_from_configs(
        cls,
        configs: list[McpServerConfig],
        *,
        session_factory: SessionFactory | None = None,
        on_tools_discovered: Callable[[str, list[dict[str, Any]]], None] | None = None,
    ) -> McpManager:
        manager = cls()
        manager.start(configs, session_factory=session_factory,
                      on_tools_discovered=on_tools_discovered)
        return manager

    # ----- lifecycle ----------------------------------------------------- #

    def start(
        self,
        configs: list[McpServerConfig],
        *,
        session_factory: SessionFactory | None = None,
        on_tools_discovered: Callable[[str, list[dict[str, Any]]], None] | None = None,
    ) -> None:
        if self._started:
            raise RuntimeError("McpManager already started")
        self._started = True
        self._on_tools_discovered = on_tools_discovered
        if not configs:
            return

        factory = session_factory or _default_session_factory
        try:
            self._bridge.start()
        except Exception as exc:  # pragma: no cover - thread/loop init failure
            _warn(f"mcp bridge failed to start: {exc}")
            return

        # Lazy: a server whose tools the host already cached is not connected
        # now; its tools come from the cache and it connects on first call.
        # Only for session types that can connect inside call_tool.
        can_lazy = getattr(factory, "supports_lazy_connect", False)
        lazy = [c for c in configs if can_lazy and c.cached_tools]
        eager = [c for c in configs if not (can_lazy and c.cached_tools)]
        for cfg in lazy:
            self._register_lazy(cfg, factory)
        if not eager:
            return

        connect_many = getattr(factory, "connect_many", None)
        if connect_many is not None:
            self._connect_servers_together(eager, factory, connect_many)
            return
        self._connect_servers_concurrently(eager, factory)

    def tools(self) -> list[McpTool]:
        return list(self._tools)

    def shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        # Close each session via the loop, then stop the bridge.
        for entry in self._servers.values():
            try:
                self._bridge.submit(entry.session.aclose(), timeout=5)
            except Exception:
                pass
        if self._bridge_owned:
            try:
                self._bridge.stop()
            except Exception:
                pass
        self._servers.clear()
        self._tools.clear()

    # ----- internals ----------------------------------------------------- #

    def _connect_servers_together(
        self,
        configs: list[McpServerConfig],
        factory: SessionFactory,
        connect_many: Callable[[list[Any]], Any],
    ) -> None:
        """Connect every server in one call, in parallel (in-sandbox path).

        A server that fails is skipped with a warning; the rest still load.
        """
        sessions = []
        for cfg in configs:
            try:
                sessions.append(factory(cfg))
            except TernoError as exc:
                _warn(f"mcp server '{cfg.name}' not loaded: {exc}")
            except Exception as exc:
                _warn(f"mcp server '{cfg.name}' failed to initialize: {exc}")
        if not sessions:
            return

        try:
            results = self._bridge.submit(connect_many(sessions), timeout=30)
        except Exception as exc:
            _warn(f"mcp servers failed to connect: {exc}")
            return

        for session in sessions:
            name = session.config.name
            error, raw_tools = results.get(name, ("no result returned", []))
            if error:
                _warn(f"mcp server '{name}' failed to connect: {error}")
                continue
            self._register(session.config, session, raw_tools)
            self._report(name, raw_tools)

    def _register_lazy(self, cfg: McpServerConfig, factory: SessionFactory) -> None:
        """Register a server's cached tools without connecting it.

        The session connects inside its first ``call_tool`` and reports the
        live tool list through ``on_tools_discovered`` then.
        """
        try:
            session = factory(cfg)
        except TernoError as exc:
            _warn(f"mcp server '{cfg.name}' not loaded: {exc}")
            return
        except Exception as exc:
            _warn(f"mcp server '{cfg.name}' failed to initialize: {exc}")
            return
        session.on_tools_discovered = self._on_tools_discovered
        self._register(cfg, session, list(cfg.cached_tools))

    def _report(self, server: str, raw_tools: list[Any]) -> None:
        """Hand a server's live tool list to the host (to refresh its cache)."""
        if self._on_tools_discovered is None:
            return
        tools = [
            {
                "name": _attr(t, "name") or "tool",
                "description": _attr(t, "description") or "",
                "inputSchema": _attr(t, "inputSchema") or {"type": "object", "properties": {}},
            }
            for t in raw_tools
        ]
        try:
            self._on_tools_discovered(server, tools)
        except Exception as exc:
            _warn(f"mcp on_tools_discovered failed for '{server}': {exc}")

    def _connect_servers_concurrently(
        self, configs: list[McpServerConfig], factory: SessionFactory
    ) -> None:
        """Connect host-side servers (stdio / HTTP from this process) at the
        same time on the bridge loop, at most ``MAX_CONCURRENT_CONNECTS`` at
        once, so startup waits for the slowest server rather than the sum.

        Each server keeps its own connect / list-tools time limits and fails
        on its own: a server that fails is skipped with a warning and closed;
        the rest still load.
        """
        sessions: list[tuple[McpServerConfig, Any]] = []
        for cfg in configs:
            try:
                sessions.append((cfg, factory(cfg)))
            except TernoError as exc:
                _warn(f"mcp server '{cfg.name}' not loaded: {exc}")
            except Exception as exc:
                _warn(f"mcp server '{cfg.name}' failed to initialize: {exc}")
        if not sessions:
            return

        # Room for every wave of MAX_CONCURRENT_CONNECTS servers to use its
        # full per-server budget before the whole batch is given up on.
        waves = -(-len(sessions) // MAX_CONCURRENT_CONNECTS)
        per_server = CONNECT_TIMEOUT_S + LIST_TIMEOUT_S + CLOSE_TIMEOUT_S
        try:
            results = self._bridge.submit(
                _connect_and_list_all(sessions), timeout=waves * per_server + 5
            )
        except Exception as exc:
            _warn(f"mcp servers failed to connect: {exc}")
            return

        for (cfg, session), (stage, error, mcp_tools) in zip(sessions, results):
            if error is not None:
                _warn(f"mcp server '{cfg.name}' failed to {stage}: {error}")
                continue
            self._register(cfg, session, mcp_tools)
            self._report(cfg.name, mcp_tools)

    def _register(self, cfg: McpServerConfig, session: Any, mcp_tools: list[Any]) -> None:
        """Wrap a connected server's listed tools as ``McpTool``s."""
        tools: list[McpTool] = []
        timeout_s = float(getattr(cfg, "timeout_s", None) or 120)
        for t in mcp_tools:
            tools.append(
                McpTool(
                    server=cfg.name,
                    tool_name=_attr(t, "name") or "tool",
                    description=_attr(t, "description") or "",
                    input_schema=_attr(t, "inputSchema") or {"type": "object", "properties": {}},
                    session=session,
                    bridge=self._bridge,
                    timeout_s=timeout_s,
                )
            )
        self._servers[cfg.name] = _ServerEntry(config=cfg, session=session, tools=tools)
        self._tools.extend(tools)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _default_session_factory(cfg: McpServerConfig) -> McpSession:
    return McpSession(cfg)


async def _connect_and_list_all(
    sessions: list[tuple[McpServerConfig, Any]],
) -> list[tuple[str, Exception | None, list[Any]]]:
    """Connect and list tools for every session concurrently (on the bridge
    loop). Returns, per session in order, ``(stage, error, tools)``: ``error``
    is None on success, else ``stage`` says which step failed."""
    limit = asyncio.Semaphore(MAX_CONCURRENT_CONNECTS)

    async def one(cfg: McpServerConfig, session: Any) -> tuple[str, Exception | None, list[Any]]:
        async with limit:
            stage = "connect"
            try:
                await asyncio.wait_for(session.connect(), timeout=CONNECT_TIMEOUT_S)
                stage = "list tools"
                tools = await asyncio.wait_for(session.list_tools(), timeout=LIST_TIMEOUT_S)
                return stage, None, list(tools)
            except Exception as exc:
                if isinstance(exc, asyncio.TimeoutError):
                    exc = TimeoutError(f"timed out during {stage}")
                try:
                    await asyncio.wait_for(session.aclose(), timeout=CLOSE_TIMEOUT_S)
                except Exception:
                    pass
                return stage, exc, []

    return await asyncio.gather(*(one(cfg, s) for cfg, s in sessions))


def _attr(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


__all__ = ["McpManager", "McpSessionLike", "SessionFactory"]
