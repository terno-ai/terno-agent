"""Tests for in-sandbox MCP execution (mcp/sandbox_session.py).

A ``FakeSandbox`` stands in for the real container: its ``run_python``
inspects the injected snippet to decide which operation is being driven and
returns a scripted sentinel-framed envelope, exactly as terno-ai's
in-container ``mcp_client`` would print. No real container is involved.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re

import pytest

from terno_agent.core.exceptions import ConfigError, ToolError
from terno_agent.mcp.config import HttpServerConfig, RunnerBlock, StdioServerConfig
from terno_agent.mcp.manager import McpManager
from terno_agent.mcp.sandbox_session import (
    _SENTINEL,
    SandboxMcpSession,
    _to_container_config,
    sandbox_mcp_session_factory,
    sandbox_mcp_supported,
)
from terno_agent.sandbox.base import ExecutionResult


def _envelope(obj) -> str:
    return f"noise before\n{_SENTINEL}{json.dumps(obj)}\ntrailing noise"


class FakeSandbox:
    """Scripted sandbox. Records every snippet it was asked to run."""

    def __init__(
        self,
        *,
        has_mcp_client: bool = True,
        connect_errors: dict | None = None,
        tools: list | None = None,
        call_output: str = "tool-said-hi",
        connect_fatal: str | None = None,
    ) -> None:
        self.has_mcp_client = has_mcp_client
        self.connect_errors = connect_errors or {}
        self.tools = tools if tools is not None else []
        self.call_output = call_output
        self.connect_fatal = connect_fatal
        self.calls: list[tuple[str, int]] = []
        # Servers the in-container client already holds a live session for.
        self.live: set[str] = set()

    def run_python(self, code: str, *, timeout_s: int = 30, **_):
        self.calls.append((code, timeout_s))
        # Capability probe.
        if "find_spec('mcp_client')" in code:
            marker = "__MCP_CLIENT_OK__" if self.has_mcp_client else "__MCP_CLIENT_NO__"
            return ExecutionResult(stdout=marker + "\n", stderr="", exit_code=0)
        if "_connected_now" in code:
            return ExecutionResult(stdout=self._connect_and_call(code), stderr="", exit_code=0)
        if "connect_all" in code:
            if self.connect_fatal:
                out = _envelope({"ok": False, "fatal": self.connect_fatal})
            elif "__CFGS_B64__" not in code and "_cfgs = json.loads" in code:
                # Batched connect: one call for every server, which also
                # returns the connected servers' tools.
                m = re.search(r'_cfgs = json\.loads\(base64\.b64decode\("([^"]+)"\)', code)
                names = {c["name"] for c in json.loads(base64.b64decode(m.group(1)).decode())}
                self.live |= names - set(self.connect_errors)
                out = _envelope({
                    "ok": True,
                    "errors": self.connect_errors,
                    "tools": [t for t in self.tools if t.get("server_name") in names
                              and t.get("server_name") not in self.connect_errors],
                })
            else:
                out = _envelope({"ok": True, "errors": self.connect_errors})
            return ExecutionResult(stdout=out, stderr="", exit_code=0)
        if "get_tool_schemas" in code:
            # Emulate the container-side filter the injected snippet performs:
            # only return tools whose server_name matches this session's server.
            server = self._decode_server(code)
            visible = [t for t in self.tools if t.get("server_name") == server]
            return ExecutionResult(
                stdout=_envelope({"ok": True, "tools": visible}),
                stderr="",
                exit_code=0,
            )
        if "call_tool" in code:
            return ExecutionResult(
                stdout=_envelope({"ok": True, "output": self.call_output}),
                stderr="",
                exit_code=0,
            )
        raise AssertionError(f"unexpected snippet:\n{code}")

    @staticmethod
    def _decode_server(code: str) -> str:
        m = re.search(r'_server = base64\.b64decode\("([^"]+)"\)', code)
        return base64.b64decode(m.group(1)).decode() if m else ""

    def _connect_and_call(self, code: str) -> str:
        """Emulate the lazy first-call snippet: connect unless live, check
        the tool exists, call it."""
        def arg(var):
            m = re.search(var + r' = (?:json\.loads\()?base64\.b64decode\("([^"]+)"\)', code)
            return base64.b64decode(m.group(1)).decode()
        name = json.loads(arg("_cfg"))["name"]
        tool = arg("_tool")
        connected_now = False
        if name not in self.live:
            if self.connect_fatal or name in self.connect_errors:
                return _envelope({"ok": False,
                                  "connect_error": self.connect_fatal or self.connect_errors[name]})
            self.live.add(name)
            connected_now = True
        schemas = [t for t in self.tools if t.get("server_name") == name]
        if tool not in {t["tool_name"] for t in schemas}:
            return _envelope({"ok": False, "missing": True, "tools": schemas,
                              "connected_now": connected_now})
        return _envelope({"ok": True, "output": self.call_output, "connected_now": connected_now,
                          "tools": schemas if connected_now else []})


# --------------------------------------------------------------------------- #
# Config translation
# --------------------------------------------------------------------------- #


def test_to_container_config_raw_command():
    cfg = StdioServerConfig(name="s", command="npx", args=("x",), env={"K": "v"})
    assert _to_container_config(cfg) == {
        "name": "s",
        "transport_type": "stdio",
        "command": "npx",
        "args": ["x"],
        "env": {"K": "v"},
    }


def test_to_container_config_http_sse_vs_streamable():
    sse = HttpServerConfig(name="a", url="http://h/sse", transport="sse")
    http = HttpServerConfig(name="b", url="http://h/mcp", transport="http")
    assert _to_container_config(sse)["transport_type"] == "sse"
    assert _to_container_config(sse)["remote_url"] == "http://h/sse"
    assert _to_container_config(http)["transport_type"] == "streamable_http"


def test_to_container_config_runner_variants():
    uvx = StdioServerConfig(name="u", runner=RunnerBlock(type="uvx", package="pkg"))
    npx = StdioServerConfig(name="n", runner=RunnerBlock(type="npx", package="srv"))
    auto = StdioServerConfig(name="a", runner=RunnerBlock(type="auto", package="@s/x"))
    assert _to_container_config(uvx)["command"] == "uvx"
    assert _to_container_config(uvx)["args"] == ["pkg"]
    assert _to_container_config(npx)["args"] == ["-y", "srv"]
    assert _to_container_config(auto)["command"] == "npx"


def test_to_container_config_docker_runner_unsupported():
    cfg = StdioServerConfig(name="d", runner=RunnerBlock(type="docker", image="img"))
    with pytest.raises(ConfigError):
        _to_container_config(cfg)


# --------------------------------------------------------------------------- #
# Session lifecycle
# --------------------------------------------------------------------------- #


def test_connect_list_call_happy_path():
    sandbox = FakeSandbox(
        tools=[
            {
                "server_name": "srv",
                "tool_name": "do_thing",
                "description": "does a thing",
                "input_schema": {"type": "object", "properties": {"a": {}}},
            },
            {"server_name": "other", "tool_name": "skip", "description": "", "input_schema": {}},
        ],
        call_output="hello from tool",
    )
    cfg = StdioServerConfig(name="srv", command="run-me")
    session = SandboxMcpSession(cfg, sandbox)

    asyncio.run(session.connect())
    tools = asyncio.run(session.list_tools())
    # Only this server's tools, adapted to name/description/inputSchema.
    assert len(tools) == 1
    assert tools[0]["name"] == "do_thing"
    assert tools[0]["inputSchema"]["type"] == "object"

    result = asyncio.run(session.call_tool("do_thing", {"a": 1}))
    assert result.content == [{"type": "text", "text": "hello from tool"}]
    assert result.isError is False


def test_connect_surfaces_per_server_error():
    sandbox = FakeSandbox(connect_errors={"srv": "auth failed"})
    session = SandboxMcpSession(StdioServerConfig(name="srv", command="x"), sandbox)
    with pytest.raises(ConfigError, match="auth failed"):
        asyncio.run(session.connect())


def test_connect_fatal_when_mcp_client_missing():
    sandbox = FakeSandbox(connect_fatal="mcp_client module not available inside sandbox")
    session = SandboxMcpSession(StdioServerConfig(name="srv", command="x"), sandbox)
    with pytest.raises(ConfigError, match="mcp_client"):
        asyncio.run(session.connect())


def test_list_requires_connect_but_call_connects_itself():
    sandbox = FakeSandbox(tools=[{"server_name": "srv", "tool_name": "t", "description": "",
                                  "input_schema": {}}])
    session = SandboxMcpSession(StdioServerConfig(name="srv", command="x"), sandbox)
    with pytest.raises(ToolError):
        asyncio.run(session.list_tools())
    # A call on an unconnected session connects it in the same sandbox call.
    result = asyncio.run(session.call_tool("t", {}))
    assert result.content[0]["text"] == "tool-said-hi"
    assert session._connected


def test_run_python_transport_failure_becomes_fatal():
    class Boom(FakeSandbox):
        def run_python(self, code, *, timeout_s=30, **_):
            # Match only the capability probe (single-quoted find_spec), not
            # the connect snippet (which also imports find_spec).
            if "find_spec('mcp_client')" in code:
                return ExecutionResult(stdout="__MCP_CLIENT_OK__", stderr="", exit_code=0)
            raise RuntimeError("kernel died")

    session = SandboxMcpSession(StdioServerConfig(name="srv", command="x"), Boom())
    with pytest.raises(ConfigError, match="run_python failed"):
        asyncio.run(session.connect())


# --------------------------------------------------------------------------- #
# Capability detection
# --------------------------------------------------------------------------- #


def test_supported_true_when_mcp_client_present():
    assert sandbox_mcp_supported(FakeSandbox(has_mcp_client=True)) is True


def test_supported_false_when_mcp_client_absent():
    assert sandbox_mcp_supported(FakeSandbox(has_mcp_client=False)) is False


def test_supported_false_for_none():
    assert sandbox_mcp_supported(None) is False


def test_supported_false_when_probe_raises():
    class Dead:
        def run_python(self, *a, **k):
            raise RuntimeError("no sandbox")

    assert sandbox_mcp_supported(Dead()) is False


# --------------------------------------------------------------------------- #
# Session-factory routing (never falls back to the host in require mode)
# --------------------------------------------------------------------------- #


def _cfg(require_sandbox: bool):
    from types import SimpleNamespace

    return SimpleNamespace(mcp_require_sandbox=require_sandbox)


def test_require_sandbox_always_uses_sandbox_never_probes():
    # A sandbox whose probe would FAIL must still get the in-sandbox factory
    # (no probe, no host fallback) when require_sandbox is set.
    from terno_agent.agents.terno import _select_mcp_session_factory

    class BadProbe:
        def run_python(self, *a, **k):
            raise RuntimeError("probe down")

    factory = _select_mcp_session_factory(_cfg(True), BadProbe())
    assert callable(factory)
    assert isinstance(
        factory(StdioServerConfig(name="s", command="x")), SandboxMcpSession
    )


def test_require_sandbox_with_no_sandbox_returns_none_never_host():
    from terno_agent.agents.terno import _select_mcp_session_factory

    # None factory here means the caller DISABLES MCP — it must never become
    # the default host factory. (from_config enforces the skip; see that flow.)
    assert _select_mcp_session_factory(_cfg(True), None) is None


def test_default_mode_uses_sandbox_only_when_capable():
    from terno_agent.agents.terno import _select_mcp_session_factory

    assert callable(_select_mcp_session_factory(_cfg(False), FakeSandbox(has_mcp_client=True)))
    # Incapable sandbox -> None so McpManager uses its host factory (allowed
    # only in the non-require default mode).
    assert _select_mcp_session_factory(_cfg(False), FakeSandbox(has_mcp_client=False)) is None


# --------------------------------------------------------------------------- #
# Integration with McpManager (the real driver)
# --------------------------------------------------------------------------- #


def test_manager_builds_tools_through_sandbox_factory():
    sandbox = FakeSandbox(
        tools=[
            {"server_name": "srv", "tool_name": "alpha", "description": "a", "input_schema": {}},
            {"server_name": "srv", "tool_name": "beta", "description": "b", "input_schema": {}},
        ]
    )
    manager = McpManager.start_from_configs(
        [StdioServerConfig(name="srv", command="run-me")],
        session_factory=sandbox_mcp_session_factory(sandbox),
    )
    try:
        names = sorted(t.schema.name for t in manager.tools())
        assert names == ["mcp__srv__alpha", "mcp__srv__beta"]
        # A tool call routes back through the sandbox and renders as text.
        tool = next(t for t in manager.tools() if t.tool_name == "alpha")
        assert tool.run() == "hello from tool" or "tool-said-hi" in tool.run()
    finally:
        manager.shutdown()


def test_manager_connects_all_sandbox_servers_in_one_call():
    """Several servers connect in ONE run_python (parallel inside the
    sandbox, like the legacy path); a failing server is skipped, others load."""
    sandbox = FakeSandbox(
        connect_errors={"broken": "401 Unauthorized"},
        tools=[
            {"server_name": "a", "tool_name": "one", "description": "", "input_schema": {}},
            {"server_name": "b", "tool_name": "two", "description": "", "input_schema": {}},
            {"server_name": "b", "tool_name": "three", "description": "", "input_schema": {}},
        ],
    )
    manager = McpManager.start_from_configs(
        [
            StdioServerConfig(name="a", command="x"),
            StdioServerConfig(name="b", command="y"),
            StdioServerConfig(name="broken", command="z"),
        ],
        session_factory=sandbox_mcp_session_factory(sandbox),
    )
    try:
        connect_calls = [c for c, _ in sandbox.calls if "connect_all" in c]
        assert len(connect_calls) == 1
        assert not [c for c, _ in sandbox.calls if "_server = base64" in c]  # no per-server list calls
        names = sorted(t.schema.name for t in manager.tools())
        assert names == ["mcp__a__one", "mcp__b__three", "mcp__b__two"]
        # Connected servers can be called afterwards.
        tool = next(t for t in manager.tools() if t.tool_name == "two")
        assert "tool-said-hi" in tool.run()
    finally:
        manager.shutdown()


def _cached(name, *tools):
    return StdioServerConfig(
        name=name, command="x",
        cached_tools=tuple({"name": t, "description": f"cached {t}", "inputSchema": {}} for t in tools),
    )


def _live(server, *tools):
    return [{"server_name": server, "tool_name": t, "description": f"live {t}", "input_schema": {}}
            for t in tools]


def test_lazy_cached_server_does_not_connect_at_start():
    sandbox = FakeSandbox(tools=_live("ads", "spend"))
    discovered = []
    manager = McpManager.start_from_configs(
        [_cached("ads", "spend")],
        session_factory=sandbox_mcp_session_factory(sandbox),
        on_tools_discovered=lambda s, t: discovered.append((s, [x["name"] for x in t])),
    )
    try:
        assert sandbox.calls == []  # nothing ran in the sandbox at startup
        assert [t.schema.name for t in manager.tools()] == ["mcp__ads__spend"]
        assert discovered == []
    finally:
        manager.shutdown()


def test_lazy_first_call_connects_once_and_reports_live_tools():
    sandbox = FakeSandbox(tools=_live("ads", "spend", "new_tool"))
    discovered = []
    manager = McpManager.start_from_configs(
        [_cached("ads", "spend")],
        session_factory=sandbox_mcp_session_factory(sandbox),
        on_tools_discovered=lambda s, t: discovered.append((s, sorted(x["name"] for x in t))),
    )
    try:
        tool = manager.tools()[0]
        assert "tool-said-hi" in tool.run()
        assert len(sandbox.calls) == 1  # connect + call in one sandbox call
        assert discovered == [("ads", ["new_tool", "spend"])]  # live list handed back
        assert "tool-said-hi" in tool.run()  # second call: plain call, no reconnect
        assert len(sandbox.calls) == 2 and "_connected_now" not in sandbox.calls[1][0]
        assert len(discovered) == 1
    finally:
        manager.shutdown()


def test_lazy_server_already_live_in_container_is_not_reported():
    """A later message in the same chat: the container still holds the
    connection, so no reconnect and no cache write."""
    sandbox = FakeSandbox(tools=_live("ads", "spend"))
    sandbox.live.add("ads")
    discovered = []
    manager = McpManager.start_from_configs(
        [_cached("ads", "spend")],
        session_factory=sandbox_mcp_session_factory(sandbox),
        on_tools_discovered=lambda s, t: discovered.append(s),
    )
    try:
        assert "tool-said-hi" in manager.tools()[0].run()
        assert discovered == []
    finally:
        manager.shutdown()


def test_lazy_removed_tool_gives_clear_error_and_refreshes_cache():
    sandbox = FakeSandbox(tools=_live("ads", "spend_v2"))
    discovered = []
    manager = McpManager.start_from_configs(
        [_cached("ads", "spend")],
        session_factory=sandbox_mcp_session_factory(sandbox),
        on_tools_discovered=lambda s, t: discovered.append([x["name"] for x in t]),
    )
    try:
        with pytest.raises(ToolError, match="no longer exists.*spend_v2"):
            manager.tools()[0].run()
        assert discovered == [["spend_v2"]]
    finally:
        manager.shutdown()


def test_lazy_connect_failure_is_a_readable_tool_error():
    sandbox = FakeSandbox(connect_errors={"ads": "401 Unauthorized"})
    manager = McpManager.start_from_configs(
        [_cached("ads", "spend")],
        session_factory=sandbox_mcp_session_factory(sandbox),
    )
    try:
        with pytest.raises(ToolError, match="unavailable right now: 401 Unauthorized"):
            manager.tools()[0].run()
    finally:
        manager.shutdown()


def test_uncached_servers_connect_at_start_and_are_reported():
    """Only servers without a cache connect at startup (one parallel call);
    their live tools go to the host so the next message can be lazy."""
    sandbox = FakeSandbox(tools=_live("ads", "spend") + _live("fresh", "a", "b"))
    discovered = []
    manager = McpManager.start_from_configs(
        [_cached("ads", "spend"), StdioServerConfig(name="fresh", command="y")],
        session_factory=sandbox_mcp_session_factory(sandbox),
        on_tools_discovered=lambda s, t: discovered.append((s, sorted(x["name"] for x in t))),
    )
    try:
        assert len(sandbox.calls) == 1
        m = re.search(r'_cfgs = json\.loads\(base64\.b64decode\("([^"]+)"\)', sandbox.calls[0][0])
        assert [c["name"] for c in json.loads(base64.b64decode(m.group(1)))] == ["fresh"]
        assert discovered == [("fresh", ["a", "b"])]
        assert sorted(t.schema.name for t in manager.tools()) == [
            "mcp__ads__spend", "mcp__fresh__a", "mcp__fresh__b"]
    finally:
        manager.shutdown()


def test_cached_tools_parsed_from_config_dict():
    from terno_agent.mcp.config import load_mcp_config_from_dict
    [cfg] = load_mcp_config_from_dict({"mcpServers": {"ads": {
        "url": "https://x.example/mcp",
        "cachedTools": [{"name": "spend", "description": "d", "inputSchema": {"type": "object"}},
                        {"bad": "entry"}],
    }}})
    assert cfg.cached_tools == ({"name": "spend", "description": "d",
                                 "inputSchema": {"type": "object"}},)


def test_manager_batched_connect_fatal_loads_nothing():
    sandbox = FakeSandbox(connect_fatal="kernel died")
    manager = McpManager.start_from_configs(
        [StdioServerConfig(name="a", command="x"), StdioServerConfig(name="b", command="y")],
        session_factory=sandbox_mcp_session_factory(sandbox),
    )
    try:
        assert manager.tools() == []
    finally:
        manager.shutdown()
