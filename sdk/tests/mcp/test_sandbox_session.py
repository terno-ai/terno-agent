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

    def run_python(self, code: str, *, timeout_s: int = 30, **_):
        self.calls.append((code, timeout_s))
        # Capability probe.
        if "find_spec('mcp_client')" in code:
            marker = "__MCP_CLIENT_OK__" if self.has_mcp_client else "__MCP_CLIENT_NO__"
            return ExecutionResult(stdout=marker + "\n", stderr="", exit_code=0)
        if "connect_all" in code:
            if self.connect_fatal:
                out = _envelope({"ok": False, "fatal": self.connect_fatal})
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


def test_list_and_call_require_connect():
    sandbox = FakeSandbox()
    session = SandboxMcpSession(StdioServerConfig(name="srv", command="x"), sandbox)
    with pytest.raises(ToolError):
        asyncio.run(session.list_tools())
    with pytest.raises(ToolError):
        asyncio.run(session.call_tool("t", {}))


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
