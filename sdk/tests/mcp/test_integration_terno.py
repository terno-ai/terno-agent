"""TernoAgent wires MCP tools into its tool list and shares the manager
with subagents."""

from __future__ import annotations

from terno_agent.agents.terno import TernoAgent
from terno_agent.mcp.config import HttpServerConfig
from terno_agent.mcp.manager import McpManager
from tests.mcp.conftest import FakeSession, FakeTool


class _DummyLLM:
    model = "dummy"

    def complete(self, *args, **kwargs):  # pragma: no cover - not called here
        raise AssertionError("LLM should not be invoked in this test")


def _manager_with_tools(tool_names):
    return McpManager.start_from_configs(
        [HttpServerConfig(name="srv", url="https://x", transport="http")],
        session_factory=lambda cfg: FakeSession([FakeTool(name=t) for t in tool_names]),
    )


def test_terno_agent_registers_mcp_tools():
    manager = _manager_with_tools(["alpha", "beta"])
    try:
        agent = TernoAgent(_DummyLLM(), mcp_manager=manager)
        assert "mcp__srv__alpha" in agent.tools
        assert "mcp__srv__beta" in agent.tools
        # Built-in tools still present:
        assert "read_file" in agent.tools
    finally:
        manager.shutdown()


def test_subagent_shares_parent_manager():
    """SpawnAgentTool must forward the parent's manager to the subagent
    rather than starting a new one."""
    manager = _manager_with_tools(["echo"])
    try:
        agent = TernoAgent(_DummyLLM(), mcp_manager=manager)
        spawn = agent.tools["spawn_agent"]
        assert getattr(spawn, "mcp_manager", None) is manager
    finally:
        manager.shutdown()
