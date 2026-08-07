"""Verify TernoAgent registers the right tools depending on whether a
sandbox is provided."""

from __future__ import annotations

from terno_agent.agents.terno import TernoAgent
from terno_agent.sandbox.local import LocalSandbox


class _DummyLLM:
    model = "dummy"

    def complete(self, *args, **kwargs):  # pragma: no cover - not called here
        raise AssertionError("LLM should not be invoked in this test")


def test_run_python_omitted_without_sandbox():
    agent = TernoAgent(_DummyLLM())
    assert "run_python" not in agent.tools
    # Core tools are sent eagerly:
    for name in ("Read", "Write", "Edit", "Bash", "Agent"):
        assert name in agent.tools
    # Task tools are deferred by default — advertised, loaded via ToolSearch.
    assert "ToolSearch" in agent.tools
    for name in ("TaskCreate", "TaskList", "TaskGet", "TaskUpdate"):
        assert name not in agent.tools
        assert name in agent.tool_registry.names


def test_run_python_registered_with_sandbox():
    agent = TernoAgent(_DummyLLM(), sandbox=LocalSandbox())
    assert "run_python" in agent.tools
