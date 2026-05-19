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
    # Core tools are always present:
    for name in (
        "read_file",
        "write_file",
        "edit_file",
        "bash",
        "task_create",
        "task_list",
        "task_get",
        "task_update",
        "spawn_agent",
    ):
        assert name in agent.tools


def test_run_python_registered_with_sandbox():
    agent = TernoAgent(_DummyLLM(), sandbox=LocalSandbox())
    assert "run_python" in agent.tools
