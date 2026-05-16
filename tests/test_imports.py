import terno_agent
from terno_agent.agents import BaseAgent, Orchestrator
from terno_agent.core import Tool, ToolSchema
from terno_agent.llm.base import LLMClient
from terno_agent.sandbox.base import Sandbox


def test_package_imports():
    assert terno_agent.__version__
    assert hasattr(BaseAgent, "run")
    assert hasattr(Orchestrator, "from_env")
    assert hasattr(LLMClient, "complete")
    assert hasattr(Sandbox, "run_python")
    assert hasattr(Tool, "run")
    assert ToolSchema.__name__ == "ToolSchema"
