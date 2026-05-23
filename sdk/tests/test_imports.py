import terno
import terno_agent
from terno_agent.agents import BaseAgent, TernoAgent
from terno_agent.core import Tool, ToolSchema
from terno_agent.llm.base import LLMClient


def test_package_imports():
    assert terno_agent.__version__
    assert hasattr(BaseAgent, "run")
    assert hasattr(TernoAgent, "from_env")
    assert hasattr(LLMClient, "complete")
    assert hasattr(Tool, "run")
    assert ToolSchema.__name__ == "ToolSchema"


def test_terno_shim_imports():
    """`from terno import Agent` is the documented SDK entry point."""
    assert terno.Agent is terno_agent.Agent
    assert hasattr(terno.Agent, "run")
    assert hasattr(terno.Agent, "ask")
    assert hasattr(terno.Agent, "from_env")
    assert hasattr(terno.Agent, "deep_research")
