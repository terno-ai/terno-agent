"""terno-agent: a coding agent CLI + SDK with file ops, shell, sandboxed
Python, task tracking, subagents, and pluggable MCP servers."""

from terno_agent.config import Config
from terno_agent.sdk import Agent

__version__ = "0.2.0"
__all__ = ["Agent", "Config", "__version__"]
