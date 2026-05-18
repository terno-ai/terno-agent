"""terno-agent: a multi-agent CLI + SDK for asking questions about your database."""

from terno_agent.config import Config
from terno_agent.sdk import Agent

__version__ = "0.1.0"
__all__ = ["Agent", "Config", "__version__"]
