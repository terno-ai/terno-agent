"""terno-agent: a multi-agent CLI that answers questions about your database."""

from terno_agent.agents.orchestrator import Orchestrator as Agent
from terno_agent.config import Config

__version__ = "0.1.0"
__all__ = ["Agent", "Config", "__version__"]
