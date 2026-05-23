class TernoError(Exception):
    """Base exception for terno-agent."""


class ConfigError(TernoError):
    """Configuration is missing or invalid."""


class LLMError(TernoError):
    """The LLM provider returned an error or unexpected response."""


class ToolError(TernoError):
    """A tool failed in a way the agent should see."""


class SandboxError(TernoError):
    """The sandbox could not start or execute the request."""


class AgentError(TernoError):
    """An agent could not complete its task."""


class AgentCancelled(TernoError):  # noqa: N818 - "Cancelled" reads better than "CancelledError" here
    """Raised when the user (or another caller) cancels the agent mid-turn."""
