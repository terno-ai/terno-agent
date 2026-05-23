"""Tool abstraction shared by all agents.

A `Tool` is a callable with a JSON-schema-described signature. Each provider
client renders the schema into the format its API expects (Anthropic
`input_schema`, OpenAI `function.parameters`, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]


@runtime_checkable
class Tool(Protocol):
    """Anything the agent can invoke as a tool."""

    @property
    def schema(self) -> ToolSchema: ...

    def run(self, **kwargs: Any) -> str:
        """Run the tool and return a string result.

        Implementations should return human/LLM-readable text. Raise
        `ToolError` for failures the agent should see as a tool error.
        """
        ...
