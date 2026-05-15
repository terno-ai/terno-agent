"""LLM client protocol.

Every provider implementation accepts our neutral `Message` list and
`ToolSchema` list and returns an `AssistantMessage`. The agent loop never sees
provider-specific objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from terno_agent.core.messages import AssistantMessage, Message
from terno_agent.core.tool import ToolSchema

TextDeltaCallback = Callable[[str], None]


@dataclass(slots=True)
class LLMResponse:
    message: AssistantMessage
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class LLMClient(Protocol):
    """A provider-agnostic chat completion client.

    Implementations always stream internally and call ``on_text_delta`` for
    each text chunk as it arrives. The returned ``LLMResponse`` contains the
    fully assembled message including any tool calls.
    """

    model: str

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> LLMResponse: ...
