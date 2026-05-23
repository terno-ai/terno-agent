"""Verify BaseAgent emits typed events and forwards text deltas from the LLM."""

from __future__ import annotations

from dataclasses import dataclass, field

from terno_agent.agents.base import BaseAgent
from terno_agent.core.events import (
    AgentEvent,
    IterationStart,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    TurnEnd,
)
from terno_agent.core.messages import AssistantMessage, Message, ToolCall
from terno_agent.core.tool import ToolSchema
from terno_agent.llm.base import LLMResponse


@dataclass
class _EchoTool:
    schema: ToolSchema = field(
        default_factory=lambda: ToolSchema(
            name="echo",
            description="echo back the value",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )
    )

    def run(self, **kwargs):
        return kwargs.get("value", "")


class _ScriptedLLM:
    """LLM that emits a scripted text stream then a tool_call, then a final text."""

    model = "scripted"

    def __init__(self):
        self.calls = 0

    def complete(
        self,
        messages: list[Message],
        tools=None,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        on_text_delta=None,
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            for piece in ["Looking ", "into ", "this..."]:
                if on_text_delta:
                    on_text_delta(piece)
            return LLMResponse(
                message=AssistantMessage(
                    content="Looking into this...",
                    tool_calls=[ToolCall(id="t1", name="echo", arguments={"value": "hi"})],
                ),
                stop_reason="tool_use",
            )
        for piece in ["Done: ", "hi"]:
            if on_text_delta:
                on_text_delta(piece)
        return LLMResponse(
            message=AssistantMessage(content="Done: hi"),
            stop_reason="end_turn",
        )


def test_base_agent_emits_typed_events():
    captured: list[AgentEvent] = []

    agent = BaseAgent(
        llm=_ScriptedLLM(),
        system_prompt="be brief",
        tools=[_EchoTool()],
        on_event=captured.append,
    )
    result = agent.run("say hi")

    assert result.answer == "Done: hi"
    types = [type(e).__name__ for e in captured]
    # Two LLM iterations: each starts with IterationStart and ends with TurnEnd.
    assert types.count("IterationStart") == 2
    assert types.count("TurnEnd") == 2
    assert types.count("ToolCallEvent") == 1
    assert types.count("ToolResultEvent") == 1

    text_chunks = [e.text for e in captured if isinstance(e, TextDelta)]
    assert "".join(text_chunks) == "Looking into this...Done: hi"

    tool_call = next(e for e in captured if isinstance(e, ToolCallEvent))
    assert tool_call.call.name == "echo"
    assert tool_call.call.arguments == {"value": "hi"}

    tool_result = next(e for e in captured if isinstance(e, ToolResultEvent))
    assert tool_result.result.content == "hi"
    assert tool_result.result.is_error is False

    final_turn = [e for e in captured if isinstance(e, TurnEnd)][-1]
    assert final_turn.message.content == "Done: hi"
    assert final_turn.message.tool_calls == []
    # First iteration should always come before the rest
    assert isinstance(captured[0], IterationStart)
