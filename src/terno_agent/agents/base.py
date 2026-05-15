"""Agent run loop.

A `BaseAgent` ties together an LLM client, a system prompt, and a set of
tools. The `run` loop is the standard "think → call tools → think" cycle,
terminating when the model produces a final assistant message with no tool
calls (or when a per-agent iteration cap is reached).

The agent emits `AgentEvent`s to an optional ``on_event`` hook: streamed
text deltas, tool calls, tool results, and turn endings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from terno_agent.core.events import (
    EventHook,
    IterationStart,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    TurnEnd,
)
from terno_agent.core.exceptions import AgentError, ToolError
from terno_agent.core.messages import (
    Message,
    SystemMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from terno_agent.core.tool import Tool
from terno_agent.llm.base import LLMClient

Trace = list[Message]


@dataclass(slots=True)
class AgentRun:
    answer: str
    trace: Trace = field(default_factory=list)
    iterations: int = 0


class BaseAgent:
    name: str = "agent"
    max_iterations: int = 12

    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str,
        tools: Iterable[Tool] = (),
        *,
        on_event: EventHook | None = None,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools: dict[str, Tool] = {t.schema.name: t for t in tools}
        self.on_event = on_event

    def run(self, task: str, *, extra_context: str | None = None) -> AgentRun:
        system = self.system_prompt
        if extra_context:
            system += "\n\n---\n" + extra_context

        messages: Trace = [SystemMessage(system), UserMessage(task)]

        for i in range(1, self.max_iterations + 1):
            self._emit(IterationStart(agent=self.name, iteration=i))

            def _on_text(text: str) -> None:
                self._emit(TextDelta(agent=self.name, text=text))

            response = self.llm.complete(
                messages,
                tools=[t.schema for t in self.tools.values()],
                on_text_delta=_on_text,
            )
            assistant = response.message
            messages.append(assistant)
            self._emit(TurnEnd(agent=self.name, message=assistant))

            if not assistant.tool_calls:
                return AgentRun(answer=assistant.content, trace=messages, iterations=i)

            results: list[ToolResult] = []
            for tc in assistant.tool_calls:
                self._emit(ToolCallEvent(agent=self.name, call=tc))
                result = self._run_tool_call(tc)
                self._emit(ToolResultEvent(agent=self.name, result=result))
                results.append(result)

            messages.append(ToolResultMessage(results=results))

        raise AgentError(
            f"{self.name} exceeded max_iterations ({self.max_iterations}) without finishing."
        )

    def _run_tool_call(self, tc: ToolCall) -> ToolResult:
        tool = self.tools.get(tc.name)
        if tool is None:
            return ToolResult(call_id=tc.id, content=f"Unknown tool: {tc.name}", is_error=True)
        try:
            output = tool.run(**tc.arguments)
            return ToolResult(call_id=tc.id, content=output, is_error=False)
        except ToolError as exc:
            return ToolResult(call_id=tc.id, content=str(exc), is_error=True)
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult(call_id=tc.id, content=f"Unhandled tool error: {exc}", is_error=True)

    def _emit(self, event) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:
            pass


__all__ = ["AgentRun", "BaseAgent", "EventHook", "Trace"]
