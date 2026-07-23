"""Agent run loop.

A `BaseAgent` ties together an LLM client, a system prompt, and a set of
tools. The `run` loop is the standard "think → call tools → think" cycle,
terminating when the model produces a final assistant message with no tool
calls (or when a per-agent iteration cap is reached).

Conversation history is maintained on the agent across multiple `run()`
calls so that `terno chat` and repeated SDK calls form a real
multi-turn conversation. A `HookManager` is dispatched at well-defined
points so users can plug in compaction, telemetry, memory extraction,
etc., without modifying the loop.

The agent also emits per-iteration `AgentEvent`s to an optional
``on_event`` callback — used by the CLI to render streamed text.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from terno_agent.core.cancel import CancelToken
from terno_agent.core.events import (
    EventHook,
    IterationStart,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    TurnEnd,
)
from terno_agent.core.exceptions import AgentCancelled, AgentError, ToolError
from terno_agent.core.hooks import (
    HookContext,
    HookEvent,
    HookManager,
    PreToolUseContext,
    UsageMeter,
)
from terno_agent.core.messages import (
    ContentPart,
    Message,
    SystemMessage,
    TextPart,
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
    cancelled: bool = False


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
        hook_manager: HookManager | None = None,
        cancel_token: CancelToken | None = None,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools: dict[str, Tool] = {t.schema.name: t for t in tools}
        self.on_event = on_event
        self.hooks = hook_manager or HookManager()
        self.cancel_token = cancel_token or CancelToken()
        self.usage = UsageMeter()
        self.history: Trace = [SystemMessage(system_prompt)]

    # ----- public history controls -------------------------------------- #

    def clear_history(self) -> None:
        """Drop the current conversation; keep the system message."""
        self.history = [SystemMessage(self.system_prompt)]
        self.usage.reset()

    def set_history(self, messages: Iterable[Message]) -> None:
        """Seed the conversation with prior turns, keeping the system message.

        Replaces everything after the system prompt with ``messages`` so a
        freshly built agent continues a conversation persisted elsewhere
        (e.g. a host application's database). The next ``run`` appends the
        new user turn after these. ``messages`` must not contain a
        ``SystemMessage`` — the system prompt is owned by the agent.

        Updated in place so existing references to ``history`` (the
        ``Agent.history`` property, compaction hooks) stay valid.
        """
        seeded = list(messages)
        if any(isinstance(m, SystemMessage) for m in seeded):
            raise ValueError(
                "set_history messages must not include a SystemMessage; "
                "the system prompt is managed by the agent."
            )
        self.history[:] = [self.history[0], *seeded]

    def add_hook(self, event: str, hook) -> None:
        """Shorthand for ``self.hooks.register(event, hook)``."""
        self.hooks.register(event, hook)

    # ----- run loop ----------------------------------------------------- #

    def run(
        self,
        task: str,
        *,
        extra_context: str | None = None,
        content_parts: list[ContentPart] | None = None,
    ) -> AgentRun:
        # Per-call context becomes part of the user message rather than the
        # persistent system prompt, so memory recall / per-task hints are
        # scoped to one turn.
        if content_parts is not None:
            user_content: str | list[ContentPart] = content_parts
            if extra_context:
                user_content = [
                    TextPart(f"<context>\n{extra_context}\n</context>"),
                    *content_parts,
                ]
        else:
            user_content = task
            if extra_context:
                user_content = f"<context>\n{extra_context}\n</context>\n\n{task}"
        self.history.append(UserMessage(user_content))

        last_iteration = 0
        run_start = len(self.history) - 1  # index of the UserMessage we just appended

        try:
            for i in range(1, self.max_iterations + 1):
                last_iteration = i
                self.cancel_token.check()
                self._emit(IterationStart(agent=self.name, iteration=i))

                response = self.llm.complete(
                    self.history,
                    tools=[t.schema for t in self.tools.values()],
                    on_text_delta=self._emit_text_delta,
                )
                self.usage.record(response)
                assistant = response.message
                self.history.append(assistant)
                self._emit(TurnEnd(agent=self.name, message=assistant))

                if not assistant.tool_calls:
                    run = AgentRun(
                        answer=assistant.content,
                        trace=self.history[run_start:],
                        iterations=i,
                    )
                    self._dispatch_chat_end(run)
                    return run

                results: list[ToolResult] = []
                for tc in assistant.tool_calls:
                    self.cancel_token.check()
                    self._emit(ToolCallEvent(agent=self.name, call=tc))
                    result = self._guarded_tool_call(tc)
                    self._emit(ToolResultEvent(agent=self.name, result=result))
                    results.append(result)

                self.history.append(ToolResultMessage(results=results))
        except AgentCancelled:
            run = AgentRun(
                answer="(cancelled by user)",
                trace=self.history[run_start:],
                iterations=last_iteration,
                cancelled=True,
            )
            self._dispatch_chat_end(run)
            return run

        raise AgentError(
            f"{self.name} exceeded max_iterations ({self.max_iterations}) without finishing."
        )

    # ----- internals ---------------------------------------------------- #

    def _guarded_tool_call(self, tc: ToolCall) -> ToolResult:
        """Dispatch ``pre_tool_use`` hooks; deny short-circuits the tool."""
        tool = self.tools.get(tc.name)
        if tool is None:
            return ToolResult(call_id=tc.id, content=f"Unknown tool: {tc.name}", is_error=True)
        if self.hooks.has(HookEvent.PRE_TOOL_USE):
            ctx = PreToolUseContext(agent=self, tool_call=tc, tool=tool)
            self.hooks.dispatch(HookEvent.PRE_TOOL_USE, ctx)
            if ctx.decision == "deny":
                return ToolResult(call_id=tc.id, content=ctx.feedback, is_error=True)
        return self._run_tool_call(tc, tool)

    def _run_tool_call(self, tc: ToolCall, tool: Tool | None = None) -> ToolResult:
        tool = tool or self.tools.get(tc.name)
        if tool is None:
            return ToolResult(call_id=tc.id, content=f"Unknown tool: {tc.name}", is_error=True)
        try:
            output = tool.run(**tc.arguments)
            return ToolResult(call_id=tc.id, content=output, is_error=False)
        except AgentCancelled:
            raise
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

    def _emit_text_delta(self, text: str) -> None:
        # Streaming is the lowest-latency place to notice cancellation;
        # raising here propagates out of the LLM client's stream loop.
        self.cancel_token.check()
        self._emit(TextDelta(agent=self.name, text=text))

    def _dispatch_chat_end(self, run: AgentRun) -> None:
        if not self.hooks.has(HookEvent.CHAT_END):
            return
        ctx = HookContext(
            event=HookEvent.CHAT_END,
            agent=self,
            history=self.history,
            usage=self.usage,
            run=run,
        )
        self.hooks.dispatch(HookEvent.CHAT_END, ctx)


__all__ = ["AgentRun", "BaseAgent", "EventHook", "Trace"]
