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

import json
import logging
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from terno_agent.core.cancel import CancelToken
from terno_agent.core.compaction import estimate_tokens
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
    to_display_messages,
)
from terno_agent.core.tool import Tool
from terno_agent.core.tool_output import ToolOutputLimiter
from terno_agent.llm.base import LLMClient

Trace = list[Message]

# Returns short messages the model should see as soon as possible (e.g. "a
# background task finished"). Called at the start of a turn and after each
# round of tool calls; an empty list means nothing to say.
NoticeSource = Callable[[], list[str]]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentRun:
    answer: str
    trace: Trace = field(default_factory=list)
    iterations: int = 0
    cancelled: bool = False


class BaseAgent:
    name: str = "agent"
    max_iterations: int = 12
    max_tools: int | None = None

    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str,
        tools: Iterable[Tool] = (),
        *,
        on_event: EventHook | None = None,
        hook_manager: HookManager | None = None,
        cancel_token: CancelToken | None = None,
        tool_output_limiter: ToolOutputLimiter | None = None,
        deferred_tools: Iterable[Tool] = (),
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools: dict[str, Tool] = {t.schema.name: t for t in tools}
        self.deferred_tools: dict[str, Tool] = {
            t.schema.name: t for t in deferred_tools if t.schema.name not in self.tools
        }
        self.on_event = on_event
        self.hooks = hook_manager or HookManager()
        self.cancel_token = cancel_token or CancelToken()
        self.usage = UsageMeter()
        self.history: Trace = [SystemMessage(system_prompt)]
        # Shortens oversized tool results (full text saved to a file) before
        # they enter the history; ``None`` passes results through unchanged.
        self.tool_output_limiter = tool_output_limiter
        self.notice_sources: list[NoticeSource] = []

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

    def load_tools(self, names: Iterable[str]) -> list[Tool]:
        """Move deferred tools into the active set; returns the ones moved.

        Appended in order, never reordered, so the tool list sent to the
        provider only grows (friendlier to prompt caching).
        """
        loaded: list[Tool] = []
        for name in names:
            if self.max_tools is not None and len(self.tools) >= self.max_tools:
                break
            tool = self.deferred_tools.pop(name, None)
            if tool is not None:
                self.tools[name] = tool
                loaded.append(tool)
        return loaded

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
        notices = self._collect_notices()
        if notices:
            extra_context = notices if not extra_context else f"{notices}\n\n---\n{extra_context}"
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
        user_message = UserMessage(user_content)
        self.history.append(user_message)

        last_iteration = 0
        turn_input = self.usage.total_input_tokens
        turn_output = self.usage.total_output_tokens
        turn_calls = self.usage.llm_calls

        try:
            for i in range(1, self.max_iterations + 1):
                last_iteration = i
                self.cancel_token.check()
                # Before the snapshot, so hosts see the history actually sent.
                self._dispatch_pre_llm()
                prompt_snapshot = (
                    to_display_messages(self.history)
                    if self.on_event is not None
                    else []
                )
                self._emit(
                    IterationStart(
                        agent=self.name, iteration=i, messages=prompt_snapshot
                    )
                )

                schemas = [t.schema for t in self.tools.values()]
                response = self.llm.complete(
                    self.history,
                    tools=schemas,
                    on_text_delta=self._emit_text_delta,
                )
                self.usage.record(response)
                self._log_call_tokens(i, response, schemas)
                assistant = response.message
                self.history.append(assistant)
                self._emit(TurnEnd(agent=self.name, message=assistant))

                if not assistant.tool_calls:
                    self._log_turn_tokens(i, turn_input, turn_output, turn_calls)
                    run = AgentRun(
                        answer=assistant.content,
                        trace=self._trace_since(user_message),
                        iterations=i,
                    )
                    self._dispatch_chat_end(run)
                    return run

                results: list[ToolResult] = []
                last_call = len(assistant.tool_calls) - 1
                for index, tc in enumerate(assistant.tool_calls):
                    self.cancel_token.check()
                    self._emit(ToolCallEvent(agent=self.name, call=tc))
                    result = self._guarded_tool_call(tc)
                    if index == last_call:
                        result = self._with_notices(result)
                    self._emit(ToolResultEvent(agent=self.name, result=result))
                    results.append(result)
                self.history.append(ToolResultMessage(results=results))
        except AgentCancelled:
            self._log_turn_tokens(last_iteration, turn_input, turn_output, turn_calls)
            run = AgentRun(
                answer="(cancelled by user)",
                trace=self._trace_since(user_message),
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
        if tool is None and tc.name in self.deferred_tools:
            loaded = self.load_tools([tc.name])
            if not loaded:
                return ToolResult(
                    call_id=tc.id,
                    content=f"Cannot load {tc.name}: the tool limit ({self.max_tools}) is reached.",
                    is_error=True,
                )
            tool = loaded[0]
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
            return ToolResult(call_id=tc.id, content=self._limit(tc, output), is_error=False)
        except AgentCancelled:
            raise
        except ToolError as exc:
            # Error text can be as large as output (an MCP error body, a long
            # traceback), so it goes through the same limit.
            return ToolResult(call_id=tc.id, content=self._limit(tc, str(exc)), is_error=True)
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult(call_id=tc.id, content=f"Unhandled tool error: {exc}", is_error=True)

    def _limit(self, tc: ToolCall, text: str) -> str:
        if self.tool_output_limiter is None or not isinstance(text, str):
            return text
        return self.tool_output_limiter.apply(tc.name, tc.arguments, text)

    def _with_notices(self, result: ToolResult) -> ToolResult:
        notices = self._collect_notices()
        if not notices:
            return result
        return ToolResult(
            call_id=result.call_id,
            content=f"{result.content}\n\n<notice>\n{notices}\n</notice>",
            is_error=result.is_error,
        )

    def _collect_notices(self) -> str:
        lines: list[str] = []
        for source in self.notice_sources:
            try:
                lines.extend(source())
            except Exception:
                logger.exception("Notice source failed")
        return "\n".join(lines)

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

    def _log_call_tokens(self, step: int, response, schemas) -> None:
        """One line per LLM call: what the provider counted, and roughly
        where it went (tool schemas / system prompt / conversation history)."""
        payload = [[s.name, s.description, s.parameters] for s in schemas]
        tools = len(json.dumps(payload, default=str)) // 4
        system = estimate_tokens(self.history[:1])
        history = estimate_tokens(self.history[1:])
        _log_tokens(
            f"{self.name} step={step} in={response.input_tokens:,} out={response.output_tokens:,} "
            f"| est tools~{tools:,} ({len(schemas)}, {len(self.deferred_tools)} deferred) "
            f"system~{system:,} "
            f"history~{history:,} ({len(self.history) - 1} msgs)"
        )

    def _log_turn_tokens(self, steps: int, start_in: int, start_out: int, start_calls: int) -> None:
        _log_tokens(
            f"{self.name} turn total: {self.usage.llm_calls - start_calls} calls, {steps} steps, "
            f"in={self.usage.total_input_tokens - start_in:,} "
            f"out={self.usage.total_output_tokens - start_out:,}"
        )

    def _trace_since(self, user_message: UserMessage) -> Trace:
        """This run's messages: from its user message on.

        Found by identity because ``pre_llm`` hooks may rewrite the history
        (compaction keeps the user message but moves it).
        """
        for idx in range(len(self.history) - 1, -1, -1):
            if self.history[idx] is user_message:
                return self.history[idx:]
        return list(self.history)

    def _dispatch_pre_llm(self) -> None:
        if not self.hooks.has(HookEvent.PRE_LLM):
            return
        ctx = HookContext(
            event=HookEvent.PRE_LLM,
            agent=self,
            history=self.history,
            usage=self.usage,
        )
        self.hooks.dispatch(HookEvent.PRE_LLM, ctx)

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


def _log_tokens(message: str) -> None:
    # stderr, like the compaction trace, so it shows in a host's console
    # without depending on logging configuration.
    print(f"[tokens] {message}", file=sys.stderr)


__all__ = ["AgentRun", "BaseAgent", "EventHook", "NoticeSource", "Trace"]
