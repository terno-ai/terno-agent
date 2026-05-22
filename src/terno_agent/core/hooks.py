"""Lifecycle hook framework for the agent run loop.

Hooks are keyed by an event name (a `HookEvent` constant). The agent
dispatches a `HookContext` to each registered callback at well-defined
points in the run loop. Hooks may *mutate* `ctx.history` in place — the
canonical use case is `CompactionHook`, which replaces older messages
with a single summary once the conversation grows past a token budget.

The design is intentionally minimal so new event names can be added
without changing the manager: register/dispatch take strings.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from terno_agent.agents.base import AgentRun, BaseAgent
    from terno_agent.core.messages import Message, ToolCall
    from terno_agent.core.tool import Tool
    from terno_agent.llm.base import LLMResponse


# --------------------------------------------------------------------------- #
# Event names
# --------------------------------------------------------------------------- #


class HookEvent:
    """String constants for hook events.

    Plain strings are used (not an Enum) so user code can register hooks
    for custom events without modifying this module.
    """

    CHAT_END = "chat_end"  # after agent.run() returns (success or cancelled)
    PRE_TOOL_USE = "pre_tool_use"  # before each tool invocation; may deny


# --------------------------------------------------------------------------- #
# Usage tracking
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class UsageMeter:
    """Aggregates token usage reported by the LLM across calls.

    The most recent call's ``input_tokens`` is the authoritative signal
    of current conversation size (the provider counts every byte of
    history we sent in). Compaction reads `last_input_tokens` to decide
    whether to summarize older history.
    """

    last_input_tokens: int = 0
    last_output_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    llm_calls: int = 0

    def record(self, response: LLMResponse) -> None:
        self.last_input_tokens = response.input_tokens
        self.last_output_tokens = response.output_tokens
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.llm_calls += 1

    def reset(self) -> None:
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.llm_calls = 0


# --------------------------------------------------------------------------- #
# Context + callable type
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class HookContext:
    """Payload passed to ``chat_end`` hooks.

    Hooks may *mutate* `history` in place (the canonical example is
    compaction). All other fields are informational. `run` is set for
    chat_end and is `None` for events that fire before a run completes.
    """

    event: str
    agent: BaseAgent
    history: list[Message]
    usage: UsageMeter
    run: AgentRun | None = None


@dataclass(slots=True)
class PreToolUseContext:
    """Payload passed to ``pre_tool_use`` hooks.

    Hooks decide whether the tool may run. Call ``allow()`` (the default)
    or ``deny(reason)``. When denied, the agent receives a
    ``ToolResult`` whose content is ``feedback`` and ``is_error`` is
    True, so the LLM sees the refusal and adapts.
    """

    agent: BaseAgent
    tool_call: ToolCall
    tool: Tool
    event: str = "pre_tool_use"
    decision: str = "allow"  # "allow" | "deny"
    feedback: str = ""

    def allow(self) -> None:
        self.decision = "allow"
        self.feedback = ""

    def deny(self, reason: str) -> None:
        self.decision = "deny"
        self.feedback = reason.strip() or "Tool call denied by the user."


Hook = Callable[[HookContext], None]
PreToolUseHook = Callable[[PreToolUseContext], None]


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class HookManager:
    """Register and dispatch lifecycle hooks.

    A single manager is owned by each `BaseAgent`. Multiple hooks may be
    registered per event; they fire in registration order. A hook that
    raises is logged and skipped — it must never break the run loop.
    """

    _hooks: dict[str, list[Hook]] = field(default_factory=dict)

    def register(self, event: str, hook: Hook) -> None:
        if not callable(hook):
            raise TypeError(f"hook for {event!r} must be callable, got {type(hook).__name__}")
        self._hooks.setdefault(event, []).append(hook)

    def unregister(self, event: str, hook: Hook) -> bool:
        """Remove a previously-registered hook. Returns True if removed."""
        lst = self._hooks.get(event)
        if not lst or hook not in lst:
            return False
        lst.remove(hook)
        return True

    def has(self, event: str) -> bool:
        return bool(self._hooks.get(event))

    def dispatch(self, event: str, ctx: object) -> None:
        """Dispatch ``ctx`` to every hook registered for ``event``.

        ``ctx`` is typed as ``object`` because different events use
        different context dataclasses (`HookContext` for ``chat_end``,
        `PreToolUseContext` for ``pre_tool_use``). Each hook is
        responsible for accepting the right shape.
        """
        for hook in list(self._hooks.get(event, ())):
            try:
                hook(ctx)
            except Exception as exc:
                # Hooks must never break the user-facing flow.
                print(f"warning: hook {event!r} raised: {exc}", file=sys.stderr)


__all__ = [
    "Hook",
    "HookContext",
    "HookEvent",
    "HookManager",
    "PreToolUseContext",
    "PreToolUseHook",
    "UsageMeter",
]
