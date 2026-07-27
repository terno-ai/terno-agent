"""Typed events emitted by an `Agent` while it runs.

The CLI subscribes to these to render streaming text and tool activity. The
library subscriber pattern is simply ``Callable[[AgentEvent], None]``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from terno_agent.core.messages import AssistantMessage, ToolCall, ToolResult


@dataclass(slots=True)
class IterationStart:
    """The agent is about to call the LLM again.

    ``messages`` carries the full prompt (the serialized message history) that
    is about to be sent to the LLM, so a subscriber can surface it — e.g. an
    app host's "view prompt" panel. It is populated only when the agent has an
    event subscriber; otherwise it is left empty to avoid needless work.
    """

    agent: str
    iteration: int
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class TextDelta:
    """A chunk of streamed assistant text."""

    agent: str
    text: str


@dataclass(slots=True)
class ToolCallEvent:
    """The model emitted a tool call (args fully assembled)."""

    agent: str
    call: ToolCall


@dataclass(slots=True)
class ToolResultEvent:
    """A tool finished executing."""

    agent: str
    result: ToolResult


@dataclass(slots=True)
class TurnEnd:
    """The assistant turn is complete (text + tool_calls finalized)."""

    agent: str
    message: AssistantMessage


@dataclass(slots=True)
class TaskListUpdate:
    """The agent's task/todo list changed.

    Emitted whenever a task is created or updated, carrying the full current
    list (non-deleted tasks, in creation order) as plain dicts so subscribers
    — the CLI renderer or an app host pushing a live todo panel to a UI — can
    mirror it without touching the store. Each dict has ``id``, ``subject``,
    ``description``, ``active_form`` and ``status``.
    """

    agent: str
    tasks: list[dict]


AgentEvent = (
    IterationStart
    | TextDelta
    | ToolCallEvent
    | ToolResultEvent
    | TurnEnd
    | TaskListUpdate
)
EventHook = Callable[[AgentEvent], None]


__all__ = [
    "AgentEvent",
    "EventHook",
    "IterationStart",
    "TaskListUpdate",
    "TextDelta",
    "ToolCallEvent",
    "ToolResultEvent",
    "TurnEnd",
]
