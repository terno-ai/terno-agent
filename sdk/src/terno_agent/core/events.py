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
class CompactionEvent:
    """History was compacted: older messages were replaced by a summary.

    Emitted right after :class:`~terno_agent.core.compaction.CompactionHook`
    rewrites the in-memory history (before an LLM call). A host can persist
    ``summary`` so the condensed context survives across turns (e.g. terno-ai
    stores it as a ``Summary`` chat message). ``preserved_turns`` is the number
    of user turns kept verbatim — the current one. ``preserved_messages`` is how
    many messages after the current user message were kept verbatim (tool calls
    with their results). The summary also covers those kept messages, so a
    host replaying its own records may keep fewer without losing anything.
    """

    agent: str
    summary: str
    preserved_turns: int
    preserved_messages: int = 0


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
    | CompactionEvent
)
EventHook = Callable[[AgentEvent], None]


__all__ = [
    "AgentEvent",
    "CompactionEvent",
    "EventHook",
    "IterationStart",
    "TaskListUpdate",
    "TextDelta",
    "ToolCallEvent",
    "ToolResultEvent",
    "TurnEnd",
]
