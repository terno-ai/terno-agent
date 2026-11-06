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
    # Estimated history size (messages only, not the system prompt or tools)
    # before and after, so a UI can show what was freed.
    tokens_before: int = 0
    tokens_after: int = 0


@dataclass(slots=True)
class CompactionStarted:
    """Compaction is about to summarize older history (an LLM call).

    Followed by :class:`CompactionEvent` on success or
    :class:`CompactionFailed` otherwise, so a UI can show "condensing…" in
    between.
    """

    agent: str
    history_tokens: int


@dataclass(slots=True)
class CompactionFailed:
    """The summary call failed; history was left as it was."""

    agent: str
    error: str = ""


@dataclass(slots=True)
class ContextUsage:
    """How full the conversation history is, emitted before each LLM call.

    ``history_tokens`` is the estimated size of the messages (not the system
    prompt or tool schemas) after any compaction; ``limit_tokens`` is the size
    at which compaction starts. A UI can draw ``history_tokens / limit_tokens``
    as a fill meter that reaches 100% when the history gets summarized.
    """

    agent: str
    history_tokens: int
    limit_tokens: int


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
    | CompactionStarted
    | CompactionFailed
    | ContextUsage
)
EventHook = Callable[[AgentEvent], None]


__all__ = [
    "AgentEvent",
    "CompactionEvent",
    "CompactionFailed",
    "CompactionStarted",
    "ContextUsage",
    "EventHook",
    "IterationStart",
    "TaskListUpdate",
    "TextDelta",
    "ToolCallEvent",
    "ToolResultEvent",
    "TurnEnd",
]
