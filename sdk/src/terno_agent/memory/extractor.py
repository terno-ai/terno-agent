"""Post-turn memory extractor.

After the main agent finishes a turn, this extractor spawns a fresh
``TernoAgent`` (system prompt = ``EXTRACTOR_SYSTEM_PROMPT``, tools = the
memory CRUD set) in a daemon thread. The user is not blocked.

The extractor's tool activity is NOT mirrored to the parent CLI; the
host only learns "memory was updated" via an optional ``on_complete``
callback once the subagent finishes. Failures are swallowed silently so
extraction never breaks the user-facing flow.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from terno_agent.core.events import AgentEvent, ToolResultEvent
from terno_agent.core.messages import (
    AssistantMessage,
    AttachmentManifestPart,
    FilePart,
    ImagePart,
    Message,
    SystemMessage,
    TextPart,
    ToolResultMessage,
    UserMessage,
)
from terno_agent.llm.base import LLMClient
from terno_agent.memory.prompts import (
    EXTRACTOR_SYSTEM_PROMPT,
    EXTRACTOR_USER_PROMPT_TEMPLATE,
)
from terno_agent.memory.store import MemoryStore
from terno_agent.memory.tools import extractor_tools

if TYPE_CHECKING:
    from terno_agent.agents.base import Trace


@dataclass(slots=True)
class ExtractionResult:
    """What the extractor did this turn."""

    saved: int = 0
    deleted: int = 0
    failed: bool = False

    @property
    def changed(self) -> bool:
        return self.saved > 0 or self.deleted > 0


ExtractionCallback = Callable[[ExtractionResult], None]


@dataclass
class MemoryExtractor:
    llm: LLMClient
    store: MemoryStore
    workdir: Path
    on_complete: ExtractionCallback | None = None
    max_iterations: int = 8
    wait: bool = False  # if True, run synchronously (used in tests)

    def extract(self, trace: Trace) -> None:
        """Run extraction. Fire-and-forget by default."""
        if self.wait:
            self._run_safely(trace)
            return
        thread = threading.Thread(
            target=self._run_safely,
            args=(trace,),
            name="memory-extractor",
            daemon=True,
        )
        thread.start()

    # ----- impl -------------------------------------------------------- #

    def _run_safely(self, trace: Trace) -> None:
        try:
            result = self._run(trace)
        except Exception:  # pragma: no cover - defensive; never break the host
            result = ExtractionResult(failed=True)
        if self.on_complete is not None:
            try:
                self.on_complete(result)
            except Exception:  # pragma: no cover - defensive
                pass

    def _run(self, trace: Trace) -> ExtractionResult:
        transcript = _format_trace(trace)
        if not transcript.strip():
            return ExtractionResult()

        result = ExtractionResult()
        tool_counter = _make_counter(result)

        # Lazy import to avoid a cycle (extractor -> TernoAgent -> agents/base).
        from terno_agent.agents.terno import TernoAgent

        subagent = TernoAgent(
            self.llm,
            system_prompt=EXTRACTOR_SYSTEM_PROMPT,
            workdir=self.workdir,
            on_event=tool_counter,  # internal-only — not propagated to CLI
        )
        # Replace the default toolset with just the memory CRUD tools — the
        # extractor must not run bash, edit files, etc.
        subagent.tools = {t.schema.name: t for t in extractor_tools(self.store)}
        subagent.max_iterations = self.max_iterations
        subagent.run(EXTRACTOR_USER_PROMPT_TEMPLATE.format(transcript=transcript))
        return result


def _make_counter(result: ExtractionResult) -> Callable[[AgentEvent], None]:
    """Count successful save/delete tool calls on the extractor subagent."""

    def _on_event(event: AgentEvent) -> None:
        if not isinstance(event, ToolResultEvent):
            return
        if event.result.is_error:
            return
        # The tool name isn't on the result event, but the surrounding
        # tool-call ID encodes it via the result content for save/delete:
        # SaveMemoryTool returns JSON with "path"; DeleteMemoryTool returns
        # "deleted" or "not_found". That's enough to count both reliably.
        content = event.result.content.strip()
        if content == "deleted":
            result.deleted += 1
        elif content.startswith("{") and '"path"' in content:
            result.saved += 1

    return _on_event


def _format_trace(trace: Trace) -> str:
    """Serialize a turn's trace into a plain transcript for the extractor."""
    parts: list[str] = []
    for msg in trace:
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, UserMessage):
            parts.append(f"USER:\n{_format_user_content(msg.content).strip()}")
        elif isinstance(msg, AssistantMessage):
            text = msg.content.strip()
            if text:
                parts.append(f"ASSISTANT:\n{text}")
            if msg.tool_calls:
                names = ", ".join(tc.name for tc in msg.tool_calls)
                parts.append(f"(assistant called tools: {names})")
        elif isinstance(msg, ToolResultMessage):
            continue
        else:  # pragma: no cover - exhaustive
            _: Message = msg
    return "\n\n".join(parts)


def _format_user_content(content) -> str:
    if isinstance(content, str):
        return content
    rendered: list[str] = []
    for part in content:
        if isinstance(part, TextPart):
            rendered.append(part.text)
        elif isinstance(part, AttachmentManifestPart):
            rendered.append(part.text)
        elif isinstance(part, ImagePart):
            rendered.append(
                f"[image attachment id={part.attachment_id} "
                f"name={part.filename} mime={part.mime_type}]"
            )
        elif isinstance(part, FilePart):
            rendered.append(
                f"[file attachment id={part.attachment_id} name={part.filename} "
                f"mime={part.mime_type} size={part.size_bytes} sha256={part.sha256}]"
            )
        else:  # pragma: no cover - defensive
            rendered.append(str(part))
    return "\n\n".join(rendered)


__all__ = ["ExtractionCallback", "ExtractionResult", "MemoryExtractor"]
