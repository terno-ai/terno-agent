"""Provider-neutral message and tool-call types.

LLM provider clients translate to/from these so the rest of the system never
sees vendor-specific payloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass(slots=True, frozen=True)
class TextPart:
    text: str


@dataclass(slots=True, frozen=True)
class ImagePart:
    attachment_id: str
    filename: str
    mime_type: str
    path: Path


@dataclass(slots=True, frozen=True)
class FilePart:
    attachment_id: str
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    text: str


@dataclass(slots=True, frozen=True)
class AttachmentManifestPart:
    text: str


ContentPart = TextPart | ImagePart | FilePart | AttachmentManifestPart
UserContent = str | list[ContentPart]


@dataclass(slots=True)
class SystemMessage:
    content: str
    role: Literal[Role.SYSTEM] = field(default=Role.SYSTEM, init=False)


@dataclass(slots=True)
class UserMessage:
    content: UserContent
    role: Literal[Role.USER] = field(default=Role.USER, init=False)


@dataclass(slots=True)
class AssistantMessage:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    role: Literal[Role.ASSISTANT] = field(default=Role.ASSISTANT, init=False)


@dataclass(slots=True)
class ToolResultMessage:
    results: list[ToolResult]
    role: Literal[Role.TOOL] = field(default=Role.TOOL, init=False)


Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage


def _user_content_to_text(content: UserContent) -> str:
    """Flatten user content (string or content parts) into readable text.

    Image parts are summarized rather than base64-inlined so the display
    payload stays light.
    """
    if isinstance(content, str):
        return content
    chunks: list[str] = []
    for part in content:
        if isinstance(part, (TextPart, AttachmentManifestPart)):
            chunks.append(part.text)
        elif isinstance(part, FilePart):
            chunks.append(
                f"[file: {part.filename} ({part.mime_type}, {part.size_bytes} bytes)]"
            )
        elif isinstance(part, ImagePart):
            chunks.append(f"[image: {part.filename} ({part.mime_type})]")
        else:  # pragma: no cover - defensive
            chunks.append(str(part))
    return "\n\n".join(chunks)


def to_display_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Serialize the message history into a plain, JSON-friendly shape for
    display — e.g. an app host's "view prompt" panel.

    Mirrors the OpenAI-style wire layout: each entry has ``role`` and
    ``content``; assistant tool calls become ``tool_calls[].function.{name,
    arguments}``; tool results become ``role: "tool"`` entries. This is the
    prompt that was (or is about to be) sent to the LLM, meant for humans to
    inspect rather than for re-sending to a provider — image bytes are
    summarized (see ``_user_content_to_text``) to keep it light.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, UserMessage):
            out.append({"role": "user", "content": _user_content_to_text(m.content)})
        elif isinstance(m, AssistantMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(entry)
        elif isinstance(m, ToolResultMessage):
            for r in m.results:
                out.append(
                    {"role": "tool", "tool_call_id": r.call_id, "content": r.content}
                )
    return out
