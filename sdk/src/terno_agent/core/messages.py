"""Provider-neutral message and tool-call types.

LLM provider clients translate to/from these so the rest of the system never
sees vendor-specific payloads.
"""

from __future__ import annotations

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
