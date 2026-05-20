from __future__ import annotations

from pathlib import Path

import pytest

from terno_agent.agents.terno import TernoAgent
from terno_agent.attachments import AttachmentManager, AttachmentPolicy, AttachmentStore
from terno_agent.core.messages import (
    AssistantMessage,
    AttachmentManifestPart,
    FilePart,
    ImagePart,
    Message,
    TextPart,
    UserMessage,
)
from terno_agent.llm.anthropic_client import _serialize_user_content as anthropic_content
from terno_agent.llm.base import LLMResponse
from terno_agent.llm.openai_client import _serialize_user_content as openai_content


class _CaptureLLM:
    model = "capture"

    def __init__(self) -> None:
        self.messages: list[Message] = []

    def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN202
        self.messages = messages
        return LLMResponse(
            message=AssistantMessage(content="ok"),
            stop_reason="stop",
        )


def test_attachment_manager_builds_manifest_and_bounded_text(tmp_path: Path) -> None:
    source = tmp_path / "large.txt"
    source.write_text("alpha\n" * 20_000, encoding="utf-8")
    manager = AttachmentManager(
        AttachmentStore(tmp_path / "attachments"),
        AttachmentPolicy(total_text_budget_chars=1024, chunk_chars=512),
    )

    parts = manager.build_parts("find alpha", [source])

    assert isinstance(parts[0], AttachmentManifestPart)
    file_parts = [part for part in parts if isinstance(part, FilePart)]
    assert len(file_parts) == 1
    assert file_parts[0].filename == "large.txt"
    assert len(file_parts[0].text) <= 1024 + len("\n\n--- chunk ---\n\n")
    assert (tmp_path / "attachments" / "attachments.sqlite3").exists()


def test_attachment_manager_uses_image_parts(tmp_path: Path) -> None:
    image = tmp_path / "pixel.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    )
    manager = AttachmentManager(AttachmentStore(tmp_path / "attachments"))

    parts = manager.build_parts("describe", [image])

    assert any(isinstance(part, ImagePart) for part in parts)


def test_terno_agent_run_accepts_attachments(tmp_path: Path) -> None:
    source = tmp_path / "note.md"
    source.write_text("# Hello\n\nattached text", encoding="utf-8")
    llm = _CaptureLLM()
    agent = TernoAgent(
        llm,
        attachment_manager=AttachmentManager(AttachmentStore(tmp_path / "attachments")),
    )

    result = agent.run("summarize", attachments=[source])

    assert result.answer == "ok"
    user = next(msg for msg in llm.messages if isinstance(msg, UserMessage))
    assert isinstance(user.content, list)
    assert any(isinstance(part, AttachmentManifestPart) for part in user.content)
    assert any(isinstance(part, FilePart) for part in user.content)


def test_provider_serializers_emit_native_image_blocks(tmp_path: Path) -> None:
    image = tmp_path / "pixel.png"
    image.write_bytes(b"png-bytes")
    parts = [
        TextPart("look"),
        ImagePart("a1", "pixel.png", "image/png", image),
    ]

    openai_blocks = openai_content(parts)
    anthropic_blocks = anthropic_content(parts)

    assert isinstance(openai_blocks, list)
    assert openai_blocks[1]["type"] == "image_url"
    assert openai_blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert isinstance(anthropic_blocks, list)
    assert anthropic_blocks[1]["type"] == "image"
    assert anthropic_blocks[1]["source"]["media_type"] == "image/png"


def test_attachment_limits_are_enforced(tmp_path: Path) -> None:
    source = tmp_path / "too-big.txt"
    source.write_text("abcdef", encoding="utf-8")
    manager = AttachmentManager(
        AttachmentStore(tmp_path / "attachments"),
        AttachmentPolicy(max_attachment_bytes=3),
    )

    with pytest.raises(ValueError, match="too large"):
        manager.build_parts("read", [source])
