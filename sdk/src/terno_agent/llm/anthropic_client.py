"""Anthropic Claude implementation of `LLMClient`."""

from __future__ import annotations

import base64
from typing import Any

from terno_agent.core.exceptions import ConfigError, LLMError
from terno_agent.core.messages import (
    AssistantMessage,
    AttachmentManifestPart,
    FilePart,
    ImagePart,
    Message,
    Role,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from terno_agent.core.tool import ToolSchema
from terno_agent.llm.base import LLMResponse, TextDeltaCallback


class AnthropicClient:
    """Wraps `anthropic.Anthropic` and translates to/from neutral messages."""

    def __init__(self, *, api_key: str | None = None, model: str) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ConfigError(
                "anthropic package not installed. "
                "Install with: pip install 'terno-agent[anthropic]'"
            ) from exc
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.model = model

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> LLMResponse:
        system, history = _split_system(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system or "",
            "messages": [_to_anthropic(m) for m in history],
        }
        tool_schemas = [_tool_to_anthropic(t) for t in (tools or [])]
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        from terno_agent.core.exceptions import AgentCancelled

        try:
            with self._client.messages.stream(**kwargs) as stream:
                if on_text_delta is not None:
                    for text in stream.text_stream:
                        if text:
                            on_text_delta(text)
                else:
                    # Still consume the stream so it completes.
                    for _ in stream.text_stream:
                        pass
                final = stream.get_final_message()
        except AgentCancelled:
            # The agent asked to stop — propagate untouched so the run
            # loop can short-circuit instead of treating it as an LLM error.
            raise
        except Exception as exc:
            raise LLMError(f"Anthropic API call failed: {exc}") from exc

        return _from_anthropic(final)


def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    system_chunks: list[str] = []
    rest: list[Message] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            system_chunks.append(m.content)
        else:
            rest.append(m)
    return "\n\n".join(system_chunks), rest


def _to_anthropic(msg: Message) -> dict[str, Any]:
    if isinstance(msg, UserMessage):
        return {"role": "user", "content": _serialize_user_content(msg.content)}
    if isinstance(msg, AssistantMessage):
        blocks: list[dict[str, Any]] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        for tc in msg.tool_calls:
            blocks.append(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
            )
        return {"role": "assistant", "content": blocks}
    if msg.role is Role.TOOL:
        assert isinstance(msg, ToolResultMessage)
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": r.call_id,
                    "content": r.content,
                    "is_error": r.is_error,
                }
                for r in msg.results
            ],
        }
    raise LLMError(f"Cannot serialize message role: {msg.role}")


def _serialize_user_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, AttachmentManifestPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, FilePart):
            blocks.append({"type": "text", "text": _file_text(part)})
        elif isinstance(part, ImagePart):
            data = base64.b64encode(part.path.read_bytes()).decode("ascii")
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.mime_type,
                        "data": data,
                    },
                }
            )
        else:  # pragma: no cover - defensive
            blocks.append({"type": "text", "text": str(part)})
    return blocks


def _file_text(part: FilePart) -> str:
    return (
        f"<attachment id={part.attachment_id!r} filename={part.filename!r} "
        f"mime_type={part.mime_type!r} size_bytes={part.size_bytes} "
        f"sha256={part.sha256!r}>\n"
        f"{part.text}\n"
        "</attachment>"
    )


def _tool_to_anthropic(tool: ToolSchema) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def _from_anthropic(response: Any) -> LLMResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "tool_use":
            tool_calls.append(
                ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            )

    usage = getattr(response, "usage", None)
    return LLMResponse(
        message=AssistantMessage(content="".join(text_parts), tool_calls=tool_calls),
        stop_reason=getattr(response, "stop_reason", "end_turn"),
        input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
        output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
    )
