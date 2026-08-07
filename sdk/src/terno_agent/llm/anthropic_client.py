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
    SystemBlock,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from terno_agent.core.tool import ToolSchema
from terno_agent.llm.base import LLMResponse, TextDeltaCallback

# Required for `role: "system"` turns inside `messages`.
MID_CONVERSATION_SYSTEM_BETA = "mid-conversation-system-2026-04-07"
# `cache_control.ttl` other than the 5-minute default.
EXTENDED_CACHE_TTL_BETA = "extended-cache-ttl-2025-04-11"
# `cache_control.scope`, which shares a cache entry across sessions.
CACHE_SCOPE_BETA = "prompt-caching-scope-2026-01-05"


class AnthropicClient:
    """Wraps `anthropic.Anthropic` and translates to/from neutral messages."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        effort: str | None = None,
        thinking: str | None = None,
        clear_thinking: bool = False,
        mid_conversation_system: bool = False,
        extended_cache_ttl: bool = False,
        cache_scope: bool = False,
    ) -> None:
        """
        `effort`, `thinking` and `clear_thinking` mirror knobs the reference
        harness sends on every request (`effort="medium"`,
        `thinking="adaptive"`, `clear_thinking=True`). They are beta features
        that a given account or API version may reject, so all three default to
        off — opt in once you've confirmed your account accepts them.

        `mid_conversation_system` keeps non-leading `SystemMessage`s in place as
        `role: "system"` turns instead of hoisting them into the top-level
        `system` param. That is how the reference harness delivers mid-run
        context (deferred-tool rosters, replayed file state, reminders) — a
        hoisted system message would instead land in the cached system prompt and
        apply to the whole conversation. Requires the
        `mid-conversation-system` beta, so it also defaults to off.

        `extended_cache_ttl` and `cache_scope` allow the `ttl` and `scope` fields
        of `cache_control`. The prompt builder always produces the captured
        values (`ttl="1h"`, `scope="global"`), but both fields are beta-gated and
        the API rejects the whole request with a 400 if they are sent without
        their beta — so they are STRIPPED unless enabled here. Caching still
        works without them, just at the 5-minute default TTL and per-session
        rather than shared.
        """
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ConfigError(
                "anthropic package not installed. "
                "Install with: pip install 'terno-agent[anthropic]'"
            ) from exc
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.model = model
        self.effort = effort
        self.thinking = thinking
        self.clear_thinking = clear_thinking
        self.mid_conversation_system = mid_conversation_system
        self.extended_cache_ttl = extended_cache_ttl
        self.cache_scope = cache_scope

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> LLMResponse:
        system, history = _split_system(
            messages,
            keep_mid_conversation=self.mid_conversation_system,
            allow_ttl=self.extended_cache_ttl,
            allow_scope=self.cache_scope,
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system or "",
            "messages": [_to_anthropic(m) for m in history],
        }
        # The reference harness sends no sampling params at all, letting the
        # model default apply. Only send one if a caller explicitly asked.
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.thinking:
            kwargs["thinking"] = {"type": self.thinking}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.clear_thinking:
            # Drop earlier thinking blocks server-side so they don't accumulate.
            kwargs["context_management"] = {
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
            }
        betas = [
            beta
            for enabled, beta in (
                (self.mid_conversation_system, MID_CONVERSATION_SYSTEM_BETA),
                (self.extended_cache_ttl, EXTENDED_CACHE_TTL_BETA),
                (self.cache_scope, CACHE_SCOPE_BETA),
            )
            if enabled
        ]
        if betas:
            kwargs["extra_headers"] = {"anthropic-beta": ",".join(betas)}
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


def system_block_to_anthropic(
    block: SystemBlock, *, allow_ttl: bool = True, allow_scope: bool = True
) -> dict[str, Any]:
    """Render one system block, carrying its cache breakpoint if it has one.

    `ttl` and `scope` are beta-gated server-side; sending either without its beta
    header 400s the entire request. Callers that haven't opted in pass False and
    get a plain `{"type": "ephemeral"}` breakpoint, which every account accepts.
    """
    out: dict[str, Any] = {"type": "text", "text": block.text}
    if block.cache_control is not None:
        cache = dict(block.cache_control)
        if not allow_ttl:
            cache.pop("ttl", None)
        if not allow_scope:
            cache.pop("scope", None)
        out["cache_control"] = cache
    return out


def _split_system(
    messages: list[Message],
    *,
    keep_mid_conversation: bool = False,
    allow_ttl: bool = True,
    allow_scope: bool = True,
) -> tuple[str | list[dict[str, Any]], list[Message]]:
    """Pull system messages out of the trace.

    Returns a block list when any system message carries structured blocks — the
    only way to express per-block cache breakpoints — and a plain joined string
    otherwise.

    With `keep_mid_conversation`, only the *leading* system message is hoisted;
    later ones stay in the message list as `role: "system"` turns, preserving
    their position in the conversation. Without it every system message is
    hoisted and merged, which is the portable behaviour but loses ordering.
    """
    chunks: list[str] = []
    blocks: list[SystemBlock] = []
    rest: list[Message] = []
    leading = True
    for m in messages:
        if isinstance(m, SystemMessage):
            if keep_mid_conversation and not leading:
                rest.append(m)
                continue
            chunks.append(m.content)
            blocks.extend(m.blocks or [SystemBlock(m.content)])
        else:
            leading = False
            rest.append(m)

    if any(b.cache_control is not None for b in blocks):
        return [
            system_block_to_anthropic(b, allow_ttl=allow_ttl, allow_scope=allow_scope)
            for b in blocks
        ], rest
    return "\n\n".join(chunks), rest


def _to_anthropic(msg: Message) -> dict[str, Any]:
    if isinstance(msg, SystemMessage):
        # Only reached when `mid_conversation_system` is on; otherwise system
        # messages never make it into the message list.
        return {"role": "system", "content": [{"type": "text", "text": msg.content}]}
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
        blocks: list[dict[str, Any]] = [
            {
                "type": "tool_result",
                "tool_use_id": r.call_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in msg.results
        ]
        # Follow-up text trails the tool_result blocks in the same user turn.
        blocks.extend(
            {"type": "text", "text": r.followup_text}
            for r in msg.results
            if r.followup_text
        )
        return {"role": "user", "content": blocks}
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
