"""OpenAI implementation of `LLMClient`."""

from __future__ import annotations

import json
from typing import Any

from terno_agent.core.exceptions import ConfigError, LLMError
from terno_agent.core.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from terno_agent.core.tool import ToolSchema
from terno_agent.llm.base import LLMResponse, TextDeltaCallback


class OpenAIClient:
    """Wraps `openai.OpenAI` and translates to/from neutral messages."""

    def __init__(self, *, api_key: str | None = None, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ConfigError(
                "openai package not installed. Install with: pip install 'terno-agent[openai]'"
            ) from exc
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _serialize_messages(messages),
            "tools": [_tool_to_openai(t) for t in (tools or [])] or None,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if _supports_custom_temperature(self.model):
            kwargs["temperature"] = temperature

        from terno_agent.core.exceptions import AgentCancelled

        try:
            stream = self._client.chat.completions.create(**kwargs)
            return _consume_openai_stream(stream, on_text_delta)
        except AgentCancelled:
            raise
        except Exception as exc:
            raise LLMError(f"OpenAI API call failed: {exc}") from exc


_FIXED_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _supports_custom_temperature(model: str) -> bool:
    """Newer OpenAI models (gpt-5.x, o1/o3/o4 series) only accept the default temperature."""
    name = model.lower()
    return not any(name.startswith(p) for p in _FIXED_TEMPERATURE_PREFIXES)


def _serialize_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AssistantMessage):
            payload: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(payload)
        elif isinstance(m, ToolResultMessage):
            for r in m.results:
                out.append({"role": "tool", "tool_call_id": r.call_id, "content": r.content})
        else:  # pragma: no cover - defensive
            raise LLMError(f"Cannot serialize message of type {type(m).__name__}")
    return out


def _tool_to_openai(tool: ToolSchema) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _consume_openai_stream(
    stream: Any, on_text_delta: TextDeltaCallback | None
) -> LLMResponse:
    """Assemble a streamed chat-completion into an `LLMResponse`.

    OpenAI streams tool calls as a sequence of partial deltas keyed by an
    integer ``index``. We accumulate ``id``, ``function.name`` and
    ``function.arguments`` per index.
    """
    text_parts: list[str] = []
    tool_partials: dict[int, dict[str, str]] = {}
    finish_reason = "stop"
    prompt_tokens = 0
    completion_tokens = 0

    for chunk in stream:
        if getattr(chunk, "usage", None):
            prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta is None:
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            continue
        if delta.content:
            text_parts.append(delta.content)
            if on_text_delta is not None:
                on_text_delta(delta.content)
        for tc_delta in (delta.tool_calls or []):
            partial = tool_partials.setdefault(
                tc_delta.index, {"id": "", "name": "", "args": ""}
            )
            if tc_delta.id:
                partial["id"] = tc_delta.id
            fn = getattr(tc_delta, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    partial["name"] += fn.name
                if getattr(fn, "arguments", None):
                    partial["args"] += fn.arguments
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    tool_calls: list[ToolCall] = []
    for _idx, partial in sorted(tool_partials.items()):
        try:
            args = json.loads(partial["args"] or "{}")
        except json.JSONDecodeError:
            args = {"_raw": partial["args"]}
        tool_calls.append(
            ToolCall(id=partial["id"], name=partial["name"], arguments=args)
        )

    return LLMResponse(
        message=AssistantMessage(content="".join(text_parts), tool_calls=tool_calls),
        stop_reason=finish_reason,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
    )
