"""Terno provisioner proxy implementation of ``LLMClient``.

Instead of calling OpenAI/Anthropic directly, this client forwards every
chat completion to the Terno *provisioner* server (``PROVISIONER_URL``).
The provisioner holds the real OpenAI credentials and generates the
response, then returns it in OpenAI ``chat.completion`` shape. This keeps
provider API keys server-side — the SDK only needs the user's Terno
``api_key``.

Mirrors ``terno-ai/terno/llm/terno_llm.py`` from the main Terno repo, but
adapted to this SDK's neutral ``Message``/``ToolSchema`` types and the
streaming ``LLMClient.complete()`` contract.

The provisioner exposes two request shapes on ``/root/llm/``:

* ``type="stream_response"`` — the provisioner replies with a
  newline-delimited JSON (NDJSON) stream of ``{"status": "streaming", ...}``
  chunks followed by a final ``{"status": "done", ...}`` chunk carrying
  usage/model metadata. This is the default path: ``on_text_delta`` is
  invoked for every text token as it arrives.
* ``type="get_response"`` — a single non-streaming JSON response. Used as an
  automatic fallback when the stream fails or yields no content.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any
import logging

from terno_agent.core.exceptions import ConfigError, LLMError
from terno_agent.core.messages import AssistantMessage, Message, ToolCall
from terno_agent.core.tool import ToolSchema
from terno_agent.llm.base import LLMResponse, TextDeltaCallback
from terno_agent.llm.openai_client import _serialize_messages, _tool_to_openai

# Matches the timeouts used by the main Terno repo's provisioner client.
PROVISIONER_CONNECT_TIMEOUT = 10
PROVISIONER_READ_TIMEOUT = 90

DEFAULT_APP_VERSION = "terno-agent"
DEFAULT_REQUEST_SOURCE = "terno-agent-sdk"

logger = logging.getLogger(__name__)

# Provisioner error codes that will fail the same way on a retry, so the
# stream → get_response fallback is skipped and the error surfaces at once.
NON_RETRYABLE_ERROR_CODES = frozenset(
    {"no_api_key", "unauthorized", "credits_exhausted"}
)


class ProvisionerError(LLMError):
    """A structured error from the provisioner's ``/root/llm/`` endpoint.

    Carries the provisioner's error envelope (``error_code``, ``message``,
    ``details``) so callers can react to the code — e.g. show a billing link
    for ``credits_exhausted`` — instead of parsing a string. ``str(exc)`` is
    just the user-facing ``message``.
    """

    def __init__(
        self,
        error_code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details = details or {}

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> ProvisionerError:
        details = data.get("details")
        return cls(
            error_code=data.get("error_code") or "unknown_error",
            message=data.get("message") or "Something went wrong. Please try again.",
            details=details if isinstance(details, dict) else None,
        )


class TernoLLMClient:
    """Proxy ``LLMClient`` that routes completions through the provisioner."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        provisioner_url: str | None = None,
        app_version: str | None = None,
        request_source: str | None = None,
        connect_timeout: int = PROVISIONER_CONNECT_TIMEOUT,
        read_timeout: int = PROVISIONER_READ_TIMEOUT,
    ) -> None:
        self.model = model
        self.api_key = api_key
        provisioner_url = (
            provisioner_url or os.getenv("PROVISIONER_URL") or ""
        ).strip().rstrip("/")
        if not provisioner_url:
            raise ConfigError(
                "No provisioner URL configured. Set PROVISIONER_URL "
                "(e.g. https://provisioner.terno.ai)."
            )
        self.provisioner_url = provisioner_url
        self.app_version = app_version or os.getenv(
            "APP_VERSION", DEFAULT_APP_VERSION
        )
        self.request_source = request_source or os.getenv(
            "REQUEST_SOURCE", DEFAULT_REQUEST_SOURCE
        )
        # urllib has a single timeout; use the (longer) read timeout as the
        # ceiling for the whole request.
        self._timeout = max(connect_timeout, read_timeout)
        # The provisioner reports the real provider/model it used; these are
        # populated after the first completion (provider is "openai" for now).
        self.last_provider: str | None = None
        self.last_model: str = model

    # ----- LLMClient protocol ------------------------------------------- #

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> LLMResponse:
        # Shared request fields for both the streaming and fallback paths.
        base_payload: dict[str, Any] = {
            "messages": _serialize_messages(messages),
            "tools": [_tool_to_openai(t) for t in (tools or [])] or None,
            "tool_choice": None,
            "priority": False,
            "summarize": False,
            "max_tokens": 25000,
        }
        # api_key is NOT in the payload yet (added inside the HTTP helpers), so
        # this print never leaks it.
        print(
            f"[terno-proxy] → provisioner {self.provisioner_url}/root/llm/ | "
            f"llm_type=terno messages={len(base_payload['messages'])} "
            f"tools={len(tools or [])}"
        )

        # Primary path: stream the response so `on_text_delta` fires per token.
        response = self._stream_completion(base_payload, on_text_delta)
        if response is not None:
            return response

        # Fallback: the stream failed to produce content (empty stream or a
        # transient error). Retry once against the non-streaming endpoint.
        print("[terno-proxy] stream produced no content — falling back to "
              "get_response")
        return self._get_response_completion(base_payload, on_text_delta)

    # ----- Streaming ---------------------------------------------------- #

    def _stream_completion(
        self,
        base_payload: dict[str, Any],
        on_text_delta: TextDeltaCallback | None,
    ) -> LLMResponse | None:
        """Consume the provisioner NDJSON stream into an ``LLMResponse``.

        Returns ``None`` (rather than raising) when the stream cannot be used
        so that ``complete`` can fall back to the non-streaming endpoint. A
        provisioner ``error`` chunk that a retry cannot fix is raised as
        ``ProvisionerError``.
        """
        payload = {**base_payload, "type": "stream_response"}

        text_parts: list[str] = []
        tool_partials: dict[int, dict[str, str]] = {}
        done: dict[str, Any] = {}
        received_done = False
        error_chunk: dict[str, Any] | None = None

        try:
            for data in self._iter_provisioner_stream(payload):
                status = data.get("status")
                if status == "streaming":
                    token = data.get("content") or ""
                    if token:
                        text_parts.append(token)
                        if on_text_delta is not None:
                            on_text_delta(token)
                    _accumulate_tool_calls(tool_partials, data.get("tool_calls"))
                elif status == "error":
                    error_chunk = data
                    break
                elif status == "done":
                    received_done = True
                    done = data
        except Exception as exc:
            # Network/parse hiccup mid-stream: let complete() fall back to the
            # non-streaming endpoint.
            print(f"[terno-proxy] stream error, will fall back: {exc!r}")
            return None

        if error_chunk is not None:
            # If tokens already reached the UI we can't cleanly re-run, and a
            # non-retryable code (e.g. credits_exhausted) would only fail
            # again, so surface the error. Otherwise (nothing streamed yet —
            # e.g. a provisioner that doesn't support stream_response, or a
            # transient stream-start failure) fall back to the non-streaming
            # endpoint, which either succeeds or raises a clean terminal error.
            if text_parts or error_chunk.get("error_code") in NON_RETRYABLE_ERROR_CODES:
                logger.error(
                    "[terno-proxy] stream returned error "
                    f"({error_chunk.get('error_code')}): {error_chunk.get('message')}"
                )
                raise ProvisionerError.from_response(error_chunk)
            logger.debug(
                "[terno-proxy] stream returned error "
                f"({error_chunk.get('error_code')}) before any tokens — "
                "falling back to get_response"
            )
            return None

        tool_calls = _build_tool_calls(tool_partials)
        content = "".join(text_parts).strip()

        if not received_done or (not content and not tool_calls):
            # Nothing usable arrived — signal complete() to fall back.
            return None

        self.last_provider = done.get("llm_provider")
        if done.get("model"):
            self.last_model = done["model"]

        print(
            f"Using provider {done.get('llm_provider')} for llm_type terno "
            f"(model={done.get('model')})"
        )
        print(
            f"[terno-proxy] ← stream done: "
            f"input_tokens={done.get('input_tokens')} "
            f"output_tokens={done.get('output_tokens')} "
            f"tool_calls={len(tool_calls)} content_len={len(content)}"
        )

        stop_reason = done.get("finish_reason") or (
            "tool_calls" if tool_calls else "stop"
        )
        return LLMResponse(
            message=AssistantMessage(content=content, tool_calls=tool_calls),
            stop_reason=stop_reason,
            input_tokens=int(done.get("input_tokens") or 0),
            output_tokens=int(done.get("output_tokens") or 0),
        )

    # ----- Non-streaming fallback --------------------------------------- #

    def _get_response_completion(
        self,
        base_payload: dict[str, Any],
        on_text_delta: TextDeltaCallback | None,
    ) -> LLMResponse:
        payload = {**base_payload, "type": "get_response"}
        data = self._call_provisioner(payload)

        if data.get("status") == "error":
            logger.debug(
                "[terno-proxy] Provisioner returned error "
                f"({data.get('error_code')}): {data.get('message')}"
            )
            raise ProvisionerError.from_response(data)

        # The provisioner returns an OpenAI-shaped ChatCompletionMessage plus
        # usage/model metadata (see TernoLLM.get_response).
        message_dict = data.get("message") or {}
        content = (message_dict.get("content") or "").strip()
        tool_calls = _parse_tool_calls(message_dict.get("tool_calls"))

        # Record which provider/model the provisioner actually used (e.g.
        # llm_provider="openai", model="o4-mini") for logging/introspection.
        self.last_provider = data.get("llm_provider")
        if data.get("model"):
            self.last_model = data["model"]

        print(
            f"Using provider {data.get('llm_provider')} for llm_type terno "
            f"(model={data.get('model')})"
        )
        print(
            f"[terno-proxy] ← provisioner response: status={data.get('status')} "
            f"input_tokens={data.get('input_tokens')} "
            f"output_tokens={data.get('output_tokens')} "
            f"tool_calls={len(tool_calls)} content_len={len(content)}"
        )

        if content and on_text_delta is not None:
            on_text_delta(content)

        stop_reason = data.get("finish_reason") or (
            "tool_calls" if tool_calls else "stop"
        )
        return LLMResponse(
            message=AssistantMessage(content=content, tool_calls=tool_calls),
            stop_reason=stop_reason,
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
        )

    # ----- HTTP --------------------------------------------------------- #

    def _iter_provisioner_stream(self, payload: dict[str, Any]):
        """Yield parsed NDJSON chunks from the streaming provisioner endpoint.

        The provisioner sends one JSON object per line (``application/x-ndjson``).
        ``urllib``'s response object is a file-like iterable that yields lines
        as they arrive, so tokens surface incrementally rather than all at once.

        Transport-level failures (HTTP error, timeout, connection error, e.g.
        an older provisioner that doesn't support ``stream_response``) are left
        to propagate so ``_stream_completion`` can fall back to the
        non-streaming endpoint, which surfaces a clean ``LLMError`` if it too
        fails.
        """
        body = json.dumps({**payload, "api_key": self.api_key}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.provisioner_url}/root/llm/",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/x-ndjson",
                "X-Terno-App-Version": self.app_version,
                "X-Terno-Request-Source": self.request_source,
            },
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Skip a malformed line rather than aborting the stream.
                    continue

    def _call_provisioner(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps({**payload, "api_key": self.api_key}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.provisioner_url}/root/llm/",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Terno-App-Version": self.app_version,
                "X-Terno-Request-Source": self.request_source,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace") if exc.fp else ""
            logger.debug(f"Provisioner HTTP {exc.code}: {(detail or str(exc.reason))[:500]}")
            raise ProvisionerError(
                "http_error",
                "The Terno server returned an error. Please try again.",
                {"status_code": exc.code, "detail": detail[:500] or str(exc.reason)},
            ) from exc
        except TimeoutError as exc:
            logger.debug("[terno-proxy] provisioner request timed out")
            raise ProvisionerError(
                "timeout_error",
                "The Terno server took too long to respond. Please try again.",
            ) from exc
        except urllib.error.URLError as exc:
            logger.debug(f"[terno-proxy] could not reach {self.provisioner_url}: {exc.reason}")
            raise ProvisionerError(
                "network_error",
                "Could not reach the Terno server. Please check your connection and try again.",
                {"reason": str(exc.reason)},
            ) from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.debug(f"[terno-proxy] non-JSON response: {raw[:200]!r}")
            raise ProvisionerError(
                "invalid_response",
                "The Terno server sent an unexpected response. Please try again.",
                {"raw": raw[:200]},
            ) from exc


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    """Map OpenAI-shaped ``tool_calls`` dicts into neutral ``ToolCall``s."""
    if not raw:
        return []
    calls: list[ToolCall] = []
    for tc in raw:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except (json.JSONDecodeError, TypeError, ValueError):
            args = {"_raw": raw_args}
        calls.append(
            ToolCall(id=tc.get("id") or "", name=fn.get("name") or "", arguments=args)
        )
    return calls


def _accumulate_tool_calls(
    partials: dict[int, dict[str, str]], raw: Any
) -> None:
    """Merge streamed OpenAI tool-call deltas into ``partials`` by index.

    Each ``streaming`` chunk carries a partial tool call keyed by ``index``;
    ``id``/``name`` arrive once and ``arguments`` stream in fragments that must
    be concatenated. Mirrors ``_consume_openai_stream`` in the OpenAI client.
    """
    for tc in raw or []:
        idx = tc.get("index", 0)
        partial = partials.setdefault(idx, {"id": "", "name": "", "args": ""})
        if tc.get("id"):
            partial["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            partial["name"] += fn["name"]
        if fn.get("arguments"):
            partial["args"] += fn["arguments"]


def _build_tool_calls(partials: dict[int, dict[str, str]]) -> list[ToolCall]:
    """Finalize accumulated tool-call partials into neutral ``ToolCall``s."""
    calls: list[ToolCall] = []
    for _idx, partial in sorted(partials.items()):
        try:
            args = json.loads(partial["args"] or "{}")
        except json.JSONDecodeError:
            args = {"_raw": partial["args"]}
        calls.append(
            ToolCall(id=partial["id"], name=partial["name"], arguments=args)
        )
    return calls


__all__ = ["NON_RETRYABLE_ERROR_CODES", "ProvisionerError", "TernoLLMClient"]
