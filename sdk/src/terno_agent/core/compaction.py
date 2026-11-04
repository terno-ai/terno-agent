"""LLM-driven history compaction hook.

:class:`CompactionHook` runs on ``pre_llm`` — before every LLM call, the
current turn included. Every call re-sends the whole history, so a long turn
pays for its early steps again on each later step. Once the history (messages
only — not the system prompt or tool schemas, which compaction can't shrink)
passes ``trigger_tokens``, older messages are summarized so it drops to about
``target_tokens``. The gap between the two keeps it from firing on every
step. The agent's `history` is rewritten in place as::

    [SystemMessage(original prompt),
     SystemMessage("Conversation summary (older turns compacted): …"),
     <current user message>,
     <last `keep_steps` tool steps verbatim>]

A step is an assistant message with its tool calls plus the results message
that follows, so tool_use ↔ tool_result pairing stays intact.

The summary also covers the kept steps (they stay verbatim in memory too), so
it is complete up to that moment. A :class:`CompactionEvent` with
``preserved_messages`` lets the host persist it and rebuild later turns from it.

If summarization fails (LLM error, no API key, etc.) the hook logs to
stderr and leaves history untouched.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from terno_agent.core.events import CompactionEvent
from terno_agent.core.hooks import HookContext
from terno_agent.core.messages import (
    AssistantMessage,
    AttachmentManifestPart,
    FilePart,
    ImagePart,
    Message,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from terno_agent.llm.base import LLMClient

_SUMMARY_PROMPT = (
    "You are compacting an in-progress coding-agent conversation so the "
    "agent can keep working with less context. Produce a tight, faithful "
    "summary of what the user asked for, what was done, what was decided, "
    "and any open threads or pending tasks. Preserve file paths, function "
    "names, and exact error strings. Do NOT add commentary or apologize for "
    "missing detail. Output plain prose, no markdown headings."
)

SUMMARY_HEADER = "Conversation summary (older turns compacted):"

# Rough chars-per-token ratio for sizing history without a tokenizer.
_CHARS_PER_TOKEN = 4
# What an image part is counted as when sizing history.
_IMAGE_TOKENS = 1_000

# In the summarizer's transcript: file bodies and edit strings are already on
# disk, so only their size is shown; other arguments and tool results are cut
# to head + tail.
_BULKY_ARGS = frozenset({"content", "old_string", "new_string"})
_ARG_CHARS = 2_000
_RESULT_CHARS = 4_000


def _log(message: str) -> None:
    """Emit a compaction trace line.

    Printed to stderr (like the module's failure warnings) so it shows up in a
    host's console — e.g. the Django runserver output — without depending on
    any logging configuration.
    """
    print(f"[compaction] {message}", file=sys.stderr)


@dataclass(slots=True)
class CompactionHook:
    """Before an LLM call, summarize older history once it passes `trigger_tokens`.

    Keeps the current user message and its last `keep_steps` tool steps; fewer
    steps when those alone would leave the history above `target_tokens`.
    """

    llm: LLMClient
    trigger_tokens: int = 20_000
    target_tokens: int = 8_000
    keep_steps: int = 3
    max_summary_tokens: int = 1_500
    # Below this there is too little to summarize to be worth an LLM call;
    # the size is then in the kept steps, which compaction leaves alone.
    min_head_tokens: int = 1_000
    # After a failed summary, wait until the history has grown to this size
    # instead of retrying on every call.
    _retry_at_tokens: int = field(default=0, init=False, repr=False)

    def __call__(self, ctx: HookContext) -> None:
        size = estimate_tokens(ctx.history[1:])
        if size < max(self.trigger_tokens, self._retry_at_tokens):
            return

        target = min(self.target_tokens, self.trigger_tokens)
        split = _split_current_turn(
            ctx.history, self.keep_steps, target - self.max_summary_tokens
        )
        if split is None:
            return
        system_msg, head, kept = split
        if all(_is_summary(m) for m in head) or estimate_tokens(head) < self.min_head_tokens:
            return

        # The kept steps go into the summary too, so a host that persists it
        # needs nothing else from before this moment.
        summary = _summarize(self.llm, head + kept[1:], self.max_summary_tokens)
        if summary is None:
            self._retry_at_tokens = size + self.trigger_tokens // 4
            return

        ctx.history[:] = [system_msg, SystemMessage(f"{SUMMARY_HEADER}\n{summary}"), *kept]
        self._retry_at_tokens = 0
        _log(
            f"history ~{size:,} -> ~{estimate_tokens(ctx.history[1:]):,} tokens; "
            f"{len(head)} messages summarized, {len(kept) - 1} kept after the user message"
        )
        if ctx.agent is not None:
            ctx.agent._emit(
                CompactionEvent(
                    agent=ctx.agent.name,
                    summary=summary,
                    preserved_turns=1,
                    preserved_messages=len(kept) - 1,
                )
            )


def estimate_tokens(messages: list[Message]) -> int:
    """Approximate token count of ``messages`` (characters / 4)."""
    chars = 0
    images = 0
    for m in messages:
        if isinstance(m, SystemMessage):
            chars += len(m.content)
        elif isinstance(m, UserMessage):
            if isinstance(m.content, str):
                chars += len(m.content)
            else:
                for part in m.content:
                    if isinstance(part, ImagePart):
                        images += 1
                    elif isinstance(part, (TextPart, AttachmentManifestPart)):
                        chars += len(part.text)
        elif isinstance(m, AssistantMessage):
            chars += len(m.content or "")
            for tc in m.tool_calls:
                chars += len(tc.name) + len(_dumps(tc.arguments))
        elif isinstance(m, ToolResultMessage):
            chars += sum(len(r.content or "") for r in m.results)
    return chars // _CHARS_PER_TOKEN + images * _IMAGE_TOKENS


# ----- internals ---------------------------------------------------------- #


def _summarize(llm: LLMClient, head: list[Message], max_tokens: int) -> str | None:
    transcript = _render_for_summary(head)
    prompt_messages: list[Message] = [
        SystemMessage(_SUMMARY_PROMPT),
        UserMessage(
            "Conversation so far (verbatim, oldest first):\n\n"
            f"{transcript}\n\n"
            "Write the summary now."
        ),
    ]
    try:
        response = llm.complete(
            prompt_messages,
            tools=None,
            max_tokens=max_tokens,
            temperature=0.2,
        )
    except Exception as exc:
        print(f"warning: compaction summarization failed: {exc}", file=sys.stderr)
        return None
    print(
        f"[tokens] summary call: in={response.input_tokens:,} out={response.output_tokens:,} "
        f"({len(head)} messages, ~{estimate_tokens(head):,} tokens summarized)",
        file=sys.stderr,
    )
    text = (response.message.content or "").strip()
    return text or None


def _is_summary(message: Message) -> bool:
    return isinstance(message, SystemMessage) and message.content.startswith(SUMMARY_HEADER)


# --------------------------------------------------------------------------- #
# History slicing
# --------------------------------------------------------------------------- #


def _split_current_turn(
    history: list[Message], keep_steps: int, kept_budget_tokens: int
) -> tuple[SystemMessage, list[Message], list[Message]] | None:
    """Return (system_msg, head_to_compact, kept).

    ``kept`` is the current (last) user message followed by its last
    ``keep_steps`` tool steps — an assistant message and the results that
    follow it — reduced to one step when more would not fit in
    ``kept_budget_tokens``. ``head`` is everything else: earlier turns and
    this turn's older steps. ``None`` when there is no user message.
    """
    if not history:
        return None
    system_msg = history[0] if isinstance(history[0], SystemMessage) else SystemMessage("")
    body = history[1:] if isinstance(history[0], SystemMessage) else list(history)

    user_idx = next(
        (i for i in range(len(body) - 1, -1, -1) if isinstance(body[i], UserMessage)), None
    )
    if user_idx is None:
        return None
    steps = [
        i for i in range(user_idx + 1, len(body)) if isinstance(body[i], AssistantMessage)
    ]
    keep = min(keep_steps, len(steps))
    user_tokens = estimate_tokens([body[user_idx]])
    while keep > 1 and user_tokens + estimate_tokens(body[steps[-keep]:]) > kept_budget_tokens:
        keep -= 1
    tail_start = steps[-keep] if keep else len(body)

    head = body[:user_idx] + body[user_idx + 1 : tail_start]
    kept = [body[user_idx], *body[tail_start:]]
    return system_msg, head, kept


def _render_for_summary(messages: list[Message]) -> str:
    parts: list[str] = []
    for m in messages:
        if isinstance(m, UserMessage):
            parts.append(f"USER:\n{_render_user_content(m.content)}")
        elif isinstance(m, AssistantMessage):
            line = f"ASSISTANT:\n{m.content}" if m.content else "ASSISTANT:"
            for tc in m.tool_calls:
                line += f"\n[tool_call] {_render_call(tc)}"
            parts.append(line)
        elif isinstance(m, ToolResultMessage):
            for r in m.results:
                marker = "ERROR" if r.is_error else "ok"
                parts.append(f"TOOL_RESULT [{marker}]:\n{_clip(r.content or '', _RESULT_CHARS)}")
        elif isinstance(m, SystemMessage):
            parts.append(f"SYSTEM_NOTE:\n{m.content}")
    return "\n\n".join(parts)


def _render_call(tc: ToolCall) -> str:
    args: list[str] = []
    for key, value in (tc.arguments or {}).items():
        text = value if isinstance(value, str) else _dumps(value)
        if key in _BULKY_ARGS and len(text) > 200:
            text = f"[{len(text):,} chars]"
        else:
            text = _clip(text, _ARG_CHARS)
        args.append(f"{key}={text}")
    return f"{tc.name}({', '.join(args)})"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n...[{len(text) - limit:,} chars omitted]...\n{text[-half:]}"


def _dumps(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _render_user_content(content) -> str:
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
                f"mime={part.mime_type} size={part.size_bytes} sha256={part.sha256}; "
                "contents omitted from compaction transcript]"
            )
        else:  # pragma: no cover - defensive
            rendered.append(str(part))
    return "\n\n".join(rendered)


__all__ = ["SUMMARY_HEADER", "CompactionHook", "estimate_tokens"]
