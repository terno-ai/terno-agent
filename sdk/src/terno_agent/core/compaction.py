"""LLM-driven history compaction hook.

When the most recent LLM call reports more than `threshold_input_tokens`
of context, the hook asks the LLM to summarize the older portion of the
conversation. The agent's `history` is rewritten in place as::

    [SystemMessage(original prompt),
     SystemMessage("Conversation summary so far: …"),
     <last `keep_last_turns` user/assistant rounds verbatim>]

`keep_last_turns` counts user turns (one user message + everything
between it and the next user message). Tool-result messages immediately
following a kept assistant message are kept too so tool_use ↔ tool_result
pairing stays intact.

If summarization fails (LLM error, no API key, etc.) the hook logs to
stderr and leaves history untouched.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from terno_agent.core.hooks import HookContext
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

_SUMMARY_PROMPT = (
    "You are compacting an in-progress coding-agent conversation so the "
    "agent can keep working with less context. Produce a tight, faithful "
    "summary of what the user asked for, what was done, what was decided, "
    "and any open threads or pending tasks. Preserve file paths, function "
    "names, and exact error strings. Do NOT add commentary or apologize for "
    "missing detail. Output plain prose, no markdown headings."
)

_SUMMARY_HEADER = "Conversation summary (older turns compacted):"


@dataclass(slots=True)
class CompactionHook:
    """Summarize older history once `last_input_tokens` exceeds the threshold."""

    llm: LLMClient
    threshold_input_tokens: int = 80_000
    keep_last_turns: int = 4
    max_summary_tokens: int = 1024

    def __call__(self, ctx: HookContext) -> None:
        if ctx.usage.last_input_tokens < self.threshold_input_tokens:
            return
        if len(ctx.history) <= 2:
            return  # nothing to compact

        system_msg, head, tail = _split_history(ctx.history, self.keep_last_turns)
        if not head:
            return  # everything is already in the "kept" window

        summary = self._summarize(system_msg, head)
        if summary is None:
            return

        # Rewrite in place so the agent's `self.history` reference stays valid.
        ctx.history[:] = [system_msg, SystemMessage(f"{_SUMMARY_HEADER}\n{summary}"), *tail]
        # Zero out `last_input_tokens` so the next no-op call doesn't re-trigger
        # before the LLM has reported actual usage on the smaller context.
        ctx.usage.last_input_tokens = 0

    # ----- internals ---------------------------------------------------- #

    def _summarize(self, system_msg: SystemMessage, head: list[Message]) -> str | None:
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
            response = self.llm.complete(
                prompt_messages,
                tools=None,
                max_tokens=self.max_summary_tokens,
                temperature=0.2,
            )
        except Exception as exc:
            print(f"warning: compaction summarization failed: {exc}", file=sys.stderr)
            return None
        text = (response.message.content or "").strip()
        return text or None


# --------------------------------------------------------------------------- #
# History slicing
# --------------------------------------------------------------------------- #


def _split_history(
    history: list[Message], keep_last_turns: int
) -> tuple[SystemMessage, list[Message], list[Message]]:
    """Return (system_msg, head_to_compact, tail_to_keep).

    `keep_last_turns` counts user messages from the tail. The tail
    starts at the Nth-from-last UserMessage so each kept turn includes
    its assistant reply (and any tool exchanges in between).
    """
    if not history:
        return SystemMessage(""), [], []
    system_msg = history[0] if isinstance(history[0], SystemMessage) else SystemMessage("")
    body_start = 1 if isinstance(history[0], SystemMessage) else 0
    body = history[body_start:]

    user_idxs = [i for i, m in enumerate(body) if isinstance(m, UserMessage)]
    if len(user_idxs) <= keep_last_turns:
        return system_msg, [], body
    boundary = user_idxs[-keep_last_turns]
    return system_msg, body[:boundary], body[boundary:]


def _render_for_summary(messages: list[Message]) -> str:
    parts: list[str] = []
    for m in messages:
        if isinstance(m, UserMessage):
            parts.append(f"USER:\n{_render_user_content(m.content)}")
        elif isinstance(m, AssistantMessage):
            line = f"ASSISTANT:\n{m.content}" if m.content else "ASSISTANT:"
            if m.tool_calls:
                names = ", ".join(f"{tc.name}" for tc in m.tool_calls)
                line += f"\n[tool_calls: {names}]"
            parts.append(line)
        elif isinstance(m, ToolResultMessage):
            for r in m.results:
                marker = "ERROR" if r.is_error else "ok"
                parts.append(f"TOOL_RESULT [{marker}]:\n{r.content}")
        elif isinstance(m, SystemMessage):
            parts.append(f"SYSTEM_NOTE:\n{m.content}")
    return "\n\n".join(parts)


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


__all__ = ["CompactionHook"]
