"""LLM-driven history compaction hook.

When the most recent LLM call reports more than `threshold_input_tokens`
of context, the hook asks the LLM to summarize the older portion of the
conversation. The agent's `history` is rewritten in place as::

    [SystemMessage(original prompt),
     UserMessage("This session is being continued … <summary> … <file state>"),
     <last `keep_last_turns` user/assistant rounds verbatim>]

The summary arrives as a *user* message, not a system one: as a system message
it would read as a standing instruction rather than as recovered conversation
(and Terno's Anthropic client hoists system messages into the top-level
`system` param, so it would land in the cached system prompt).

`keep_last_turns` counts user turns (one user message + everything
between it and the next user message). Tool-result messages immediately
following a kept assistant message are kept too so tool_use ↔ tool_result
pairing stays intact.

If summarization fails (LLM error, no API key, etc.) the hook logs to
stderr and leaves history untouched.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from terno_agent.core.hooks import HookContext
from terno_agent.core.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    TextPart,
    ToolResultMessage,
    UserMessage,
)
from terno_agent.llm.base import LLMClient
from terno_agent.prompts.compaction import (
    extract_summary,
    render_read_replay,
    with_instructions,
    wrap_summary,
)


@dataclass(slots=True)
class CompactionHook:
    """Summarize older history once `last_input_tokens` exceeds the threshold."""

    llm: LLMClient
    threshold_input_tokens: int = 80_000
    keep_last_turns: int = 4
    # The reference summary is ~1.4k tokens of structured output, and the
    # <analysis> block is generated before it and then discarded — so this
    # needs real headroom. Truncation here silently loses the tail sections
    # (Pending Tasks, Current Work), which are the ones worth keeping.
    max_summary_tokens: int = 8192
    # Project-specific summarisation guidance, e.g. "focus on test output and
    # code changes". The prompt already tells the model to honour instructions
    # "provided in the included context"; this is what provides them.
    extra_instructions: str = ""
    # Re-read files that were Read in the compacted turns and carry their current
    # contents across the boundary. Without this the agent remembers only THAT it
    # read a file, not what the file said — the summary rarely preserves whole
    # files. Only reads are replayed: Edit/Bash output is transient, and the
    # reference harness replays reads alone (verified in the capture).
    replay_file_reads: bool = True
    # When the client keeps mid-conversation system turns in place (see
    # `AnthropicClient(mid_conversation_system=True)`), deliver the replayed file
    # state as its own trailing system message — that is the captured shape.
    # Otherwise it is folded into the summary's user message, because a hoisted
    # system message would land in the cached system prompt instead.
    replay_as_system_message: bool = False
    max_replay_chars: int = 20_000
    # Injected for testing; defaults to a plain UTF-8 read.
    file_reader: Callable[[str], str | None] | None = None

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

        replay = self._read_replay(head) if self.replay_file_reads else ""
        # Skill bodies are instructions the agent is mid-way through following.
        # A summary rarely reproduces them faithfully, so carry them verbatim.
        skills = _skill_replay(head)
        if skills:
            replay = f"{skills}\n\n{replay}" if replay else skills

        # Rewrite in place so the agent's `self.history` reference stays valid.
        # The summary comes back as a USER message, matching the reference
        # harness — a system message would read as a standing instruction
        # rather than as recovered conversation.
        if replay and self.replay_as_system_message:
            ctx.history[:] = [
                system_msg,
                UserMessage(wrap_summary(summary)),
                *tail,
                SystemMessage(replay),
            ]
        else:
            ctx.history[:] = [
                system_msg,
                UserMessage(wrap_summary(summary, replay)),
                *tail,
            ]
        # Zero out `last_input_tokens` so the next no-op call doesn't re-trigger
        # before the LLM has reported actual usage on the smaller context.
        ctx.usage.last_input_tokens = 0

    # ----- internals ---------------------------------------------------- #

    def _summarize(self, system_msg: SystemMessage, head: list[Message]) -> str | None:
        # Ask in-conversation, the way the reference harness does: replay the
        # turns being compacted and append the request as a trailing user
        # message, so the model summarises what it can already see rather than a
        # transcript we flatten for it. `tools=None` makes a tool call
        # impossible rather than merely forbidden.
        prompt_messages: list[Message] = [
            system_msg,
            *head,
            UserMessage(with_instructions(self.extra_instructions)),
        ]
        try:
            response = self.llm.complete(
                prompt_messages,
                tools=None,
                max_tokens=self.max_summary_tokens,
            )
        except Exception as exc:
            print(f"warning: compaction summarization failed: {exc}", file=sys.stderr)
            return None
        text = extract_summary(response.message.content or "")
        return text or None

    def _read_replay(self, head: list[Message]) -> str:
        """Re-read every file the compacted turns Read, newest path order last.

        Paths are deduplicated — a file read three times is carried once — and
        the whole block is capped, because the point of compaction is to shrink
        the context, not to smuggle the whole repo back in.
        """
        paths = _read_paths(head)
        if not paths:
            return ""

        reader = self.file_reader or _read_text
        entries: list[tuple[str, str]] = []
        budget = self.max_replay_chars
        for path in paths:
            content = reader(path)
            if content is None:
                continue  # deleted, renamed, or unreadable since it was read
            numbered = _number_lines(content)
            if len(numbered) > budget:
                break  # keep whole files only; a truncated file is misleading
            budget -= len(numbered)
            entries.append((json.dumps({"file_path": path}), numbered))
        return render_read_replay(entries)


# --------------------------------------------------------------------------- #
# History slicing
# --------------------------------------------------------------------------- #


_SKILL_BLOCK = re.compile(
    r"<skill_content name=\"([^\"]+)\">.*?</skill_content>", re.S
)


def _skill_replay(messages: list[Message]) -> str:
    """Verbatim bodies of skills activated in the turns being compacted.

    Keyed by skill name so a skill activated twice is carried once, keeping the
    latest copy. Without this the agent loses instructions it is still meant to
    be following, and — because the loss is silent — simply stops following them.
    """
    found: dict[str, str] = {}
    for m in messages:
        texts: list[str] = []
        if isinstance(m, UserMessage) and isinstance(m.content, str):
            texts.append(m.content)
        elif isinstance(m, UserMessage):
            texts.extend(p.text for p in m.content if isinstance(p, TextPart))
        elif isinstance(m, ToolResultMessage):
            texts.extend(r.followup_text for r in m.results if r.followup_text)
        for text in texts:
            for match in _SKILL_BLOCK.finditer(text):
                found[match.group(1)] = match.group(0)
    return "\n\n".join(found.values())


def _read_paths(messages: list[Message]) -> list[str]:
    """`file_path` args of every successful Read call, in first-seen order."""
    failed: set[str] = set()
    for m in messages:
        if isinstance(m, ToolResultMessage):
            failed.update(r.call_id for r in m.results if r.is_error)

    seen: dict[str, None] = {}
    for m in messages:
        if not isinstance(m, AssistantMessage):
            continue
        for tc in m.tool_calls or ():
            if tc.name != "Read" or tc.id in failed:
                continue
            path = (tc.arguments or {}).get("file_path")
            if isinstance(path, str) and path:
                seen.setdefault(path, None)
    return list(seen)


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _number_lines(text: str) -> str:
    """`cat -n` style, matching what the Read tool returns."""
    return "\n".join(f"{i}\t{line}" for i, line in enumerate(text.splitlines(), start=1))


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


__all__ = ["CompactionHook"]
