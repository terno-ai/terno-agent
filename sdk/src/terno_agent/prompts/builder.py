"""Assemble the system prompt as an ordered list of cacheable blocks.

The captured wire shape is four blocks with two cache breakpoints:

    [0] billing/telemetry header      no cache    (Terno omits this)
    [1] identity one-liner            no cache
    [2] core + security + harness     ephemeral 1h, scope=global
    [3] session template              ephemeral 1h  (session-local)

`scope="global"` on block 2 is deliberate: that text is byte-identical for every
session and every user, so the cache entry is shared. Block 3 embeds the working
directory and git snapshot, so its entry is session-local.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from terno_agent.core.messages import SystemBlock
from terno_agent.prompts import blocks
from terno_agent.prompts.context import PromptContext

# Cache breakpoints, matching the capture.
_GLOBAL_CACHE: dict[str, Any] = {"type": "ephemeral", "ttl": "1h", "scope": "global"}
_SESSION_CACHE: dict[str, Any] = {"type": "ephemeral", "ttl": "1h"}


@dataclass(slots=True)
class SystemPrompt:
    """The full system prompt as blocks, plus conveniences."""

    blocks: list[SystemBlock] = field(default_factory=list)

    def to_anthropic(self) -> list[dict[str, Any]]:
        from terno_agent.llm.anthropic_client import system_block_to_anthropic

        return [system_block_to_anthropic(b) for b in self.blocks]

    def render(self) -> str:
        """Flatten to one string, for providers without multi-block system."""
        return "\n\n".join(b.text for b in self.blocks)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.render()


def _fill(template: str, **tokens: str) -> str:
    for key, value in tokens.items():
        template = template.replace(f"{{{{{key}}}}}", value)
    return template


def render_session_block(
    ctx: PromptContext, *, skill_section: str | None = None
) -> str:
    """Render block 3 — the session-scoped template.

    Sections appear in the captured order. Each is omitted rather than left
    empty when it has nothing to say, so the prompt never advertises a
    capability this build lacks.
    """
    sections: list[str] = [
        blocks.CODE_STYLE,
        blocks.PRONOUNS,
        blocks.CARE_AND_REPORTING,
    ]

    guidance = []
    if ctx.supports_bang_prefix:
        guidance.append(blocks.GUIDANCE_BANG_PREFIX)
    if ctx.supports_skills:
        guidance.append(blocks.GUIDANCE_SKILLS)
    if guidance:
        bullets = "\n".join(f" - {g}" for g in guidance)
        sections.append(f"# Session-specific guidance\n{bullets}")

    sections.append(_fill(blocks.MEMORY, MEMORY_DIR=str(ctx.memory_dir)))
    sections.append(
        _fill(
            blocks.ENVIRONMENT,
            ENV_LINES="\n".join(f" - {line}" for line in ctx.env_lines()),
        )
    )
    sections.append(_fill(blocks.LANGUAGE, LANG=ctx.language))
    sections.append(_fill(blocks.SCRATCHPAD, SCRATCHPAD_DIR=str(ctx.scratchpad_dir)))
    sections.append(blocks.CONTEXT_MANAGEMENT)
    sections.append(blocks.DELIVERING_WORK)
    sections.append(blocks.CORRECTIONS)

    restrictions = []
    if ctx.has_agent_tool:
        restrictions.append(blocks.NO_UNPROMPTED_AGENT)
    if ctx.has_workflow_tool:
        restrictions.append(blocks.NO_UNPROMPTED_WORKFLOW)
    if restrictions:
        sections.append("\n".join(restrictions))

    if skill_section:
        sections.append(skill_section)

    git = ctx.git.render()
    if git:
        sections.append(_fill(blocks.GIT_STATUS, GIT_STATUS=git))

    return "\n\n".join(sections)


def build_system_prompt(
    ctx: PromptContext | None = None,
    *,
    skill_section: str | None = None,
    extra: str | None = None,
) -> SystemPrompt:
    """Build the block list for a session.

    `extra` is appended as its own uncached block — that's where a caller's
    custom system prompt (e.g. a subagent brief) goes, so it never invalidates
    the cached blocks above it.
    """
    ctx = ctx or PromptContext.detect()
    result = [
        SystemBlock(blocks.IDENTITY),
        SystemBlock(blocks.CORE, cache_control=_GLOBAL_CACHE),
        SystemBlock(
            render_session_block(ctx, skill_section=skill_section),
            cache_control=_SESSION_CACHE,
        ),
    ]
    if extra:
        result.append(SystemBlock(extra))
    return SystemPrompt(result)


def render_system_prompt(
    ctx: PromptContext | None = None, *, skill_section: str | None = None
) -> str:
    """Flattened system prompt, for callers that still want one string."""
    return build_system_prompt(ctx, skill_section=skill_section).render()


__all__ = [
    "SystemBlock",
    "SystemPrompt",
    "build_system_prompt",
    "render_session_block",
    "render_system_prompt",
]
