"""Slash invocation of skills — `/name` injects the body with no tool call.

The reference harness has two ways into a skill, and they are structurally
different (both verified on the wire):

* auto-trigger — the model emits `Skill(skill="…")`, gets a short stub back, and
  the body arrives as a sibling text part. Costs a round-trip.
* slash — the user types `/name`, and the body is injected straight into the
  user turn. **No tool call at all.**

This module is the second path. The user turn carries two text parts, matching
the capture byte-for-byte:

    <command-message>simplify</command-message>
    <command-name>/simplify</command-name>

then the skill body as its own part.
"""

from __future__ import annotations

from terno_agent.core.messages import TextPart
from terno_agent.skills.manager import SkillCatalog
from terno_agent.skills.tool import format_skill_content

# Names the CLI handles itself; a skill may not shadow them.
RESERVED = frozenset(
    {"clear", "reset", "attach", "deep_research", "research", "knowledge"}
)


def wrapper_text(name: str) -> str:
    """The `<command-…>` preamble, verbatim per the capture (note: no args)."""
    return f"<command-message>{name}</command-message>\n<command-name>/{name}</command-name>\n"


def parse(line: str) -> tuple[str, str] | None:
    """Split `/name rest` into `(name, rest)`, or None if it isn't a slash call."""
    if not line.startswith("/") or len(line) < 2:
        return None
    body = line[1:].strip()
    if not body:
        return None
    name, _, rest = body.partition(" ")
    if not name or "/" in name:
        return None
    return name, rest.strip()


def resolve(
    line: str, catalog: SkillCatalog, *, max_resources: int = 50
) -> list[TextPart] | None:
    """Content parts for a `/name` invocation, or None if it isn't one of ours.

    Returning None (rather than raising) lets the caller fall through to its own
    slash commands and to plain chat — an unknown `/foo` should reach the model
    as text, not become an error.
    """
    parsed = parse(line)
    if parsed is None:
        return None
    name, args = parsed
    if name in RESERVED:
        return None
    skill = catalog.skills.get(name)
    if skill is None:
        return None

    body = format_skill_content(skill, max_resources=max_resources)
    if args:
        # The capture carries no `<command-args>` for skill-backed commands, so
        # arguments are appended as ordinary turn text rather than invented into
        # a tag the reference harness doesn't use here.
        body = f"{body}\n\n{args}"
    return [TextPart(wrapper_text(name)), TextPart(body)]


__all__ = ["RESERVED", "parse", "resolve", "wrapper_text"]
