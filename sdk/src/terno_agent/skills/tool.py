"""Tool for activating discovered Agent Skills."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.skills.manager import Skill, SkillCatalog


class ActivateSkillTool:
    def __init__(self, catalog: SkillCatalog, *, max_resources: int = 50) -> None:
        self.catalog = catalog
        self.max_resources = max_resources
        # Body of the most recently activated skill, awaiting pickup by the run
        # loop via `take_pending_context`.
        self._pending = ""

    @property
    def schema(self) -> ToolSchema:
        names = sorted(self.catalog.skills)
        return ToolSchema(
            name="Skill",
            # Ported from the reference harness. Dropped: plugin (`plugin:skill`)
            # and directory-scoped (`apps/web:deploy`) name forms, skills that run
            # in a subagent and return asynchronously, the `args` passthrough, and
            # the `<command-name>` block check — Terno has none of these. The
            # `enum` of known names is kept, which is stricter than the reference
            # tool and makes "do not guess names" enforceable rather than advisory.
            description=(
                "Invoke a skill.\n"
                "\n"
                "A skill is a packaged set of instructions the user or project has"
                " set up for a particular kind of task (deploy steps, a review"
                " checklist, a repo-specific workflow). Available skills appear in"
                " the system prompt with one-line descriptions. When the task at"
                " hand is one a listed skill covers, call this tool first — the"
                " skill's instructions load into the turn for you to follow in"
                " place of your default approach. Users may also ask for one by"
                ' name (`/<name>`, or "slash command"); that\'s a request to'
                " invoke it.\n"
                "\n"
                "- `skill`: exact name from the listing, no leading slash.\n"
                "\n"
                "Only names from the listing (or that the user typed explicitly)"
                " are valid."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "description": (
                            "The name of a skill from the available-skills list."
                            " Do not guess names."
                        ),
                        "enum": names,
                    }
                },
                "required": ["skill"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        name = str(kwargs.get("skill") or "").strip()
        skill = self.catalog.skills.get(name)
        if skill is None:
            available = ", ".join(sorted(self.catalog.skills)) or "(none)"
            raise ToolError(f"Unknown skill: {name}. Available skills: {available}")
        # The body does NOT come back as the tool result. The reference harness
        # returns a short stub and delivers the body as a sibling text part in
        # the same user turn; `take_pending_context` is how the run loop picks
        # it up. Keeping the body out of the tool result matters for compaction:
        # tool results are summarised away, ordinary turn text is not.
        self._pending = format_skill_content(skill, max_resources=self.max_resources)
        return f"Launching skill: {skill.name}"

    def take_pending_context(self) -> str:
        """Hand the skill body to the run loop, once."""
        pending, self._pending = self._pending, ""
        return pending


def format_skill_content(skill: Skill, *, max_resources: int) -> str:
    references, scripts = _list_resources(skill.directory, max_resources=max_resources)
    lines = [
        f'<skill_content name="{skill.name}">',
        skill.body,
        "",
        f"Skill directory: {skill.directory}",
        "Relative paths in this skill are relative to the skill directory.",
    ]
    # References and scripts are split because they are used differently:
    # a reference is Read into context, a script is EXECUTED and its source
    # never needs to be read. Listing them together invites the model to read
    # a script it should have run.
    if references:
        lines.append("<skill_references>")
        lines.append("  Read these only when you need them.")
        lines.extend(f"  <file>{path}</file>" for path in references)
        lines.append("</skill_references>")
    if scripts:
        lines.append("<skill_scripts>")
        lines.append("  Run these with Bash from the skill directory. Do not Read them.")
        lines.extend(f"  <file>{path}</file>" for path in scripts)
        lines.append("</skill_scripts>")
    lines.append("</skill_content>")
    return "\n".join(lines)


def _list_resources(
    directory: Path, *, max_resources: int
) -> tuple[list[str], list[str]]:
    """Split a skill's bundled files into (references, scripts).

    Anything under a `scripts/` directory is a script; everything else is a
    reference. The cap counts both together so a script-heavy skill can't
    crowd out its own documentation.
    """
    references: list[str] = []
    scripts: list[str] = []
    for path in sorted(directory.rglob("*"), key=lambda p: p.as_posix()):
        if len(references) + len(scripts) >= max_resources:
            references.append(f"... resource list capped at {max_resources} files")
            break
        if not path.is_file() or path.name == "SKILL.md":
            continue
        if any(part in {".git", "node_modules", "__pycache__"} for part in path.parts):
            continue
        rel = path.relative_to(directory)
        (scripts if "scripts" in rel.parts[:-1] else references).append(rel.as_posix())
    return references, scripts


__all__ = ["ActivateSkillTool", "format_skill_content"]
