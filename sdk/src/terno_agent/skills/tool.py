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
        return _format_skill_content(skill, max_resources=self.max_resources)


def _format_skill_content(skill: Skill, *, max_resources: int) -> str:
    resources = _list_resources(skill.directory, max_resources=max_resources)
    lines = [
        f'<skill_content name="{skill.name}">',
        skill.body,
        "",
        f"Skill directory: {skill.directory}",
        "Relative paths in this skill are relative to the skill directory.",
    ]
    if resources:
        lines.append("<skill_resources>")
        lines.extend(f"  <file>{path}</file>" for path in resources)
        lines.append("</skill_resources>")
    lines.append("</skill_content>")
    return "\n".join(lines)


def _list_resources(directory: Path, *, max_resources: int) -> list[str]:
    resources: list[str] = []
    for path in sorted(directory.rglob("*"), key=lambda p: p.as_posix()):
        if len(resources) >= max_resources:
            resources.append(f"... resource list capped at {max_resources} files")
            break
        if not path.is_file() or path.name == "SKILL.md":
            continue
        if any(part in {".git", "node_modules", "__pycache__"} for part in path.parts):
            continue
        resources.append(path.relative_to(directory).as_posix())
    return resources


__all__ = ["ActivateSkillTool"]
