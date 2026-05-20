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
            name="activate_skill",
            description=(
                "Load full instructions for an available Agent Skill. Use this when "
                "the user's task matches a skill description from the system prompt."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the skill to activate.",
                        "enum": names,
                    }
                },
                "required": ["name"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        name = str(kwargs.get("name") or "").strip()
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
