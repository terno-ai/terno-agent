"""Filesystem-backed Agent Skills discovery.

The implementation follows the Agent Skills progressive-disclosure model:
discover only `name` and `description` up front, then let the model load a
skill's full instructions through a tool when the current task needs them.
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SKILL_FILE = "SKILL.md"
_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_DEFAULT_PROJECT_DIRS = (".terno/skills", ".agents/skills", ".claude/skills")
_DEFAULT_USER_DIRS = (".terno/skills", ".agents/skills", ".claude/skills")
_BUILTIN_SKILLS_DIR = Path(__file__).parent / "builtin"


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    path: Path
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        return self.path.parent


@dataclass(frozen=True, slots=True)
class SkillDiagnostic:
    path: Path
    message: str


@dataclass(slots=True)
class SkillCatalog:
    skills: dict[str, Skill] = field(default_factory=dict)
    diagnostics: list[SkillDiagnostic] = field(default_factory=list)

    def prompt_section(self) -> str:
        """Return the compact system-prompt catalog for discovered skills."""
        if not self.skills:
            return ""
        lines = [
            "# Agent Skills",
            "",
            "The following skills provide specialized instructions for specific tasks.",
            "When a task matches a skill's description, call `activate_skill` with the",
            "skill name before proceeding. After activation, follow the skill instructions",
            "and load any referenced files only when needed.",
            "",
            "<available_skills>",
        ]
        for skill in sorted(self.skills.values(), key=lambda s: s.name):
            lines.extend(
                [
                    "  <skill>",
                    f"    <name>{html.escape(skill.name)}</name>",
                    f"    <description>{html.escape(skill.description)}</description>",
                    "  </skill>",
                ]
            )
        lines.append("</available_skills>")
        return "\n".join(lines)


def discover_skills(
    workdir: Path,
    *,
    include_builtin: bool = True,
    include_user: bool = True,
    extra_roots: list[Path] | None = None,
) -> SkillCatalog:
    """Discover skills from built-in, user, project, and configured roots.

    Later roots override earlier roots, so project skills naturally shadow
    built-in and user skills with the same name.
    """
    diagnostics: list[SkillDiagnostic] = []
    skills: dict[str, Skill] = {}

    for root in _ordered_skill_roots(
        workdir,
        include_builtin=include_builtin,
        include_user=include_user,
        extra_roots=extra_roots,
    ):
        if not root.is_dir():
            continue
        for skill_file in _iter_skill_files(root):
            parsed = _parse_skill_file(skill_file)
            if isinstance(parsed, SkillDiagnostic):
                diagnostics.append(parsed)
                continue
            if parsed.name in skills:
                diagnostics.append(
                    SkillDiagnostic(
                        skill_file,
                        f"skill {parsed.name!r} shadows {skills[parsed.name].path}",
                    )
                )
            skills[parsed.name] = parsed

    return SkillCatalog(skills=skills, diagnostics=diagnostics)


def _ordered_skill_roots(
    workdir: Path,
    *,
    include_builtin: bool,
    include_user: bool,
    extra_roots: list[Path] | None,
) -> list[Path]:
    roots: list[Path] = []
    if include_builtin:
        roots.append(_BUILTIN_SKILLS_DIR)
    if include_user:
        home = Path.home()
        roots.extend(home / rel for rel in _DEFAULT_USER_DIRS)

    ancestors = list(_project_ancestors(workdir.resolve()))
    for base in reversed(ancestors):
        roots.extend(base / rel for rel in _DEFAULT_PROJECT_DIRS)

    roots.extend(extra_roots or [])
    return _dedupe_paths(roots)


def _project_ancestors(workdir: Path) -> list[Path]:
    ancestors = [workdir]
    current = workdir
    while current.parent != current:
        if (current / ".git").exists():
            break
        current = current.parent
        ancestors.append(current)
    return ancestors


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def _iter_skill_files(root: Path) -> list[Path]:
    skill_files: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.name in {".git", "node_modules", "__pycache__"}:
            continue
        skill_file = child / _SKILL_FILE
        if child.is_dir() and skill_file.is_file():
            skill_files.append(skill_file)
    return skill_files


def _parse_skill_file(path: Path) -> Skill | SkillDiagnostic:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return SkillDiagnostic(path, f"could not read skill: {exc}")

    parsed = _split_frontmatter(raw)
    if parsed is None:
        return SkillDiagnostic(path, "missing YAML frontmatter")
    frontmatter, body = parsed

    metadata = _parse_simple_yaml(frontmatter)
    name = str(metadata.get("name") or "").strip()
    description = str(metadata.get("description") or "").strip()
    if not name:
        return SkillDiagnostic(path, "missing required `name` field")
    if not description:
        return SkillDiagnostic(path, "missing required `description` field")
    if len(description) > 1024:
        return SkillDiagnostic(path, "`description` exceeds 1024 characters")

    diagnostics: list[str] = []
    if len(name) > 64 or not _NAME_RE.fullmatch(name) or "--" in name:
        diagnostics.append("`name` does not match Agent Skills naming constraints")
    if name != path.parent.name:
        diagnostics.append("`name` does not match parent directory name")
    if diagnostics:
        metadata["_warnings"] = diagnostics

    return Skill(
        name=name,
        description=description,
        path=path.resolve(),
        body=body.strip(),
        metadata=metadata,
    )


def _split_frontmatter(raw: str) -> tuple[str, str] | None:
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            frontmatter = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            return frontmatter, body
    return None


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by SKILL.md frontmatter.

    This deliberately avoids a new dependency. It supports top-level scalar
    fields and one-level mappings, which covers the required fields and common
    `metadata` usage.
    """
    data: dict[str, Any] = {}
    current_map: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent and current_map and ":" in line:
            key, value = line.split(":", 1)
            nested = data.setdefault(current_map, {})
            if isinstance(nested, dict):
                nested[key.strip()] = _unquote(value.strip())
            continue
        current_map = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            data[key] = {}
            current_map = key
        else:
            data[key] = _unquote(value)
    return data


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return os.path.expandvars(value)
