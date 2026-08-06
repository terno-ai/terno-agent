"""Runtime facts injected into the system prompt.

Claude Code's largest system block is a template plus a handful of
independently injected sections — working directory, model identity, memory and
scratchpad paths, git snapshot. `PromptContext` collects exactly those, so
`builder` stays pure string assembly and is trivially testable.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from terno_agent.memory.paths import memory_dir


def _run(args: list[str], cwd: Path) -> str:
    """Run a git command, returning stripped stdout or "" on any failure."""
    try:
        out = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _slug(path: Path) -> str:
    """`/Users/x/terno/agent` -> `-Users-x-terno-agent`, matching CC's scheme."""
    return str(path).replace(os.sep, "-")


@dataclass(slots=True)
class GitSnapshot:
    """Git state at the start of the conversation."""

    is_repo: bool = False
    branch: str = ""
    main_branch: str = ""
    user: str = ""
    status: str = ""
    recent_commits: str = ""

    def render(self) -> str:
        if not self.is_repo:
            return ""
        parts = [f"Current branch: {self.branch}"]
        if self.main_branch:
            parts.append(
                f"Main branch (you will usually use this for PRs): {self.main_branch}"
            )
        if self.user:
            parts.append(f"Git user: {self.user}")
        parts.append(f"Status:\n{self.status or '(clean)'}")
        if self.recent_commits:
            parts.append(f"Recent commits:\n{self.recent_commits}")
        return "\n\n".join(parts)

    @classmethod
    def detect(cls, cwd: Path) -> GitSnapshot:
        inside = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd)
        if inside != "true":
            return cls(is_repo=False)

        main = ""
        for candidate in ("main", "master"):
            if _run(["git", "rev-parse", "--verify", candidate], cwd):
                main = candidate
                break

        status = _run(["git", "status", "--short"], cwd)
        return cls(
            is_repo=True,
            branch=_run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd),
            main_branch=main,
            user=_run(["git", "config", "user.name"], cwd),
            status=status or "(clean)",
            recent_commits=_run(["git", "log", "-5", "--oneline"], cwd),
        )


@dataclass(slots=True)
class PromptContext:
    """Everything the system prompt needs that isn't static text."""

    cwd: Path
    session_id: str = ""
    model_name: str = ""
    model_id: str = ""
    knowledge_cutoff: str = ""
    language: str = "en"
    memory_path: Path | None = None
    scratchpad_path: Path | None = None
    git: GitSnapshot = field(default_factory=GitSnapshot)

    # Harness capabilities — each gates one prompt section, so the prompt never
    # describes a feature this build doesn't have.
    supports_bang_prefix: bool = False
    supports_skills: bool = True
    has_agent_tool: bool = True
    has_workflow_tool: bool = False
    extra_env_lines: list[str] = field(default_factory=list)

    @property
    def memory_dir(self) -> Path:
        return self.memory_path or memory_dir(self.cwd)

    @property
    def scratchpad_dir(self) -> Path:
        if self.scratchpad_path is not None:
            return self.scratchpad_path
        uid = getattr(os, "getuid", lambda: 0)()  # getuid is POSIX-only
        root = Path(tempfile.gettempdir()) / f"terno-{uid}"
        return root / _slug(self.cwd) / (self.session_id or "session") / "scratchpad"

    def env_lines(self) -> list[str]:
        """The bullet list under "# Environment", in captured order."""
        lines = [f"Primary working directory: {self.cwd}"]
        lines.append(f"Is a git repository: {str(self.git.is_repo).lower()}")
        lines.append(f"Platform: {sys.platform}")
        lines.append(f"Shell: {Path(os.environ.get('SHELL', '')).name or 'unknown'}")
        lines.append(f"OS Version: {platform.system()} {platform.release()}")
        if self.model_name:
            model = f"You are powered by the model named {self.model_name}."
            if self.model_id:
                model += f" The exact model ID is {self.model_id}."
            lines.append(model)
        if self.knowledge_cutoff:
            lines.append(f"Assistant knowledge cutoff is {self.knowledge_cutoff}.")
        lines.extend(self.extra_env_lines)
        return lines

    @classmethod
    def detect(
        cls,
        cwd: Path | str | None = None,
        *,
        session_id: str = "",
        model_name: str = "",
        model_id: str = "",
        **kwargs: object,
    ) -> PromptContext:
        """Build a context by probing the environment."""
        resolved = Path(cwd or Path.cwd()).resolve()
        return cls(
            cwd=resolved,
            session_id=session_id,
            model_name=model_name,
            model_id=model_id,
            git=GitSnapshot.detect(resolved),
            **kwargs,  # type: ignore[arg-type]
        )


__all__ = ["GitSnapshot", "PromptContext"]
