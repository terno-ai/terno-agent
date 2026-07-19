"""Filesystem search tools: glob (file patterns) and grep (regex content).

Both are read-only and scoped to the agent's working directory by
default; callers can pass an explicit ``path`` to widen the search. When a
``sandbox`` is also injected, a `path` argument outside the agent's local
`workdir` is searched *inside the sandbox* instead — see `files.py` for why
(the same container-mounted-directory reasoning applies here).
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.sandbox.base import Sandbox

_GLOB_DEFAULT_LIMIT = 200
_GREP_DEFAULT_LIMIT = 200
_GREP_TIMEOUT_S = 30
_SANDBOX_TIMEOUT_S = 30


def _resolve_root(root_arg: Any, workdir: Path) -> Path:
    if not root_arg:
        return workdir
    path = Path(str(root_arg)).expanduser()
    if not path.is_absolute():
        path = (workdir / path).resolve()
    return path


def _use_sandbox(path: Path, workdir: Path | None, sandbox: Sandbox | None) -> bool:
    """See `files._use_sandbox` — same rule, duplicated to avoid a cross-file
    dependency for two tools that already stand alone."""
    if sandbox is None or not path.is_absolute():
        return False
    if workdir is not None:
        try:
            path.relative_to(workdir)
            return False
        except ValueError:
            pass
    return True


def _run_json(sandbox: Sandbox, code_template: str, payload: dict) -> Any:
    code = code_template % (json.dumps(payload),)
    result = sandbox.run_python(code, timeout_s=_SANDBOX_TIMEOUT_S)
    if result.exit_code != 0:
        raise ToolError(f"sandbox operation failed: {result.stderr or result.stdout}")
    try:
        return json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ToolError(
            f"sandbox operation returned unparsable output: {result.stdout!r}"
        ) from exc


@dataclass
class GlobTool:
    workdir: Path
    sandbox: Sandbox | None = None

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="glob",
            description=(
                "List files matching a glob pattern, sorted by most recently "
                "modified first. Supports '**' for recursive descent. Use "
                "this to find files by name; use `grep` to search file "
                "contents."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Glob pattern, e.g. '**/*.py' or 'src/**/*.tsx'."
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Root directory to search from. Defaults to the "
                            "agent's working directory."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            f"Maximum number of matches to return "
                            f"(default {_GLOB_DEFAULT_LIMIT})."
                        ),
                    },
                },
                "required": ["pattern"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        pattern = (kwargs.get("pattern") or "").strip()
        if not pattern:
            raise ToolError("glob requires a 'pattern'.")
        root = _resolve_root(kwargs.get("path"), self.workdir)
        limit = int(kwargs.get("limit") or _GLOB_DEFAULT_LIMIT)
        if limit <= 0:
            raise ToolError("limit must be positive.")

        if _use_sandbox(root, self.workdir, self.sandbox):
            data = _run_json(
                self.sandbox,
                _GLOB_TEMPLATE,
                {"root": str(root), "pattern": pattern, "limit": limit},
            )
            if "error" in data:
                raise ToolError(data["error"])
            matches, total = data["matches"], data["total"]
        else:
            if not root.exists():
                raise ToolError(f"Path not found: {root}")
            if not root.is_dir():
                raise ToolError(f"Path is not a directory: {root}")
            try:
                found = [p for p in root.glob(pattern) if p.is_file()]
            except (OSError, ValueError) as exc:
                raise ToolError(f"glob failed: {exc}") from exc
            try:
                found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            except OSError:
                found.sort()
            total = len(found)
            matches = [str(p) for p in found[:limit]]

        if not matches:
            return f"(no files matched {pattern!r} under {root})"
        body = "\n".join(matches)
        if total > limit:
            body += f"\n... ({total - limit} more matches truncated)"
        return body


_GLOB_TEMPLATE = """
import json, pathlib
data = json.loads(%r)
root = pathlib.Path(data["root"])
pattern = data["pattern"]
limit = data["limit"]
if not root.exists():
    print(json.dumps({"error": "Path not found: " + str(root)}))
elif not root.is_dir():
    print(json.dumps({"error": "Path is not a directory: " + str(root)}))
else:
    try:
        found = [p for p in root.glob(pattern) if p.is_file()]
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": "glob failed: " + str(exc)}))
    else:
        try:
            found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            found.sort()
        total = len(found)
        shown = [str(p) for p in found[:limit]]
        print(json.dumps({"matches": shown, "total": total}))
"""


@dataclass
class GrepTool:
    workdir: Path
    sandbox: Sandbox | None = None

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="grep",
            description=(
                "Search file contents for a regex pattern. Returns matching "
                "lines as 'path:line:text'. Uses ripgrep when installed and "
                "falls back to a pure-Python walk otherwise. Combine with "
                "the `glob` filter to restrict by filename."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory or single file to search. Defaults to "
                            "the agent's working directory."
                        ),
                    },
                    "glob": {
                        "type": "string",
                        "description": (
                            "Optional filename filter, e.g. '*.py' or "
                            "'src/**/*.ts'."
                        ),
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Match case-insensitively (default false).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            f"Maximum number of match lines to return "
                            f"(default {_GREP_DEFAULT_LIMIT})."
                        ),
                    },
                },
                "required": ["pattern"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        pattern = kwargs.get("pattern")
        if not pattern:
            raise ToolError("grep requires a 'pattern'.")
        root = _resolve_root(kwargs.get("path"), self.workdir)
        glob_filter = kwargs.get("glob")
        case_insensitive = bool(kwargs.get("case_insensitive"))
        limit = int(kwargs.get("limit") or _GREP_DEFAULT_LIMIT)
        if limit <= 0:
            raise ToolError("limit must be positive.")

        if _use_sandbox(root, self.workdir, self.sandbox):
            data = _run_json(
                self.sandbox,
                _GREP_TEMPLATE,
                {
                    "root": str(root),
                    "pattern": str(pattern),
                    "glob_filter": glob_filter,
                    "case_insensitive": case_insensitive,
                    "limit": limit,
                },
            )
            if "error" in data:
                raise ToolError(data["error"])
            matches, total = data["matches"], data["total"]
            if not matches:
                return f"(no matches for {pattern!r} under {root})"
            body = "\n".join(matches)
            if total > limit:
                body += f"\n... ({total - limit} more matches truncated)"
            return body

        if not root.exists():
            raise ToolError(f"Path not found: {root}")

        rg_result = _ripgrep_search(
            pattern=str(pattern),
            root=root,
            glob_filter=glob_filter,
            case_insensitive=case_insensitive,
            limit=limit,
        )
        if rg_result is not None:
            return rg_result
        return _python_grep(
            pattern=str(pattern),
            root=root,
            glob_filter=glob_filter,
            case_insensitive=case_insensitive,
            limit=limit,
        )


_GREP_TEMPLATE = """
import json, pathlib, re
data = json.loads(%r)
root = pathlib.Path(data["root"])
pattern = data["pattern"]
glob_filter = data["glob_filter"]
flags = re.IGNORECASE if data["case_insensitive"] else 0
limit = data["limit"]
if not root.exists():
    print(json.dumps({"error": "Path not found: " + str(root)}))
else:
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        print(json.dumps({"error": "invalid regex: " + str(exc)}))
    else:
        if root.is_file():
            candidates = [root]
        elif glob_filter:
            candidates = [p for p in root.rglob(glob_filter) if p.is_file()]
        else:
            candidates = [p for p in root.rglob("*") if p.is_file()]

        matches = []
        for f in candidates:
            try:
                with f.open("r", encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh, start=1):
                        if rx.search(line):
                            matches.append(str(f) + ":" + str(i) + ":" + line.rstrip())
                            if len(matches) > limit:
                                break
            except OSError:
                continue
            if len(matches) > limit:
                break

        total = len(matches)
        print(json.dumps({"matches": matches[:limit], "total": total}))
"""


def _ripgrep_search(
    *,
    pattern: str,
    root: Path,
    glob_filter: str | None,
    case_insensitive: bool,
    limit: int,
) -> str | None:
    """Try ripgrep; return None to signal "fall back to Python"."""
    rg = which("rg")
    if rg is None:
        return None
    cmd: list[str] = [rg, "--line-number", "--no-heading", "--color=never"]
    if case_insensitive:
        cmd.append("-i")
    if glob_filter:
        cmd.extend(["--glob", glob_filter])
    cmd.extend(["--", pattern, str(root)])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_GREP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"(ripgrep timed out after {_GREP_TIMEOUT_S}s)"
    except OSError:
        return None
    # rg returncodes: 0 = matches, 1 = no matches, 2 = error.
    if proc.returncode == 2:
        return None
    lines = (proc.stdout or "").splitlines()
    if not lines:
        return f"(no matches for {pattern!r} under {root})"
    if len(lines) > limit:
        head = "\n".join(lines[:limit])
        return head + f"\n... ({len(lines) - limit} more matches truncated)"
    return "\n".join(lines)


def _python_grep(
    *,
    pattern: str,
    root: Path,
    glob_filter: str | None,
    case_insensitive: bool,
    limit: int,
) -> str:
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        raise ToolError(f"invalid regex: {exc}") from exc

    if root.is_file():
        candidates: list[Path] = [root]
    elif glob_filter:
        candidates = [p for p in root.rglob(glob_filter) if p.is_file()]
    else:
        candidates = [p for p in root.rglob("*") if p.is_file()]

    matches: list[str] = []
    for f in candidates:
        try:
            with f.open("r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, start=1):
                    if rx.search(line):
                        matches.append(f"{f}:{i}:{line.rstrip()}")
                        if len(matches) > limit:
                            break
        except OSError:
            continue
        if len(matches) > limit:
            break

    if not matches:
        return f"(no matches for {pattern!r} under {root})"
    if len(matches) > limit:
        head = "\n".join(matches[:limit])
        return head + "\n... (more matches truncated)"
    return "\n".join(matches)


__all__ = ["GlobTool", "GrepTool"]
