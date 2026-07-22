"""Filesystem search tools: glob (file patterns) and grep (regex content).

Both are read-only and scoped to the agent's working directory by
default; callers can pass an explicit ``path`` to widen the search.

``glob`` follows the same rule as `files.py`'s file tools: when a
``sandbox`` is injected, a `path` outside the agent's local `workdir` is
searched *inside the sandbox* instead of failing with "not found" on the
host.

``grep`` is always sandbox-backed, mirroring `shell.py`'s `BashTool`: with
no sandbox configured it falls back to a `LocalSandbox`; a real sandbox
runs the search inside its own container. Just like `bash`, `grep` never
resolves or checks its `path` argument against the host filesystem — it
passes the string straight to the sandbox (as `cwd` when omitted, or as
the search root when given) and lets the sandbox interpret it in its own
filesystem, which may have nothing to do with the host's directory
layout.
"""

from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.sandbox.base import Sandbox
from terno_agent.sandbox.local import LocalSandbox

logger = logging.getLogger(__name__)

_GLOB_DEFAULT_LIMIT = 200
_GREP_DEFAULT_LIMIT = 200
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


def _run_shell_json(sandbox: Sandbox, code_template: str, payload: dict, *, cwd: str) -> Any:
    """Like `_run_json`, but goes through `run_shell` (as `python3 -c ...`)
    instead of `run_python`, so `cwd` is forwarded the same way BashTool
    forwards it — honored by sandboxes that support it, ignored by those
    that don't."""
    code = code_template % (json.dumps(payload),)
    command = f"python3 -c {shlex.quote(code)}"
    result = sandbox.run_shell(command, timeout_s=_SANDBOX_TIMEOUT_S, cwd=cwd)
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


class GrepTool:
    def __init__(self, *, workdir: Path, sandbox: Sandbox | None = None) -> None:
        self.workdir = workdir
        # Grep is always sandbox-backed, same as BashTool: with no sandbox
        # configured we run on the host via LocalSandbox (in `workdir`); a
        # Docker sandbox searches inside its container.
        self.sandbox: Sandbox = sandbox or LocalSandbox()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="grep",
            description=(
                "Search file contents for a regex pattern. Returns matching "
                "lines as 'path:line:text'. Combine with the `glob` filter "
                "to restrict by filename."
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
                            "Directory or single file to search, as it "
                            "resolves inside the sandbox (not necessarily "
                            "the host filesystem) — e.g. an absolute "
                            "sandbox-side path, or a path relative to the "
                            "sandbox's working directory. Defaults to the "
                            "sandbox's working directory."
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
        # Passed straight through to the sandbox, unresolved — the caller is
        # responsible for giving a path that makes sense there, not on the
        # host (same contract as BashTool's `command`).
        root = str(kwargs.get("path") or ".")
        glob_filter = kwargs.get("glob")
        case_insensitive = bool(kwargs.get("case_insensitive"))
        limit = int(kwargs.get("limit") or _GREP_DEFAULT_LIMIT)
        if limit <= 0:
            raise ToolError("limit must be positive.")

        logger.info(
            "grep tool called: pattern=%r path=%s glob=%s case_insensitive=%s limit=%s",
            pattern, root, glob_filter, case_insensitive, limit,
        )

        data = _run_shell_json(
            self.sandbox,
            _GREP_TEMPLATE,
            {
                "root": root,
                "pattern": str(pattern),
                "glob_filter": glob_filter,
                "case_insensitive": case_insensitive,
                "limit": limit,
            },
            cwd=str(self.workdir),
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


__all__ = ["GlobTool", "GrepTool"]
