"""File-system tools: read_file, write_file, edit_file.

Each tool is scoped to the agent's local ``workdir`` by default. When a
``sandbox`` is also injected (e.g. terno-ai's container-backed bridge), an
absolute path that falls outside ``workdir`` is resolved *inside the
sandbox* instead of failing with "not found" on the host — this is how
these tools reach paths that only exist in a mounted container directory
(an attachments folder, say), the same way ``run_python``/``bash`` already
do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.sandbox.base import Sandbox

_SANDBOX_TIMEOUT_S = 30


def _resolve(path_str: str, workdir: Path | None = None) -> Path:
    if not path_str:
        raise ToolError("path is required.")
    path = Path(path_str).expanduser()
    if path.is_absolute() or workdir is None:
        return path
    return workdir / path


def _use_sandbox(path: Path, workdir: Path | None, sandbox: Sandbox | None) -> bool:
    """True when `path` should be resolved inside `sandbox` rather than the
    local host filesystem: a sandbox is available, the path is absolute,
    and it doesn't fall under the agent's own local `workdir` (which stays
    on the host regardless — e.g. the file-based memory directory)."""
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
    """Fill a ``%r``-style template with the JSON-encoded `payload`, run it
    in `sandbox`, and return the parsed JSON object it printed."""
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
class ReadFileTool:
    workdir: Path | None = None
    sandbox: Sandbox | None = None

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_file",
            description=(
                "Read a UTF-8 text file from disk. Returns the contents "
                "prefixed with 1-indexed line numbers ('N\\tline'). Use "
                "'offset' and 'limit' to read a slice of a large file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-indexed line to start reading from (default 1).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default 2000).",
                    },
                },
                "required": ["path"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        path = _resolve(kwargs.get("path", ""), self.workdir)
        offset = max(1, int(kwargs.get("offset") or 1))
        limit = int(kwargs.get("limit") or 2000)
        if limit <= 0:
            raise ToolError("limit must be positive.")

        if _use_sandbox(path, self.workdir, self.sandbox):
            data = _run_json(self.sandbox, _READ_FILE_TEMPLATE, {"path": str(path)})
            if "error" in data:
                raise ToolError(data["error"])
            text = data["text"]
        else:
            if not path.exists():
                raise ToolError(f"File not found: {path}")
            if path.is_dir():
                raise ToolError(f"Path is a directory, not a file: {path}")
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise ToolError(f"Failed to read {path}: {exc}") from exc

        lines = text.splitlines()
        end = min(len(lines), offset - 1 + limit)
        slice_ = lines[offset - 1 : end]
        numbered = [f"{i}\t{line}" for i, line in enumerate(slice_, start=offset)]
        if not numbered:
            return f"(file has {len(lines)} lines; offset {offset} is past the end)"
        suffix = ""
        if end < len(lines):
            suffix = f"\n... ({len(lines) - end} more lines)"
        return "\n".join(numbered) + suffix


_READ_FILE_TEMPLATE = """
import json, pathlib
data = json.loads(%r)
p = pathlib.Path(data["path"])
if not p.exists():
    print(json.dumps({"error": "File not found: " + str(p)}))
elif p.is_dir():
    print(json.dumps({"error": "Path is a directory, not a file: " + str(p)}))
else:
    print(json.dumps({"text": p.read_text(encoding="utf-8", errors="replace")}))
"""


@dataclass
class WriteFileTool:
    workdir: Path | None = None
    sandbox: Sandbox | None = None

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_file",
            description=(
                "Create a NEW file with the given content. Parent directories "
                "are created if they don't exist. For ANY change to an "
                "existing file use `edit_file` — it's almost always what you "
                "want. If you truly need to replace a file end-to-end (e.g. "
                "a generated artefact), pass `overwrite=true` explicitly."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to write."},
                    "content": {"type": "string", "description": "File contents."},
                    "overwrite": {
                        "type": "boolean",
                        "description": (
                            "Required (true) when the file already exists. "
                            "Defaults to false; calling on an existing file "
                            "without this flag errors and directs you to "
                            "edit_file."
                        ),
                    },
                },
                "required": ["path", "content"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        path = _resolve(kwargs.get("path", ""), self.workdir)
        content = kwargs.get("content")
        if content is None:
            raise ToolError("write_file requires a 'content' argument.")
        overwrite = bool(kwargs.get("overwrite", False))

        if _use_sandbox(path, self.workdir, self.sandbox):
            data = _run_json(
                self.sandbox,
                _WRITE_FILE_TEMPLATE,
                {"path": str(path), "content": content, "overwrite": overwrite},
            )
            if "error" in data:
                raise ToolError(data["error"])
            return f"Wrote {data['bytes']} bytes to {path}"

        if path.exists():
            if path.is_dir():
                raise ToolError(f"Path is a directory, not a file: {path}")
            if not overwrite:
                raise ToolError(
                    f"{path} already exists. For targeted changes use "
                    "edit_file. If you genuinely need to replace the whole "
                    "file, retry with overwrite=true."
                )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"Failed to write {path}: {exc}") from exc
        return f"Wrote {len(content)} bytes to {path}"


_WRITE_FILE_TEMPLATE = """
import json, pathlib
data = json.loads(%r)
p = pathlib.Path(data["path"])
content = data["content"]
overwrite = data["overwrite"]
if p.exists() and p.is_dir():
    print(json.dumps({"error": "Path is a directory, not a file: " + str(p)}))
elif p.exists() and not overwrite:
    print(json.dumps({"error": str(p) + " already exists. For targeted changes use edit_file. If you genuinely need to replace the whole file, retry with overwrite=true."}))
else:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    print(json.dumps({"bytes": len(content)}))
"""


@dataclass
class EditFileTool:
    workdir: Path | None = None
    sandbox: Sandbox | None = None

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="edit_file",
            description=(
                "Replace an exact string in a file. 'old_string' must match "
                "uniquely unless 'replace_all' is true. Fails if the file is "
                "missing or the string is not found."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to edit."},
                    "old_string": {"type": "string", "description": "Text to replace."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence (default false).",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        path = _resolve(kwargs.get("path", ""), self.workdir)
        old = kwargs.get("old_string")
        new = kwargs.get("new_string")
        if old is None or new is None:
            raise ToolError("edit_file requires 'old_string' and 'new_string'.")
        if old == new:
            raise ToolError("edit_file: old_string and new_string are identical.")
        replace_all = bool(kwargs.get("replace_all"))

        if _use_sandbox(path, self.workdir, self.sandbox):
            data = _run_json(
                self.sandbox,
                _EDIT_FILE_TEMPLATE,
                {
                    "path": str(path),
                    "old": old,
                    "new": new,
                    "replace_all": replace_all,
                },
            )
            if "error" in data:
                raise ToolError(data["error"])
            return f"Replaced {data['replaced']} occurrence(s) in {path}"

        if not path.exists():
            raise ToolError(f"File not found: {path}")

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"Failed to read {path}: {exc}") from exc

        count = text.count(old)
        if count == 0:
            raise ToolError(f"old_string not found in {path}.")
        if count > 1 and not replace_all:
            raise ToolError(
                f"old_string is not unique in {path} ({count} matches). "
                "Provide more surrounding context or pass replace_all=true."
            )

        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"Failed to write {path}: {exc}") from exc
        replaced = count if replace_all else 1
        return f"Replaced {replaced} occurrence(s) in {path}"


_EDIT_FILE_TEMPLATE = """
import json, pathlib
data = json.loads(%r)
p = pathlib.Path(data["path"])
old, new, replace_all = data["old"], data["new"], data["replace_all"]
if not p.exists():
    print(json.dumps({"error": "File not found: " + str(p)}))
else:
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        print(json.dumps({"error": "old_string not found in " + str(p) + "."}))
    elif count > 1 and not replace_all:
        print(json.dumps({"error": "old_string is not unique in " + str(p) + " (" + str(count) + " matches). Provide more surrounding context or pass replace_all=true."}))
    else:
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        p.write_text(updated, encoding="utf-8")
        print(json.dumps({"replaced": count if replace_all else 1}))
"""


__all__ = ["EditFileTool", "ReadFileTool", "WriteFileTool"]
