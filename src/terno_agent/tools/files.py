"""File-system tools: read_file, write_file, edit_file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema


def _resolve(path_str: str) -> Path:
    if not path_str:
        raise ToolError("path is required.")
    return Path(path_str).expanduser()


class ReadFileTool:
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
        path = _resolve(kwargs.get("path", ""))
        if not path.exists():
            raise ToolError(f"File not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory, not a file: {path}")

        offset = max(1, int(kwargs.get("offset") or 1))
        limit = int(kwargs.get("limit") or 2000)
        if limit <= 0:
            raise ToolError("limit must be positive.")

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


class WriteFileTool:
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_file",
            description=(
                "Create or overwrite a file with the given content. Parent "
                "directories are created if they don't exist. Use edit_file "
                "for targeted changes to existing files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to write."},
                    "content": {"type": "string", "description": "File contents."},
                },
                "required": ["path", "content"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        path = _resolve(kwargs.get("path", ""))
        content = kwargs.get("content")
        if content is None:
            raise ToolError("write_file requires a 'content' argument.")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"Failed to write {path}: {exc}") from exc
        return f"Wrote {len(content)} bytes to {path}"


class EditFileTool:
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
        path = _resolve(kwargs.get("path", ""))
        old = kwargs.get("old_string")
        new = kwargs.get("new_string")
        if old is None or new is None:
            raise ToolError("edit_file requires 'old_string' and 'new_string'.")
        if old == new:
            raise ToolError("edit_file: old_string and new_string are identical.")
        if not path.exists():
            raise ToolError(f"File not found: {path}")

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"Failed to read {path}: {exc}") from exc

        count = text.count(old)
        if count == 0:
            raise ToolError(f"old_string not found in {path}.")
        replace_all = bool(kwargs.get("replace_all"))
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


__all__ = ["EditFileTool", "ReadFileTool", "WriteFileTool"]
