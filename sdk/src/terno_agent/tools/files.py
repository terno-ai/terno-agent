"""File-system tools: Read, Write, Edit.

Names, parameters and descriptions are ported from the reference harness so the
model sees the surface it was trained against. Two deliberate deviations, both
because a verbatim description would promise behaviour Terno doesn't have:

* `Read` drops the bullet about images and PDFs — Terno reads UTF-8 text only,
  so the `pages` parameter is omitted too. Notebooks ARE supported (see
  `tools/notebook.py`), so that half of the bullet is kept.
* `Read` drops the bullet about the harness tracking file state for you; Terno's
  tracking is `FileStateTracker` below, which is per-agent rather than global.

The read-before-write rule those descriptions rely on IS implemented: `Edit`
refuses a file that hasn't been read this conversation, and `Write` refuses to
clobber an existing file that hasn't been read. That's what makes "overwriting
if one exists" safe to say.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema


def _resolve(path_str: str, workdir: Path | None = None) -> Path:
    if not path_str:
        raise ToolError("file_path is required.")
    path = Path(path_str).expanduser()
    if path.is_absolute() or workdir is None:
        return path
    return workdir / path


class FileStateTracker:
    """Remembers which files have been read during this conversation.

    `Edit` and `Write` consult it so the model can't blind-write over a file it
    has never looked at. One instance is shared by the three tools.
    """

    def __init__(self) -> None:
        self._read: set[Path] = set()

    def mark_read(self, path: Path) -> None:
        self._read.add(path.resolve())

    def has_read(self, path: Path) -> bool:
        return path.resolve() in self._read

    def forget(self, path: Path) -> None:
        self._read.discard(path.resolve())

    def reset(self) -> None:
        self._read.clear()


@dataclass
class ReadFileTool:
    workdir: Path | None = None
    tracker: FileStateTracker = field(default_factory=FileStateTracker)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="Read",
            description=(
                "Reads a file from the local filesystem.\n"
                "\n"
                "- `file_path` must be an absolute path.\n"
                "- Reads up to 2000 lines by default.\n"
                "- When you already know which part of the file you need, only"
                " read that part. This can be important for larger files.\n"
                "- Results are returned using cat -n format, with line numbers"
                " starting at 1\n"
                "- Reads Jupyter notebooks (.ipynb) as cells with outputs.\n"
                "- Reading a directory, a missing file, or an empty file returns"
                " an error rather than content.\n"
                "- Do NOT re-read a file you just edited to verify — Edit/Write"
                " would have errored if the change failed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to read",
                    },
                    "offset": {
                        "type": "integer",
                        "description": (
                            "The line number to start reading from. Only provide"
                            " if the file is too large to read at once"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "The number of lines to read. Only provide if the"
                            " file is too large to read at once."
                        ),
                    },
                },
                "required": ["file_path"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        path = _resolve(kwargs.get("file_path", ""), self.workdir)
        if not path.exists():
            raise ToolError(f"File not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory, not a file: {path}")

        offset = max(1, int(kwargs.get("offset") or 1))
        limit = int(kwargs.get("limit") or 2000)
        if limit <= 0:
            raise ToolError("limit must be positive.")

        if path.suffix == ".ipynb":
            # Notebooks are rendered as cells, not numbered lines: NotebookEdit
            # addresses cells by the id shown here, so line numbers are useless.
            from terno_agent.tools.notebook import load_notebook, render_notebook

            rendered = render_notebook(load_notebook(path))
            self.tracker.mark_read(path)
            return rendered

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ToolError(f"Failed to read {path}: {exc}") from exc

        # Reading a slice still counts: the model has seen the file.
        self.tracker.mark_read(path)

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


@dataclass
class WriteFileTool:
    workdir: Path | None = None
    tracker: FileStateTracker = field(default_factory=FileStateTracker)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="Write",
            description=(
                "Writes a file to the local filesystem, overwriting if one"
                " exists.\n"
                "\n"
                "When to use: creating a new file, or fully replacing one you've"
                " already Read. Overwriting an existing file you haven't Read"
                " will fail. For partial changes, use Edit instead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "The absolute path to the file to write (must be"
                            " absolute, not relative)"
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
                "required": ["file_path", "content"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        path = _resolve(kwargs.get("file_path", ""), self.workdir)
        content = kwargs.get("content")
        if content is None:
            raise ToolError("Write requires a 'content' argument.")
        if path.exists():
            if path.is_dir():
                raise ToolError(f"Path is a directory, not a file: {path}")
            if not self.tracker.has_read(path):
                raise ToolError(
                    f"{path} already exists and has not been read in this "
                    "conversation. Read it first, or use Edit for a targeted "
                    "change."
                )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"Failed to write {path}: {exc}") from exc
        # The file's contents are now known — a follow-up Write needn't re-read.
        self.tracker.mark_read(path)
        return f"Wrote {len(content)} bytes to {path}"


@dataclass
class EditFileTool:
    workdir: Path | None = None
    tracker: FileStateTracker = field(default_factory=FileStateTracker)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="Edit",
            description=(
                "Performs exact string replacement in a file.\n"
                "\n"
                "- You must Read the file in this conversation before editing,"
                " or the call will fail.\n"
                "- `old_string` must match the file exactly, including"
                " indentation, and be unique — the edit fails otherwise. Strip"
                " the Read line prefix (line number + tab) before matching.\n"
                "- `replace_all: true` replaces every occurrence instead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to modify",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The text to replace",
                    },
                    "new_string": {
                        "type": "string",
                        "description": (
                            "The text to replace it with (must be different from"
                            " old_string)"
                        ),
                    },
                    "replace_all": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Replace all occurrences of old_string (default"
                            " false)"
                        ),
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        path = _resolve(kwargs.get("file_path", ""), self.workdir)
        old = kwargs.get("old_string")
        new = kwargs.get("new_string")
        if old is None or new is None:
            raise ToolError("Edit requires 'old_string' and 'new_string'.")
        if old == new:
            raise ToolError("Edit: old_string and new_string are identical.")
        if not path.exists():
            raise ToolError(f"File not found: {path}")
        if not self.tracker.has_read(path):
            raise ToolError(
                f"{path} has not been read in this conversation. Read it before "
                "editing."
            )

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


__all__ = ["EditFileTool", "FileStateTracker", "ReadFileTool", "WriteFileTool"]
