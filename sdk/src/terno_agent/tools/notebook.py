"""Jupyter notebook support: rendering for `Read`, and the `NotebookEdit` tool.

`NotebookEdit`'s captured description says `cell_id` is "the `id` attribute shown
in the Read tool's `<cell id="...">` output" — so notebook rendering and notebook
editing are one feature, not two. `render_notebook` is what `Read` uses when the
path ends in `.ipynb`, and it emits exactly the ids `NotebookEdit` accepts.

The `<cell …>` rendering itself is NOT captured — only the fact that ids appear
in it. The format here is a reasonable reconstruction from that hint; if a future
capture shows the real one, this is the piece to correct.

Notebooks below nbformat 4.5 have no per-cell `id`, so positional `cell-N` ids
are synthesised for them. Those shift if cells are inserted or deleted, which is
why a `Read` is required before every edit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.tools.files import FileStateTracker, _resolve

# Truncate long outputs; a single cell can hold megabytes of base64 image data.
_MAX_OUTPUT_CHARS = 2000


def _source_to_text(source: Any) -> str:
    """nbformat stores source as a string or a list of lines."""
    if isinstance(source, list):
        return "".join(str(s) for s in source)
    return str(source or "")


def load_notebook(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ToolError(f"Failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ToolError(f"{path} is not valid JSON (corrupt notebook?): {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("cells"), list):
        raise ToolError(f"{path} is not a Jupyter notebook (no 'cells' array).")
    return data


def cell_id_at(nb: dict[str, Any], index: int) -> str:
    """The id for a cell, synthesising one for pre-4.5 notebooks."""
    cell = nb["cells"][index]
    cid = cell.get("id")
    return str(cid) if cid else f"cell-{index}"


def find_cell(nb: dict[str, Any], cell_id: str) -> int:
    for i in range(len(nb["cells"])):
        if cell_id_at(nb, i) == cell_id:
            return i
    known = ", ".join(cell_id_at(nb, i) for i in range(len(nb["cells"]))) or "(none)"
    raise ToolError(f"No cell with id {cell_id!r}. Known ids: {known}")


def _render_outputs(cell: dict[str, Any]) -> str:
    chunks: list[str] = []
    for out in cell.get("outputs") or ():
        if not isinstance(out, dict):
            continue
        kind = out.get("output_type")
        if kind == "stream":
            chunks.append(_source_to_text(out.get("text")))
        elif kind in ("execute_result", "display_data"):
            data = out.get("data") or {}
            if "text/plain" in data:
                chunks.append(_source_to_text(data["text/plain"]))
            else:
                # Images and rich mime types can't be shown as text.
                chunks.append(f"[{', '.join(sorted(data)) or 'no data'}]")
        elif kind == "error":
            name = out.get("ename", "Error")
            value = out.get("evalue", "")
            chunks.append(f"{name}: {value}")
    text = "".join(chunks).rstrip()
    if not text:
        return ""
    if len(text) > _MAX_OUTPUT_CHARS:
        text = f"{text[:_MAX_OUTPUT_CHARS]}\n… (output truncated)"
    return f"<output>\n{text}\n</output>\n"


def render_notebook(nb: dict[str, Any]) -> str:
    """Cells with their ids, types, source and outputs."""
    parts: list[str] = []
    for i, cell in enumerate(nb["cells"]):
        cid = cell_id_at(nb, i)
        ctype = cell.get("cell_type", "code")
        parts.append(
            f'<cell id="{cid}" type="{ctype}">\n'
            f"{_source_to_text(cell.get('source'))}\n"
            f"{_render_outputs(cell)}"
            "</cell>"
        )
    return "\n\n".join(parts) if parts else "(notebook has no cells)"


@dataclass
class NotebookEditTool:
    workdir: Path | None = None
    tracker: FileStateTracker = field(default_factory=FileStateTracker)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="NotebookEdit",
            description=(
                "Replaces, inserts, or deletes a single cell in a Jupyter"
                " notebook (.ipynb file).\n"
                "\n"
                "Usage:\n"
                "- You must use the Read tool on the notebook in this"
                " conversation before editing — this tool will fail otherwise.\n"
                "- `notebook_path` must be an absolute path.\n"
                "- `cell_id` is the `id` attribute shown in the Read tool's"
                ' `<cell id="...">` output. It is required for `replace` and'
                " `delete`.\n"
                "- `edit_mode` defaults to `replace`. Use `insert` to add a new"
                " cell after the cell with the given `cell_id` (or at the"
                " beginning of the notebook if `cell_id` is omitted) —"
                " `cell_type` is required when inserting. Use `delete` to remove"
                " the cell."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "notebook_path": {
                        "type": "string",
                        "description": (
                            "The absolute path to the Jupyter notebook file to"
                            " edit (must be absolute, not relative)"
                        ),
                    },
                    "new_source": {
                        "type": "string",
                        "description": "The new source for the cell",
                    },
                    "cell_id": {
                        "type": "string",
                        "description": (
                            "The ID of the cell to edit. When inserting a new"
                            " cell, the new cell will be inserted after the cell"
                            " with this ID, or at the beginning if not specified."
                        ),
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": ["code", "markdown"],
                        "description": (
                            "The type of the cell (code or markdown). If not"
                            " specified, it defaults to the current cell type. If"
                            " using edit_mode=insert, this is required."
                        ),
                    },
                    "edit_mode": {
                        "type": "string",
                        "enum": ["replace", "insert", "delete"],
                        "description": (
                            "The type of edit to make (replace, insert, delete)."
                            " Defaults to replace."
                        ),
                    },
                },
                "required": ["notebook_path", "new_source"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        path = _resolve(kwargs.get("notebook_path", ""), self.workdir)
        if not path.exists():
            raise ToolError(f"File not found: {path}")
        if not self.tracker.has_read(path):
            raise ToolError(
                f"{path} has not been read in this conversation. Read it before "
                "editing — cell ids come from the Read output."
            )

        mode = kwargs.get("edit_mode") or "replace"
        if mode not in ("replace", "insert", "delete"):
            raise ToolError(f"Invalid edit_mode {mode!r}.")
        cell_id = kwargs.get("cell_id")
        new_source = kwargs.get("new_source")
        cell_type = kwargs.get("cell_type")

        nb = load_notebook(path)

        if mode == "insert":
            if not cell_type:
                raise ToolError("cell_type is required when edit_mode=insert.")
            index = find_cell(nb, str(cell_id)) + 1 if cell_id else 0
            nb["cells"].insert(index, _new_cell(cell_type, new_source or ""))
            what = f"Inserted a {cell_type} cell at index {index}"
        else:
            if not cell_id:
                raise ToolError(f"cell_id is required when edit_mode={mode}.")
            index = find_cell(nb, str(cell_id))
            if mode == "delete":
                nb["cells"].pop(index)
                what = f"Deleted cell {cell_id}"
            else:
                if new_source is None:
                    raise ToolError("new_source is required when replacing a cell.")
                cell = nb["cells"][index]
                cell["source"] = new_source
                if cell_type:
                    cell["cell_type"] = cell_type
                # The stored outputs described the old source, so they are stale.
                if cell.get("cell_type") == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
                else:
                    cell.pop("outputs", None)
                    cell.pop("execution_count", None)
                what = f"Replaced cell {cell_id}"

        try:
            path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"Failed to write {path}: {exc}") from exc
        # Cell ids shift after an insert or delete, so the model must re-Read
        # before its next edit.
        if mode in ("insert", "delete"):
            self.tracker.forget(path)
        return f"{what} in {path}"


def _new_cell(cell_type: str, source: str) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


__all__ = [
    "NotebookEditTool",
    "cell_id_at",
    "find_cell",
    "load_notebook",
    "render_notebook",
]
