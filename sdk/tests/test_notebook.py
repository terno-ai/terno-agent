"""Notebook rendering (via Read) and NotebookEdit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.tools.files import FileStateTracker, ReadFileTool
from terno_agent.tools.notebook import NotebookEditTool, render_notebook


def _nb(*cells: dict) -> dict:
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": list(cells)}


def _code(cid: str, source: str, outputs: list | None = None) -> dict:
    return {
        "id": cid,
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": outputs or [],
        "execution_count": 1,
    }


def _write(tmp_path: Path, nb: dict, name: str = "n.ipynb") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(nb), encoding="utf-8")
    return p


def _tools(tmp_path: Path, nb: dict) -> tuple[Path, ReadFileTool, NotebookEditTool]:
    path = _write(tmp_path, nb)
    tracker = FileStateTracker()
    return (
        path,
        ReadFileTool(workdir=tmp_path, tracker=tracker),
        NotebookEditTool(workdir=tmp_path, tracker=tracker),
    )


# ----- rendering ----------------------------------------------------------- #


def test_read_renders_notebooks_as_cells_not_numbered_lines(tmp_path: Path) -> None:
    path, read, _ = _tools(tmp_path, _nb(_code("abc", "print(1)")))
    out = read.run(file_path=str(path))

    # NotebookEdit addresses cells by this id, so it has to be visible.
    assert '<cell id="abc" type="code">' in out
    assert "print(1)" in out
    assert "1\t" not in out  # not cat -n format


def test_render_includes_stream_and_error_outputs() -> None:
    nb = _nb(
        _code("a", "print(1)", [{"output_type": "stream", "text": ["1\n"]}]),
        _code("b", "boom()", [{"output_type": "error", "ename": "NameError",
                               "evalue": "boom is not defined"}]),
    )
    out = render_notebook(nb)

    assert "<output>\n1\n</output>" in out
    assert "NameError: boom is not defined" in out


def test_render_names_mime_types_it_cannot_show() -> None:
    nb = _nb(_code("a", "plot()", [
        {"output_type": "display_data", "data": {"image/png": "iVBOR..."}}
    ]))
    out = render_notebook(nb)

    # Better to say an image exists than to dump base64 or silently drop it.
    assert "[image/png]" in out
    assert "iVBOR" not in out


def test_render_handles_source_as_a_list_of_lines() -> None:
    # nbformat allows either; a list must not render as a Python repr.
    out = render_notebook(_nb(_code("a", ["import os\n", "os.getcwd()\n"])))
    assert "import os\nos.getcwd()" in out
    assert "['import os" not in out


def test_pre_4_5_notebooks_get_positional_ids() -> None:
    nb = {"nbformat": 4, "nbformat_minor": 2, "cells": [
        {"cell_type": "code", "source": "x = 1", "outputs": []},
    ]}
    assert '<cell id="cell-0"' in render_notebook(nb)


def test_empty_notebook_renders_a_note() -> None:
    assert "no cells" in render_notebook(_nb())


# ----- editing ------------------------------------------------------------- #


def test_edit_requires_a_read_first(tmp_path: Path) -> None:
    path, _read, edit = _tools(tmp_path, _nb(_code("a", "x = 1")))

    with pytest.raises(ToolError, match="has not been read"):
        edit.run(notebook_path=str(path), cell_id="a", new_source="x = 2")


def test_replace_updates_source_and_clears_stale_outputs(tmp_path: Path) -> None:
    nb = _nb(_code("a", "x = 1", [{"output_type": "stream", "text": ["old\n"]}]))
    path, read, edit = _tools(tmp_path, nb)
    read.run(file_path=str(path))

    edit.run(notebook_path=str(path), cell_id="a", new_source="x = 2")

    cell = json.loads(path.read_text())["cells"][0]
    assert cell["source"] == "x = 2"
    # The old output described the old source, so keeping it would be a lie.
    assert cell["outputs"] == []
    assert cell["execution_count"] is None


def test_insert_places_the_cell_after_the_named_one(tmp_path: Path) -> None:
    path, read, edit = _tools(tmp_path, _nb(_code("a", "first"), _code("b", "second")))
    read.run(file_path=str(path))

    edit.run(
        notebook_path=str(path), cell_id="a", new_source="# note",
        cell_type="markdown", edit_mode="insert",
    )

    cells = json.loads(path.read_text())["cells"]
    assert [c["source"] for c in cells] == ["first", "# note", "second"]
    assert cells[1]["cell_type"] == "markdown"


def test_insert_without_a_cell_id_goes_to_the_top(tmp_path: Path) -> None:
    path, read, edit = _tools(tmp_path, _nb(_code("a", "first")))
    read.run(file_path=str(path))

    edit.run(
        notebook_path=str(path), new_source="# title",
        cell_type="markdown", edit_mode="insert",
    )

    assert json.loads(path.read_text())["cells"][0]["source"] == "# title"


def test_insert_requires_a_cell_type(tmp_path: Path) -> None:
    path, read, edit = _tools(tmp_path, _nb(_code("a", "x")))
    read.run(file_path=str(path))

    with pytest.raises(ToolError, match="cell_type is required"):
        edit.run(notebook_path=str(path), new_source="y", edit_mode="insert")


def test_delete_removes_the_cell(tmp_path: Path) -> None:
    path, read, edit = _tools(tmp_path, _nb(_code("a", "x"), _code("b", "y")))
    read.run(file_path=str(path))

    edit.run(notebook_path=str(path), cell_id="a", new_source="", edit_mode="delete")

    cells = json.loads(path.read_text())["cells"]
    assert [c["id"] for c in cells] == ["b"]


def test_insert_and_delete_force_a_re_read(tmp_path: Path) -> None:
    # Positional ids shift, so a second blind edit would target the wrong cell.
    path, read, edit = _tools(tmp_path, _nb(_code("a", "x"), _code("b", "y")))
    read.run(file_path=str(path))
    edit.run(notebook_path=str(path), cell_id="a", new_source="", edit_mode="delete")

    with pytest.raises(ToolError, match="has not been read"):
        edit.run(notebook_path=str(path), cell_id="b", new_source="z")

    # A replace does not shift ids, so it stays readable.
    read.run(file_path=str(path))
    edit.run(notebook_path=str(path), cell_id="b", new_source="z")
    edit.run(notebook_path=str(path), cell_id="b", new_source="zz")


def test_unknown_cell_id_lists_the_known_ones(tmp_path: Path) -> None:
    path, read, edit = _tools(tmp_path, _nb(_code("a", "x"), _code("b", "y")))
    read.run(file_path=str(path))

    with pytest.raises(ToolError, match="Known ids: a, b"):
        edit.run(notebook_path=str(path), cell_id="nope", new_source="z")


def test_replace_and_delete_require_a_cell_id(tmp_path: Path) -> None:
    path, read, edit = _tools(tmp_path, _nb(_code("a", "x")))
    read.run(file_path=str(path))

    for mode in ("replace", "delete"):
        with pytest.raises(ToolError, match="cell_id is required"):
            edit.run(notebook_path=str(path), new_source="y", edit_mode=mode)


def test_a_corrupt_or_non_notebook_file_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.ipynb"
    bad.write_text("not json at all", encoding="utf-8")
    read = ReadFileTool(workdir=tmp_path)

    with pytest.raises(ToolError, match="not valid JSON"):
        read.run(file_path=str(bad))

    bad.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(ToolError, match="not a Jupyter notebook"):
        read.run(file_path=str(bad))


def test_agent_registers_notebook_edit() -> None:
    from terno_agent.agents.terno import TernoAgent
    from terno_agent.core.messages import AssistantMessage
    from terno_agent.llm.base import LLMResponse

    class _LLM:
        model = "dummy"

        def complete(self, *_a, **_k):
            return LLMResponse(message=AssistantMessage(content="x"), stop_reason="stop")

    # Deferred by default — advertised by name, schema fetched via ToolSearch.
    agent = TernoAgent(llm=_LLM())
    assert "NotebookEdit" not in agent.tools
    assert "NotebookEdit" in agent.tool_registry.names
    assert "ToolSearch" in agent.tools

    # ...and eager when nothing is deferred.
    assert "NotebookEdit" in TernoAgent(llm=_LLM(), defer_tools=()).tools
