import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.tools.files import EditFileTool, ReadFileTool, WriteFileTool


def test_write_and_read_roundtrip(tmp_path):
    target = tmp_path / "sub" / "hello.txt"
    msg = WriteFileTool().run(path=str(target), content="line 1\nline 2\n")
    assert "Wrote" in msg
    assert target.read_text() == "line 1\nline 2\n"

    out = ReadFileTool().run(path=str(target))
    assert "1\tline 1" in out
    assert "2\tline 2" in out


def test_read_offset_and_limit(tmp_path):
    target = tmp_path / "many.txt"
    target.write_text("\n".join(f"row {i}" for i in range(1, 11)))
    out = ReadFileTool().run(path=str(target), offset=3, limit=2)
    assert "3\trow 3" in out
    assert "4\trow 4" in out
    assert "row 5" not in out


def test_read_missing_file(tmp_path):
    with pytest.raises(ToolError):
        ReadFileTool().run(path=str(tmp_path / "nope.txt"))


def test_edit_replaces_unique_string(tmp_path):
    target = tmp_path / "doc.txt"
    target.write_text("hello world\n")
    EditFileTool().run(path=str(target), old_string="world", new_string="terno")
    assert target.read_text() == "hello terno\n"


def test_edit_requires_unique_match(tmp_path):
    target = tmp_path / "doc.txt"
    target.write_text("foo foo foo")
    with pytest.raises(ToolError):
        EditFileTool().run(path=str(target), old_string="foo", new_string="bar")
    # replace_all bypasses the uniqueness check
    EditFileTool().run(
        path=str(target),
        old_string="foo",
        new_string="bar",
        replace_all=True,
    )
    assert target.read_text() == "bar bar bar"


def test_edit_missing_string(tmp_path):
    target = tmp_path / "doc.txt"
    target.write_text("hello")
    with pytest.raises(ToolError):
        EditFileTool().run(path=str(target), old_string="missing", new_string="x")
