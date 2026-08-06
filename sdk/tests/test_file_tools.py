import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.tools.files import (
    EditFileTool,
    FileStateTracker,
    ReadFileTool,
    WriteFileTool,
)


def _read(path) -> FileStateTracker:
    """A tracker that has already seen `path` — Edit/Write require this."""
    tracker = FileStateTracker()
    tracker.mark_read(path)
    return tracker


def test_write_and_read_roundtrip(tmp_path):
    target = tmp_path / "sub" / "hello.txt"
    msg = WriteFileTool().run(file_path=str(target), content="line 1\nline 2\n")
    assert "Wrote" in msg
    assert target.read_text() == "line 1\nline 2\n"

    out = ReadFileTool().run(file_path=str(target))
    assert "1\tline 1" in out
    assert "2\tline 2" in out


def test_read_offset_and_limit(tmp_path):
    target = tmp_path / "many.txt"
    target.write_text("\n".join(f"row {i}" for i in range(1, 11)))
    out = ReadFileTool().run(file_path=str(target), offset=3, limit=2)
    assert "3\trow 3" in out
    assert "4\trow 4" in out
    assert "row 5" not in out


def test_relative_paths_resolve_against_tool_workdir(tmp_path):
    # Read/Write/Edit share one tracker, the way the agent wires them.
    tracker = FileStateTracker()
    WriteFileTool(workdir=tmp_path, tracker=tracker).run(
        file_path="nested/hello.txt", content="hi\n"
    )
    assert (tmp_path / "nested" / "hello.txt").read_text() == "hi\n"

    out = ReadFileTool(workdir=tmp_path, tracker=tracker).run(
        file_path="nested/hello.txt"
    )
    assert "1\thi" in out

    EditFileTool(workdir=tmp_path, tracker=tracker).run(
        file_path="nested/hello.txt",
        old_string="hi",
        new_string="bye",
    )
    assert (tmp_path / "nested" / "hello.txt").read_text() == "bye\n"


def test_read_missing_file(tmp_path):
    with pytest.raises(ToolError):
        ReadFileTool().run(file_path=str(tmp_path / "nope.txt"))


def test_edit_replaces_unique_string(tmp_path):
    target = tmp_path / "doc.txt"
    target.write_text("hello world\n")
    EditFileTool(tracker=_read(target)).run(
        file_path=str(target), old_string="world", new_string="terno"
    )
    assert target.read_text() == "hello terno\n"


def test_edit_requires_unique_match(tmp_path):
    target = tmp_path / "doc.txt"
    target.write_text("foo foo foo")
    tracker = _read(target)
    with pytest.raises(ToolError):
        EditFileTool(tracker=tracker).run(
            file_path=str(target), old_string="foo", new_string="bar"
        )
    # replace_all bypasses the uniqueness check
    EditFileTool(tracker=tracker).run(
        file_path=str(target),
        old_string="foo",
        new_string="bar",
        replace_all=True,
    )
    assert target.read_text() == "bar bar bar"


def test_edit_missing_string(tmp_path):
    target = tmp_path / "doc.txt"
    target.write_text("hello")
    with pytest.raises(ToolError):
        EditFileTool(tracker=_read(target)).run(
            file_path=str(target), old_string="missing", new_string="x"
        )


def test_write_refuses_to_clobber_existing_file(tmp_path):
    target = tmp_path / "existing.txt"
    target.write_text("original\n")
    with pytest.raises(ToolError, match="Edit"):
        WriteFileTool().run(file_path=str(target), content="replaced\n")
    # File is untouched.
    assert target.read_text() == "original\n"


def test_write_replaces_an_existing_file_once_it_has_been_read(tmp_path):
    target = tmp_path / "regen.txt"
    target.write_text("v1\n")
    WriteFileTool(tracker=_read(target)).run(file_path=str(target), content="v2\n")
    assert target.read_text() == "v2\n"


def test_edit_refuses_a_file_that_was_never_read(tmp_path):
    target = tmp_path / "doc.txt"
    target.write_text("hello\n")
    with pytest.raises(ToolError, match="has not been read"):
        EditFileTool().run(file_path=str(target), old_string="hello", new_string="bye")
    assert target.read_text() == "hello\n"


def test_write_marks_the_file_read_so_a_second_write_succeeds(tmp_path):
    target = tmp_path / "new.txt"
    tool = WriteFileTool()
    tool.run(file_path=str(target), content="v1\n")
    tool.run(file_path=str(target), content="v2\n")
    assert target.read_text() == "v2\n"


def test_write_rejects_directory_target(tmp_path):
    with pytest.raises(ToolError, match="directory"):
        WriteFileTool().run(file_path=str(tmp_path), content="x")
