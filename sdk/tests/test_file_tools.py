import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.sandbox.base import ExecutionResult
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


def test_relative_paths_resolve_against_tool_workdir(tmp_path):
    WriteFileTool(workdir=tmp_path).run(path="nested/hello.txt", content="hi\n")
    assert (tmp_path / "nested" / "hello.txt").read_text() == "hi\n"

    out = ReadFileTool(workdir=tmp_path).run(path="nested/hello.txt")
    assert "1\thi" in out

    EditFileTool(workdir=tmp_path).run(
        path="nested/hello.txt",
        old_string="hi",
        new_string="bye",
    )
    assert (tmp_path / "nested" / "hello.txt").read_text() == "bye\n"


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


def test_write_refuses_to_clobber_existing_file(tmp_path):
    target = tmp_path / "existing.txt"
    target.write_text("original\n")
    with pytest.raises(ToolError, match="edit_file"):
        WriteFileTool().run(path=str(target), content="replaced\n")
    # File is untouched.
    assert target.read_text() == "original\n"


def test_write_overwrite_flag_replaces_existing_file(tmp_path):
    target = tmp_path / "regen.txt"
    target.write_text("v1\n")
    WriteFileTool().run(path=str(target), content="v2\n", overwrite=True)
    assert target.read_text() == "v2\n"


def test_write_rejects_directory_target(tmp_path):
    with pytest.raises(ToolError, match="directory"):
        WriteFileTool().run(path=str(tmp_path), content="x")


# --------------------------------------------------------------------------- #
# Native FileOpSandbox path: a sandbox that implements read_file/write_file/   #
# edit_file is used directly, instead of the run_python + stdout template.     #
# --------------------------------------------------------------------------- #

_SANDBOX_PATH = "/home/runner/user_workspace/report.txt"


class _NativeSandbox:
    """A sandbox implementing FileOpSandbox. run_python raises so a test
    fails loudly if the tools ever fall back to the stdout template path."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def run_python(self, code, **kwargs):
        raise AssertionError("native file op should not use run_python")

    def read_file(self, path, *, offset=1, limit=2000):
        self.calls.append(("read", path, offset, limit))
        return self.results["read"]

    def write_file(self, path, content, *, overwrite=False):
        self.calls.append(("write", path, content, overwrite))
        return self.results["write"]

    def edit_file(self, path, old, new, *, replace_all=False):
        self.calls.append(("edit", path, old, new, replace_all))
        return self.results["edit"]


def test_native_write_is_preferred_over_template():
    sandbox = _NativeSandbox(
        {"write": {"ok": True, "output": "Wrote 5 bytes to " + _SANDBOX_PATH}}
    )
    msg = WriteFileTool(sandbox=sandbox).run(
        path=_SANDBOX_PATH, content="hello", overwrite=True
    )
    assert msg == "Wrote 5 bytes to " + _SANDBOX_PATH
    assert sandbox.calls == [("write", _SANDBOX_PATH, "hello", True)]


def test_native_read_output_returned_verbatim():
    # The native op already applies offset/limit and numbers lines; the tool
    # must NOT re-number it.
    sandbox = _NativeSandbox(
        {"read": {"ok": True, "output": "     1\thello\n     2\tworld"}}
    )
    out = ReadFileTool(sandbox=sandbox).run(path=_SANDBOX_PATH, offset=1, limit=10)
    assert out == "     1\thello\n     2\tworld"
    assert sandbox.calls == [("read", _SANDBOX_PATH, 1, 10)]


def test_native_edit_is_preferred():
    sandbox = _NativeSandbox(
        {"edit": {"ok": True, "output": "Replaced 1 occurrence(s) in " + _SANDBOX_PATH}}
    )
    msg = EditFileTool(sandbox=sandbox).run(
        path=_SANDBOX_PATH, old_string="a", new_string="b"
    )
    assert "Replaced 1 occurrence(s)" in msg
    assert sandbox.calls == [("edit", _SANDBOX_PATH, "a", "b", False)]


def test_native_error_result_raises_toolerror():
    sandbox = _NativeSandbox(
        {"write": {"ok": False, "error": "path already exists; use overwrite"}}
    )
    with pytest.raises(ToolError, match="already exists"):
        WriteFileTool(sandbox=sandbox).run(path=_SANDBOX_PATH, content="x")


def test_native_unexpected_result_raises_toolerror():
    sandbox = _NativeSandbox({"write": "not a dict"})
    with pytest.raises(ToolError, match="unexpected result"):
        WriteFileTool(sandbox=sandbox).run(
            path=_SANDBOX_PATH, content="x", overwrite=True
        )


class _TemplateSandbox:
    """A bare Sandbox (only run_python) — exercises the fallback path. Stands
    in for the pre-fix behaviour and for the SDK's own Docker sandbox."""

    def __init__(self, stdout):
        self.stdout = stdout
        self.ran = False

    def run_python(self, code, **kwargs):
        self.ran = True
        return ExecutionResult(stdout=self.stdout, stderr="", exit_code=0)


def test_falls_back_to_template_when_no_native_methods():
    # {"bytes": 5} is what _WRITE_FILE_TEMPLATE prints on success.
    sandbox = _TemplateSandbox('{"bytes": 5}')
    msg = WriteFileTool(sandbox=sandbox).run(
        path=_SANDBOX_PATH, content="hello", overwrite=True
    )
    assert msg == "Wrote 5 bytes to " + _SANDBOX_PATH
    assert sandbox.ran


def test_template_empty_stdout_still_errors_without_native():
    # The original bug: empty stdout (lost kernel output) -> unparsable. This
    # is exactly what the native path exists to avoid; without it, the
    # fallback still surfaces the error rather than silently succeeding.
    sandbox = _TemplateSandbox("")
    with pytest.raises(ToolError, match="unparsable output"):
        WriteFileTool(sandbox=sandbox).run(
            path=_SANDBOX_PATH, content="hello", overwrite=True
        )
