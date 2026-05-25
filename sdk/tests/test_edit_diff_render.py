from __future__ import annotations

import io

from rich.console import Console
from rich.text import Text

from terno_agent.cli import _format_call_body, _format_edit_diff


def _render(text: Text) -> str:
    """Capture Rich-styled ANSI output of ``text`` for snapshot-style asserts."""
    console = Console(
        file=io.StringIO(),
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        width=120,
    )
    console.print(text, end="")
    return console.file.getvalue()


def test_edit_call_body_renders_unified_diff() -> None:
    body = _format_call_body(
        "edit_file",
        {
            "path": "src/foo.py",
            "old_string": "def add(a, b):\n    return a+b\n",
            "new_string": "def add(a, b):\n    return a + b\n",
        },
    )
    assert isinstance(body, Text)
    plain = body.plain
    assert "path: src/foo.py" in plain
    assert "--- a/src/foo.py" in plain
    assert "+++ b/src/foo.py" in plain
    assert "-    return a+b" in plain
    assert "+    return a + b" in plain


def test_diff_lines_are_coloured() -> None:
    body = _format_edit_diff(
        path="x.txt",
        old="hello\n",
        new="world\n",
        replace_all=False,
    )
    rendered = _render(body)
    # Rich emits ANSI escape sequences; green/red codes mark the changed lines.
    assert "\x1b[32m" in rendered  # green for +
    assert "\x1b[31m" in rendered  # red for -
    assert "\x1b[36m" in rendered  # cyan for @@ hunk header


def test_diff_handles_no_change_gracefully() -> None:
    body = _format_edit_diff(
        path="same.txt",
        old="unchanged\n",
        new="unchanged\n",
        replace_all=False,
    )
    assert "no textual difference" in body.plain


def test_diff_truncates_huge_changes() -> None:
    old = "\n".join(f"old-{i}" for i in range(500))
    new = "\n".join(f"new-{i}" for i in range(500))
    body = _format_edit_diff(path="big.txt", old=old, new=new, replace_all=True)
    assert "diff truncated" in body.plain
    # replace_all marker bubbles up into the header.
    assert "(replace_all)" in body.plain


def test_diff_shows_path_when_missing() -> None:
    body = _format_edit_diff(path="", old="a\n", new="b\n", replace_all=False)
    assert "path: (unspecified)" in body.plain
    assert "--- a/(unspecified)" in body.plain


def test_write_body_renders_syntax_for_new_file(tmp_path) -> None:
    target = tmp_path / "fresh.py"
    body = _format_call_body(
        "write_file",
        {"path": str(target), "content": "def hello():\n    return 1\n"},
    )
    # New file → Syntax-highlighted content, not a diff.
    assert not isinstance(body, Text)


def test_write_body_renders_diff_when_overwriting_existing_file(tmp_path) -> None:
    target = tmp_path / "regen.py"
    target.write_text("def hello():\n    return 1\n")
    body = _format_call_body(
        "write_file",
        {
            "path": str(target),
            "content": "def hello():\n    return 2\n",
            "overwrite": True,
        },
    )
    assert isinstance(body, Text)
    plain = body.plain
    assert "(replace_all)" in plain  # the overwrite marker
    assert "-    return 1" in plain
    assert "+    return 2" in plain
