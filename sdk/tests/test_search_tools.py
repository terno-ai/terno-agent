from pathlib import Path

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.tools.search import GlobTool, GrepTool


def _seed(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("import os\nprint('hello')\n")
    (tmp_path / "src" / "b.py").write_text("import sys\nprint('world')\n")
    (tmp_path / "src" / "c.txt").write_text("not python\n")
    (tmp_path / "README.md").write_text("# project\n")


def test_glob_matches_recursive(tmp_path: Path):
    _seed(tmp_path)
    out = GlobTool(workdir=tmp_path).run(pattern="**/*.py")
    assert "a.py" in out
    assert "b.py" in out
    assert "c.txt" not in out


def test_glob_no_matches(tmp_path: Path):
    _seed(tmp_path)
    out = GlobTool(workdir=tmp_path).run(pattern="**/*.rs")
    assert "no files matched" in out


def test_glob_respects_explicit_path(tmp_path: Path):
    _seed(tmp_path)
    out = GlobTool(workdir=tmp_path).run(pattern="*.py", path="src")
    assert "a.py" in out
    assert "b.py" in out
    # README.md (at root) should not appear because we scoped to src/
    assert "README.md" not in out


def test_glob_requires_pattern(tmp_path: Path):
    with pytest.raises(ToolError):
        GlobTool(workdir=tmp_path).run(pattern="")


def test_glob_rejects_missing_root(tmp_path: Path):
    with pytest.raises(ToolError, match="not found"):
        GlobTool(workdir=tmp_path).run(pattern="*", path=str(tmp_path / "nope"))


def test_grep_finds_matches(tmp_path: Path):
    _seed(tmp_path)
    out = GrepTool(workdir=tmp_path).run(pattern=r"print\('hello'\)")
    assert "a.py" in out
    assert "hello" in out
    assert "b.py" not in out


def test_grep_filter_by_glob(tmp_path: Path):
    _seed(tmp_path)
    out = GrepTool(workdir=tmp_path).run(pattern="python", glob="*.txt")
    assert "c.txt" in out
    assert "a.py" not in out


def test_grep_case_insensitive(tmp_path: Path):
    _seed(tmp_path)
    out = GrepTool(workdir=tmp_path).run(
        pattern="HELLO",
        case_insensitive=True,
    )
    assert "hello" in out


def test_grep_no_match(tmp_path: Path):
    _seed(tmp_path)
    out = GrepTool(workdir=tmp_path).run(pattern="qzqzqz")
    assert "no matches" in out


def test_grep_invalid_regex_python_fallback(tmp_path: Path, monkeypatch):
    # Force the Python fallback by hiding rg, then send an invalid regex.
    monkeypatch.setattr("terno_agent.tools.search.which", lambda _name: None)
    _seed(tmp_path)
    with pytest.raises(ToolError, match="invalid regex"):
        GrepTool(workdir=tmp_path).run(pattern="(")
