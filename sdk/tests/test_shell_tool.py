from pathlib import Path

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.tools.shell import BashTool


def test_bash_returns_stdout_and_exit_code(tmp_path: Path):
    out = BashTool(workdir=tmp_path).run(command="echo hello")
    assert "exit_code=0" in out
    assert "hello" in out


def test_bash_captures_nonzero_exit(tmp_path: Path):
    out = BashTool(workdir=tmp_path).run(command="false")
    assert "exit_code=1" in out


def test_bash_runs_in_workdir(tmp_path: Path):
    (tmp_path / "marker.txt").write_text("x")
    out = BashTool(workdir=tmp_path).run(command="ls")
    assert "marker.txt" in out


def test_bash_timeout(tmp_path: Path):
    out = BashTool(workdir=tmp_path).run(command="sleep 5", timeout_s=1)
    assert "exit_code=124" in out
    assert "timed out" in out


def test_bash_empty_command(tmp_path: Path):
    with pytest.raises(ToolError):
        BashTool(workdir=tmp_path).run(command="")
