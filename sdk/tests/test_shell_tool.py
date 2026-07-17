from pathlib import Path

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.sandbox.base import ExecutionResult
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


# ----- sandbox delegation ---------------------------------------------- #
#
# Bash always runs through Sandbox.run_shell. Without a sandbox it defaults
# to LocalSandbox (host); the tests above exercise that path. These cover an
# explicitly-injected sandbox.


class _RecordingSandbox:
    """Captures the run_shell call and returns a canned result."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run_python(self, code, *, timeout_s=30, env=None):  # pragma: no cover
        raise NotImplementedError

    def run_shell(self, command, *, timeout_s=30, cwd=None, env=None, cancel_token=None):
        self.calls.append({"command": command, "timeout_s": timeout_s, "cwd": cwd})
        return ExecutionResult(stdout="ran", stderr="", exit_code=0)


def test_bash_delegates_to_injected_sandbox(tmp_path: Path):
    sb = _RecordingSandbox()
    out = BashTool(workdir=tmp_path, sandbox=sb).run(command="echo hi")
    assert "exit_code=0" in out
    assert "ran" in out
    assert sb.calls == [{"command": "echo hi", "timeout_s": 120, "cwd": str(tmp_path)}]


class _TimeoutSandbox:
    """Sandbox stub whose run_shell always reports a timeout."""

    def run_python(self, code, *, timeout_s=30, env=None):  # pragma: no cover
        raise NotImplementedError

    def run_shell(self, command, *, timeout_s=30, cwd=None, env=None):
        return ExecutionResult(stdout="", stderr="", exit_code=124, timed_out=True)


def test_bash_sandbox_timeout_is_surfaced(tmp_path: Path):
    tool = BashTool(workdir=tmp_path, sandbox=_TimeoutSandbox())
    out = tool.run(command="sleep 100", timeout_s=1)
    assert "exit_code=124" in out
    assert "timed out after 1s" in out


class _NoShellSandbox:
    """Older-protocol sandbox that only supports run_python."""

    def run_python(self, code, *, timeout_s=30, env=None):  # pragma: no cover
        raise NotImplementedError


def test_bash_sandbox_without_run_shell_raises(tmp_path: Path):
    tool = BashTool(workdir=tmp_path, sandbox=_NoShellSandbox())
    with pytest.raises(ToolError):
        tool.run(command="echo hi")
