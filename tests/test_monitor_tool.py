from pathlib import Path

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.tools.monitor import MonitorTool


def test_monitor_returns_on_exit(tmp_path: Path):
    out = MonitorTool(workdir=tmp_path).run(command="echo hello; echo world")
    assert "status=exited" in out
    assert "exit_code=0" in out
    assert "hello" in out
    assert "world" in out


def test_monitor_matches_regex_and_stops_early(tmp_path: Path):
    out = MonitorTool(workdir=tmp_path).run(
        command="echo first; echo READY; sleep 5; echo never",
        until_regex="READY",
        timeout_s=3,
    )
    assert "status=matched" in out
    assert "matched_line='READY'" in out
    assert "never" not in out


def test_monitor_times_out(tmp_path: Path):
    out = MonitorTool(workdir=tmp_path).run(
        command="sleep 5",
        timeout_s=1,
    )
    assert "status=timeout" in out


def test_monitor_requires_command(tmp_path: Path):
    with pytest.raises(ToolError):
        MonitorTool(workdir=tmp_path).run(command="   ")


def test_monitor_rejects_invalid_regex(tmp_path: Path):
    with pytest.raises(ToolError, match="invalid until_regex"):
        MonitorTool(workdir=tmp_path).run(command="echo hi", until_regex="(")
