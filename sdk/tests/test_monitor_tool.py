import sys
from pathlib import Path

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.tools.monitor import MonitorTool

# `monitor` shells out via `sh -c` on POSIX and `cmd /c` on Windows (see
# _spawn() in monitor.py) - each platform's tests use that shell's own
# syntax rather than one lowest-common-denominator command string.
_IS_WINDOWS = sys.platform == "win32"


@pytest.mark.skipif(_IS_WINDOWS, reason="POSIX shell syntax (sh -c)")
def test_monitor_returns_on_exit_posix(tmp_path: Path):
    out = MonitorTool(workdir=tmp_path).run(command="echo hello; echo world")
    assert "status=exited" in out
    assert "exit_code=0" in out
    assert "hello" in out
    assert "world" in out


@pytest.mark.skipif(not _IS_WINDOWS, reason="Windows shell syntax (cmd /c)")
def test_monitor_returns_on_exit_windows(tmp_path: Path):
    out = MonitorTool(workdir=tmp_path).run(command="echo hello & echo world")
    assert "status=exited" in out
    assert "exit_code=0" in out
    assert "hello" in out
    assert "world" in out


@pytest.mark.skipif(_IS_WINDOWS, reason="POSIX shell syntax (sh -c)")
def test_monitor_matches_regex_and_stops_early_posix(tmp_path: Path):
    out = MonitorTool(workdir=tmp_path).run(
        command="echo first; echo READY; sleep 5; echo never",
        until_regex="READY",
        timeout_s=3,
    )
    assert "status=matched" in out
    assert "matched_line='READY'" in out
    assert "never" not in out


@pytest.mark.skipif(not _IS_WINDOWS, reason="Windows: cmd.exe buffers piped stdout until exit, so this needs an explicitly-flushed writer")
def test_monitor_matches_regex_and_stops_early_windows(tmp_path: Path):
    code = (
        "import time; "
        "print('first', flush=True); "
        "print('READY', flush=True); "
        "time.sleep(5); "
        "print('never', flush=True)"
    )
    out = MonitorTool(workdir=tmp_path).run(
        command=f'"{sys.executable}" -u -c "{code}"',
        until_regex="READY",
        timeout_s=3,
    )
    assert "status=matched" in out
    assert "matched_line='READY'" in out
    assert "never" not in out


@pytest.mark.skipif(_IS_WINDOWS, reason="POSIX shell syntax (sh -c)")
def test_monitor_times_out_posix(tmp_path: Path):
    out = MonitorTool(workdir=tmp_path).run(
        command="sleep 5",
        timeout_s=1,
    )
    assert "status=timeout" in out


@pytest.mark.skipif(not _IS_WINDOWS, reason="Windows shell syntax (cmd /c)")
def test_monitor_times_out_windows(tmp_path: Path):
    out = MonitorTool(workdir=tmp_path).run(
        command="ping -n 6 127.0.0.1 >NUL",
        timeout_s=1,
    )
    assert "status=timeout" in out


def test_monitor_requires_command(tmp_path: Path):
    with pytest.raises(ToolError):
        MonitorTool(workdir=tmp_path).run(command="   ")


def test_monitor_rejects_invalid_regex(tmp_path: Path):
    with pytest.raises(ToolError, match="invalid until_regex"):
        MonitorTool(workdir=tmp_path).run(command="echo hi", until_regex="(")
