import sys

import pytest

from terno_agent.sandbox.local import LocalSandbox

# run_shell shells out via `sh -c` on POSIX and shell=True on Windows (see
# LocalSandbox._exec) - each platform's tests use that shell's own commands
# rather than one lowest-common-denominator command string.
_IS_WINDOWS = sys.platform == "win32"


def test_local_sandbox_captures_stdout():
    sb = LocalSandbox()
    result = sb.run_python("print('hello, terno')")
    assert result.ok
    assert "hello, terno" in result.stdout


def test_local_sandbox_reports_nonzero_exit():
    sb = LocalSandbox()
    result = sb.run_python("import sys; sys.exit(7)")
    assert result.exit_code == 7
    assert not result.ok


def test_local_sandbox_timeout():
    sb = LocalSandbox()
    result = sb.run_python("import time; time.sleep(5)", timeout_s=1)
    assert result.timed_out
    assert result.exit_code == 124


def test_local_sandbox_run_shell_captures_stdout():
    sb = LocalSandbox()
    result = sb.run_shell("echo hello, shell")
    assert result.ok
    assert "hello, shell" in result.stdout


def test_local_sandbox_run_shell_reports_nonzero_exit():
    sb = LocalSandbox()
    result = sb.run_shell("exit 7")
    assert result.exit_code == 7
    assert not result.ok


@pytest.mark.skipif(_IS_WINDOWS, reason="POSIX command (sleep)")
def test_local_sandbox_run_shell_timeout_posix():
    sb = LocalSandbox()
    result = sb.run_shell("sleep 5", timeout_s=1)
    assert result.timed_out
    assert result.exit_code == 124


@pytest.mark.skipif(not _IS_WINDOWS, reason="Windows delay (ping)")
def test_local_sandbox_run_shell_timeout_windows():
    # `timeout` fails under redirected/non-interactive stdin, which is
    # exactly how tests spawn this process - ping as a delay instead.
    sb = LocalSandbox()
    result = sb.run_shell("ping -n 6 127.0.0.1 >NUL", timeout_s=1)
    assert result.timed_out
    assert result.exit_code == 124


@pytest.mark.skipif(_IS_WINDOWS, reason="POSIX command (ls)")
def test_local_sandbox_run_shell_runs_in_cwd_posix(tmp_path):
    (tmp_path / "marker.txt").write_text("x")
    sb = LocalSandbox()
    result = sb.run_shell("ls", cwd=str(tmp_path))
    assert result.ok
    assert "marker.txt" in result.stdout


@pytest.mark.skipif(not _IS_WINDOWS, reason="Windows command (dir)")
def test_local_sandbox_run_shell_runs_in_cwd_windows(tmp_path):
    (tmp_path / "marker.txt").write_text("x")
    sb = LocalSandbox()
    result = sb.run_shell("dir", cwd=str(tmp_path))
    assert result.ok
    assert "marker.txt" in result.stdout
