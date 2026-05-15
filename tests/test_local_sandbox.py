from terno_agent.sandbox.local import LocalSandbox


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
