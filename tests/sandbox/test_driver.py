"""Driver protocol — runs the snippet executor in a real subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest

from terno_agent.sandbox._driver import DRIVER_SOURCE, SENTINEL


@pytest.fixture
def driver() -> Iterator[subprocess.Popen]:
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", DRIVER_SOURCE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()


def _call(driver: subprocess.Popen, code: str, timeout: float = 5.0) -> dict:
    driver.stdin.write(json.dumps({"code": code}) + "\n")
    driver.stdin.flush()
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            raise AssertionError("driver did not respond in time")
        line = driver.stdout.readline()
        if not line:
            raise AssertionError("driver closed stdout unexpectedly")
        idx = line.find(SENTINEL)
        if idx == -1:
            continue
        return json.loads(line[idx + len(SENTINEL):])


def test_basic_run(driver):
    r = _call(driver, "print(1 + 1)")
    assert r["stdout"].strip() == "2"
    assert r["stderr"] == ""
    assert r["exit_code"] == 0


def test_variables_persist_across_calls(driver):
    _call(driver, "x = 41; y = 1")
    r = _call(driver, "print(x + y)")
    assert r["stdout"].strip() == "42"


def test_imports_persist_across_calls(driver):
    _call(driver, "import math")
    r = _call(driver, "print(round(math.pi, 2))")
    assert r["stdout"].strip() == "3.14"


def test_function_defs_persist(driver):
    _call(driver, "def double(n):\n    return n * 2")
    r = _call(driver, "print(double(21))")
    assert r["stdout"].strip() == "42"


def test_exception_yields_nonzero_but_keeps_driver_alive(driver):
    bad = _call(driver, "1 / 0")
    assert bad["exit_code"] == 1
    assert "ZeroDivisionError" in bad["stderr"]
    # Driver must still respond after a snippet exception.
    ok = _call(driver, "print('still here')")
    assert ok["stdout"].strip() == "still here"
    assert ok["exit_code"] == 0


def test_system_exit_does_not_kill_driver(driver):
    r = _call(driver, "raise SystemExit(7)")
    assert r["exit_code"] == 7
    # Next call still works.
    r2 = _call(driver, "print('ok')")
    assert r2["stdout"].strip() == "ok"


def test_malformed_request_does_not_kill_driver(driver):
    driver.stdin.write("not-json\n")
    driver.stdin.flush()
    # Read until SENTINEL appears.
    deadline = time.monotonic() + 5
    response = None
    while time.monotonic() < deadline:
        line = driver.stdout.readline()
        if not line:
            break
        idx = line.find(SENTINEL)
        if idx != -1:
            response = json.loads(line[idx + len(SENTINEL):])
            break
    assert response is not None
    assert response["exit_code"] == 1
    assert "protocol error" in response["stderr"]
    # Driver still alive.
    r = _call(driver, "print(2 * 2)")
    assert r["stdout"].strip() == "4"
