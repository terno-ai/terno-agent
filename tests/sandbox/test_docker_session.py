"""Live integration tests for the session-aware DockerSandbox.

Skipped automatically when docker-py isn't installed or the daemon
isn't reachable. They're cheap (one tiny image) but real.
"""

from __future__ import annotations

import pytest

from terno_agent.sandbox.base import ExecutionResult


pytest.importorskip("docker")


@pytest.fixture(scope="module")
def docker_client():
    import docker

    from terno_agent.sandbox.docker import discover_docker_base_url

    base_url = discover_docker_base_url()
    try:
        client = docker.DockerClient(base_url=base_url) if base_url else docker.from_env()
        client.ping()
    except Exception as exc:
        pytest.skip(f"Docker daemon not reachable: {exc}")
    return client


@pytest.fixture
def fresh_sandbox(docker_client):
    """A DockerSandbox with a unique non-persisted name so tests don't collide."""
    import uuid

    from terno_agent.sandbox.docker import DockerSandbox

    sb = DockerSandbox(container_name=f"terno-test-{uuid.uuid4().hex[:8]}")
    yield sb
    sb.close()


def test_state_persists_within_session(fresh_sandbox):
    r1 = fresh_sandbox.run_python("x = 7")
    assert r1.exit_code == 0
    r2 = fresh_sandbox.run_python("print(x * 6)")
    assert r2.exit_code == 0
    assert r2.stdout.strip() == "42"


def test_imports_persist_within_session(fresh_sandbox):
    fresh_sandbox.run_python("import math")
    r = fresh_sandbox.run_python("print(math.floor(2.9))")
    assert r.stdout.strip() == "2"


def test_files_persist_in_work_within_session(fresh_sandbox):
    fresh_sandbox.run_python(
        "open('/work/note.txt', 'w').write('hello-session')"
    )
    r = fresh_sandbox.run_python("print(open('/work/note.txt').read())")
    assert r.stdout.strip() == "hello-session"


def test_exception_does_not_break_session(fresh_sandbox):
    bad = fresh_sandbox.run_python("raise RuntimeError('boom')")
    assert bad.exit_code == 1
    assert "RuntimeError" in bad.stderr
    ok = fresh_sandbox.run_python("print('still alive')")
    assert ok.exit_code == 0
    assert ok.stdout.strip() == "still alive"


def test_close_removes_container_when_not_persisted(docker_client):
    import uuid

    from terno_agent.sandbox.docker import DockerSandbox

    name = f"terno-test-{uuid.uuid4().hex[:8]}"
    sb = DockerSandbox(container_name=name, persist=False)
    sb.run_python("x = 1")  # trigger container creation
    sb.close()
    # Container is gone.
    with pytest.raises(Exception):
        docker_client.containers.get(name)


def test_persist_keeps_container_alive_across_close(docker_client):
    import uuid

    from terno_agent.sandbox.docker import DockerSandbox

    name = f"terno-test-persist-{uuid.uuid4().hex[:8]}"
    try:
        sb1 = DockerSandbox(container_name=name, persist=True)
        r = sb1.run_python("session_token = 'kept-around'")
        assert r.exit_code == 0
        sb1.close()
        # Container should still exist.
        c = docker_client.containers.get(name)
        assert c.status in {"running", "created"}

        # New sandbox reuses the same container; state is preserved
        # because the python driver process never restarted.
        sb2 = DockerSandbox(container_name=name, persist=True)
        r2 = sb2.run_python("print(session_token)")
        assert r2.stdout.strip() == "kept-around"
        sb2.close()
    finally:
        try:
            c = docker_client.containers.get(name)
            c.remove(force=True)
        except Exception:
            pass


def test_run_python_returns_execution_result(fresh_sandbox):
    r = fresh_sandbox.run_python("print('ok')")
    assert isinstance(r, ExecutionResult)
    assert r.ok is True
    assert r.timed_out is False


def test_timeout_kills_and_recovers(fresh_sandbox):
    r = fresh_sandbox.run_python(
        "import time; time.sleep(60)", timeout_s=1
    )
    assert r.timed_out is True
    assert r.exit_code == 124
    # The container was killed; next call must work on a fresh container.
    r2 = fresh_sandbox.run_python("print('recovered')")
    assert r2.exit_code == 0
    assert r2.stdout.strip() == "recovered"
