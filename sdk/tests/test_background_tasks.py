"""Background tasks: store rules, the service, the sandbox runner, and how
the agent is told about finished work."""

from __future__ import annotations

import os
import sys
import time
import uuid
from unittest import mock

import pytest

from terno_agent.agents.base import BaseAgent
from terno_agent.agents.terno import TernoAgent
from terno_agent.background import (
    COMPLETED,
    FAILED,
    RUNNING,
    STOPPED,
    TIMED_OUT,
    BackgroundRunnerError,
    BackgroundTasks,
    InMemoryBackgroundTaskStore,
    RunInBackgroundTool,
    RunnerState,
    SandboxBackgroundRunner,
    _sandbox_ops,
)
from terno_agent.background.runner import build_call
from terno_agent.config import Config
from terno_agent.core.exceptions import ToolError
from terno_agent.core.messages import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from terno_agent.core.tool import ToolSchema
from terno_agent.llm.base import LLMResponse
from terno_agent.sandbox.local import LocalSandbox

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")


class _FakeRunner:
    def __init__(self) -> None:
        self.states: dict[str, RunnerState] = {}
        self.stopped: list[str] = []
        self.fail_start = False

    def start(self, task, payload, *, max_runtime_s):
        if self.fail_start:
            raise BackgroundRunnerError("sandbox is down")
        handle = f"h{task.id}"
        self.states[handle] = RunnerState(running=True)
        return handle

    def poll(self, handle):
        return self.states[handle]

    def read_output(self, handle, *, max_chars):
        return f"output of {handle}"[-max_chars:]

    def stop(self, handle):
        self.stopped.append(handle)


class _Clock:
    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now


def _service(**kwargs) -> tuple[BackgroundTasks, _FakeRunner, _Clock]:
    runner = _FakeRunner()
    clock = _Clock()
    service = BackgroundTasks({"sandbox": runner}, clock=clock, poll_interval_s=0, **kwargs)
    return service, runner, clock


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class TestStore:
    def test_create_needs_a_description(self):
        with pytest.raises(ToolError):
            InMemoryBackgroundTaskStore().create("  ", kind="python", runner="sandbox")

    def test_first_outcome_wins(self):
        store = InMemoryBackgroundTaskStore()
        task = store.create("job", kind="python", runner="sandbox")
        store.finish(task.id, STOPPED)
        assert store.finish(task.id, COMPLETED, exit_code=0).status == STOPPED
        assert store.get(task.id).finished_at is not None

    def test_finished_tasks_are_claimed_once(self):
        store = InMemoryBackgroundTaskStore()
        a = store.create("a", kind="python", runner="sandbox")
        store.create("b", kind="python", runner="sandbox")
        store.finish(a.id, COMPLETED, exit_code=0)
        assert [t.id for t in store.claim_unnotified()] == [a.id]
        assert store.claim_unnotified() == []

    def test_observer_sees_every_change(self):
        store = InMemoryBackgroundTaskStore()
        seen = []
        store.set_on_change(lambda t: seen.append(t.status))
        task = store.create("job", kind="python", runner="sandbox")
        store.finish(task.id, FAILED, exit_code=1)
        assert seen == [RUNNING, FAILED]

    def test_rejects_unknown_fields_and_statuses(self):
        store = InMemoryBackgroundTaskStore()
        task = store.create("job", kind="python", runner="sandbox")
        with pytest.raises(ValueError):
            store.update(task.id, description="other")
        with pytest.raises(ValueError):
            store.update(task.id, status="done")


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


class TestService:
    def test_finished_task_becomes_one_notice(self):
        service, runner, _ = _service()
        task = service.start(kind="python", payload="x", description="train model")
        assert service.take_notices() == []

        runner.states[task.handle] = RunnerState(running=False, exit_code=0)
        notices = service.take_notices()
        assert len(notices) == 1
        assert "train model" in notices[0] and "completed" in notices[0]
        assert 'background_output(task_id="1")' in notices[0]
        assert service.take_notices() == []

    def test_finish_time_is_when_the_work_ended(self):
        service, runner, clock = _service()
        task = service.start(kind="python", payload="x", description="train")
        clock.now += 100
        runner.states[task.handle] = RunnerState(running=False, exit_code=0, ended_at=clock.now - 30)
        service.refresh(force=True)
        assert service.store.get(task.id).finished_at == clock.now - 30

    def test_finish_time_stays_between_start_and_now(self):
        service, runner, clock = _service()
        early = service.start(kind="python", payload="x", description="a")
        late = service.start(kind="python", payload="x", description="b")
        clock.now += 100
        # Another machine's clock: before the start, or in the future.
        runner.states[early.handle] = RunnerState(running=False, exit_code=0, ended_at=early.started_at - 50)
        runner.states[late.handle] = RunnerState(running=False, exit_code=0, ended_at=clock.now + 50)
        service.refresh(force=True)
        assert service.store.get(early.id).finished_at == early.started_at
        assert service.store.get(late.id).finished_at == clock.now

    def test_nonzero_exit_and_lost_process_are_failures(self):
        service, runner, _ = _service()
        a = service.start(kind="shell", payload="x", description="a")
        b = service.start(kind="shell", payload="y", description="b")
        runner.states[a.handle] = RunnerState(running=False, exit_code=2)
        runner.states[b.handle] = RunnerState(running=False, error="gone")
        service.refresh(force=True)
        assert service.store.get(a.id).status == FAILED
        assert service.store.get(a.id).exit_code == 2
        assert service.store.get(b.id).status == FAILED
        assert service.store.get(b.id).error == "gone"

    def test_running_limit(self):
        service, _, _ = _service(max_running=1)
        service.start(kind="python", payload="x", description="a")
        with pytest.raises(ToolError, match="limit"):
            service.start(kind="python", payload="x", description="b")

    def test_time_limit_stops_the_task(self):
        service, runner, clock = _service(max_runtime_s=60)
        task = service.start(kind="python", payload="x", description="slow")
        clock.now += 61
        service.refresh(force=True)
        assert service.store.get(task.id).status == TIMED_OUT
        assert runner.stopped == [task.handle]
        assert "timed out" in service.take_notices()[0]

    def test_stopping_is_not_announced(self):
        service, runner, _ = _service()
        task = service.start(kind="python", payload="x", description="a")
        assert service.stop(task.id).status == STOPPED
        assert runner.stopped == [task.handle]
        assert service.take_notices() == []

    def test_failed_start_is_recorded(self):
        service, runner, _ = _service()
        runner.fail_start = True
        with pytest.raises(ToolError, match="sandbox is down"):
            service.start(kind="python", payload="x", description="a")
        assert service.store.list()[0].status == FAILED
        assert service.take_notices() == []

    def test_poll_errors_leave_the_task_running(self):
        service, runner, _ = _service()
        task = service.start(kind="python", payload="x", description="a")

        def boom(handle):
            raise BackgroundRunnerError("busy")

        runner.poll = boom
        service.refresh(force=True)
        assert service.store.get(task.id).status == RUNNING

    def test_context_lists_running_tasks(self):
        service, runner, _ = _service()
        assert service.context_block() == ""
        task = service.start(kind="python", payload="x", description="crunch")
        assert "crunch" in service.context_block()
        runner.states[task.handle] = RunnerState(running=False, exit_code=0)
        assert service.context_block() == ""

    def test_output_includes_status(self):
        service, _, _ = _service()
        task = service.start(kind="python", payload="x", description="a")
        text = service.output(task.id)
        assert "running" in text and "output of h1" in text


# --------------------------------------------------------------------------- #
# Sandbox runner (real processes via LocalSandbox)
# --------------------------------------------------------------------------- #


# Python payloads run the same on every OS, so these cover Windows too.
SLEEP = "import time; time.sleep(30)"
# Starts a child that would outlive the task if stop missed it.
SLEEP_WITH_CHILD = (
    "import os, subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "open(os.path.join(os.getcwd(), 'child.pid'), 'w').write(str(child.pid))\n"
    "time.sleep(60)\n"
)


def _wait(runner, handle, timeout=15.0) -> RunnerState:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = runner.poll(handle)
        if not state.running:
            return state
        time.sleep(0.2)
    raise AssertionError("background task did not finish")


def _wait_for_file(path, timeout=15.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            text = open(path).read().strip()
            if text:
                return text
        time.sleep(0.1)
    raise AssertionError(f"{path} never appeared")


class TestSandboxRunner:
    def _start(self, tmp_path, kind, payload, task_id="1"):
        store = InMemoryBackgroundTaskStore()
        runner = SandboxBackgroundRunner(LocalSandbox(), root_dir=str(tmp_path), cwd=str(tmp_path))
        task = store.create("t", kind=kind, runner="sandbox")
        task.id = task_id
        return runner, runner.start(task, payload, max_runtime_s=600)

    def test_python_runs_detached_and_output_is_kept(self, tmp_path):
        runner, handle = self._start(tmp_path, "python", "print('hello from bg')")
        state = _wait(runner, handle)
        assert state.exit_code == 0
        assert "hello from bg" in runner.read_output(handle, max_chars=1000)

    def test_end_time_is_reported(self, tmp_path):
        before = time.time()
        runner, handle = self._start(tmp_path, "python", "print('done')")
        state = _wait(runner, handle)
        # Allow for filesystems that keep whole-second times.
        assert state.ended_at is not None and before - 2 <= state.ended_at <= time.time() + 1

    def test_exit_code_is_reported(self, tmp_path):
        runner, handle = self._start(tmp_path, "python", "import sys; print('oops'); sys.exit(3)")
        assert _wait(runner, handle).exit_code == 3
        assert "oops" in runner.read_output(handle, max_chars=1000)

    @posix_only
    def test_shell_command(self, tmp_path):
        runner, handle = self._start(tmp_path, "shell", "echo oops >&2; exit 3")
        assert _wait(runner, handle).exit_code == 3
        assert "oops" in runner.read_output(handle, max_chars=1000)

    def test_output_is_tailed(self, tmp_path):
        runner, handle = self._start(tmp_path, "python", "print('a' * 5000 + 'END')")
        _wait(runner, handle)
        text = runner.read_output(handle, max_chars=10)
        assert text.rstrip().endswith("END") and "earlier output omitted" in text

    def test_stop_kills_the_whole_process_tree(self, tmp_path):
        runner, handle = self._start(tmp_path, "python", SLEEP_WITH_CHILD)
        child_pid = int(_wait_for_file(str(tmp_path / "child.pid")))
        assert runner.poll(handle).running
        runner.stop(handle)
        state = _wait(runner, handle, timeout=10)
        assert state.exit_code is None or state.exit_code != 0
        assert _wait_dead(child_pid)

    def test_unknown_kind_is_rejected(self, tmp_path):
        with pytest.raises(BackgroundRunnerError):
            self._start(tmp_path, "ruby", "puts 1")

    def test_start_without_a_cwd(self, tmp_path):
        # LocalSandbox runs each call in a temporary directory it deletes
        # afterwards; on Windows that failed while the detached process sat in
        # it. The work runs where the call ran; the wrapper in its task dir.
        runner = SandboxBackgroundRunner(LocalSandbox(), root_dir=str(tmp_path))
        task = InMemoryBackgroundTaskStore().create("t", kind="python", runner="sandbox")
        handle = runner.start(task, "import os; print(os.getcwd())", max_runtime_s=600)
        assert _wait(runner, handle).exit_code == 0
        output = runner.read_output(handle, max_chars=1000).strip()
        assert os.path.realpath(output) == os.path.realpath(os.getcwd())

    def test_missing_cwd_falls_back_to_the_task_directory(self, tmp_path):
        gone = tmp_path / "gone"
        handle = _sandbox_ops.start(
            root=str(tmp_path / "bg"), name="1-x", kind="python",
            payload="import os; print(os.getcwd())", cwd=str(gone), max_log_bytes=10_000,
            max_runtime_s=600,
        )["handle"]
        deadline = time.time() + 15
        while _sandbox_ops.poll(handle=handle)["running"] and time.time() < deadline:
            time.sleep(0.1)
        assert _sandbox_ops.poll(handle=handle)["exit_code"] == 0
        output = _sandbox_ops.read_output(handle=handle, max_chars=2000)["output"]
        assert "is gone" in output
        assert os.path.realpath(output.strip().splitlines()[-1]) == os.path.realpath(handle)


def _pid_alive(pid) -> bool:
    if sys.platform == "win32":
        import subprocess

        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True).stdout
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_dead(pid, timeout=10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.2)
    return False


# --------------------------------------------------------------------------- #
# In-sandbox operations, called directly
# --------------------------------------------------------------------------- #


class TestSandboxOps:
    def _start(self, tmp_path, kind, payload, max_log_bytes=1_000_000, name=None,
               max_runtime_s=600):
        return _sandbox_ops.start(
            root=str(tmp_path),
            name=name or f"7-{uuid.uuid4().hex[:8]}",
            kind=kind,
            payload=payload,
            cwd=str(tmp_path),
            max_log_bytes=max_log_bytes,
            max_runtime_s=max_runtime_s,
        )["handle"]

    def _wait(self, handle):
        deadline = time.time() + 15
        while time.time() < deadline:
            reply = _sandbox_ops.poll(handle=handle)
            if not reply["running"]:
                return reply
            time.sleep(0.1)
        raise AssertionError("did not finish")

    def test_task_directory_layout(self, tmp_path):
        handle = self._start(tmp_path, "python", "print('hi')")
        self._wait(handle)
        names = set(os.listdir(handle))
        assert {"spec.json", "pid", "output.log", "exit_code", "_sandbox_ops.py"} <= names

    def test_runs_in_the_given_directory(self, tmp_path):
        handle = self._start(tmp_path, "python", "import os; print(os.getcwd())")
        self._wait(handle)
        output = _sandbox_ops.read_output(handle=handle, max_chars=1000)["output"]
        assert os.path.realpath(output.strip()) == os.path.realpath(str(tmp_path))

    def test_large_output_keeps_the_tail(self, tmp_path):
        code = "for i in range(20000):\n    print('line', i)"
        handle = self._start(tmp_path, "python", code, max_log_bytes=4096)
        assert self._wait(handle)["exit_code"] == 0
        assert os.path.getsize(os.path.join(handle, "output.log")) <= 4096
        reply = _sandbox_ops.read_output(handle=handle, max_chars=100)
        assert "line 19999" in reply["output"] and reply["truncated"]

    @posix_only
    def test_missing_command_reports_exit_code(self, tmp_path):
        handle = self._start(tmp_path, "shell", "exec /nonexistent/binary")
        assert self._wait(handle)["exit_code"] != 0

    @posix_only
    def test_lost_process_is_an_error(self, tmp_path):
        handle = self._start(tmp_path, "python", SLEEP)
        pid = int(open(os.path.join(handle, "pid")).read())
        os.killpg(pid, 9)
        # Started straight from this test process, the wrapper is our child:
        # reap it, as the sandbox's init would.
        os.waitpid(pid, 0)
        reply = self._wait(handle)
        assert reply.get("exit_code") is None and "without an exit code" in reply["error"]

    def test_stop_on_finished_task_is_harmless(self, tmp_path):
        handle = self._start(tmp_path, "python", "pass")
        self._wait(handle)
        assert _sandbox_ops.stop(handle=handle) == {}

    def test_a_task_directory_is_never_reused(self, tmp_path):
        first = self._start(tmp_path, "python", "import sys; sys.exit(4)", name="1-same")
        assert self._wait(first)["exit_code"] == 4
        with pytest.raises(FileExistsError):
            self._start(tmp_path, "python", SLEEP, name="1-same")

    def test_stop_after_finish_never_signals(self, tmp_path, monkeypatch):
        handle = self._start(tmp_path, "python", "pass")
        self._wait(handle)
        # Even if the pid now belongs to a live process, a finished task is
        # left alone.
        def fail(*_args, **_kwargs):
            pytest.fail("signalled a finished task")

        monkeypatch.setattr(_sandbox_ops, "_wrapper_alive", lambda pid, task_dir: True)
        monkeypatch.setattr(_sandbox_ops.os, "killpg", fail, raising=False)
        monkeypatch.setattr(_sandbox_ops.subprocess, "run", fail)
        assert _sandbox_ops.stop(handle=handle) == {}

    def test_old_finished_tasks_are_pruned(self, tmp_path):
        old = self._start(tmp_path, "python", "pass")
        self._wait(old)
        long_ago = time.time() - _sandbox_ops.RETENTION_S - 60
        os.utime(old, (long_ago, long_ago))
        running = self._start(tmp_path, "python", SLEEP)
        try:
            assert not os.path.exists(old)
            assert os.path.exists(running)
        finally:
            _sandbox_ops.stop(handle=running, grace_s=0.5)

    def test_unknown_kind(self, tmp_path):
        with pytest.raises(ValueError):
            self._start(tmp_path, "ruby", "puts 1")


class TestBuildCall:
    def test_leaves_no_names_behind(self, tmp_path, capsys):
        # A Jupyter kernel keeps globals between calls: the call must not add
        # or overwrite any.
        namespace = {"start": "user value"}
        code = build_call("poll", handle=str(tmp_path / "missing"))
        exec(code, namespace)
        assert set(namespace) == {"start", "__builtins__"}
        assert namespace["start"] == "user value"
        assert "task files are missing" in capsys.readouterr().out

    def test_rejects_unknown_operations(self):
        with pytest.raises(ValueError):
            build_call("rm_rf", handle="/")

    def test_arguments_are_data_not_code(self, tmp_path, capsys):
        handle = str(tmp_path / "x\"); print('injected')  # %(source)r")
        exec(build_call("poll", handle=handle), {})
        assert "injected" not in capsys.readouterr().out.replace(handle, "")


# --------------------------------------------------------------------------- #
# Agent wiring
# --------------------------------------------------------------------------- #


class _EchoTool:
    @property
    def schema(self):
        return ToolSchema(name="echo", description="", parameters={"type": "object"})

    def run(self, **_kwargs):
        return "echoed"


class _ScriptedLLM:
    model = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    def complete(self, messages, tools=None, **_kwargs):
        self.seen.append(list(messages))
        return self.responses.pop(0)


def _call(name="echo"):
    return LLMResponse(
        message=AssistantMessage(
            content="", tool_calls=[ToolCall(id="c1", name=name, arguments={})]
        ),
        stop_reason="tool_use",
    )


def _final():
    return LLMResponse(message=AssistantMessage(content="done"), stop_reason="end_turn")


class TestAgentNotices:
    def test_notice_rides_on_the_last_tool_result(self):
        pending = [[], ["task 1 finished"]]
        llm = _ScriptedLLM([_call(), _final()])
        agent = BaseAgent(llm, "sys", [_EchoTool()])
        agent.notice_sources.append(lambda: pending.pop(0))
        agent.run("go")
        result = next(m for m in agent.history if isinstance(m, ToolResultMessage))
        assert result.results[0].content.startswith("echoed")
        assert "<notice>\ntask 1 finished\n</notice>" in result.results[0].content

    def test_notice_at_turn_start_goes_in_the_context(self):
        llm = _ScriptedLLM([_final()])
        agent = BaseAgent(llm, "sys", [])
        agent.notice_sources.append(lambda: ["task 2 finished"])
        agent.run("hi")
        user = next(m for m in agent.history if isinstance(m, UserMessage))
        assert "<context>\ntask 2 finished\n</context>" in user.content

    def test_broken_notice_source_does_not_break_the_turn(self):
        def broken():
            raise RuntimeError("db down")

        agent = BaseAgent(_ScriptedLLM([_final()]), "sys", [])
        agent.notice_sources.append(broken)
        assert agent.run("hi").answer == "done"


class _DummyLLM:
    model = "dummy"

    def complete(self, *args, **kwargs):  # pragma: no cover - not called
        raise AssertionError("LLM should not be invoked")


class TestTernoAgentWiring:
    def test_tools_off_by_default(self):
        agent = TernoAgent(_DummyLLM())
        assert "run_in_background" not in agent.tools
        assert agent.tools["bash"].timeout_hint == ""

    def test_tools_and_hints_with_a_service(self):
        service, _, _ = _service()
        agent = TernoAgent(_DummyLLM(), sandbox=LocalSandbox(), background=service)
        for name in ("run_in_background", "background_output", "stop_background_task"):
            assert name in agent.tools
        assert "run_in_background" in agent.tools["bash"].timeout_hint
        assert "run_in_background" in agent.tools["run_python"].timeout_hint
        assert service.take_notices in agent.notice_sources

    @staticmethod
    def _config(monkeypatch, **overrides) -> Config:
        import terno_agent.agents.terno as terno_mod

        monkeypatch.setattr(terno_mod, "create_llm_client", lambda **_: _DummyLLM())
        return Config(
            llm_api_key="k",
            sandbox="none",
            mcp_enabled=False,
            memory_enabled=False,
            file_memory_enabled=False,
            skills_enabled=False,
            background_tasks_enabled=True,
            **overrides,
        )

    def test_config_switch_builds_a_service(self, monkeypatch):
        config = self._config(monkeypatch, background_max_running=2)
        sandbox = LocalSandbox()
        agent = TernoAgent.from_config(config, sandbox=sandbox)
        assert "run_in_background" in agent.tools
        assert agent.background.max_running == 2
        assert agent.background.runners["sandbox"].sandbox is sandbox

    def test_never_falls_back_to_the_host(self, monkeypatch):
        agent = TernoAgent.from_config(self._config(monkeypatch))
        assert agent.background is None
        assert "run_in_background" not in agent.tools
        assert agent.tools["bash"].timeout_hint == ""

    def test_run_in_background_needs_exactly_one_payload(self):
        service, _, _ = _service()
        tool = TernoAgent(_DummyLLM(), background=service).tools["run_in_background"]
        with pytest.raises(ToolError):
            tool.run(description="x")
        with pytest.raises(ToolError):
            tool.run(description="x", code="1", command="ls")

        assert "Started background task 1" in tool.run(description="x", code="print(1)")

    def test_background_output_caps_max_chars(self):
        service, _, _ = _service()
        tool = TernoAgent(_DummyLLM(), background=service).tools["background_output"]
        with pytest.raises(ToolError, match="between"):
            tool.run(task_id="1", max_chars=10**9)


def test_trim_is_skipped_while_the_log_is_busy(tmp_path, monkeypatch):
    log = tmp_path / "output.log"
    log.write_bytes(b"x" * 100)

    def busy(*_args):
        raise PermissionError("in use")

    monkeypatch.setattr(_sandbox_ops.os, "replace", busy)
    _sandbox_ops._keep_tail(str(log), 10)
    assert log.read_bytes() == b"x" * 100
    assert sorted(p.name for p in tmp_path.iterdir()) == ["output.log"]


# --------------------------------------------------------------------------- #
# Review regressions
# --------------------------------------------------------------------------- #


class TestServiceFailures:
    def test_work_that_ended_unpolled_is_not_reported_as_timed_out(self):
        service, runner, clock = _service(max_runtime_s=60)
        task = service.start(kind="python", payload="x", description="quick")
        runner.states[task.handle] = RunnerState(running=False, exit_code=0)
        clock.now += 4 * 3600  # nobody polled for hours
        service.refresh(force=True)
        assert service.store.get(task.id).status == COMPLETED
        assert runner.stopped == []

    def test_runner_reported_time_out(self):
        service, runner, _ = _service()
        task = service.start(kind="python", payload="x", description="slow")
        runner.states[task.handle] = RunnerState(running=False, timed_out=True)
        service.refresh(force=True)
        assert service.store.get(task.id).status == TIMED_OUT

    def test_any_start_error_frees_the_slot(self):
        service, runner, _ = _service(max_running=1)
        runner.start = mock.Mock(side_effect=RuntimeError("sandbox exploded"))
        with pytest.raises(ToolError, match="sandbox exploded"):
            service.start(kind="python", payload="x", description="a")
        runner.start = _FakeRunner().start
        service.start(kind="python", payload="x", description="b")  # slot is free

    def test_any_poll_error_is_transient(self):
        service, runner, _ = _service()
        task = service.start(kind="python", payload="x", description="a")
        runner.poll = mock.Mock(side_effect=RuntimeError("kernel restarting"))
        service.refresh(force=True)
        assert service.store.get(task.id).status == RUNNING
        assert "a" in service.context_block()

    def test_stop_is_recorded_even_if_the_sandbox_fails(self):
        service, runner, _ = _service()
        task = service.start(kind="python", payload="x", description="a")
        runner.stop = mock.Mock(side_effect=RuntimeError("gone"))
        stopped = service.stop(task.id)
        assert (stopped.status, stopped.notified) == (STOPPED, True)

    def test_stop_never_hides_a_finish_that_won(self):
        store = InMemoryBackgroundTaskStore()
        task = store.create("a", kind="python", runner="sandbox")
        store.finish(task.id, COMPLETED, exit_code=0)
        store.update(task.id, status=STOPPED, notified=True)
        got = store.get(task.id)
        assert (got.status, got.notified) == (COMPLETED, False)


class TestRunnerErrors:
    def test_every_sandbox_error_becomes_a_runner_error(self):
        class Broken:
            def run_python(self, code, **kwargs):
                raise TimeoutError("no reply")

        runner = SandboxBackgroundRunner(Broken())
        with pytest.raises(BackgroundRunnerError):
            runner.poll("/x")

    def test_bad_reply_becomes_a_runner_error(self):
        from terno_agent.sandbox.base import ExecutionResult

        class Garbled:
            def run_python(self, code, **kwargs):
                reply = _sandbox_ops.MARKER + "{not json"
                return ExecutionResult(stdout=reply, stderr="", exit_code=0)

        with pytest.raises(BackgroundRunnerError, match="bad reply"):
            SandboxBackgroundRunner(Garbled()).poll("/x")

    def test_failed_start_stops_whatever_did_start(self):
        calls = []

        class LostReply:
            def run_python(self, code, **kwargs):
                calls.append(code)
                raise TimeoutError("reply lost")

        runner = SandboxBackgroundRunner(LostReply(), root_dir="/r")
        task = InMemoryBackgroundTaskStore().create("t", kind="python", runner="sandbox")
        with pytest.raises(BackgroundRunnerError):
            runner.start(task, "print(1)", max_runtime_s=60)
        assert len(calls) == 2 and "'stop'" in calls[1]


@posix_only
class TestProcessLifecycle:
    def _start(self, tmp_path, payload, kind="shell", max_runtime_s=600):
        return _sandbox_ops.start(
            root=str(tmp_path), name=f"t-{uuid.uuid4().hex[:8]}", kind=kind, payload=payload,
            cwd=str(tmp_path), max_log_bytes=100_000, max_runtime_s=max_runtime_s,
        )["handle"]

    def _wait(self, handle, timeout=15.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            reply = _sandbox_ops.poll(handle=handle)
            if not reply["running"]:
                return reply
            time.sleep(0.1)
        raise AssertionError("still running")

    def _group_gone(self, handle, timeout=10.0):
        group = int(open(os.path.join(handle, "pid")).read())
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Started from this test, the wrapper is our child: reap it as
            # init would, or it lingers as a zombie in the group.
            try:
                os.waitpid(group, os.WNOHANG)
            except ChildProcessError:
                pass
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass  # macOS: only zombies left, not yet reaped
            time.sleep(0.1)
        return False

    def test_stop_kills_work_that_ignores_sigterm(self, tmp_path):
        handle = self._start(tmp_path, "trap '' TERM; sleep 30")
        time.sleep(0.5)
        _sandbox_ops.stop(handle=handle, grace_s=0.5)
        assert self._group_gone(handle)

    def test_polite_stop_lets_the_wrapper_record_the_end(self, tmp_path):
        handle = self._start(tmp_path, "sleep 30")
        time.sleep(0.5)
        _sandbox_ops.stop(handle=handle, grace_s=2)
        reply = self._wait(handle)
        assert reply.get("exit_code") not in (None, 0)

    def test_wrapper_enforces_the_time_limit(self, tmp_path):
        handle = self._start(tmp_path, "sleep 30", max_runtime_s=1)
        reply = self._wait(handle, timeout=10)
        assert reply["running"] is False and reply["timed_out"] is True
        assert reply["ended_at"] is not None
        assert self._group_gone(handle)
        assert "time limit" in _sandbox_ops.read_output(handle=handle, max_chars=500)["output"]

    def test_task_ends_when_the_work_exits_even_if_a_child_holds_the_output(self, tmp_path):
        handle = self._start(tmp_path, "(sleep 20 &); echo bye")
        started = time.time()
        assert self._wait(handle, timeout=10)["exit_code"] == 0
        assert time.time() - started < 8
        assert "bye" in _sandbox_ops.read_output(handle=handle, max_chars=500)["output"]

    def test_reused_pid_is_not_taken_for_the_wrapper(self, tmp_path):
        handle = self._start(tmp_path, "sleep 30")
        try:
            if not os.path.isdir("/proc"):
                pytest.skip("needs /proc to tell processes apart")
            # Our own pid is alive but is not this task's wrapper.
            assert not _sandbox_ops._wrapper_alive(os.getpid(), handle)
        finally:
            _sandbox_ops.stop(handle=handle, grace_s=0.5)


class TestSandboxOpsSafety:
    def test_log_write_errors_never_raise(self, tmp_path):
        log = _sandbox_ops._Log(str(tmp_path / "missing" / "output.log"), 1000)
        log.note("hello")  # directory does not exist: the write fails
        assert log.failed

    @pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")
    def test_refuses_a_root_owned_by_someone_else(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_sandbox_ops.os, "getuid", lambda: os.stat(tmp_path).st_uid + 1)
        with pytest.raises(PermissionError):
            _sandbox_ops._prepare_root(str(tmp_path))

    def test_default_root_is_per_user(self):
        root = _sandbox_ops._prepare_root("")
        owner = str(os.getuid()) if hasattr(os, "getuid") else os.environ.get("USERNAME", "user")
        assert root.endswith(f"terno-background-{owner}")

    def test_windows_shell_command_is_one_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_sandbox_ops, "_IS_WINDOWS", True)
        assert _sandbox_ops._command(str(tmp_path), "shell", 'echo "a b"') == 'cmd /c echo "a b"'

    def test_pid_files_cannot_point_at_init(self, tmp_path):
        (tmp_path / "pid").write_text("1")
        assert _sandbox_ops._read_pid(str(tmp_path)) is None


class TestNoticePersistence:
    def test_hosts_see_the_notice_in_the_result_event(self):
        events = []
        pending = [[], ["task 1 finished"]]
        llm = _ScriptedLLM([_call(), _final()])
        agent = BaseAgent(llm, "sys", [_EchoTool()], on_event=events.append)
        agent.notice_sources.append(lambda: pending.pop(0))
        agent.run("go")
        from terno_agent.core.events import ToolResultEvent

        result = next(e for e in events if isinstance(e, ToolResultEvent)).result
        assert "task 1 finished" in result.content


class TestRicherNotices:
    def test_notice_carries_the_end_of_the_output(self):
        service, runner, _ = _service()
        task = service.start(kind="python", payload="x", description="report")
        runner.states[task.handle] = RunnerState(running=False, exit_code=0)
        (notice,) = service.take_notices()
        assert "End of its output:\noutput of h1" in notice
        assert 'background_output(task_id="1")' in notice

    def test_notice_survives_unreadable_output(self):
        service, runner, _ = _service()
        task = service.start(kind="python", payload="x", description="report")
        runner.states[task.handle] = RunnerState(running=False, exit_code=0)
        runner.read_output = mock.Mock(side_effect=RuntimeError("sandbox gone"))
        (notice,) = service.take_notices()
        assert "completed" in notice and "End of its output" not in notice

    def test_output_file_is_named_when_the_runner_has_one(self):
        class FileRunner(_FakeRunner):
            def output_path(self, handle):
                return f"/bg/{handle}/output.log"

        service = BackgroundTasks({"sandbox": FileRunner()}, poll_interval_s=0)
        tool = RunInBackgroundTool(service)
        assert "Its output is written to /bg/h1/output.log." in tool.run(
            description="x", code="print(1)"
        )
        service.runners["sandbox"].states["h1"] = RunnerState(running=False, exit_code=0)
        assert "or open /bg/h1/output.log" in service.take_notices()[0]

    def test_no_file_named_for_runners_without_one(self):
        service, _, _ = _service()
        assert "written to" not in RunInBackgroundTool(service).run(description="x", code="1")

    def test_sandbox_runner_output_path_follows_the_sandbox_style(self):
        runner = SandboxBackgroundRunner(LocalSandbox())
        assert runner.output_path("/tmp/bg/1-ab") == "/tmp/bg/1-ab/output.log"
        assert runner.output_path(r"C:\Temp\bg\1-ab") == r"C:\Temp\bg\1-ab\output.log"
