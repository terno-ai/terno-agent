"""Background shell tasks, TaskStop and TaskOutput."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from terno_agent.core.background import BackgroundTaskRegistry
from terno_agent.core.exceptions import ToolError
from terno_agent.tools.background_tasks import TaskOutputTool, TaskStopTool
from terno_agent.tools.shell import BashTool


def _wait_for(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ----- the registry -------------------------------------------------------- #


def test_output_streams_to_a_file_while_the_task_runs(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("echo first; sleep 5; echo second")

    # Unbuffered, so a read mid-flight sees partial output rather than nothing.
    assert _wait_for(lambda: "first" in task.read_output())
    assert task.status == "running"
    assert "second" not in task.read_output()
    task.stop()


def test_completed_task_reports_status_and_exit_code(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("echo hi")

    assert task.wait(5.0)
    assert task.status == "completed"
    assert task.exit_code == 0
    assert "hi" in task.read_output()


def test_failing_task_is_reported_as_failed(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("echo boom >&2; exit 3")

    assert task.wait(5.0)
    assert task.status == "failed"
    assert task.exit_code == 3
    # stderr is merged into the same file.
    assert "boom" in task.read_output()


def test_stopping_kills_the_whole_process_group(tmp_path: Path) -> None:
    # A bare `proc.kill()` would leave the `sleep` orphaned behind `sh`.
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("sleep 30 & sleep 30; echo done")

    assert task.stop() is True
    assert task.status == "stopped"
    assert "done" not in task.read_output()


def test_stopping_a_finished_task_is_a_no_op(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("true")
    task.wait(5.0)

    assert task.stop() is False
    assert task.status == "completed"


def test_output_survives_the_task(tmp_path: Path) -> None:
    # The file must outlive the process, or Read after completion finds nothing.
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("echo persisted")
    task.wait(5.0)

    assert task.output_path.exists()
    assert "persisted" in task.output_path.read_text()


def test_ids_are_unique_and_stop_all_reports_a_count(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    a = reg.launch_shell("sleep 30")
    b = reg.launch_shell("sleep 30")

    assert a.id != b.id
    assert len(reg.running()) == 2
    assert reg.stop_all() == 2
    assert reg.running() == []


def test_a_command_with_unbalanced_quotes_still_gets_a_label(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("echo 'unclosed")
    task.wait(5.0)

    assert task.description  # shlex would raise; the fallback splits naively


# ----- Bash(run_in_background) --------------------------------------------- #


def test_bash_background_returns_the_output_path(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    out = BashTool(workdir=tmp_path, background=reg).run(
        command="echo hello", run_in_background=True
    )

    assert "output file:" in out
    assert "TaskStop" in out
    task = reg.list()[0]
    assert task.wait(5.0)
    assert "hello" in task.read_output()


def test_bash_background_errors_when_unsupported(tmp_path: Path) -> None:
    # Better a clear error than silently running in the foreground, which would
    # block the turn on a command the model expected to detach.
    with pytest.raises(ToolError, match="not available"):
        BashTool(workdir=tmp_path).run(command="echo hi", run_in_background=True)


def test_bash_foreground_is_unaffected(tmp_path: Path) -> None:
    out = BashTool(workdir=tmp_path, background=BackgroundTaskRegistry(tmp_path)).run(
        command="echo sync"
    )
    assert "sync" in out


# ----- the tools ----------------------------------------------------------- #


def test_task_stop_stops_a_running_task(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("sleep 30")

    out = TaskStopTool(registry=reg).run(task_id=task.id)

    assert "Stopped" in out
    assert task.status == "stopped"


def test_task_stop_accepts_the_deprecated_shell_id(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("sleep 30")

    assert "Stopped" in TaskStopTool(registry=reg).run(shell_id=task.id)


def test_task_stop_on_a_finished_task_is_not_an_error(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("true")
    task.wait(5.0)

    # The model asked for it to be over, and it is.
    assert "already completed" in TaskStopTool(registry=reg).run(task_id=task.id)


def test_unknown_task_id_lists_the_known_ones(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("true")

    with pytest.raises(ToolError, match=f"Known tasks: {task.id}"):
        TaskStopTool(registry=reg).run(task_id="nope")


def test_missing_task_id_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="task_id"):
        TaskStopTool(registry=BackgroundTaskRegistry(tmp_path)).run()


def test_task_output_blocks_until_completion(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("sleep 0.2; echo late")

    out = TaskOutputTool(registry=reg).run(task_id=task.id, block=True, timeout=5000)

    assert "completed" in out
    assert "late" in out


def test_task_output_non_blocking_returns_immediately(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("sleep 30")

    started = time.monotonic()
    out = TaskOutputTool(registry=reg).run(task_id=task.id, block=False, timeout=5000)

    assert time.monotonic() - started < 1.0
    assert "running" in out
    task.stop()


def test_task_output_reports_a_timeout_without_failing(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("sleep 30")

    out = TaskOutputTool(registry=reg).run(task_id=task.id, block=True, timeout=100)

    # Still running: a timeout is a status report, not an error.
    assert "running" in out
    task.stop()


def test_task_output_always_names_the_file(tmp_path: Path) -> None:
    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("true")

    out = TaskOutputTool(registry=reg).run(task_id=task.id, block=True, timeout=5000)
    assert str(task.output_path) in out


def test_task_output_advertises_its_own_deprecation() -> None:
    # The notice is what steers the model to Read the file instead; dropping it
    # would make this tool more attractive than intended.
    schema = TaskOutputTool(registry=BackgroundTaskRegistry(Path("/tmp"))).schema
    assert schema.description.startswith("DEPRECATED:")
    assert "prefer using the Read tool" in schema.description


# ----- agent integration --------------------------------------------------- #


def test_agent_registers_the_background_tools() -> None:
    from terno_agent.agents.terno import TernoAgent
    from terno_agent.core.messages import AssistantMessage
    from terno_agent.llm.base import LLMResponse

    class _LLM:
        model = "dummy"

        def complete(self, *_a, **_k):
            return LLMResponse(message=AssistantMessage(content="x"), stop_reason="stop")

    agent = TernoAgent(llm=_LLM())
    # Deferred, like the reference harness.
    assert {"TaskStop", "TaskOutput"} <= set(agent.tool_registry.names)
    assert agent.background_tasks is not None

    eager = TernoAgent(llm=_LLM(), defer_tools=())
    assert "TaskStop" in eager.tools
    assert "TaskOutput" in eager.tools


def test_closing_the_sdk_stops_running_background_tasks(tmp_path: Path) -> None:
    """Detached process groups would otherwise outlive the session."""
    from terno_agent.sdk import Agent

    reg = BackgroundTaskRegistry(tmp_path)
    task = reg.launch_shell("sleep 30")
    assert task.status == "running"

    # `Agent.__init__` builds a whole stack (config, sandbox, MCP); this exercises
    # the close path alone against a stand-in that only has what close() touches.
    sdk = Agent.__new__(Agent)
    sdk._closed = False
    sdk._agent = type("_Stub", (), {"background_tasks": reg, "sandbox": None})()
    sdk.close()

    assert task.status == "stopped"
