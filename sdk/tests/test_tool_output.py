"""Oversized tool results: head + tail to the model, full text to a file."""

from __future__ import annotations

from pathlib import Path

from terno_agent.agents.base import BaseAgent
from terno_agent.core.exceptions import ToolError
from terno_agent.core.messages import AssistantMessage, Message, ToolCall, ToolResultMessage
from terno_agent.core.tool import ToolSchema
from terno_agent.core.tool_output import ToolOutputLimiter, make_output_saver
from terno_agent.llm.base import LLMResponse
from terno_agent.sandbox.base import ExecutionResult
from terno_agent.tools.code_exec import RunPythonTool
from terno_agent.tools.shell import BashTool


def _limiter(saved: dict, **kw) -> ToolOutputLimiter:
    def save(name: str, text: str) -> str:
        path = f"/out/{name}"
        saved[path] = text
        return path

    return ToolOutputLimiter.build(save, output_dir="/out", **kw)


# ----- limiter ---------------------------------------------------------------- #


def test_small_output_passes_through_unsaved():
    saved: dict = {}
    assert _limiter(saved).apply("grep", {}, "hello") == "hello"
    assert saved == {}


def test_large_output_keeps_head_tail_and_saves_full_text():
    saved: dict = {}
    text = "HEAD" + "x" * 20000 + "TAIL"
    out = _limiter(saved).apply("mcp__ga__report", {"q": 1}, text)

    (path, content), = saved.items()
    assert content == text
    assert path.endswith("-mcp__ga__report.txt")
    assert out.startswith("HEAD") and "TAIL" in out and path in out
    assert len(out) < 4000


def test_failed_save_still_shortens():
    def save(name, text):
        raise OSError("disk full")

    out = ToolOutputLimiter.build(save).apply("web_fetch", {}, "y" * 20000)
    assert len(out) < 4000
    assert "could not be saved" in out


def test_read_file_is_shortened_but_not_saved_again():
    saved: dict = {}
    out = _limiter(saved).apply("read_file", {"path": "/big.csv"}, "z" * 40000)
    assert saved == {}
    assert "offset/limit" in out
    assert len(out) < 13000


def test_reading_saved_output_back_through_code_is_not_saved_again():
    saved: dict = {}
    out = _limiter(saved).apply(
        "run_python", {"code": "print(open('/out/abc.txt').read())"}, "z" * 40000
    )
    assert saved == {}
    assert "smaller range" in out


def test_exempt_tools_are_untouched():
    saved: dict = {}
    body = "s" * 20000
    assert _limiter(saved).apply("activate_skill", {"name": "x"}, body) == body
    assert saved == {}


def test_install_noise_gets_tight_budget():
    saved: dict = {}
    out = _limiter(saved).apply("bash", {"command": "pip install pandas"}, "Collecting a\n" * 200)
    assert len(saved) == 1
    assert len(out) < 1200


def test_large_patterns_get_larger_budget():
    saved: dict = {}
    limiter = _limiter(saved, large_patterns=[r"\blist_tables\s*\("])
    text = "t" * 9000
    assert limiter.apply("run_python", {"code": "print(list_tables(1))"}, text) == text
    assert limiter.apply("run_python", {"code": "print(other())"}, text) != text


# ----- savers ----------------------------------------------------------------- #


class _FileOpSandbox:
    def __init__(self, ok=True):
        self.ok = ok
        self.writes = []

    def write_file(self, path, content, *, overwrite=False):
        self.writes.append((path, content, overwrite))
        return {"ok": True, "output": "w"} if self.ok else {"ok": False, "error": "quota"}


def test_sandbox_dir_is_written_through_sandbox(tmp_path: Path):
    box = _FileOpSandbox()
    save, resolved = make_output_saver("/home/runner/ws/logs", workdir=tmp_path, sandbox=box)
    assert resolved == "/home/runner/ws/logs"
    assert save("a.txt", "data") == "/home/runner/ws/logs/a.txt"
    assert box.writes == [("/home/runner/ws/logs/a.txt", "data", True)]


def test_sandbox_write_failure_returns_none(tmp_path: Path):
    save, _ = make_output_saver("/home/runner/ws/logs", workdir=tmp_path, sandbox=_FileOpSandbox(ok=False))
    assert save("a.txt", "data") is None


def test_relative_dir_is_written_on_host_under_workdir(tmp_path: Path):
    save, resolved = make_output_saver("", workdir=tmp_path, sandbox=_FileOpSandbox())
    path = save("a.txt", "data")
    assert Path(path).read_text() == "data"
    assert Path(resolved) == tmp_path / ".terno" / "tool_outputs"


# ----- agent loop ------------------------------------------------------------- #


class _BigTool:
    def __init__(self, text: str, error: bool = False):
        self.text = text
        self.error = error

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(name="big", description="", parameters={"type": "object"})

    def run(self, **_kw) -> str:
        if self.error:
            raise ToolError(self.text)
        return self.text


class _LLM:
    model = "scripted"

    def __init__(self):
        self.calls = 0

    def complete(self, messages: list[Message], tools=None, **_kw) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            msg = AssistantMessage(content="", tool_calls=[ToolCall(id="c1", name="big", arguments={})])
            return LLMResponse(message=msg, stop_reason="tool_use")
        return LLMResponse(message=AssistantMessage(content="done"), stop_reason="end_turn")


def _tool_result(agent: BaseAgent):
    return next(m for m in agent.history if isinstance(m, ToolResultMessage)).results[0]


def test_agent_loop_shortens_tool_results_in_history():
    saved: dict = {}
    agent = BaseAgent(_LLM(), "sys", [_BigTool("r" * 30000)], tool_output_limiter=_limiter(saved))
    agent.run("go")
    result = _tool_result(agent)
    assert len(result.content) < 4000
    assert list(saved.values()) == ["r" * 30000]


def test_agent_loop_shortens_tool_errors_too():
    saved: dict = {}
    agent = BaseAgent(
        _LLM(), "sys", [_BigTool("e" * 30000, error=True)], tool_output_limiter=_limiter(saved)
    )
    agent.run("go")
    result = _tool_result(agent)
    assert result.is_error
    assert len(result.content) < 4000


def test_agent_without_limiter_is_unchanged():
    agent = BaseAgent(_LLM(), "sys", [_BigTool("r" * 30000)])
    agent.run("go")
    assert _tool_result(agent).content == "r" * 30000


# ----- code tools hand the limiter the whole output -------------------------- #


class _Box:
    def run_python(self, code, *, timeout_s=30):
        return ExecutionResult(stdout="a" * 20000 + "END", stderr="", exit_code=0)

    def run_shell(self, command, *, timeout_s=30):
        return ExecutionResult(stdout="b" * 100000 + "END", stderr="", exit_code=0)


def test_run_python_cap_is_configurable():
    assert "END" not in RunPythonTool(_Box()).run(code="x")
    assert "END" in RunPythonTool(_Box(), max_output_chars=50000).run(code="x")


def test_bash_cap_is_configurable(tmp_path: Path):
    out = BashTool(workdir=tmp_path, sandbox=_Box(), max_output_chars=200000).run(command="x")
    assert "truncated" not in out and out.endswith("END")
