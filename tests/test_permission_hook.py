from __future__ import annotations

import io

import pytest
from rich.console import Console

from terno_agent.agents.terno import TernoAgent
from terno_agent.core.hooks import HookEvent, PreToolUseContext
from terno_agent.core.messages import AssistantMessage, ToolCall
from terno_agent.core.tool import ToolSchema
from terno_agent.llm.base import LLMResponse

# --------------------------------------------------------------------------- #
# PreToolUseContext API
# --------------------------------------------------------------------------- #


def test_context_default_decision_is_allow() -> None:
    ctx = PreToolUseContext(
        agent=None,  # type: ignore[arg-type]
        tool_call=ToolCall(id="t1", name="bash", arguments={"command": "ls"}),
        tool=None,  # type: ignore[arg-type]
    )
    assert ctx.decision == "allow"
    assert ctx.feedback == ""


def test_context_deny_sets_decision_and_feedback() -> None:
    ctx = PreToolUseContext(
        agent=None,  # type: ignore[arg-type]
        tool_call=ToolCall(id="t1", name="bash", arguments={}),
        tool=None,  # type: ignore[arg-type]
    )
    ctx.deny("dangerous")
    assert ctx.decision == "deny"
    assert ctx.feedback == "dangerous"


def test_context_deny_falls_back_to_generic_reason() -> None:
    ctx = PreToolUseContext(
        agent=None,  # type: ignore[arg-type]
        tool_call=ToolCall(id="t1", name="bash", arguments={}),
        tool=None,  # type: ignore[arg-type]
    )
    ctx.deny("   ")
    assert ctx.decision == "deny"
    assert ctx.feedback == "Tool call denied by the user."


# --------------------------------------------------------------------------- #
# BaseAgent: pre_tool_use dispatch + deny short-circuits the tool
# --------------------------------------------------------------------------- #


class _EchoTool:
    """Records that it was called and returns a fixed string."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="echo",
            description="echo",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def run(self, **_kwargs: object) -> str:
        self.calls += 1
        return "ran"


class _ScriptedLLM:
    model = "scripted"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    def complete(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
        if not self._responses:
            return LLMResponse(message=AssistantMessage(content="done"), stop_reason="stop")
        return self._responses.pop(0)


def _two_turn_echo_then_done() -> list[LLMResponse]:
    return [
        LLMResponse(
            message=AssistantMessage(
                content="",
                tool_calls=[ToolCall(id="t1", name="echo", arguments={})],
            ),
            stop_reason="tool_use",
        ),
        LLMResponse(
            message=AssistantMessage(content="done"),
            stop_reason="stop",
        ),
    ]


def test_permission_hook_fires_before_tool_runs() -> None:
    tool = _EchoTool()
    seen: list[PreToolUseContext] = []

    def hook(ctx: PreToolUseContext) -> None:
        seen.append(ctx)
        ctx.allow()

    agent = TernoAgent(_ScriptedLLM(_two_turn_echo_then_done()), permission_hook=hook)
    agent.tools = {tool.schema.name: tool}
    agent.run("hi")

    assert tool.calls == 1
    assert len(seen) == 1
    assert seen[0].tool_call.name == "echo"


def test_deny_short_circuits_tool_and_surfaces_feedback() -> None:
    tool = _EchoTool()

    def hook(ctx: PreToolUseContext) -> None:
        ctx.deny("not allowed in tests")

    agent = TernoAgent(_ScriptedLLM(_two_turn_echo_then_done()), permission_hook=hook)
    agent.tools = {tool.schema.name: tool}
    result = agent.run("hi")

    assert tool.calls == 0
    # The denial becomes a tool result with is_error=True in the agent's trace.
    tool_results = [
        r
        for msg in result.trace
        for r in (getattr(msg, "results", []) or [])
    ]
    assert len(tool_results) == 1
    assert tool_results[0].is_error is True
    assert "not allowed" in tool_results[0].content


def test_no_hook_runs_tool_unimpeded() -> None:
    tool = _EchoTool()
    agent = TernoAgent(_ScriptedLLM(_two_turn_echo_then_done()))
    agent.tools = {tool.schema.name: tool}
    agent.run("hi")
    assert tool.calls == 1


def test_permission_hook_registered_for_pre_tool_use_event() -> None:
    def hook(_ctx: PreToolUseContext) -> None:
        pass

    agent = TernoAgent(_ScriptedLLM([]), permission_hook=hook)
    assert agent.hooks.has(HookEvent.PRE_TOOL_USE)


# --------------------------------------------------------------------------- #
# CliPermissionPrompter
# --------------------------------------------------------------------------- #


def _silent_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=80)


def _make_ctx(name: str = "bash", args: dict | None = None) -> PreToolUseContext:
    return PreToolUseContext(
        agent=None,  # type: ignore[arg-type]
        tool_call=ToolCall(id="t1", name=name, arguments=args or {"command": "ls"}),
        tool=None,  # type: ignore[arg-type]
    )


def test_cli_prompter_allow_once(monkeypatch) -> None:
    from terno_agent.cli import CliPermissionPrompter

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["1"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    prompter = CliPermissionPrompter(_silent_console())
    ctx = _make_ctx()
    prompter(ctx)
    assert ctx.decision == "allow"
    assert "bash" not in prompter.session_allowed


def test_cli_prompter_allow_for_session_remembers_tool(monkeypatch) -> None:
    from terno_agent.cli import CliPermissionPrompter

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    prompter = CliPermissionPrompter(_silent_console())
    ctx1 = _make_ctx()
    prompter(ctx1)
    assert ctx1.decision == "allow"
    assert "bash" in prompter.session_allowed

    # A second call to the same tool must NOT prompt — input iterator is empty,
    # so a prompt would StopIteration.
    ctx2 = _make_ctx()
    prompter(ctx2)
    assert ctx2.decision == "allow"


def test_cli_prompter_deny_with_reason(monkeypatch) -> None:
    from terno_agent.cli import CliPermissionPrompter

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["3", "use uv instead"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    prompter = CliPermissionPrompter(_silent_console())
    ctx = _make_ctx(name="bash", args={"command": "pip install foo"})
    prompter(ctx)
    assert ctx.decision == "deny"
    assert ctx.feedback == "use uv instead"


def test_cli_prompter_reprompts_on_bad_choice(monkeypatch) -> None:
    from terno_agent.cli import CliPermissionPrompter

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["banana", "9", "1"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    prompter = CliPermissionPrompter(_silent_console())
    ctx = _make_ctx()
    prompter(ctx)
    assert ctx.decision == "allow"


def test_cli_prompter_skips_always_allowed_tools(monkeypatch) -> None:
    from terno_agent.cli import CliPermissionPrompter

    # No isatty/input mocks needed — read-only tools must not prompt at all.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    prompter = CliPermissionPrompter(_silent_console())
    for safe in ("read_file", "task_list", "search_memory", "ask_user"):
        ctx = _make_ctx(name=safe, args={})
        prompter(ctx)
        assert ctx.decision == "allow"


def test_cli_prompter_allows_when_no_tty(monkeypatch) -> None:
    from terno_agent.cli import CliPermissionPrompter

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    def boom(_prompt: str = "") -> str:
        raise AssertionError("input() must not be called when stdin isn't a TTY")

    monkeypatch.setattr("builtins.input", boom)
    prompter = CliPermissionPrompter(_silent_console())
    ctx = _make_ctx()
    prompter(ctx)
    assert ctx.decision == "allow"


def test_cli_prompter_defaults_to_deny_on_eof(monkeypatch) -> None:
    from terno_agent.cli import CliPermissionPrompter

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def boom(_prompt: str = "") -> str:
        raise EOFError()

    monkeypatch.setattr("builtins.input", boom)
    prompter = CliPermissionPrompter(_silent_console())
    ctx = _make_ctx()
    prompter(ctx)
    assert ctx.decision == "deny"
    assert ctx.feedback == "Denied by user."


@pytest.fixture(autouse=True)
def _fast_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable persistent memory side-effects from the bare TernoAgent in tests."""
    # Tests construct TernoAgent directly (no from_config), so the extractor is
    # never wired in. Nothing to disable here — fixture is a placeholder for
    # future expansion. Kept so additional teardown can be added in one spot.
