"""Hook framework, persistent history, usage tracking, compaction."""

from __future__ import annotations

from terno_agent.agents.base import BaseAgent
from terno_agent.core.compaction import CompactionHook
from terno_agent.core.hooks import HookContext, HookEvent, HookManager, UsageMeter
from terno_agent.core.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from terno_agent.llm.base import LLMResponse


class _ScriptedLLM:
    """LLM stub that returns pre-scripted responses on each call."""

    model = "scripted"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.seen_message_counts: list[int] = []

    def complete(self, messages: list[Message], tools=None, **_kwargs) -> LLMResponse:
        self.seen_message_counts.append(len(messages))
        if not self._responses:
            return LLMResponse(message=AssistantMessage(content="done"), stop_reason="end_turn")
        return self._responses.pop(0)


def _final(answer: str, *, inp: int = 100, out: int = 10) -> LLMResponse:
    return LLMResponse(
        message=AssistantMessage(content=answer),
        stop_reason="end_turn",
        input_tokens=inp,
        output_tokens=out,
    )


# --------------------------------------------------------------------------- #
# HookManager
# --------------------------------------------------------------------------- #


def test_hook_manager_dispatch_in_order():
    mgr = HookManager()
    seen: list[str] = []
    mgr.register("e", lambda _ctx: seen.append("a"))
    mgr.register("e", lambda _ctx: seen.append("b"))
    ctx = HookContext(event="e", agent=None, history=[], usage=UsageMeter())  # type: ignore[arg-type]
    mgr.dispatch("e", ctx)
    assert seen == ["a", "b"]


def test_hook_manager_swallows_errors():
    mgr = HookManager()
    seen: list[str] = []

    def bad(_ctx):
        raise RuntimeError("boom")

    mgr.register("e", bad)
    mgr.register("e", lambda _ctx: seen.append("after"))
    ctx = HookContext(event="e", agent=None, history=[], usage=UsageMeter())  # type: ignore[arg-type]
    mgr.dispatch("e", ctx)
    # Second hook still ran despite first raising.
    assert seen == ["after"]


def test_hook_manager_unregister():
    mgr = HookManager()
    fn = lambda _ctx: None  # noqa: E731
    mgr.register("e", fn)
    assert mgr.has("e")
    assert mgr.unregister("e", fn) is True
    assert not mgr.has("e")


# --------------------------------------------------------------------------- #
# UsageMeter
# --------------------------------------------------------------------------- #


def test_usage_meter_aggregates_and_tracks_last():
    u = UsageMeter()
    u.record(_final("a", inp=10, out=2))
    u.record(_final("b", inp=20, out=3))
    assert u.last_input_tokens == 20
    assert u.last_output_tokens == 3
    assert u.total_input_tokens == 30
    assert u.total_output_tokens == 5
    assert u.llm_calls == 2


# --------------------------------------------------------------------------- #
# BaseAgent persistent history + chat_end hook
# --------------------------------------------------------------------------- #


def test_history_persists_across_run_calls():
    llm = _ScriptedLLM([_final("first"), _final("second")])
    agent = BaseAgent(llm, system_prompt="sys")
    agent.run("hello")
    # system + user + assistant = 3
    assert len(agent.history) == 3
    agent.run("again")
    # +2 (user + assistant) = 5
    assert len(agent.history) == 5
    # System message must still be at index 0 and unchanged.
    assert isinstance(agent.history[0], SystemMessage)
    assert agent.history[0].content == "sys"
    # Second LLM call must have seen the accumulated context.
    assert llm.seen_message_counts == [2, 4]


def test_clear_history_resets_history_and_usage():
    llm = _ScriptedLLM([_final("first", inp=50)])
    agent = BaseAgent(llm, system_prompt="sys")
    agent.run("hi")
    assert agent.usage.total_input_tokens == 50
    agent.clear_history()
    assert len(agent.history) == 1
    assert agent.usage.total_input_tokens == 0
    assert agent.usage.llm_calls == 0


def test_chat_end_hook_fires_with_correct_context():
    llm = _ScriptedLLM([_final("ok", inp=42, out=7)])
    agent = BaseAgent(llm, system_prompt="sys")
    seen: list[HookContext] = []
    agent.add_hook(HookEvent.CHAT_END, seen.append)
    run = agent.run("ping")
    assert len(seen) == 1
    ctx = seen[0]
    assert ctx.event == HookEvent.CHAT_END
    assert ctx.agent is agent
    assert ctx.history is agent.history
    assert ctx.usage.last_input_tokens == 42
    assert ctx.run is run


def test_extra_context_lands_in_user_message_not_system():
    llm = _ScriptedLLM([_final("ok")])
    agent = BaseAgent(llm, system_prompt="sys")
    agent.run("task", extra_context="hint")
    # System prompt is untouched.
    assert isinstance(agent.history[0], SystemMessage)
    assert agent.history[0].content == "sys"
    # User message now carries the hint inline.
    assert isinstance(agent.history[1], UserMessage)
    assert "hint" in agent.history[1].content
    assert "task" in agent.history[1].content


# --------------------------------------------------------------------------- #
# CompactionHook
# --------------------------------------------------------------------------- #


def _build_history(num_turns: int) -> list[Message]:
    history: list[Message] = [SystemMessage("sys")]
    for i in range(num_turns):
        history.append(UserMessage(f"q{i}"))
        history.append(AssistantMessage(content=f"a{i}"))
    return history


def test_compaction_noop_below_threshold():
    summarizer = _ScriptedLLM([_final("SUMMARY")])
    hook = CompactionHook(llm=summarizer, threshold_input_tokens=1_000_000)
    history = _build_history(5)
    original = list(history)
    ctx = HookContext(
        event=HookEvent.CHAT_END,
        agent=None,  # type: ignore[arg-type]
        history=history,
        usage=UsageMeter(last_input_tokens=100),
    )
    hook(ctx)
    assert ctx.history == original
    # Summarizer must not have been called.
    assert summarizer.seen_message_counts == []


def test_compaction_summarizes_above_threshold():
    summarizer = _ScriptedLLM([_final("SUMMARY OF OLD TURNS")])
    hook = CompactionHook(
        llm=summarizer,
        threshold_input_tokens=1000,
        keep_last_turns=2,
    )
    history = _build_history(6)  # 6 user turns; keep last 2 verbatim
    ctx = HookContext(
        event=HookEvent.CHAT_END,
        agent=None,  # type: ignore[arg-type]
        history=history,
        usage=UsageMeter(last_input_tokens=5000),
    )
    hook(ctx)
    # Expect: system + summary + 2 user turns × (user + assistant) = 6 messages.
    assert len(ctx.history) == 6
    assert isinstance(ctx.history[0], SystemMessage)
    assert ctx.history[0].content == "sys"
    assert isinstance(ctx.history[1], SystemMessage)
    assert "SUMMARY OF OLD TURNS" in ctx.history[1].content
    # Last two user turns must be kept verbatim.
    assert isinstance(ctx.history[2], UserMessage)
    assert ctx.history[2].content == "q4"
    assert isinstance(ctx.history[4], UserMessage)
    assert ctx.history[4].content == "q5"


def test_compaction_safe_when_summarizer_fails():
    class _BoomLLM:
        model = "boom"

        def complete(self, *_a, **_kw):
            raise RuntimeError("network down")

    hook = CompactionHook(llm=_BoomLLM(), threshold_input_tokens=10)
    history = _build_history(6)
    snapshot = list(history)
    ctx = HookContext(
        event=HookEvent.CHAT_END,
        agent=None,  # type: ignore[arg-type]
        history=history,
        usage=UsageMeter(last_input_tokens=10_000),
    )
    hook(ctx)
    # Summarization failed — history is left untouched, no exception escapes.
    assert ctx.history == snapshot


def test_compaction_keeps_tool_result_pairing():
    """A kept assistant message with tool calls keeps its following tool result."""
    summarizer = _ScriptedLLM([_final("SUM")])
    hook = CompactionHook(
        llm=summarizer,
        threshold_input_tokens=1,
        keep_last_turns=1,
    )
    history: list[Message] = [
        SystemMessage("sys"),
        UserMessage("q0"),
        AssistantMessage(content="a0"),
        UserMessage("q1"),
        AssistantMessage(content="", tool_calls=[]),
        ToolResultMessage(results=[ToolResult(call_id="x", content="ok")]),
        AssistantMessage(content="a1-final"),
    ]
    ctx = HookContext(
        event=HookEvent.CHAT_END,
        agent=None,  # type: ignore[arg-type]
        history=history,
        usage=UsageMeter(last_input_tokens=5000),
    )
    hook(ctx)
    # System + summary + (q1 onwards) = 2 + 5 = 7? But q1 onwards has 5 items above.
    # Actually: tail starts at last UserMessage (q1), so 4 messages in tail.
    # Final = system + summary + tail(4) = 6.
    assert len(ctx.history) == 6
    assert isinstance(ctx.history[2], UserMessage)
    assert ctx.history[2].content == "q1"
    # ToolResultMessage must be preserved in the tail.
    assert any(isinstance(m, ToolResultMessage) for m in ctx.history)
