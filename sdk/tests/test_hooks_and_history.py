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
    # The summary returns as a USER message, matching the reference harness — as
    # a system message it would read as a standing instruction.
    assert isinstance(ctx.history[1], UserMessage)
    assert "SUMMARY OF OLD TURNS" in ctx.history[1].content
    assert ctx.history[1].content.startswith("This session is being continued")
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


# ----- compaction prompt fidelity ----------------------------------------- #


def test_extract_summary_keeps_only_the_summary_block():
    from terno_agent.prompts.compaction import extract_summary

    response = (
        "<analysis>\nthinking out loud, should be dropped\n</analysis>\n"
        "<summary>\n1. Primary Request and Intent:\n   Do the thing.\n</summary>"
    )
    out = extract_summary(response)

    assert out.startswith("1. Primary Request and Intent:")
    assert "thinking out loud" not in out


def test_extract_summary_falls_back_to_the_whole_response():
    from terno_agent.prompts.compaction import extract_summary

    # A model that ignores the format should still yield something usable.
    assert extract_summary("  just prose, no tags  ") == "just prose, no tags"


def test_extract_summary_tolerates_a_missing_closing_tag():
    from terno_agent.prompts.compaction import extract_summary

    assert extract_summary("<summary>\ntruncated mid-stream") == "truncated mid-stream"


def test_wrap_summary_brackets_the_summary_for_replay():
    from terno_agent.prompts.compaction import wrap_summary

    out = wrap_summary("1. Primary Request and Intent:\n   Do the thing.")

    assert out.startswith("This session is being continued from a previous")
    assert "1. Primary Request and Intent:" in out
    # The "resume without acknowledging" tail is what stops a visible
    # get-back-up-to-speed turn after compaction.
    assert "do not acknowledge the summary" in out


def test_summary_request_forbids_tool_use_and_asks_for_nine_sections():
    from terno_agent.prompts.compaction import SUMMARY_REQUEST

    assert SUMMARY_REQUEST.startswith("CRITICAL: Respond with TEXT ONLY.")
    for section in (
        "1. Primary Request and Intent",
        "6. All user messages",
        "8. Current Work",
        "9. Optional Next Step",
    ):
        assert section in SUMMARY_REQUEST
    # Terno has no Grep/Glob; the tool list must name tools that exist here.
    assert "Grep" not in SUMMARY_REQUEST
    assert "Glob" not in SUMMARY_REQUEST


def test_compaction_asks_in_conversation_without_tools():
    """The turns being compacted are replayed, not flattened into a transcript."""
    from terno_agent.core.compaction import CompactionHook
    from terno_agent.prompts.compaction import SUMMARY_REQUEST

    seen = {}

    class _LLM:
        model = "dummy"

        def complete(self, messages, tools=None, **kwargs):
            seen["messages"] = messages
            seen["tools"] = tools
            seen["max_tokens"] = kwargs.get("max_tokens")
            seen["temperature"] = kwargs.get("temperature")
            return LLMResponse(
                message=AssistantMessage(content="<summary>ok</summary>"),
                stop_reason="stop",
            )

    hook = CompactionHook(llm=_LLM(), threshold_input_tokens=1, keep_last_turns=2)
    ctx = HookContext(
        event=HookEvent.CHAT_END,
        agent=None,  # type: ignore[arg-type]
        history=_build_history(6),
        usage=UsageMeter(last_input_tokens=5000),
    )
    hook(ctx)

    msgs = seen["messages"]
    assert seen["tools"] is None  # a tool call must be impossible, not just banned
    assert seen["temperature"] is None  # no sampling params, per the capture
    assert seen["max_tokens"] >= 8192  # the structured summary needs headroom
    # System prompt first, replayed turns next, request appended last.
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[-1], UserMessage)
    assert msgs[-1].content == SUMMARY_REQUEST
    assert len(msgs) > 2


def test_summary_request_is_byte_exact_apart_from_the_tool_list():
    """The only permitted deviation is the CRITICAL block's tool list."""
    import difflib
    from pathlib import Path

    from terno_agent.prompts.compaction import SUMMARY_REQUEST

    captured = Path("/Users/navin/terno/cc-corpus/replay/compaction-request.md")
    if not captured.exists():
        return  # corpus not present on this machine

    changed = [
        line[1:]
        for line in difflib.unified_diff(
            captured.read_text().splitlines(), SUMMARY_REQUEST.splitlines(), n=0
        )
        if line[:1] in "+-" and line[:3] not in ("---", "+++")
    ]
    assert len(changed) == 2, f"unexpected deviations: {changed}"
    assert all("or ANY other tool." in line for line in changed)


def test_extra_instructions_are_appended_under_the_expected_header():
    from terno_agent.prompts.compaction import (
        INSTRUCTIONS_HEADER,
        SUMMARY_REQUEST,
        with_instructions,
    )

    # No instructions -> the prompt is untouched.
    assert with_instructions("") is SUMMARY_REQUEST
    assert with_instructions(None) is SUMMARY_REQUEST
    assert with_instructions("   ") is SUMMARY_REQUEST

    out = with_instructions("Focus on SQL changes.")
    assert out.startswith(SUMMARY_REQUEST)
    assert out.endswith(f"{INSTRUCTIONS_HEADER}\nFocus on SQL changes.")


def test_hook_forwards_extra_instructions_to_the_prompt():
    from terno_agent.core.compaction import CompactionHook

    seen = {}

    class _LLM:
        model = "dummy"

        def complete(self, messages, tools=None, **kwargs):
            seen["last"] = messages[-1].content
            return LLMResponse(
                message=AssistantMessage(content="<summary>ok</summary>"),
                stop_reason="stop",
            )

    hook = CompactionHook(
        llm=_LLM(),
        threshold_input_tokens=1,
        keep_last_turns=2,
        extra_instructions="Focus on SQL changes.",
    )
    hook(
        HookContext(
            event=HookEvent.CHAT_END,
            agent=None,  # type: ignore[arg-type]
            history=_build_history(6),
            usage=UsageMeter(last_input_tokens=5000),
        )
    )
    assert "## Compact Instructions\nFocus on SQL changes." in seen["last"]


# ----- file-state replay across the compaction boundary -------------------- #


def _history_with_read(path: str, *, is_error: bool = False, tool: str = "Read"):
    """6 turns, with a tool call on the first so it lands in the compacted head."""
    from terno_agent.core.messages import ToolCall

    history: list[Message] = [SystemMessage("sys")]
    history.append(UserMessage("q0"))
    history.append(
        AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="c1", name=tool, arguments={"file_path": path})],
        )
    )
    history.append(
        ToolResultMessage(results=[ToolResult(call_id="c1", content="old", is_error=is_error)])
    )
    for i in range(1, 6):
        history.append(UserMessage(f"q{i}"))
        history.append(AssistantMessage(content=f"a{i}"))
    return history


def _compact(history, **kwargs) -> str:
    """Run the hook with a stub LLM; return the resulting summary message body."""
    from terno_agent.core.compaction import CompactionHook

    class _LLM:
        model = "dummy"

        def complete(self, messages, tools=None, **_kw):
            return LLMResponse(
                message=AssistantMessage(content="<summary>S</summary>"),
                stop_reason="stop",
            )

    hook = CompactionHook(
        llm=_LLM(), threshold_input_tokens=1, keep_last_turns=2, **kwargs
    )
    ctx = HookContext(
        event=HookEvent.CHAT_END,
        agent=None,  # type: ignore[arg-type]
        history=history,
        usage=UsageMeter(last_input_tokens=5000),
    )
    hook(ctx)
    return ctx.history[1].content


def test_replay_carries_current_file_contents_not_the_original_output():
    # The capture showed a file edited after being read coming back POST-edit,
    # so the replay must re-read from disk rather than replay stale tool output.
    body = _compact(
        _history_with_read("/x/a.py"),
        file_reader=lambda p: "line one\nline two\n",
    )

    assert "Called the Read tool with the following input:" in body
    assert '"file_path": "/x/a.py"' in body
    assert "1\tline one" in body and "2\tline two" in body
    assert "old" not in body  # the stale tool output must not survive


def test_replay_deduplicates_repeated_reads_of_one_file():
    from terno_agent.core.messages import ToolCall

    history = _history_with_read("/x/a.py")
    history[2].tool_calls.append(
        ToolCall(id="c2", name="Read", arguments={"file_path": "/x/a.py"})
    )
    body = _compact(history, file_reader=lambda p: "content\n")

    assert body.count("Called the Read tool") == 1


def test_replay_skips_failed_reads_and_non_read_tools():
    assert "Called the Read tool" not in _compact(
        _history_with_read("/x/a.py", is_error=True), file_reader=lambda p: "c\n"
    )
    # Edit/Bash output is transient; only reads are replayed.
    assert "Called the Read tool" not in _compact(
        _history_with_read("/x/a.py", tool="Edit"), file_reader=lambda p: "c\n"
    )


def test_replay_skips_files_that_no_longer_read():
    # Deleted or renamed since it was read — omit it rather than inventing text.
    body = _compact(_history_with_read("/x/gone.py"), file_reader=lambda p: None)
    assert "Called the Read tool" not in body
    assert "S" in body  # the summary itself still lands


def test_replay_respects_its_budget_and_never_truncates_a_file():
    big = "x" * 500
    body = _compact(
        _history_with_read("/x/a.py"),
        file_reader=lambda p: big,
        max_replay_chars=10,
    )
    # A half-file would misrepresent its contents, so it is dropped entirely.
    assert "Called the Read tool" not in body


def test_replay_can_be_switched_off():
    body = _compact(
        _history_with_read("/x/a.py"),
        replay_file_reads=False,
        file_reader=lambda p: "c\n",
    )
    assert "Called the Read tool" not in body


def test_resume_instruction_stays_last_after_the_replay():
    body = _compact(_history_with_read("/x/a.py"), file_reader=lambda p: "c\n")

    # "continue from where you left off" must be the final instruction, or the
    # file dump becomes the model's most recent instruction instead.
    assert body.index("Called the Read tool") < body.index(
        "Continue the conversation from where it left off"
    )
    assert body.strip().endswith("as if the break never happened.")
