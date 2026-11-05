"""Compaction: summarize older history before an LLM call."""

from __future__ import annotations

from terno_agent.agents.base import BaseAgent
from terno_agent.core.compaction import (
    SUMMARY_HEADER,
    CompactionHook,
    estimate_tokens,
)
from terno_agent.core.hooks import HookContext, HookEvent, UsageMeter
from terno_agent.core.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from terno_agent.core.tool import ToolSchema
from terno_agent.llm.base import LLMResponse


class _SummaryLLM:
    """Summarizer stub: records each transcript it is asked to summarize."""

    model = "summary"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.transcripts: list[str] = []

    def complete(self, messages: list[Message], tools=None, **_kw) -> LLMResponse:
        self.transcripts.append(messages[-1].content)
        if self.fail:
            raise RuntimeError("provider down")
        return LLMResponse(message=AssistantMessage(content="SUMMARY"), stop_reason="end_turn")


def _step(n: int, result_chars: int, **args) -> list[Message]:
    call = ToolCall(id=f"c{n}", name="run_python", arguments=args or {"code": f"step({n})"})
    return [
        AssistantMessage(content="", tool_calls=[call]),
        ToolResultMessage(results=[ToolResult(call_id=f"c{n}", content="r" * result_chars)]),
    ]


def _history(steps: int, result_chars: int = 8_000) -> list[Message]:
    history: list[Message] = [SystemMessage("sys"), UserMessage("question")]
    for n in range(steps):
        history += _step(n, result_chars)
    return history


def _fire(hook: CompactionHook, history: list[Message], agent=None) -> None:
    ctx = HookContext(event=HookEvent.PRE_LLM, agent=agent, history=history, usage=UsageMeter())
    hook(ctx)


def _assert_paired(history: list[Message]) -> None:
    for i, m in enumerate(history):
        if isinstance(m, AssistantMessage) and m.tool_calls:
            nxt = history[i + 1]
            assert isinstance(nxt, ToolResultMessage)
            assert {r.call_id for r in nxt.results} == {tc.id for tc in m.tool_calls}
        if isinstance(m, ToolResultMessage):
            assert isinstance(history[i - 1], AssistantMessage)


# ----- the hook --------------------------------------------------------------- #


def test_below_trigger_does_nothing():
    llm = _SummaryLLM()
    history = _history(steps=3)  # ~6k tokens
    before = list(history)
    _fire(CompactionHook(llm=llm), history)
    assert history == before and llm.transcripts == []


def test_over_trigger_keeps_request_and_last_steps():
    llm = _SummaryLLM()
    history = _history(steps=12)  # ~24k tokens
    user = history[1]
    _fire(CompactionHook(llm=llm), history)

    assert history[0].content == "sys"
    # Legacy-style: the summary is an assistant message; the system prompt is untouched.
    assert isinstance(history[1], AssistantMessage) and not history[1].tool_calls
    assert history[1].content.startswith(SUMMARY_HEADER)
    assert sum(isinstance(m, SystemMessage) for m in history) == 1
    assert history[2] is user
    kept = [m for m in history[3:] if isinstance(m, AssistantMessage)]
    assert [tc.id for m in kept for tc in m.tool_calls] == ["c9", "c10", "c11"]
    assert estimate_tokens(history[1:]) <= 8_000
    _assert_paired(history)


def test_fewer_steps_kept_when_they_do_not_fit_the_target():
    history = _history(steps=8, result_chars=12_000)  # each step ~3k tokens
    _fire(CompactionHook(llm=_SummaryLLM(), target_tokens=6_000), history)
    kept_calls = [m for m in history if isinstance(m, AssistantMessage) and m.tool_calls]
    assert len(kept_calls) == 1
    _assert_paired(history)


def test_earlier_turns_are_summarized_on_the_first_call_of_a_turn():
    llm = _SummaryLLM()
    history: list[Message] = [SystemMessage("sys")]
    for turn in range(6):
        history += [UserMessage(f"q{turn}"), AssistantMessage(content="a" * 16_000)]
    history.append(UserMessage("new question"))
    _fire(CompactionHook(llm=llm), history)
    assert len(history) == 3 and history[2].content == "new question"
    assert "q0" in llm.transcripts[0]


def test_previous_summary_is_carried_into_the_next_one():
    llm = _SummaryLLM()
    hook = CompactionHook(llm=llm)
    history = _history(steps=12)
    _fire(hook, history)
    for n in range(100, 112):
        history += _step(n, 8_000)
    _fire(hook, history)
    assert SUMMARY_HEADER in llm.transcripts[1]
    summaries = [m for m in history if isinstance(m, AssistantMessage) and not m.tool_calls]
    assert len(summaries) == 1 and summaries[0].content.startswith(SUMMARY_HEADER)


def test_does_not_resummarize_when_only_the_kept_steps_are_large():
    llm = _SummaryLLM()
    hook = CompactionHook(llm=llm)
    history: list[Message] = [
        SystemMessage("sys"),
        AssistantMessage(content=f"{SUMMARY_HEADER}\nold"),
        UserMessage("question"),
        *_step(1, 100_000),
    ]
    _fire(hook, history)
    assert llm.transcripts == []


def test_failure_leaves_history_and_waits_before_retrying():
    llm = _SummaryLLM(fail=True)
    hook = CompactionHook(llm=llm)
    history = _history(steps=12)
    before = list(history)
    _fire(hook, history)
    assert history == before
    _fire(hook, history)  # same size: no second attempt
    assert len(llm.transcripts) == 1


def test_transcript_shows_file_sizes_not_bodies_and_keeps_sql():
    llm = _SummaryLLM()
    history: list[Message] = [SystemMessage("sys"), UserMessage("question")]
    big = "<html>" + "x" * 40_000
    write = ToolCall(id="w", name="write_file", arguments={"path": "/ws/d.html", "content": big})
    history += [
        AssistantMessage(tool_calls=[write]),
        ToolResultMessage(results=[ToolResult(call_id="w", content="Wrote 40 KB")]),
    ]
    history += _step(1, 100, code="execute_sql(1, 'SELECT region, SUM(x) FROM s GROUP BY 1')")
    for n in range(2, 8):
        history += _step(n, 8_000)
    _fire(CompactionHook(llm=llm), history)

    transcript = llm.transcripts[0]
    assert "/ws/d.html" in transcript and "[40,006 chars]" in transcript
    assert "x" * 300 not in transcript
    assert "SELECT region, SUM(x) FROM s GROUP BY 1" in transcript


# ----- agent loop ------------------------------------------------------------- #


class _BigTool:
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(name="big", description="", parameters={"type": "object"})

    def run(self, **_kw) -> str:
        return "b" * 8_000


class _LoopLLM:
    """Calls the tool `steps` times, then answers. Records history sizes."""

    model = "loop"

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.calls = 0
        self.sizes: list[int] = []

    def complete(self, messages: list[Message], tools=None, **_kw) -> LLMResponse:
        if tools is None:  # the summarizer
            return LLMResponse(message=AssistantMessage(content="SUMMARY"), stop_reason="end_turn")
        self.calls += 1
        self.sizes.append(estimate_tokens(messages[1:]))
        if self.calls <= self.steps:
            call = ToolCall(id=f"c{self.calls}", name="big", arguments={})
            return LLMResponse(message=AssistantMessage(tool_calls=[call]), stop_reason="tool_use")
        return LLMResponse(message=AssistantMessage(content="done"), stop_reason="end_turn")


def test_long_turn_compacts_once_and_stays_bounded():
    llm = _LoopLLM(steps=15)
    agent = BaseAgent(llm, "sys", [_BigTool()])
    agent.max_iterations = 30
    agent.add_hook(HookEvent.PRE_LLM, CompactionHook(llm=llm))
    run = agent.run("go")

    assert run.answer == "done"
    assert max(llm.sizes) < 20_000 + 2_100  # never far past the trigger
    drops = sum(1 for a, b in zip(llm.sizes, llm.sizes[1:], strict=False) if b < a)
    assert 1 <= drops <= 2  # summarized rarely, not every step
    assert isinstance(run.trace[0], UserMessage) and run.trace[0].content == "go"
    _assert_paired(agent.history)


def test_summary_covers_kept_steps_and_is_reported_for_saving():
    llm = _SummaryLLM()
    events: list = []
    agent = BaseAgent(llm, "sys", [], on_event=events.append)
    history = _history(steps=12)
    agent.history[:] = history
    hook = CompactionHook(llm=llm)
    _fire(hook, agent.history, agent=agent)

    # The kept steps stay verbatim and are also in what was summarized.
    assert "[tool_call] run_python(code=step(11))" in llm.transcripts[0]
    (event,) = [e for e in events if type(e).__name__ == "CompactionEvent"]
    assert event.summary == "SUMMARY"
    assert event.preserved_messages == 6


def test_huge_question_does_not_cause_a_summary_every_step():
    # The question alone is over the trigger, so history never gets under it;
    # summaries must still wait for enough new steps instead of firing each call.
    llm = _SummaryLLM()
    hook = CompactionHook(llm=llm)
    history: list[Message] = [SystemMessage("sys"), UserMessage("q" * 100_000)]
    for n in range(9):
        history += _step(n, 4_000)
        _fire(hook, history)
    assert 1 <= len(llm.transcripts) <= 3


def test_anthropic_moves_a_leading_summary_into_the_system_prompt():
    from terno_agent.llm.anthropic_client import _split_system

    system, rest = _split_system([
        SystemMessage("prompt"),
        AssistantMessage(content=f"{SUMMARY_HEADER}\nolder work"),
        UserMessage("question"),
    ])
    assert system == f"prompt\n\n{SUMMARY_HEADER}\nolder work"
    assert [type(m) for m in rest] == [UserMessage]
