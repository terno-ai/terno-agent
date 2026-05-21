from __future__ import annotations

import io
import json

import pytest
from rich.console import Console

from terno_agent.agents.terno import TernoAgent
from terno_agent.core.exceptions import ToolError
from terno_agent.core.messages import AssistantMessage, ToolCall
from terno_agent.llm.base import LLMResponse
from terno_agent.tools.ask_user import (
    Answer,
    AskUserTool,
    Question,
    QuestionOption,
)

# --------------------------------------------------------------------------- #
# AskUserTool — schema parsing + dispatch
# --------------------------------------------------------------------------- #


def test_schema_advertises_questions_array() -> None:
    tool = AskUserTool()
    schema = tool.schema
    assert schema.name == "ask_user"
    questions_field = schema.parameters["properties"]["questions"]
    assert questions_field["type"] == "array"
    assert questions_field["minItems"] == 1
    assert questions_field["maxItems"] == 4


def test_run_dispatches_to_callback_and_returns_json() -> None:
    captured: dict[str, list[Question]] = {}

    def cb(questions: list[Question]) -> list[Answer]:
        captured["seen"] = questions
        return [
            Answer(question=questions[0].question, selected=["Postgres"]),
            Answer(question=questions[1].question, selected=[], other_text="AWS Lambda"),
        ]

    tool = AskUserTool(ask_callback=cb)
    raw = tool.run(
        questions=[
            {
                "question": "Which database?",
                "options": [
                    {"label": "Postgres", "description": "Default."},
                    {"label": "SQLite"},
                ],
            },
            {
                "question": "Where to deploy?",
                "options": [
                    {"label": "Fly.io"},
                    {"label": "Render"},
                ],
            },
        ]
    )

    assert len(captured["seen"]) == 2
    assert captured["seen"][0].options[0].description == "Default."
    payload = json.loads(raw)
    assert payload["answers"][0]["selected"] == ["Postgres"]
    assert payload["answers"][1]["other_text"] == "AWS Lambda"


def test_run_strips_llm_supplied_other_option() -> None:
    seen: dict[str, list[Question]] = {}

    def cb(questions: list[Question]) -> list[Answer]:
        seen["q"] = questions
        return [Answer(question=questions[0].question, selected=["A"])]

    tool = AskUserTool(ask_callback=cb)
    tool.run(
        questions=[
            {
                "question": "Pick one?",
                "options": ["A", "B", "Other"],
            }
        ]
    )
    labels = [opt.label for opt in seen["q"][0].options]
    assert labels == ["A", "B"]


def test_run_rejects_too_few_options() -> None:
    tool = AskUserTool()
    with pytest.raises(ToolError, match="at least 2"):
        tool.run(questions=[{"question": "x?", "options": ["only-one"]}])


def test_run_rejects_too_many_questions() -> None:
    tool = AskUserTool()
    payload = [
        {"question": f"q{i}?", "options": ["a", "b"]} for i in range(5)
    ]
    with pytest.raises(ToolError, match="at most 4 questions"):
        tool.run(questions=payload)


def test_default_callback_errors_for_non_interactive() -> None:
    tool = AskUserTool()
    with pytest.raises(ToolError, match="no interactive channel"):
        tool.run(questions=[{"question": "x?", "options": ["a", "b"]}])


def test_callback_returning_wrong_count_errors() -> None:
    tool = AskUserTool(ask_callback=lambda _qs: [])
    with pytest.raises(ToolError, match="0 answers for 1"):
        tool.run(questions=[{"question": "x?", "options": ["a", "b"]}])


# --------------------------------------------------------------------------- #
# CliPrompter — input parsing flow
# --------------------------------------------------------------------------- #


def _silent_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=80)


def test_cli_prompter_single_select_reads_one_number(monkeypatch) -> None:
    from terno_agent.cli import CliPrompter

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    prompter = CliPrompter(_silent_console())
    answer = prompter(
        [
            Question(
                question="Pick?",
                options=[QuestionOption("Alpha"), QuestionOption("Beta")],
            )
        ]
    )
    assert len(answer) == 1
    assert answer[0].selected == ["Beta"]
    assert answer[0].other_text is None


def test_cli_prompter_multi_select_parses_comma_list(monkeypatch) -> None:
    from terno_agent.cli import CliPrompter

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["1, 3"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    prompter = CliPrompter(_silent_console())
    answer = prompter(
        [
            Question(
                question="Pick all that apply?",
                options=[
                    QuestionOption("Cache"),
                    QuestionOption("Retry"),
                    QuestionOption("Tracing"),
                ],
                multi_select=True,
            )
        ]
    )
    assert answer[0].selected == ["Cache", "Tracing"]


def test_cli_prompter_other_prompts_for_free_text(monkeypatch) -> None:
    from terno_agent.cli import CliPrompter

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["3", "AWS Lambda"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    prompter = CliPrompter(_silent_console())
    answer = prompter(
        [
            Question(
                question="Where to deploy?",
                options=[QuestionOption("Fly.io"), QuestionOption("Render")],
            )
        ]
    )
    assert answer[0].selected == []
    assert answer[0].other_text == "AWS Lambda"


def test_cli_prompter_reprompts_on_bad_input(monkeypatch) -> None:
    from terno_agent.cli import CliPrompter

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["banana", "9", "", "1"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    prompter = CliPrompter(_silent_console())
    answer = prompter(
        [
            Question(
                question="Pick?",
                options=[QuestionOption("A"), QuestionOption("B")],
            )
        ]
    )
    assert answer[0].selected == ["A"]


def test_cli_prompter_errors_without_tty(monkeypatch) -> None:
    from terno_agent.cli import CliPrompter

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    prompter = CliPrompter(_silent_console())
    with pytest.raises(ToolError, match="not a TTY"):
        prompter([Question("x?", [QuestionOption("a"), QuestionOption("b")])])


# --------------------------------------------------------------------------- #
# TernoAgent integration — tool registration gated on ask_callback
# --------------------------------------------------------------------------- #


class _ScriptedLLM:
    """LLM stub that emits the queued responses in order."""

    model = "scripted"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.tools_seen: list[list] = []

    def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN202
        self.tools_seen.append([t.name for t in (tools or [])])
        if not self._responses:
            return LLMResponse(message=AssistantMessage(content="done"), stop_reason="stop")
        return self._responses.pop(0)


def test_terno_agent_registers_ask_user_when_callback_provided() -> None:
    llm = _ScriptedLLM([])
    agent = TernoAgent(llm, ask_callback=lambda qs: [Answer(q.question) for q in qs])
    assert "ask_user" in agent.tools


def test_terno_agent_omits_ask_user_when_no_callback() -> None:
    llm = _ScriptedLLM([])
    agent = TernoAgent(llm)
    assert "ask_user" not in agent.tools


def test_terno_agent_routes_ask_user_calls_to_callback() -> None:
    asked: dict[str, list[Question]] = {}

    def cb(questions: list[Question]) -> list[Answer]:
        asked["q"] = questions
        return [Answer(question=q.question, selected=[q.options[0].label]) for q in questions]

    llm = _ScriptedLLM(
        [
            LLMResponse(
                message=AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="ask_user",
                            arguments={
                                "questions": [
                                    {
                                        "question": "Style?",
                                        "options": ["Tabs", "Spaces"],
                                    }
                                ]
                            },
                        )
                    ],
                ),
                stop_reason="tool_use",
            ),
            LLMResponse(
                message=AssistantMessage(content="done"),
                stop_reason="stop",
            ),
        ]
    )
    agent = TernoAgent(llm, ask_callback=cb)
    result = agent.run("Configure formatting")
    assert result.answer == "done"
    assert asked["q"][0].question == "Style?"
