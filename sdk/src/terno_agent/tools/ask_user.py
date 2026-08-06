"""Ask-the-user tool — synchronous human-in-the-loop.

When the agent hits a fork that needs a user decision (ambiguous spec,
risky destructive step, missing input) it can call ``ask_user`` to pose
one or more multiple-choice questions and pause for a reply.

The tool's `ask_callback` is invoked with the parsed questions and must
return one answer per question. The CLI provides a console-driven
callback that prints questions one at a time; non-interactive contexts
get the default callback which errors out so the agent proceeds with a
documented assumption.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema


@dataclass(slots=True, frozen=True)
class QuestionOption:
    label: str
    description: str = ""


@dataclass(slots=True, frozen=True)
class Question:
    question: str
    options: list[QuestionOption]
    multi_select: bool = False


@dataclass(slots=True)
class Answer:
    question: str
    selected: list[str] = field(default_factory=list)
    other_text: str | None = None


AskCallback = Callable[[list[Question]], list[Answer]]


def _no_interactive_callback(_questions: list[Question]) -> list[Answer]:
    raise ToolError(
        "ask_user is unavailable: no interactive channel is configured. "
        "Make a reasonable assumption and state it in your response."
    )


@dataclass
class AskUserTool:
    """Pose 1–4 multiple-choice questions to the user and block until answered."""

    ask_callback: AskCallback = field(default=_no_interactive_callback)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="AskUserQuestion",
            # Ported from the reference harness. Dropped: the plan-mode
            # paragraph (no EnterPlanMode/ExitPlanMode here) and the whole
            # `preview` feature, which needs a side-by-side rendering UI the
            # terminal prompter doesn't have. The return-shape sentence is kept —
            # it is Terno-specific and the model needs it to read the result.
            description=(
                "Use this tool only when you are blocked on a decision that is"
                " genuinely the user's to make: one you cannot resolve from the"
                " request, the code, or sensible defaults.\n"
                "\n"
                "Usage notes:\n"
                '- Users will always be able to select "Other" to provide custom'
                " text input\n"
                "- Use multiSelect: true to allow multiple answers to be selected"
                " for a question\n"
                "- If you recommend a specific option, make that the first option"
                ' in the list and add "(Recommended)" at the end of the label\n'
                "\n"
                "Reserve this for decisions where the user's answer changes what"
                " you do next — not for choices with a conventional default or"
                " facts you can verify in the codebase yourself. In those cases"
                " pick the obvious option, mention it in your response, and"
                " proceed.\n"
                "\n"
                'Returns JSON: {"answers": [{"question", "selected": [labels],'
                ' "other_text": str|null}]}.'
            ),
            parameters={
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "description": "1–4 questions; the user sees them one at a time.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": (
                                        "Full question text shown to the user. "
                                        "End with a question mark."
                                    ),
                                },
                                "multiSelect": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": (
                                        "true when more than one option may be selected; "
                                        "defaults to single-select."
                                    ),
                                },
                                "options": {
                                    "type": "array",
                                    "minItems": 2,
                                    "maxItems": 4,
                                    "description": (
                                        "2–4 distinct options. Do NOT include an 'Other' "
                                        "option — one is appended automatically."
                                    ),
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": {
                                                "type": "string",
                                                "description": "Short option label (1–5 words).",
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "Optional one-line explanation.",
                                            },
                                        },
                                        "required": ["label"],
                                    },
                                },
                            },
                            "required": ["question", "options"],
                        },
                    }
                },
                "required": ["questions"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        questions = _parse_questions(kwargs.get("questions"))
        answers = self.ask_callback(questions)
        if len(answers) != len(questions):
            raise ToolError(
                f"ask_user callback returned {len(answers)} answers for "
                f"{len(questions)} questions."
            )
        payload = {
            "answers": [
                {
                    "question": a.question,
                    "selected": list(a.selected),
                    "other_text": a.other_text,
                }
                for a in answers
            ]
        }
        return json.dumps(payload, ensure_ascii=False)


def _parse_questions(raw: Any) -> list[Question]:
    if not isinstance(raw, list) or not raw:
        raise ToolError("ask_user requires a non-empty 'questions' list.")
    if len(raw) > 4:
        raise ToolError("ask_user accepts at most 4 questions per call.")

    questions: list[Question] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ToolError(f"questions[{idx}] must be an object.")
        text = (item.get("question") or "").strip()
        if not text:
            raise ToolError(f"questions[{idx}].question is empty.")

        raw_opts = item.get("options")
        if not isinstance(raw_opts, list) or len(raw_opts) < 2:
            raise ToolError(f"questions[{idx}].options must have at least 2 entries.")
        if len(raw_opts) > 4:
            raise ToolError(f"questions[{idx}].options accepts at most 4 entries.")

        options: list[QuestionOption] = []
        seen: set[str] = set()
        for j, opt in enumerate(raw_opts):
            if isinstance(opt, str):
                label, description = opt.strip(), ""
            elif isinstance(opt, dict):
                label = (opt.get("label") or "").strip()
                description = (opt.get("description") or "").strip()
            else:
                raise ToolError(
                    f"questions[{idx}].options[{j}] must be a string or object."
                )
            if not label:
                raise ToolError(f"questions[{idx}].options[{j}].label is empty.")
            # CLI always appends Other itself, so silently drop any LLM-supplied one.
            if label.lower() == "other":
                continue
            if label.lower() in seen:
                continue
            seen.add(label.lower())
            options.append(QuestionOption(label=label, description=description))

        if len(options) < 2:
            raise ToolError(
                f"questions[{idx}] needs at least 2 distinct non-'Other' options."
            )

        questions.append(
            Question(
                question=text,
                options=options,
                multi_select=bool(item.get("multiSelect", False)),
            )
        )
    return questions


__all__ = [
    "Answer",
    "AskCallback",
    "AskUserTool",
    "Question",
    "QuestionOption",
]
