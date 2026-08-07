"""ReportFindings: code-review results as typed data."""

from __future__ import annotations

from typing import Any

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.tools.report_findings import ReportFindingsTool, render_findings


def _finding(**over: Any) -> dict[str, Any]:
    base = {
        "file": "src/a.py",
        "summary": "Off-by-one in the loop bound.",
        "failure_scenario": "n=0 -> IndexError on the first iteration.",
    }
    base.update(over)
    return base


# ----- validation ---------------------------------------------------------- #


def test_the_three_required_fields_are_enforced() -> None:
    tool = ReportFindingsTool()
    for missing in ("file", "summary", "failure_scenario"):
        bad = _finding()
        del bad[missing]
        with pytest.raises(ToolError, match=missing):
            tool.run(findings=[bad])


def test_blank_required_fields_are_rejected() -> None:
    # A present-but-empty `failure_scenario` is the whole value of the field.
    with pytest.raises(ToolError, match="failure_scenario"):
        ReportFindingsTool().run(findings=[_finding(failure_scenario="   ")])


def test_an_empty_array_is_valid() -> None:
    # "empty array if nothing survived verification" — a clean review.
    out = ReportFindingsTool().run(findings=[])
    assert "No findings" in out


def test_a_missing_findings_key_is_an_error() -> None:
    with pytest.raises(ToolError, match=r"use \[\] for none"):
        ReportFindingsTool().run()


def test_non_array_and_non_object_shapes_are_rejected() -> None:
    tool = ReportFindingsTool()
    with pytest.raises(ToolError, match="must be an array"):
        tool.run(findings="a finding")
    with pytest.raises(ToolError, match="must be an object"):
        tool.run(findings=["a finding"])


def test_the_short_summary_limit_is_enforced() -> None:
    # The cap exists so a compact UI can render one row per finding; accepting
    # more would push the breakage into the renderer.
    with pytest.raises(ToolError, match="the limit is 60"):
        ReportFindingsTool().run(findings=[_finding(short_summary="x" * 61)])

    ReportFindingsTool().run(findings=[_finding(short_summary="x" * 60)])


def test_the_findings_cap_is_enforced() -> None:
    tool = ReportFindingsTool()
    tool.run(findings=[_finding() for _ in range(32)])

    with pytest.raises(ToolError, match="At most 32"):
        tool.run(findings=[_finding() for _ in range(33)])


def test_category_length_is_capped() -> None:
    with pytest.raises(ToolError, match="the limit is 40"):
        ReportFindingsTool().run(findings=[_finding(category="c" * 41)])


def test_verdict_and_outcome_are_constrained() -> None:
    tool = ReportFindingsTool()
    with pytest.raises(ToolError, match="verdict must be one of"):
        tool.run(findings=[_finding(verdict="MAYBE")])
    with pytest.raises(ToolError, match="outcome must be one of"):
        tool.run(findings=[_finding(outcome="done")])

    tool.run(findings=[_finding(verdict="CONFIRMED", outcome="fixed")])


def test_line_must_be_an_integer() -> None:
    with pytest.raises(ToolError, match="line must be an integer"):
        ReportFindingsTool().run(findings=[_finding(line="12")])


# ----- rendering and the host hook ----------------------------------------- #


def test_text_rendering_includes_location_and_tags() -> None:
    out = render_findings(
        [_finding(line=12, category="correctness", verdict="CONFIRMED")]
    )

    assert "src/a.py:12" in out
    assert "CONFIRMED" in out and "correctness" in out
    assert "Off-by-one" in out
    assert "failure:" in out


def test_rendering_omits_the_line_when_absent() -> None:
    out = render_findings([_finding()])
    assert "src/a.py" in out
    assert "src/a.py:" not in out


def test_findings_go_to_the_host_hook_when_one_is_wired() -> None:
    seen: list[list[dict[str, Any]]] = []
    tool = ReportFindingsTool(on_findings=seen.append)

    out = tool.run(findings=[_finding(line=3)])

    assert len(seen) == 1
    assert seen[0][0]["line"] == 3
    # The host renders them; echoing the text too would duplicate exactly what
    # the description says not to do.
    assert out == "Reported 1 finding(s)."
    assert "Off-by-one" not in out


def test_without_a_hook_the_text_rendering_is_returned() -> None:
    # Keeps the SDK usable headless.
    out = ReportFindingsTool().run(findings=[_finding()])
    assert "Off-by-one" in out


def test_the_hook_receives_a_copy_not_the_callers_dicts() -> None:
    original = _finding()
    seen: list[list[dict[str, Any]]] = []
    ReportFindingsTool(on_findings=seen.append).run(findings=[original])

    seen[0][0]["summary"] = "mutated"
    assert original["summary"] == "Off-by-one in the loop bound."


# ----- wiring -------------------------------------------------------------- #


def test_agent_registers_it_eagerly() -> None:
    from terno_agent.agents.terno import TernoAgent
    from terno_agent.core.messages import AssistantMessage
    from terno_agent.llm.base import LLMResponse

    class _LLM:
        model = "dummy"

        def complete(self, *_a, **_k):
            return LLMResponse(message=AssistantMessage(content="x"), stop_reason="stop")

    # Eager, not deferred: a review is exactly when an extra ToolSearch
    # round-trip is unwelcome.
    assert "ReportFindings" in TernoAgent(llm=_LLM()).tools


def test_the_code_review_skill_activates_the_tool() -> None:
    from pathlib import Path

    import terno_agent

    skill = (
        Path(terno_agent.__file__).parent
        / "skills"
        / "builtin"
        / "code-review"
        / "SKILL.md"
    )
    body = skill.read_text()
    # The description says to use the tool "only when the active code-review
    # instructions tell you to" — so the skill has to say so.
    assert "ReportFindings" in body
    assert "failure_scenario" in body
