"""`ReportFindings` — code-review results as typed data rather than prose.

The point of the tool is that findings arrive as structured records the host can
render, sort and act on, instead of prose the host has to parse. Terno's host is
the CLI, so `on_findings` is the render hook; without one the tool still
validates and returns a readable summary, so the SDK is usable headless.

The captured description says to call it "only when the active code-review
instructions tell you to report findings with this tool" — so the built-in
`code-review` skill is what activates it. A review that wasn't asked for in this
format should stay prose.

Validation is deliberately strict about the two limits the captured schema
carries (`short_summary` ≤ 60 chars, ≤ 32 findings): they exist so a compact UI
can render a row per finding, and silently accepting longer values would push the
breakage into the renderer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema

FindingsCallback = Callable[[list[dict[str, Any]]], None]

_MAX_FINDINGS = 32
_MAX_SHORT_SUMMARY = 60
_MAX_CATEGORY = 40
_VERDICTS = ("CONFIRMED", "PLAUSIBLE")
_OUTCOMES = ("fixed", "skipped", "no_change_needed")
_REQUIRED = ("file", "summary", "failure_scenario")

# Most-severe-first is the caller's job; this is only used to render a stable
# order when the model supplies verdicts.
_VERDICT_RANK = {"CONFIRMED": 0, "PLAUSIBLE": 1}


def _validate(findings: Any) -> list[dict[str, Any]]:
    if findings is None:
        raise ToolError("ReportFindings requires a 'findings' array (use [] for none).")
    if not isinstance(findings, list):
        raise ToolError("'findings' must be an array.")
    if len(findings) > _MAX_FINDINGS:
        raise ToolError(
            f"At most {_MAX_FINDINGS} findings may be reported at once; got "
            f"{len(findings)}. Report the most severe ones."
        )

    out: list[dict[str, Any]] = []
    for i, raw in enumerate(findings):
        if not isinstance(raw, dict):
            raise ToolError(f"findings[{i}] must be an object.")
        missing = [k for k in _REQUIRED if not str(raw.get(k) or "").strip()]
        if missing:
            raise ToolError(f"findings[{i}] is missing: {', '.join(missing)}.")

        short = str(raw.get("short_summary") or "")
        if len(short) > _MAX_SHORT_SUMMARY:
            raise ToolError(
                f"findings[{i}].short_summary is {len(short)} chars; the limit is "
                f"{_MAX_SHORT_SUMMARY}. State the claim alone, without rationale."
            )
        category = str(raw.get("category") or "")
        if len(category) > _MAX_CATEGORY:
            raise ToolError(
                f"findings[{i}].category is {len(category)} chars; the limit is "
                f"{_MAX_CATEGORY}."
            )
        verdict = raw.get("verdict")
        if verdict is not None and verdict not in _VERDICTS:
            raise ToolError(
                f"findings[{i}].verdict must be one of {', '.join(_VERDICTS)}."
            )
        outcome = raw.get("outcome")
        if outcome is not None and outcome not in _OUTCOMES:
            raise ToolError(
                f"findings[{i}].outcome must be one of {', '.join(_OUTCOMES)}."
            )
        line = raw.get("line")
        if line is not None and not isinstance(line, int):
            raise ToolError(f"findings[{i}].line must be an integer.")

        out.append(dict(raw))
    return out


def render_findings(findings: list[dict[str, Any]]) -> str:
    """A plain-text rendering, for hosts without a findings UI."""
    if not findings:
        return "No findings survived verification."
    lines: list[str] = []
    for i, f in enumerate(findings, start=1):
        where = f["file"]
        if f.get("line") is not None:
            where += f":{f['line']}"
        tags = [t for t in (f.get("verdict"), f.get("category"), f.get("outcome")) if t]
        head = f"{i}. {where}"
        if tags:
            head += f"  [{' | '.join(str(t) for t in tags)}]"
        lines.append(head)
        lines.append(f"   {f['summary']}")
        lines.append(f"   failure: {f['failure_scenario']}")
    return "\n".join(lines)


@dataclass
class ReportFindingsTool:
    # The CLI supplies this to render findings itself; headless callers leave it
    # unset and get the text rendering back as the tool result.
    on_findings: FindingsCallback | None = None

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="ReportFindings",
            description=(
                "Report code-review findings as a typed list so the host can"
                " render them. Use this only when the active code-review"
                " instructions tell you to report findings with this tool;"
                " otherwise follow whatever output format those instructions"
                " specify. When reporting a review's results, call it once with"
                " the verified findings ranked most-severe first (empty array if"
                " nothing survived verification) and do not also print the"
                " findings as text. When re-reporting after applying fixes (only"
                " if the apply instructions ask for it), set `outcome` on each"
                " finding to what actually happened."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "maxItems": _MAX_FINDINGS,
                        "description": (
                            "Verified findings, most-severe first; empty if none"
                            " survived"
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {
                                    "type": "string",
                                    "description": (
                                        "Repo-relative path of the file the"
                                        " finding is in"
                                    ),
                                },
                                "summary": {
                                    "type": "string",
                                    "description": "One-sentence statement of the defect",
                                },
                                "failure_scenario": {
                                    "type": "string",
                                    "description": (
                                        "Concrete inputs/state -> wrong"
                                        " output/crash"
                                    ),
                                },
                                "line": {
                                    "type": "integer",
                                    "description": "1-indexed line the finding anchors to",
                                },
                                "short_summary": {
                                    "type": "string",
                                    "maxLength": _MAX_SHORT_SUMMARY,
                                    "description": (
                                        "Compressed label for compact UI (<=60"
                                        " chars): the claim alone, no rationale or"
                                        " consequence clause"
                                    ),
                                },
                                "category": {
                                    "type": "string",
                                    "maxLength": _MAX_CATEGORY,
                                    "description": (
                                        "Short kebab-case slug of the finding"
                                        ' type, e.g. "correctness",'
                                        ' "simplification", "efficiency",'
                                        ' "test-coverage"'
                                    ),
                                },
                                "verdict": {
                                    "type": "string",
                                    "enum": list(_VERDICTS),
                                    "description": (
                                        "Set when a verify pass ran; absent on"
                                        " inline-only reviews"
                                    ),
                                },
                                "outcome": {
                                    "type": "string",
                                    "enum": list(_OUTCOMES),
                                    "description": (
                                        "Set ONLY when re-reporting after applying"
                                        " fixes: what happened to this finding"
                                    ),
                                },
                            },
                            "required": list(_REQUIRED),
                        },
                    },
                },
                "required": ["findings"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        findings = _validate(kwargs.get("findings"))
        if self.on_findings is not None:
            self.on_findings(findings)
            # The host has rendered them; echoing the text too would duplicate
            # exactly what the description says not to do.
            return f"Reported {len(findings)} finding(s)."
        return render_findings(findings)


__all__ = ["FindingsCallback", "ReportFindingsTool", "render_findings"]
