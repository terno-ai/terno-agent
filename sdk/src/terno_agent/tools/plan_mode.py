"""Plan mode: explore read-only, write a plan, get it approved.

Descriptions are ported from the reference harness. Its plan flow keeps the plan
in a file rather than passing it as a tool argument — `ExitPlanMode` reads what
the agent wrote — and that shape is reproduced here: `EnterPlanMode` names the
plan file, and the permission policy allows writes to that one path while
denying every other mutation.

Trimmed from the captured text: the `allowedPrompts` parameter (marked
"Deprecated: no longer used" in the capture itself) and references to features
Terno lacks. Each tool's description says where.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from terno_agent.core.exceptions import ToolError
from terno_agent.core.permissions import PermissionMode, PermissionPolicy
from terno_agent.core.tool import ToolSchema

# Called with the plan text; returns True to approve. Without one there is
# nobody to approve, so ExitPlanMode says so rather than silently proceeding.
PlanApproval = Callable[[str], bool]

DEFAULT_PLAN_FILE = ".terno/plan.md"


@dataclass
class EnterPlanModeTool:
    policy: PermissionPolicy
    workdir: Path
    plan_file: str = DEFAULT_PLAN_FILE

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="EnterPlanMode",
            description=(
                "Use this tool proactively when you're about to start a"
                " non-trivial implementation task. Getting user sign-off on your"
                " approach before writing code prevents wasted effort and ensures"
                " alignment. This tool transitions you into plan mode where you"
                " can explore the codebase and design an implementation approach"
                " for user approval.\n"
                "\n"
                "## When to Use This Tool\n"
                "\n"
                "**Prefer using EnterPlanMode** for implementation tasks unless"
                " they're simple. Use it when ANY of these conditions apply:\n"
                "\n"
                "1. **New Feature Implementation**: Adding meaningful new"
                " functionality\n"
                "2. **Multiple Valid Approaches**: The task can be solved in"
                " several different ways\n"
                "3. **Code Modifications**: Changes that affect existing behavior"
                " or structure\n"
                "4. **Architectural Decisions**: The task requires choosing"
                " between patterns or technologies\n"
                "5. **Multi-File Changes**: The task will likely touch more than"
                " 2-3 files\n"
                "\n"
                "## While In Plan Mode\n"
                "\n"
                "Only read-only tools work — every mutating tool is denied until"
                " a plan is approved. Explore, then write your plan to the file"
                " this tool names, then call ExitPlanMode to request approval."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def run(self, **_kwargs: Any) -> str:
        target = self.workdir / self.plan_file
        self.policy.plan_file = str(target)
        self.policy.mode = PermissionMode.PLAN
        return (
            "Entered plan mode. Mutating tools are now denied.\n"
            f"Write your plan to {target}, then call ExitPlanMode to request "
            "approval. Do not ask whether the plan is acceptable with "
            "AskUserQuestion — ExitPlanMode is what requests approval."
        )


@dataclass
class ExitPlanModeTool:
    policy: PermissionPolicy
    workdir: Path
    plan_file: str = DEFAULT_PLAN_FILE
    on_approval: PlanApproval | None = None
    # Mode to return to once the plan is approved.
    approved_mode: PermissionMode = PermissionMode.ASK

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="ExitPlanMode",
            description=(
                "Use this tool when you are in plan mode and have finished"
                " writing your plan to the plan file and are ready for user"
                " approval.\n"
                "\n"
                "## How This Tool Works\n"
                "- You should have already written your plan to the plan file"
                " named by EnterPlanMode\n"
                "- This tool does NOT take the plan content as a parameter — it"
                " will read the plan from the file you wrote\n"
                "- This tool simply signals that you're done planning and ready"
                " for the user to review and approve\n"
                "- The user will see the contents of your plan file when they"
                " review it\n"
                "\n"
                "## When to Use This Tool\n"
                "IMPORTANT: Only use this tool when the task requires planning the"
                " implementation steps of a task that requires writing code. For"
                " research tasks where you're gathering information, searching"
                " files, reading files or in general trying to understand the"
                " codebase - do NOT use this tool.\n"
                "\n"
                "## Before Using This Tool\n"
                "Ensure your plan is complete and unambiguous:\n"
                "- If you have unresolved questions about requirements or"
                " approach, use AskUserQuestion first (in earlier phases)\n"
                "- Once your plan is finalized, use THIS tool to request"
                " approval\n"
                "\n"
                '**Important:** Do NOT use AskUserQuestion to ask "Is this plan'
                ' okay?" or "Should I proceed?" - that\'s exactly what THIS tool'
                " does. ExitPlanMode inherently requests user approval of your"
                " plan."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def run(self, **_kwargs: Any) -> str:
        if self.policy.mode is not PermissionMode.PLAN:
            raise ToolError("Not in plan mode — nothing to exit.")

        target = Path(self.policy.plan_file or (self.workdir / self.plan_file))
        try:
            plan = target.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ToolError(
                f"No plan found at {target}: {exc}. Write your plan there first."
            ) from exc
        if not plan:
            raise ToolError(f"The plan file {target} is empty. Write your plan first.")

        if self.on_approval is None:
            raise ToolError(
                "No approval callback is configured, so the plan cannot be "
                "approved. Staying in plan mode."
            )
        if not self.on_approval(plan):
            # Deliberately stay in plan mode: a rejected plan means revise, not
            # proceed.
            return (
                "The user did not approve the plan. Still in plan mode — revise "
                f"{target} and call ExitPlanMode again."
            )

        self.policy.mode = self.approved_mode
        self.policy.plan_file = ""
        return (
            "Plan approved. Plan mode exited; mutating tools are available "
            "again. Begin implementing the approved plan."
        )


__all__ = [
    "DEFAULT_PLAN_FILE",
    "EnterPlanModeTool",
    "ExitPlanModeTool",
    "PlanApproval",
]
