"""Plan mode: read-only exploration until a written plan is approved."""

from __future__ import annotations

from pathlib import Path

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.core.permissions import (
    PermissionMode,
    PermissionPolicy,
    PermissionRequest,
)
from terno_agent.tools.plan_mode import EnterPlanModeTool, ExitPlanModeTool


def _policy() -> PermissionPolicy:
    return PermissionPolicy.build(mode=PermissionMode.ALLOW_ALL)


def _enter(tmp_path: Path) -> PermissionPolicy:
    policy = _policy()
    EnterPlanModeTool(policy=policy, workdir=tmp_path).run()
    return policy


# ----- enforcement --------------------------------------------------------- #


def test_entering_plan_mode_switches_the_policy(tmp_path: Path) -> None:
    policy = _policy()
    out = EnterPlanModeTool(policy=policy, workdir=tmp_path).run()

    assert policy.mode is PermissionMode.PLAN
    assert policy.plan_file == str(tmp_path / ".terno" / "plan.md")
    assert "plan.md" in out


def test_read_only_tools_still_work_in_plan_mode(tmp_path: Path) -> None:
    policy = _enter(tmp_path)
    for tool in ("Read", "TaskList", "WebSearch", "AskUserQuestion"):
        assert policy.decide(PermissionRequest(tool, {})).kind == "allow_once"


def test_mutating_tools_are_denied_not_prompted(tmp_path: Path) -> None:
    # Denied outright: the point is that nothing changes before approval, so a
    # prompt the user could accept would defeat it.
    policy = _enter(tmp_path)
    for tool in ("Bash", "Edit", "Agent"):
        decision = policy.decide(PermissionRequest(tool, {}))
        assert decision.kind == "deny"
        assert "plan mode" in decision.feedback
        assert "ExitPlanMode" in decision.feedback


def test_writing_the_plan_file_is_the_one_permitted_write(tmp_path: Path) -> None:
    policy = _enter(tmp_path)
    plan = policy.plan_file

    assert policy.decide(
        PermissionRequest("Write", {"file_path": plan})
    ).kind == "allow_once"
    assert policy.decide(
        PermissionRequest("Write", {"file_path": str(tmp_path / "app.py")})
    ).kind == "deny"


def test_plan_mode_tools_are_never_locked_out(tmp_path: Path) -> None:
    policy = _enter(tmp_path)
    for tool in ("EnterPlanMode", "ExitPlanMode"):
        assert policy.decide(PermissionRequest(tool, {})).kind == "allow_once"


# ----- exiting ------------------------------------------------------------- #


def _write_plan(policy: PermissionPolicy, text: str = "1. Do the thing.") -> None:
    p = Path(policy.plan_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_approved_plan_restores_the_previous_mode(tmp_path: Path) -> None:
    policy = _enter(tmp_path)
    _write_plan(policy)
    seen: list[str] = []

    out = ExitPlanModeTool(
        policy=policy,
        workdir=tmp_path,
        on_approval=lambda plan: (seen.append(plan), True)[1],
        approved_mode=PermissionMode.ALLOW_ALL,
    ).run()

    assert policy.mode is PermissionMode.ALLOW_ALL
    assert policy.plan_file == ""
    assert seen == ["1. Do the thing."]  # the reviewer sees the file's contents
    assert "approved" in out.lower()


def test_rejected_plan_stays_in_plan_mode(tmp_path: Path) -> None:
    policy = _enter(tmp_path)
    _write_plan(policy)

    out = ExitPlanModeTool(
        policy=policy, workdir=tmp_path, on_approval=lambda _plan: False
    ).run()

    # Rejection means revise, not proceed.
    assert policy.mode is PermissionMode.PLAN
    assert "not approve" in out
    assert policy.decide(PermissionRequest("Edit", {})).kind == "deny"


def test_exit_requires_a_plan_that_actually_exists(tmp_path: Path) -> None:
    policy = _enter(tmp_path)
    tool = ExitPlanModeTool(
        policy=policy, workdir=tmp_path, on_approval=lambda _p: True
    )

    with pytest.raises(ToolError, match="No plan found"):
        tool.run()

    _write_plan(policy, "   \n")
    with pytest.raises(ToolError, match="empty"):
        tool.run()

    assert policy.mode is PermissionMode.PLAN  # still locked down


def test_exit_without_an_approver_cannot_self_approve(tmp_path: Path) -> None:
    policy = _enter(tmp_path)
    _write_plan(policy)

    with pytest.raises(ToolError, match="approval callback"):
        ExitPlanModeTool(policy=policy, workdir=tmp_path, on_approval=None).run()

    assert policy.mode is PermissionMode.PLAN


def test_exit_outside_plan_mode_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="Not in plan mode"):
        ExitPlanModeTool(
            policy=_policy(), workdir=tmp_path, on_approval=lambda _p: True
        ).run()


# ----- registration -------------------------------------------------------- #


def test_agent_registers_the_plan_tools() -> None:
    from terno_agent.agents.terno import TernoAgent
    from terno_agent.core.messages import AssistantMessage
    from terno_agent.llm.base import LLMResponse

    class _LLM:
        model = "dummy"

        def complete(self, *_a, **_k):
            return LLMResponse(
                message=AssistantMessage(content="x"), stop_reason="stop"
            )

    # Deferred by default, like the reference harness.
    agent = TernoAgent(llm=_LLM(), permission_mode=PermissionMode.ALLOW_ALL)
    assert {"EnterPlanMode", "ExitPlanMode"} <= set(agent.tool_registry.names)

    eager = TernoAgent(
        llm=_LLM(), permission_mode=PermissionMode.ALLOW_ALL, defer_tools=()
    )
    assert "EnterPlanMode" in eager.tools
    assert "ExitPlanMode" in eager.tools


def test_a_decoy_file_with_the_plans_name_is_still_denied(tmp_path: Path) -> None:
    """Basename matching would let any `plan.md` through; paths are resolved."""
    policy = _enter(tmp_path)
    decoy = tmp_path / "elsewhere" / "plan.md"
    decoy.parent.mkdir(parents=True, exist_ok=True)

    assert policy.decide(
        PermissionRequest("Write", {"file_path": str(decoy)})
    ).kind == "deny"
    # ...but a non-canonical spelling of the real plan file is accepted.
    weird = str(tmp_path / "." / ".terno" / ".." / ".terno" / "plan.md")
    assert policy.decide(
        PermissionRequest("Write", {"file_path": weird})
    ).kind == "allow_once"
