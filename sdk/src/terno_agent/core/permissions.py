"""Generic permission system for tool calls.

Decides — across CLI, SDK, and backend front-ends — whether each tool
call may run. Three modes:

- ``ALLOW_ALL``: run every tool. (Default; SDK behavior pre-dating this
  module.)
- ``ALLOW_LIST``: run only tools/commands that match a configured rule;
  deny the rest without prompting.
- ``ASK``: run tools matched by a rule; for anything else, consult an
  ``on_request`` callback. The callback may pre-approve future calls by
  returning ``PermissionDecision.allow_always(...)``, which the policy
  persists as a new rule.

A `PermissionPolicy` is itself a ``PreToolUseHook`` — registering it on
the agent (or passing it to ``TernoAgent(permission_policy=...)``) is
sufficient. Rules are mutable at runtime so the front-end can add or
remove allow-rules between turns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from terno_agent.core.hooks import PreToolUseContext


# Tools that ship with terno and are intrinsically safe — read-only file
# helpers, in-memory task list, user-driven prompts, etc. They never
# trigger a permission prompt regardless of mode. Callers can override
# the set via ``PermissionPolicy(always_allow_tools=...)``.
DEFAULT_ALWAYS_ALLOW_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "glob",
        "grep",
        "task_list",
        "task_get",
        "task_create",
        "task_update",
        "search_memory",
        "web_fetch",
        "web_search",
        "ask_user",
        "activate_skill",
    }
)


class PermissionMode(str, Enum):
    ASK = "ask"
    ALLOW_LIST = "allow_list"
    ALLOW_ALL = "allow_all"


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    """A pending tool call the policy is deciding on."""

    tool_name: str
    arguments: dict


@dataclass(frozen=True, slots=True)
class AllowRule:
    """Pre-approves a tool call without prompting.

    ``tool_name`` is required. ``command_prefix``, if set, additionally
    requires ``arguments["command"]`` to start with the prefix — used
    for narrow allowlists like "anything starting with ``uv run``".
    """

    tool_name: str
    command_prefix: str | None = None

    def matches(self, request: PermissionRequest) -> bool:
        if request.tool_name != self.tool_name:
            return False
        if self.command_prefix is None:
            return True
        cmd = request.arguments.get("command", "")
        return isinstance(cmd, str) and cmd.startswith(self.command_prefix)

    def label(self) -> str:
        if self.command_prefix:
            return f"{self.tool_name}({self.command_prefix}...)"
        return self.tool_name


RuleLike = "AllowRule | str | tuple[str, str | None] | tuple[str]"


@dataclass(slots=True)
class PermissionDecision:
    """Outcome returned by an ASK-mode prompter.

    ``kind`` is one of:
      - ``"allow_once"``  — run this call only.
      - ``"allow_always"`` — run this call and persist ``rule`` so future
        matching calls skip the prompt.
      - ``"deny"`` — refuse the call; ``feedback`` is shown to the LLM.
    """

    kind: str
    feedback: str = ""
    rule: AllowRule | None = None

    @classmethod
    def allow_once(cls) -> PermissionDecision:
        return cls(kind="allow_once")

    @classmethod
    def allow_always(
        cls,
        tool: str,
        *,
        command_prefix: str | None = None,
    ) -> PermissionDecision:
        return cls(
            kind="allow_always",
            rule=AllowRule(tool_name=tool, command_prefix=command_prefix),
        )

    @classmethod
    def deny(cls, feedback: str = "") -> PermissionDecision:
        return cls(
            kind="deny",
            feedback=feedback.strip() or "Tool call denied by the user.",
        )


PermissionCallback = Callable[[PermissionRequest], PermissionDecision]


@dataclass(slots=True)
class PermissionPolicy:
    """Per-agent permission state. Mutable; also a ``PreToolUseHook``.

    The user-facing front-end (CLI, web UI, SDK caller) supplies an
    ``on_request`` callable. Whatever decision it returns is honored:
    ``allow_always`` decisions are persisted as new rules so the policy
    doesn't ask again for matching calls.
    """

    mode: PermissionMode = PermissionMode.ALLOW_ALL
    on_request: PermissionCallback | None = None
    always_allow_tools: set[str] = field(
        default_factory=lambda: set(DEFAULT_ALWAYS_ALLOW_TOOLS)
    )
    _rules: list[AllowRule] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        mode: PermissionMode | str = PermissionMode.ALLOW_ALL,
        allow_rules: Iterable[object] = (),
        on_request: PermissionCallback | None = None,
        always_allow_tools: Iterable[str] | None = None,
    ) -> PermissionPolicy:
        """Construct a policy from possibly-loose argument types.

        ``allow_rules`` accepts ``AllowRule`` instances, bare strings
        (tool name only), or ``(tool, command_prefix)`` tuples.
        """
        resolved_mode = PermissionMode(mode) if isinstance(mode, str) else mode
        policy = cls(
            mode=resolved_mode,
            on_request=on_request,
            always_allow_tools=(
                set(DEFAULT_ALWAYS_ALLOW_TOOLS)
                if always_allow_tools is None
                else set(always_allow_tools)
            ),
        )
        for rule in allow_rules:
            policy._add_rule(rule)
        return policy

    # ----- public mutation -------------------------------------------------- #

    @property
    def rules(self) -> tuple[AllowRule, ...]:
        return tuple(self._rules)

    def set_mode(self, mode: PermissionMode | str) -> None:
        self.mode = PermissionMode(mode) if isinstance(mode, str) else mode

    def allow(self, tool: str, *, command_prefix: str | None = None) -> AllowRule:
        return self._add_rule(AllowRule(tool_name=tool, command_prefix=command_prefix))

    def revoke(self, tool: str, *, command_prefix: str | None = None) -> bool:
        target = AllowRule(tool_name=tool, command_prefix=command_prefix)
        for i, r in enumerate(self._rules):
            if r == target:
                del self._rules[i]
                return True
        return False

    def clear_rules(self) -> None:
        self._rules.clear()

    # ----- core decision logic --------------------------------------------- #

    def is_allowed(self, request: PermissionRequest) -> bool:
        if request.tool_name in self.always_allow_tools:
            return True
        return any(r.matches(request) for r in self._rules)

    def decide(self, request: PermissionRequest) -> PermissionDecision:
        if self.mode == PermissionMode.ALLOW_ALL:
            return PermissionDecision.allow_once()
        if self.is_allowed(request):
            return PermissionDecision.allow_once()
        if self.mode == PermissionMode.ALLOW_LIST:
            return PermissionDecision.deny(
                f"Tool {request.tool_name!r} is not on the allow list."
            )
        # ASK mode and unmatched.
        if self.on_request is None:
            return PermissionDecision.deny(
                f"Tool {request.tool_name!r} needs permission but no prompter is configured."
            )
        decision = self.on_request(request)
        if decision.kind == "allow_always" and decision.rule is not None:
            self._add_rule(decision.rule)
        return decision

    # ----- PreToolUseHook protocol ---------------------------------------- #

    def __call__(self, ctx: "PreToolUseContext") -> None:
        request = PermissionRequest(
            tool_name=ctx.tool_call.name,
            arguments=dict(ctx.tool_call.arguments),
        )
        decision = self.decide(request)
        if decision.kind == "deny":
            ctx.deny(decision.feedback)
        else:
            ctx.allow()

    # ----- internals ------------------------------------------------------- #

    def _add_rule(self, rule: object) -> AllowRule:
        if isinstance(rule, str):
            rule = AllowRule(tool_name=rule)
        elif isinstance(rule, tuple):
            if len(rule) == 1:
                rule = AllowRule(tool_name=rule[0])
            elif len(rule) == 2:
                rule = AllowRule(tool_name=rule[0], command_prefix=rule[1])
            else:
                raise ValueError(f"Bad rule tuple: {rule!r}")
        if not isinstance(rule, AllowRule):
            raise TypeError(f"Bad allow rule: {rule!r}")
        if rule not in self._rules:
            self._rules.append(rule)
        return rule


__all__ = [
    "AllowRule",
    "DEFAULT_ALWAYS_ALLOW_TOOLS",
    "PermissionCallback",
    "PermissionDecision",
    "PermissionMode",
    "PermissionPolicy",
    "PermissionRequest",
]
