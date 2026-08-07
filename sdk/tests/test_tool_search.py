"""Deferred tools and ToolSearch."""

from __future__ import annotations

import json
import re

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema
from terno_agent.tools.tool_search import (
    DeferredToolPlaceholderTool,
    DeferredToolRegistry,
    ToolSearchTool,
    roster_text,
)


class _Fake:
    def __init__(self, name: str, description: str = "") -> None:
        self._name = name
        self._description = description

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description=self._description,
            parameters={"type": "object", "properties": {}},
        )

    def run(self, **_kw: object) -> str:
        return f"ran {self._name}"


def _registry(**tools: str) -> DeferredToolRegistry:
    return DeferredToolRegistry(
        deferred={n: _Fake(n, d) for n, d in tools.items()}, active={}
    )


def _names(block: str) -> list[str]:
    return re.findall(r'"name": "(\w+)"', block)


# ----- the roster ---------------------------------------------------------- #


def test_roster_lists_names_sorted_with_the_captured_header() -> None:
    out = roster_text(["WebSearch", "Monitor"])
    lines = out.splitlines()

    assert "deferred tools" in lines[0]
    assert "InputValidationError" in lines[0]
    assert "select:" in lines[0]
    assert lines[1:] == ["Monitor", "WebSearch"]


def test_roster_is_static_and_keeps_loaded_tools() -> None:
    # Verified against the capture: a loaded tool appears in BOTH the roster and
    # the tools array; the roster is written once and never filtered.
    reg = _registry(Monitor="watch", WebSearch="search")
    reg.load("Monitor")

    assert reg.names == ["Monitor", "WebSearch"]


# ----- query forms --------------------------------------------------------- #


def test_select_fetches_exact_names() -> None:
    reg = _registry(Monitor="watch", WebSearch="search", WebFetch="fetch")
    block = ToolSearchTool(reg).run(query="select:WebFetch,Monitor", max_results=5)

    assert _names(block) == ["WebFetch", "Monitor"]  # requested order preserved
    assert reg.loaded == {"WebFetch", "Monitor"}


def test_select_ignores_max_results() -> None:
    # The caller named exactly what it wants; truncating would silently drop it.
    reg = _registry(A="a", B="b", C="c")
    block = ToolSearchTool(reg).run(query="select:A,B,C", max_results=1)

    assert len(_names(block)) == 3


def test_select_skips_unknown_names_without_failing() -> None:
    reg = _registry(Monitor="watch")
    block = ToolSearchTool(reg).run(query="select:Monitor,Nope", max_results=5)

    assert _names(block) == ["Monitor"]
    assert "Nope" not in reg.loaded


def test_keyword_search_ranks_name_matches_above_description() -> None:
    reg = _registry(
        NotebookEdit="edit a jupyter notebook cell",
        WebFetch="fetch a url; unrelated to notebooks",
    )
    block = ToolSearchTool(reg).run(query="notebook jupyter", max_results=5)

    assert _names(block)[0] == "NotebookEdit"


def test_keyword_search_respects_max_results() -> None:
    reg = _registry(A="alpha task", B="beta task", C="gamma task")
    assert len(_names(ToolSearchTool(reg).run(query="task", max_results=2))) == 2


def test_plus_prefix_requires_the_term_in_the_name() -> None:
    reg = _registry(
        TaskCreate="create a task", WebSearch="search the web for a task"
    )
    block = ToolSearchTool(reg).run(query="+task create", max_results=5)

    # WebSearch mentions "task" in its description but not its name.
    assert _names(block) == ["TaskCreate"]


def test_a_bare_plus_query_still_matches() -> None:
    reg = _registry(TaskCreate="c", TaskList="l", WebFetch="f")
    block = ToolSearchTool(reg).run(query="+task", max_results=5)

    assert sorted(_names(block)) == ["TaskCreate", "TaskList"]


def test_no_match_reports_what_is_available() -> None:
    reg = _registry(Monitor="watch")
    out = ToolSearchTool(reg).run(query="zzzz", max_results=5)

    assert "No deferred tools matched" in out
    assert "Monitor" in out


def test_empty_query_is_an_error() -> None:
    with pytest.raises(ToolError, match="non-empty"):
        ToolSearchTool(_registry(A="a")).run(query="  ", max_results=5)


# ----- result encoding ----------------------------------------------------- #


def test_result_is_a_functions_block_of_one_line_per_tool() -> None:
    reg = _registry(Monitor="watch things")
    block = ToolSearchTool(reg).run(query="select:Monitor", max_results=5)

    assert block.startswith("<functions>\n")
    assert block.endswith("\n</functions>")
    inner = block.split("\n")[1]
    payload = json.loads(inner[len("<function>") : -len("</function>")])
    # Same encoding as the eager tool list: description / name / parameters.
    assert set(payload) == {"description", "name", "parameters"}
    assert payload["name"] == "Monitor"
    assert payload["description"] == "watch things"


# ----- loading makes a tool callable --------------------------------------- #


def test_loading_inserts_the_tool_into_the_live_dispatch_map() -> None:
    active: dict[str, object] = {}
    reg = DeferredToolRegistry(deferred={"Monitor": _Fake("Monitor")}, active=active)

    assert "Monitor" not in active
    ToolSearchTool(reg).run(query="select:Monitor", max_results=5)
    # Callable on the very next iteration, not the next turn.
    assert "Monitor" in active


def test_loading_is_idempotent() -> None:
    reg = _registry(Monitor="watch")
    ToolSearchTool(reg).run(query="select:Monitor", max_results=5)
    ToolSearchTool(reg).run(query="select:Monitor", max_results=5)

    assert reg.loaded == {"Monitor"}


# ----- the placeholder ----------------------------------------------------- #


def test_placeholder_is_advertised_but_not_callable() -> None:
    tool = DeferredToolPlaceholderTool()

    assert tool.schema.name == "DeferredToolPlaceholder"
    assert "never call this tool" in tool.schema.description
    with pytest.raises(ToolError, match="not callable"):
        tool.run()


# ----- agent integration --------------------------------------------------- #


def _agent(**kwargs):
    from terno_agent.agents.terno import TernoAgent
    from terno_agent.core.messages import AssistantMessage
    from terno_agent.llm.base import LLMResponse

    class _LLM:
        model = "dummy"

        def complete(self, *_a, **_k):
            return LLMResponse(
                message=AssistantMessage(content="x"), stop_reason="stop"
            )

    return TernoAgent(llm=_LLM(), **kwargs)


def test_agent_defers_the_default_set_and_adds_the_machinery() -> None:
    agent = _agent()

    assert "ToolSearch" in agent.tools
    assert "DeferredToolPlaceholder" in agent.tools
    # Core tools stay eager.
    for name in ("Read", "Write", "Edit", "Bash", "Agent"):
        assert name in agent.tools
    assert "WebSearch" in agent.tool_registry.names
    assert "WebSearch" not in agent.tools


def test_defer_tools_empty_disables_the_whole_mechanism() -> None:
    agent = _agent(defer_tools=())

    assert agent.tool_registry is None
    assert "ToolSearch" not in agent.tools
    assert "DeferredToolPlaceholder" not in agent.tools
    assert "WebSearch" in agent.tools


def test_defer_tools_accepts_an_explicit_set() -> None:
    agent = _agent(defer_tools={"WebSearch"})

    assert agent.tool_registry.names == ["WebSearch"]
    assert "WebFetch" in agent.tools  # no longer deferred


def test_roster_is_injected_after_the_first_user_message() -> None:
    from terno_agent.core.messages import SystemMessage, UserMessage

    agent = _agent()
    agent.run("hello")

    # [0] system prompt, [1] the user turn, [2] the roster.
    assert isinstance(agent.history[1], UserMessage)
    assert isinstance(agent.history[2], SystemMessage)
    assert "deferred tools" in agent.history[2].content


def test_roster_is_not_repeated_on_later_turns() -> None:
    from terno_agent.core.messages import SystemMessage

    agent = _agent()
    agent.run("first")
    agent.run("second")

    rosters = [
        m
        for m in agent.history
        if isinstance(m, SystemMessage) and "deferred tools" in m.content
    ]
    assert len(rosters) == 1


def test_a_loaded_tool_becomes_callable_on_the_agent() -> None:
    agent = _agent()
    assert "WebSearch" not in agent.tools

    agent.tools["ToolSearch"].run(query="select:WebSearch", max_results=5)

    assert "WebSearch" in agent.tools
    assert agent.tools["WebSearch"].schema.name == "WebSearch"
