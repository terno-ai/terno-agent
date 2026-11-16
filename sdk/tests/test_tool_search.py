"""Deferred MCP tools behind ``tool_search``."""

from __future__ import annotations

from dataclasses import dataclass

from terno_agent.agents.terno import TernoAgent
from terno_agent.core.messages import ToolCall
from terno_agent.core.tool import ToolSchema
from terno_agent.tools.tool_search import TOOL_SEARCH_NAME


class _DummyLLM:
    model = "dummy"

    def complete(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("LLM should not be invoked in this test")


@dataclass
class _FakeMcpTool:
    server: str
    tool_name: str
    description: str

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=f"mcp__{self.server}__{self.tool_name}",
            description=f"[mcp:{self.server}] {self.description}",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        )

    def run(self, **kwargs):
        return f"{self.tool_name} ran with {kwargs}"


class _FakeManager:
    def __init__(self, tools):
        self._tools = tools

    def tools(self):
        return list(self._tools)


def _manager():
    return _FakeManager(
        [
            _FakeMcpTool("github", "create_issue", "Create a GitHub issue in a repo"),
            _FakeMcpTool("github", "list_pulls", "List pull requests"),
            _FakeMcpTool("slack", "post_message", "Post a message to a Slack channel"),
        ]
    )


def _agent(**kw):
    return TernoAgent(_DummyLLM(), mcp_manager=_manager(), **kw)


def test_default_sends_all_mcp_tools():
    agent = _agent()
    assert "mcp__github__create_issue" in agent.tools
    assert TOOL_SEARCH_NAME not in agent.tools
    assert not agent.deferred_tools


def test_below_threshold_is_not_deferred():
    agent = _agent(defer_mcp_tools=True, defer_mcp_min_tokens=10_000_000)
    assert "mcp__slack__post_message" in agent.tools
    assert TOOL_SEARCH_NAME not in agent.tools


def test_deferred_tools_hidden_and_indexed():
    agent = _agent(defer_mcp_tools=True, defer_mcp_min_tokens=0)
    assert TOOL_SEARCH_NAME in agent.tools
    assert not any(n.startswith("mcp__") for n in agent.tools)
    assert set(agent.deferred_tools) == {
        "mcp__github__create_issue",
        "mcp__github__list_pulls",
        "mcp__slack__post_message",
    }
    assert "Deferred tools" in agent.system_prompt
    assert "- slack: mcp__slack__post_message" in agent.system_prompt


def test_select_loads_exact_names_and_appends():
    agent = _agent(defer_mcp_tools=True, defer_mcp_min_tokens=0)
    before = list(agent.tools)
    out = agent.tools[TOOL_SEARCH_NAME].run(query="select:mcp__slack__post_message,nope")
    assert "mcp__slack__post_message" in out and "Not found: nope" in out
    assert list(agent.tools)[: len(before)] == before  # order kept, appended
    assert list(agent.tools)[-1] == "mcp__slack__post_message"
    assert "mcp__slack__post_message" not in agent.deferred_tools


def test_server_query_loads_whole_server():
    agent = _agent(defer_mcp_tools=True, defer_mcp_min_tokens=0)
    agent.tools[TOOL_SEARCH_NAME].run(query="server:github")
    assert "mcp__github__create_issue" in agent.tools
    assert "mcp__github__list_pulls" in agent.tools
    assert "mcp__slack__post_message" in agent.deferred_tools


def test_keyword_search_ranks_and_limits():
    agent = _agent(defer_mcp_tools=True, defer_mcp_min_tokens=0)
    out = agent.tools[TOOL_SEARCH_NAME].run(query="create issue", max_results=1)
    assert "mcp__github__create_issue" in out
    assert "mcp__github__create_issue" in agent.tools
    assert "mcp__github__list_pulls" in agent.deferred_tools


def test_no_match_message():
    agent = _agent(defer_mcp_tools=True, defer_mcp_min_tokens=0)
    out = agent.tools[TOOL_SEARCH_NAME].run(query="zzzz")
    assert "No deferred tools matched" in out


def test_direct_call_to_deferred_tool_autoloads():
    agent = _agent(defer_mcp_tools=True, defer_mcp_min_tokens=0)
    result = agent._guarded_tool_call(
        ToolCall(id="1", name="mcp__slack__post_message", arguments={"x": "hi"})
    )
    assert not result.is_error
    assert "post_message ran" in result.content
    assert "mcp__slack__post_message" in agent.tools


def test_subagent_inherits_deferral():
    agent = _agent(defer_mcp_tools=True, defer_mcp_min_tokens=0)
    spawn = agent.tools["spawn_agent"]
    assert spawn.defer_mcp_tools is True
    assert spawn.defer_mcp_min_tokens == 0


def test_max_tools_caps_loading():
    agent = _agent(defer_mcp_tools=True, defer_mcp_min_tokens=0)
    agent.max_tools = len(agent.tools) + 1
    out = agent.tools[TOOL_SEARCH_NAME].run(query="server:github")
    assert "tool limit reached" in out
    assert sum(n.startswith("mcp__github") for n in agent.tools) == 1
    result = agent._guarded_tool_call(
        ToolCall(id="2", name="mcp__slack__post_message", arguments={})
    )
    assert result.is_error and "tool limit" in result.content
