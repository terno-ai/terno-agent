"""TernoAgent wiring: read-only memory tools exposed, curator runs each turn."""

from __future__ import annotations

from pathlib import Path

import pytest

import terno_agent.agents.terno as terno_mod
from terno_agent.agents.terno import TernoAgent
from terno_agent.config import Config
from terno_agent.core.messages import AssistantMessage
from terno_agent.llm.base import LLMResponse


class _FakeLLM:
    model = "fake"

    def complete(self, messages, tools=None, **_kwargs) -> LLMResponse:
        return LLMResponse(
            message=AssistantMessage(content="ok"), stop_reason="end_turn"
        )


def _config(**over) -> Config:
    base = dict(
        llm_provider="anthropic",
        llm_api_key="x",
        database_url="",
        sandbox="none",
        sandbox_fallback="none",
        memory_enabled=False,
        mcp_enabled=False,
        attachments_enabled=False,
        skills_enabled=False,
        wiki_datasource="sales_db",
    )
    base.update(over)
    return Config(**base)


@pytest.fixture
def agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TernoAgent:
    monkeypatch.setattr(terno_mod, "create_llm_client", lambda **kw: _FakeLLM())
    return TernoAgent.from_config(_config(), workdir=tmp_path)


def test_main_agent_gets_readonly_memory_tools_only(agent: TernoAgent):
    names = set(agent.tools)
    assert {"list_memory", "search_memory", "read_memory"} <= names
    # The curator is the only writer — write tools are NOT on the main agent.
    assert not ({"write_memory", "edit_memory"} & names)


def test_curator_and_context_attached(agent: TernoAgent):
    assert agent.wiki_memory_agent is not None
    assert agent.wiki_memory_context is not None


def test_memory_stored_in_workspace_memory_folder_not_terno(
    agent: TernoAgent, tmp_path: Path
):
    # With no workspace roots configured, memory falls back to <workdir>/memory.
    user_root = agent.wiki_memory_context.user_root
    assert user_root == (tmp_path / "memory").resolve()
    assert ".terno" not in user_root.parts


def test_explicit_user_memory_root_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(terno_mod, "create_llm_client", lambda **kw: _FakeLLM())
    target = tmp_path / "users" / "acme" / "ada" / "memory"
    agent = TernoAgent.from_config(
        _config(user_memory_root=str(target)), workdir=tmp_path
    )
    assert agent.wiki_memory_context.user_root == target.resolve()


def test_run_invokes_curator_each_turn(
    agent: TernoAgent, monkeypatch: pytest.MonkeyPatch
):
    seen: list[str] = []
    monkeypatch.setattr(
        agent.wiki_memory_agent,
        "curate_async",
        lambda task, **kw: seen.append(task),
    )
    agent.run("how many users are active?")
    assert seen == ["how many users are active?"]


def test_curator_failure_never_breaks_main_turn(
    agent: TernoAgent, monkeypatch: pytest.MonkeyPatch
):
    def boom(task, **kw):
        raise RuntimeError("curation exploded")

    monkeypatch.setattr(agent.wiki_memory_agent, "curate_async", boom)
    run = agent.run("hello")  # must still complete
    assert run.answer == "ok"


def test_context_injected_when_memory_exists(agent: TernoAgent):
    from terno_agent.wiki.tools import MemoryWriteTool

    assert agent.wiki_memory_context.context_block() == ""  # nothing yet
    MemoryWriteTool(agent.wiki_memory_context.user_root).run(
        datasource="sales_db",
        memory_id="metrics/active_user",
        title="Active user",
        type="metric",
        scope="global",
        summary="status = 1",
    )
    block = agent.wiki_memory_context.context_block()
    assert "sales_db" in block and "Active user" in block
