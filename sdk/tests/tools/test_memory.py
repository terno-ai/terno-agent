"""File-based memory: the per-turn context provider and the agent wiring.

Memory works terno-ai style — the agent reads/writes it with the ordinary file
tools, and only the dynamic context (folder paths + MEMORY.md index + session
id) is injected. There are deliberately no memory-specific tools.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import terno_agent.agents.terno as terno_mod
from terno_agent.agents.terno import TernoAgent
from terno_agent.config import Config
from terno_agent.core.messages import AssistantMessage
from terno_agent.llm.base import LLMResponse
from terno_agent.tools.memory import (
    MEMORY_INDEX_FILENAME,
    MemoryContextProvider,
    read_memory_index,
)

_INDEX = """# Memory Index

## Global
- [Active User](active-user.md) — status = 1
"""


# ----- read_memory_index ---------------------------------------------------- #


def test_read_index_missing_returns_none(tmp_path: Path):
    assert read_memory_index(tmp_path / "memory") is None


def test_read_index_returns_contents(tmp_path: Path):
    root = tmp_path / "memory"
    root.mkdir()
    (root / MEMORY_INDEX_FILENAME).write_text(_INDEX, encoding="utf-8")
    assert read_memory_index(root) == _INDEX.strip()


# ----- MemoryContextProvider ------------------------------------------------ #


def test_block_always_names_user_folder_even_when_empty(tmp_path: Path):
    user_root = tmp_path / "user" / "memory"
    block = MemoryContextProvider(user_root).context_block()
    assert str(user_root) in block
    assert "empty — no memories saved yet" in block
    # No org section when no org root is configured.
    assert "Organisation-shared" not in block


def test_block_includes_index_and_session_id(tmp_path: Path):
    user_root = tmp_path / "user" / "memory"
    user_root.mkdir(parents=True)
    (user_root / MEMORY_INDEX_FILENAME).write_text(_INDEX, encoding="utf-8")
    block = MemoryContextProvider(user_root, session_id="sess-42").context_block()
    assert "Active User" in block  # index content surfaced
    assert "currentSessionId: sess-42" in block
    assert "originSessionId" in block  # tells the agent how to stamp it


def test_block_adds_org_section_when_configured(tmp_path: Path):
    user_root = tmp_path / "user" / "memory"
    org_root = tmp_path / "org" / "memory"
    org_root.mkdir(parents=True)
    (org_root / MEMORY_INDEX_FILENAME).write_text(_INDEX, encoding="utf-8")
    block = MemoryContextProvider(user_root, org_root=org_root).context_block()
    assert str(org_root) in block
    assert "read-only unless you are an org admin" in block


# ----- TernoAgent wiring ---------------------------------------------------- #


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
    )
    base.update(over)
    return Config(**base)


@pytest.fixture
def agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TernoAgent:
    monkeypatch.setattr(terno_mod, "create_llm_client", lambda **kw: _FakeLLM())
    return TernoAgent.from_config(_config(), workdir=tmp_path)


def test_agent_uses_generic_file_tools_no_memory_tools(agent: TernoAgent):
    names = set(agent.tools)
    # Memory is read/written with the ordinary file tools.
    assert {"read_file", "write_file", "edit_file", "grep"} <= names
    # There are NO memory-specific tools, and no background curator.
    assert not ({"list_memory", "search_memory", "read_memory",
                 "write_memory", "edit_memory"} & names)
    assert not hasattr(agent, "wiki_memory_agent")


def test_memory_context_attached_and_defaults_under_workdir(
    agent: TernoAgent, tmp_path: Path
):
    assert agent.memory_context is not None
    # With no workspace roots configured, memory falls back to <workdir>/memory.
    assert agent.memory_context.user_root == (tmp_path / "memory").resolve()
    assert ".terno" not in agent.memory_context.user_root.parts


def test_explicit_user_memory_root_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(terno_mod, "create_llm_client", lambda **kw: _FakeLLM())
    target = tmp_path / "users" / "acme" / "ada" / "memory"
    agent = TernoAgent.from_config(
        _config(user_memory_root=str(target)), workdir=tmp_path
    )
    assert agent.memory_context.user_root == target.resolve()


def test_context_block_injected_and_names_memory_folder(agent: TernoAgent):
    block = agent.memory_context.context_block()
    assert "Persistent memory" in block
    assert str(agent.memory_context.user_root) in block
