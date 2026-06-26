"""TernoAgent wiring: knowledge tools exposed + context injected."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa

import terno_agent.agents.terno as terno_mod
from terno_agent.agents.terno import TernoAgent, _derive_datasource_name
from terno_agent.config import Config
from terno_agent.core.messages import AssistantMessage
from terno_agent.llm.base import LLMResponse


class _FakeLLM:
    model = "fake"

    def complete(self, messages, tools=None, **_kwargs) -> LLMResponse:
        return LLMResponse(
            message=AssistantMessage(content="ok"), stop_reason="end_turn"
        )


def _config(url: str) -> Config:
    return Config(
        llm_provider="anthropic",
        llm_api_key="x",
        database_url=url,
        sandbox="none",
        sandbox_fallback="none",
        memory_enabled=False,
        mcp_enabled=False,
        attachments_enabled=False,
        skills_enabled=False,
        knowledge_datasource="sales_db",
    )


@pytest.fixture
def agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TernoAgent:
    url = f"sqlite:///{tmp_path / 'sales.db'}"
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)"))
    monkeypatch.setattr(terno_mod, "create_llm_client", lambda **kw: _FakeLLM())
    return TernoAgent.from_config(_config(url), workdir=tmp_path)


def test_derive_datasource_name():
    assert _derive_datasource_name("postgresql://u:p@h/mydb") == "mydb"
    assert _derive_datasource_name("") == "datasource"


def test_knowledge_agent_attached_not_as_tools(agent: TernoAgent):
    # The knowledge agent is a separate per-turn curator, NOT a main-agent tool.
    assert agent.knowledge_agent is not None
    assert agent.knowledge_datasource == "sales_db"
    names = set(agent.tools)
    assert not (
        {"build_datasource_knowledge", "read_concept", "list_datasource_knowledge",
         "write_concept"}
        & names
    )


def test_run_invokes_knowledge_agent_each_turn(
    agent: TernoAgent, monkeypatch: pytest.MonkeyPatch
):
    seen: list[str] = []
    monkeypatch.setattr(
        agent.knowledge_agent, "run_turn", lambda task: seen.append(task)
    )
    agent.run("how many users are active?")
    assert seen == ["how many users are active?"]


def test_knowledge_failure_never_breaks_main_turn(
    agent: TernoAgent, monkeypatch: pytest.MonkeyPatch
):
    def boom(_task):
        raise RuntimeError("knowledge exploded")

    monkeypatch.setattr(agent.knowledge_agent, "run_turn", boom)
    run = agent.run("hello")  # must still complete
    assert run.answer == "ok"


def test_context_injected_when_bundle_exists(agent: TernoAgent):
    from terno_agent.wiki.builder import DatasourceKnowledgeAgent
    from terno_agent.wiki.bundle import KnowledgeBundle
    from terno_agent.wiki.paths import bundle_dir

    assert agent.knowledge_provider.context_block() == ""  # nothing built yet
    bundle = KnowledgeBundle(
        bundle_dir(agent.workdir, "sales_db"), name="sales_db"
    )
    DatasourceKnowledgeAgent(db=agent.db, bundle=bundle, llm=None).build()
    block = agent.knowledge_provider.context_block()
    assert "sales_db" in block and "users" in block


def test_disabled_when_no_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Knowledge stays enabled (read/curate) even without a live DB; the agent
    # exists but its build tool will report no datasource if it tries to build.
    monkeypatch.setattr(terno_mod, "create_llm_client", lambda **kw: _FakeLLM())
    agent = TernoAgent.from_config(_config(""), workdir=tmp_path)
    assert agent.knowledge_agent is not None
    assert agent.db is None
