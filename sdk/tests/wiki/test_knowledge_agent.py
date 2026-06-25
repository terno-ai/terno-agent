"""KnowledgeAgent: the per-turn curator loop decides build/read/write."""

from __future__ import annotations

from pathlib import Path

from terno_agent.core.messages import AssistantMessage, ToolCall
from terno_agent.db.connection import Database
from terno_agent.llm.base import LLMResponse
from terno_agent.okf.agent import KnowledgeAgent
from terno_agent.okf.bundle import KnowledgeBundle
from terno_agent.okf.paths import bundle_dir


class LoopLLM:
    """Drives the agent loop with pre-scripted responses (one per .complete)."""

    model = "loop"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    def complete(self, messages, tools=None, **_kwargs) -> LLMResponse:
        if not self._responses:
            return LLMResponse(
                message=AssistantMessage(content="done"), stop_reason="end_turn"
            )
        return self._responses.pop(0)


def _tool_call(name: str, args: dict) -> LLMResponse:
    return LLMResponse(
        message=AssistantMessage(
            content="", tool_calls=[ToolCall(id="1", name=name, arguments=args)]
        ),
        stop_reason="tool_use",
    )


def _final(text: str) -> LLMResponse:
    return LLMResponse(message=AssistantMessage(content=text), stop_reason="end_turn")


def test_run_turn_builds_when_missing(sqlite_db: Database, workdir: Path):
    llm = LoopLLM(
        [
            _tool_call("build_datasource_knowledge", {"datasource": "sales_db"}),
            _final("built the bundle"),
        ]
    )
    ka = KnowledgeAgent(
        llm=llm, workdir=workdir, datasource="sales_db", db=sqlite_db, enrich=False
    )
    ka.run_turn("how many active users?")

    bundle = KnowledgeBundle(bundle_dir(workdir, "sales_db"), name="sales_db")
    assert bundle.exists()
    assert bundle.read_concept("tables/users") is not None


def test_run_turn_can_write_a_concept(sqlite_db: Database, workdir: Path):
    llm = LoopLLM(
        [
            _tool_call(
                "write_concept",
                {
                    "datasource": "sales_db",
                    "concept_id": "concepts/active_user",
                    "title": "Active user",
                    "type": "metric",
                    "summary": "users.status = 1",
                    "body": "An active user has status = 1.",
                },
            ),
            _final("captured the metric"),
        ]
    )
    ka = KnowledgeAgent(
        llm=llm, workdir=workdir, datasource="sales_db", db=sqlite_db, enrich=False
    )
    ka.run_turn("define active user as status = 1")

    bundle = KnowledgeBundle(bundle_dir(workdir, "sales_db"), name="sales_db")
    concept = bundle.read_concept("concepts/active_user")
    assert concept is not None and concept.type == "metric"
    assert "status = 1" in concept.body


def test_run_turn_noop_leaves_no_bundle(sqlite_db: Database, workdir: Path):
    llm = LoopLLM([_final("no changes needed")])
    ka = KnowledgeAgent(
        llm=llm, workdir=workdir, datasource="sales_db", db=sqlite_db, enrich=False
    )
    ka.run_turn("hi there")
    bundle = KnowledgeBundle(bundle_dir(workdir, "sales_db"), name="sales_db")
    assert not bundle.exists()


def test_fresh_history_each_turn(sqlite_db: Database, workdir: Path):
    llm = LoopLLM([_final("a"), _final("b")])
    ka = KnowledgeAgent(
        llm=llm, workdir=workdir, datasource="sales_db", db=sqlite_db, enrich=False
    )
    ka.run_turn("first")
    ka.run_turn("second")
    # History was cleared between turns: system + one user + one assistant.
    assert len(ka._agent.history) == 3
