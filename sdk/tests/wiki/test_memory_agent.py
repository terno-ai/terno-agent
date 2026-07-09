"""MemoryAgent: the per-turn background curator writes to workspace memory."""

from __future__ import annotations

from pathlib import Path

from terno_agent.core.messages import AssistantMessage, ToolCall
from terno_agent.llm.base import LLMResponse
from terno_agent.wiki.agent import MemoryAgent
from terno_agent.wiki.bundle import KnowledgeBundle


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
    return LLMResponse(
        message=AssistantMessage(content=text), stop_reason="end_turn"
    )


def _write_call(**over) -> LLMResponse:
    args = dict(
        memory_id="active_user",
        title="Active user",
        type="metric",
        scope="datasource:1",
        datasource_name="sales_db",
        body="An active user has status = 1.",
    )
    args.update(over)
    return _tool_call("write_memory", args)


def test_curator_writes_private_memory_into_workspace_not_terno(tmp_path: Path):
    user_root = tmp_path / "user_workspace" / "memory"
    agent = MemoryAgent(
        llm=LoopLLM([_write_call(), _final("recorded the metric")]),
        user_root=user_root,
        datasource="memory",
    )
    agent.run_turn("define active user", assistant_answer="status = 1")

    bundle = KnowledgeBundle(user_root, name="memory")
    concept = bundle.read_concept("active_user")
    assert concept is not None and concept.type == "metric"
    assert "status = 1" in concept.body
    # Landed directly under the memory folder — no `.terno`, no nested subdir.
    written = user_root / "active_user.md"
    assert written.exists()
    assert ".terno" not in written.parts
    assert not (tmp_path / "org_workspace").exists()


def test_curator_noop_leaves_no_bundle(tmp_path: Path):
    user_root = tmp_path / "user_workspace" / "memory"
    agent = MemoryAgent(
        llm=LoopLLM([_final("no changes needed")]),
        user_root=user_root,
        datasource="memory",
    )
    agent.run_turn("hi there")
    assert not KnowledgeBundle(user_root, name="memory").exists()


def test_curator_shared_write_refused_without_admin(tmp_path: Path):
    user_root = tmp_path / "user_workspace" / "memory"
    org_root = tmp_path / "org_workspace" / "memory"
    # A non-admin curator that tries a shared write gets a tool error; the loop
    # continues and nothing lands in the org folder.
    agent = MemoryAgent(
        llm=LoopLLM([_write_call(shared=True), _final("fell back")]),
        user_root=user_root,
        datasource="memory",
        org_root=org_root,
        is_org_admin=False,
    )
    agent.run_turn("try to share org memory")
    assert not org_root.exists()


def test_curator_admin_can_write_shared(tmp_path: Path):
    user_root = tmp_path / "user_workspace" / "memory"
    org_root = tmp_path / "org_workspace" / "memory"
    agent = MemoryAgent(
        llm=LoopLLM([_write_call(shared=True), _final("shared it")]),
        user_root=user_root,
        datasource="memory",
        org_root=org_root,
        is_org_admin=True,
        session_id="sess-42",
    )
    agent.run_turn("record the org metric")
    concept = KnowledgeBundle(org_root, name="memory").read_concept(
        "active_user"
    )
    assert concept is not None
    # session_id is stamped as provenance.
    assert concept.metadata.get("originSessionId") == "sess-42"


def test_fresh_history_each_turn(tmp_path: Path):
    agent = MemoryAgent(
        llm=LoopLLM([_final("a"), _final("b")]),
        user_root=tmp_path / "memory",
        datasource="memory",
    )
    agent.run_turn("first")
    agent.run_turn("second")
    # History cleared between turns: system + one user + one assistant.
    assert len(agent._agent.history) == 3
