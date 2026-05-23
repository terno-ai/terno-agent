"""End-to-end wiring: a scripted LLM + memory pipeline through TernoAgent."""

from __future__ import annotations

from pathlib import Path

from terno_agent.agents.terno import TernoAgent
from terno_agent.core.messages import AssistantMessage, Message, UserMessage
from terno_agent.llm.base import LLMResponse
from terno_agent.memory.extractor import ExtractionResult, MemoryExtractor
from terno_agent.memory.retriever import MemoryRetriever
from terno_agent.memory.store import MemoryStore
from terno_agent.memory.types import MemoryEntry, MemoryType


class _OneShotLLM:
    """Returns a single final assistant message — no tool calls."""

    model = "scripted"

    def __init__(self, answer: str = "ok") -> None:
        self.answer = answer
        self.seen_systems: list[str] = []
        self.seen_users: list[str] = []

    def complete(
        self,
        messages: list[Message],
        tools=None,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        on_text_delta=None,
    ) -> LLMResponse:
        # Capture system + user content so the test can inspect recall injection
        # (which now rides on the user message rather than the system prompt).
        system = next((m.content for m in messages if m.__class__.__name__ == "SystemMessage"), "")
        user = next(
            (m.content for m in reversed(messages) if m.__class__.__name__ == "UserMessage"),
            "",
        )
        self.seen_systems.append(system)
        self.seen_users.append(user)
        return LLMResponse(
            message=AssistantMessage(content=self.answer),
            stop_reason="end_turn",
        )


def test_search_memory_tool_registered_when_store_present(
    isolated_memory_dirs: Path, stub_embedder
) -> None:
    workdir = isolated_memory_dirs / "p"
    workdir.mkdir()
    store = MemoryStore(workdir, stub_embedder)
    llm = _OneShotLLM()
    agent = TernoAgent(llm, workdir=workdir, memory_store=store)
    assert "search_memory" in agent.tools


def test_memory_retriever_injects_recall_into_user_turn(
    isolated_memory_dirs: Path, stub_embedder
) -> None:
    workdir = isolated_memory_dirs / "p"
    workdir.mkdir()
    store = MemoryStore(workdir, stub_embedder)
    store.save(
        MemoryEntry(
            name="user-role",
            description="data engineer focused on Snowflake",
            type=MemoryType.USER,
            body="User is a senior data engineer.",
        )
    )
    retriever = MemoryRetriever(store=store, k=3)

    llm = _OneShotLLM(answer="hello")
    agent = TernoAgent(
        llm,
        workdir=workdir,
        memory_store=store,
        memory_retriever=retriever,
    )
    result = agent.run("Tell me about my role")
    assert result.answer == "hello"
    # Recalled memory now rides on the user message (extra_context is per-turn,
    # not part of the persistent system prompt).
    assert any("Relevant memories" in u for u in llm.seen_users)
    assert any("user-role" in u for u in llm.seen_users)


def test_post_turn_hook_fires(isolated_memory_dirs: Path, stub_embedder) -> None:
    workdir = isolated_memory_dirs / "p"
    workdir.mkdir()
    store = MemoryStore(workdir, stub_embedder)

    calls: list[int] = []

    class _RecordingExtractor:
        def extract(self, trace) -> None:
            calls.append(len(trace))

    llm = _OneShotLLM(answer="done")
    agent = TernoAgent(
        llm,
        workdir=workdir,
        memory_store=store,
        memory_extractor=_RecordingExtractor(),  # type: ignore[arg-type]
    )
    agent.run("hi")
    assert len(calls) == 1
    assert calls[0] >= 2  # at minimum system + user + assistant


def test_extractor_swallows_errors(
    isolated_memory_dirs: Path, stub_embedder
) -> None:
    workdir = isolated_memory_dirs / "p"
    workdir.mkdir()
    store = MemoryStore(workdir, stub_embedder)

    class _BadLLM:
        model = "bad"

        def complete(self, *_args, **_kwargs):
            raise RuntimeError("network down")

    received: list[ExtractionResult] = []
    extractor = MemoryExtractor(
        llm=_BadLLM(),  # type: ignore[arg-type]
        store=store,
        workdir=workdir,
        on_complete=received.append,
        wait=True,
    )
    # Should NOT raise even though the LLM blows up.
    extractor.extract(
        [UserMessage("hi"), AssistantMessage(content="hello")]
    )
    assert received and received[0].failed


def test_extractor_signals_on_complete_with_counts(
    isolated_memory_dirs: Path, stub_embedder
) -> None:
    """When the subagent calls save_memory successfully, the result counts it."""
    from terno_agent.core.messages import ToolCall

    workdir = isolated_memory_dirs / "p"
    workdir.mkdir()
    store = MemoryStore(workdir, stub_embedder)

    class _ScriptedLLM:
        """Issues one save_memory tool call, then a final text response."""

        model = "scripted"

        def __init__(self) -> None:
            self.turns = 0

        def complete(self, *_args, **_kwargs):
            self.turns += 1
            if self.turns == 1:
                return LLMResponse(
                    message=AssistantMessage(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="t1",
                                name="save_memory",
                                arguments={
                                    "name": "insight-prod-db",
                                    "description": "prod DB host",
                                    "type": "insight",
                                    "body": "db.terno-prod.us-east-1",
                                },
                            )
                        ],
                    ),
                    stop_reason="tool_use",
                )
            return LLMResponse(
                message=AssistantMessage(content="Saved 1 memory."),
                stop_reason="end_turn",
            )

    received: list[ExtractionResult] = []
    extractor = MemoryExtractor(
        llm=_ScriptedLLM(),  # type: ignore[arg-type]
        store=store,
        workdir=workdir,
        on_complete=received.append,
        wait=True,
    )
    extractor.extract(
        [
            UserMessage("Where's prod DB?"),
            AssistantMessage(content="db.terno-prod.us-east-1"),
        ]
    )

    assert len(received) == 1
    assert received[0].saved == 1
    assert received[0].changed is True
    # The insight memory actually landed in the store.
    fetched = store.read("insight-prod-db")
    assert fetched is not None
    assert fetched.type.value == "insight"
    assert "db.terno-prod" in fetched.body
