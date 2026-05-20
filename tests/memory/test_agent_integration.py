"""End-to-end wiring: a scripted LLM + memory pipeline through TernoAgent."""

from __future__ import annotations

from pathlib import Path

from terno_agent.agents.terno import TernoAgent
from terno_agent.core.messages import AssistantMessage, Message, UserMessage
from terno_agent.llm.base import LLMResponse
from terno_agent.memory.extractor import MemoryExtractor
from terno_agent.memory.retriever import MemoryRetriever
from terno_agent.memory.store import MemoryStore
from terno_agent.memory.types import MemoryEntry, MemoryType


class _OneShotLLM:
    """Returns a single final assistant message — no tool calls."""

    model = "scripted"

    def __init__(self, answer: str = "ok") -> None:
        self.answer = answer
        self.seen_systems: list[str] = []

    def complete(
        self,
        messages: list[Message],
        tools=None,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        on_text_delta=None,
    ) -> LLMResponse:
        # Capture the system prompt so the test can assert recall injection.
        system = next((m.content for m in messages if m.__class__.__name__ == "SystemMessage"), "")
        self.seen_systems.append(system)
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


def test_memory_retriever_injects_into_system_prompt(
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
    # First (and only) LLM call should have seen the recalled memory.
    assert any("Relevant memories" in s for s in llm.seen_systems)
    assert any("user-role" in s for s in llm.seen_systems)


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

    extractor = MemoryExtractor(
        llm=_BadLLM(),  # type: ignore[arg-type]
        store=store,
        workdir=workdir,
        wait=True,
    )
    # Should NOT raise even though the LLM blows up.
    extractor.extract(
        [UserMessage("hi"), AssistantMessage(content="hello")]
    )
