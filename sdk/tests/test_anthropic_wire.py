"""Wire-shape tests for the Anthropic client.

These pin the request payload to the shape captured from the reference harness:
a multi-block `system` carrying cache breakpoints, and no sampling params unless
a caller asks for them.
"""

from __future__ import annotations

from typing import Any

import pytest

from terno_agent.core.messages import (
    SystemBlock,
    SystemMessage,
    UserMessage,
)
from terno_agent.llm.anthropic_client import _split_system, system_block_to_anthropic


def test_block_without_cache_control_omits_the_key() -> None:
    assert system_block_to_anthropic(SystemBlock("hi")) == {"type": "text", "text": "hi"}


def test_block_carries_its_cache_breakpoint() -> None:
    cc = {"type": "ephemeral", "ttl": "1h", "scope": "global"}
    out = system_block_to_anthropic(SystemBlock("hi", cache_control=cc))

    assert out == {"type": "text", "text": "hi", "cache_control": cc}
    # Copied, so a caller mutating the payload can't corrupt the block.
    out["cache_control"]["ttl"] = "5m"
    assert cc["ttl"] == "1h"


def test_plain_system_messages_still_join_into_one_string() -> None:
    system, rest = _split_system(
        [SystemMessage("a"), SystemMessage("b"), UserMessage("hello")]
    )

    assert system == "a\n\nb"
    assert [type(m) for m in rest] == [UserMessage]


def test_blocks_are_sent_when_any_carries_a_breakpoint() -> None:
    msg = SystemMessage(
        "flat",
        blocks=[
            SystemBlock("identity"),
            SystemBlock("core", cache_control={"type": "ephemeral", "ttl": "1h"}),
        ],
    )

    system, _ = _split_system([msg, UserMessage("hi")])

    assert isinstance(system, list)
    assert [b["text"] for b in system] == ["identity", "core"]
    assert "cache_control" not in system[0]
    assert system[1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_a_later_uncached_system_message_becomes_a_trailing_block() -> None:
    # This is what compaction does: it keeps the block-carrying system message
    # and inserts a summary as a second, plain one.
    blocks = [SystemBlock("core", cache_control={"type": "ephemeral", "ttl": "1h"})]
    system, _ = _split_system(
        [SystemMessage("core", blocks=blocks), SystemMessage("summary: ...")]
    )

    assert isinstance(system, list)
    assert [b["text"] for b in system] == ["core", "summary: ..."]
    assert "cache_control" not in system[1]


# ----- request payload ---------------------------------------------------- #


class _FakeStream:
    def __init__(self, sink: dict[str, Any]) -> None:
        self._sink = sink

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    @property
    def text_stream(self):
        return iter(())

    def get_final_message(self):
        class _Msg:
            content: list[Any] = []
            stop_reason = "end_turn"
            usage = None

        return _Msg()


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    """An AnthropicClient whose transport records the request kwargs."""
    anthropic = pytest.importorskip("anthropic")
    captured: dict[str, Any] = {}

    class _Messages:
        def stream(self, **kwargs: Any) -> _FakeStream:
            captured.clear()
            captured.update(kwargs)
            return _FakeStream(captured)

    class _Client:
        messages = _Messages()

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: _Client())

    from terno_agent.llm.anthropic_client import AnthropicClient

    def build(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
        return AnthropicClient(model="m", api_key="k", **kwargs), captured

    return build


def test_no_sampling_params_by_default(client) -> None:
    llm, captured = client()
    llm.complete([SystemMessage("s"), UserMessage("u")])

    # The reference harness sends none of these; the model default applies.
    for key in ("temperature", "top_p", "top_k", "tool_choice", "thinking"):
        assert key not in captured
    assert "output_config" not in captured
    assert "context_management" not in captured


def test_explicit_temperature_is_forwarded(client) -> None:
    llm, captured = client()
    llm.complete([SystemMessage("s"), UserMessage("u")], temperature=0.2)

    assert captured["temperature"] == 0.2


def test_opt_in_knobs_match_the_captured_values(client) -> None:
    llm, captured = client(effort="medium", thinking="adaptive", clear_thinking=True)
    llm.complete([SystemMessage("s"), UserMessage("u")])

    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["output_config"] == {"effort": "medium"}
    assert captured["context_management"] == {
        "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
    }


def test_system_blocks_reach_the_request(client) -> None:
    llm, captured = client()
    llm.complete(
        [
            SystemMessage(
                "flat",
                blocks=[
                    SystemBlock("identity"),
                    SystemBlock(
                        "core",
                        cache_control={
                            "type": "ephemeral",
                            "ttl": "1h",
                            "scope": "global",
                        },
                    ),
                ],
            ),
            UserMessage("u"),
        ]
    )

    assert isinstance(captured["system"], list)
    assert captured["system"][1]["cache_control"]["scope"] == "global"


def test_agent_sends_its_blocks(client) -> None:
    """End to end: a default TernoAgent's prompt reaches the wire as blocks."""
    from terno_agent.agents.terno import TernoAgent

    llm, captured = client()
    agent = TernoAgent(llm=llm)
    agent.run("hi")

    system = captured["system"]
    assert isinstance(system, list)
    # identity / core / session / tool-guide, with two breakpoints.
    assert len(system) == 4
    assert sum("cache_control" in b for b in system) == 2
    assert system[1]["cache_control"]["scope"] == "global"
    assert "scope" not in system[2]["cache_control"]
