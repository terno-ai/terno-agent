"""Tests for the cancellation pipeline.

Covers the four layers that have to cooperate:
* `CancelToken` primitive
* `BaseAgent.run` returns a cancelled `AgentRun` instead of raising
* `BashTool` kills its subprocess group when the token flips
* SDK-level `Agent.cancel()` plumbs through to the underlying agent
"""

from __future__ import annotations

import time
from pathlib import Path

from terno_agent.agents.base import BaseAgent
from terno_agent.core.cancel import CancelToken
from terno_agent.core.exceptions import AgentCancelled
from terno_agent.core.messages import AssistantMessage, Message
from terno_agent.llm.base import LLMResponse
from terno_agent.tools.shell import BashTool

# --------------------------------------------------------------------------- #
# CancelToken primitive
# --------------------------------------------------------------------------- #


def test_token_starts_uncancelled():
    t = CancelToken()
    assert not t.is_cancelled
    t.check()  # must not raise


def test_token_cancel_raises_via_check():
    t = CancelToken()
    t.cancel()
    assert t.is_cancelled
    import pytest

    with pytest.raises(AgentCancelled):
        t.check()


def test_token_clear_resets():
    t = CancelToken()
    t.cancel()
    t.clear()
    assert not t.is_cancelled
    t.check()


def test_token_wait_unblocks_on_cancel():
    t = CancelToken()
    import threading

    threading.Timer(0.05, t.cancel).start()
    start = time.monotonic()
    assert t.wait(timeout=1.0)
    assert time.monotonic() - start < 0.5


# --------------------------------------------------------------------------- #
# BaseAgent — cancelled run returns AgentRun(cancelled=True)
# --------------------------------------------------------------------------- #


class _OneshotLLM:
    """Emits one text delta then returns a final message — no tool calls."""

    model = "scripted"

    def complete(
        self,
        messages: list[Message],
        tools=None,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        on_text_delta=None,
    ) -> LLMResponse:
        if on_text_delta is not None:
            on_text_delta("hi")
        return LLMResponse(
            message=AssistantMessage(content="ok"),
            stop_reason="end_turn",
        )


def test_baseagent_returns_cancelled_run_when_token_set_upfront():
    token = CancelToken()
    token.cancel()
    agent = BaseAgent(_OneshotLLM(), system_prompt="x", cancel_token=token)
    result = agent.run("anything")
    assert result.cancelled is True
    assert result.answer == "(cancelled by user)"


class _TextDeltaLLM:
    """Calls on_text_delta in a loop; the cancel check inside the delta
    handler is what should abort the call."""

    model = "scripted"

    def complete(
        self, messages, tools=None, *, max_tokens=4096, temperature=0.2, on_text_delta=None,
    ):
        if on_text_delta:
            for piece in ["a", "b", "c", "d", "e"]:
                on_text_delta(piece)
        return LLMResponse(
            message=AssistantMessage(content="abcde"),
            stop_reason="end_turn",
        )


def test_cancel_during_streaming_returns_cancelled():
    token = CancelToken()

    class _SlowLLM:
        model = "scripted"

        def complete(
        self, messages, tools=None, *, max_tokens=4096, temperature=0.2, on_text_delta=None,
    ):
            # Flip the token mid-stream; the next emit should raise.
            for i, piece in enumerate(["a", "b", "c", "d", "e"]):
                if i == 2:
                    token.cancel()
                if on_text_delta:
                    on_text_delta(piece)
            return LLMResponse(
                message=AssistantMessage(content="abcde"),
                stop_reason="end_turn",
            )

    agent = BaseAgent(_SlowLLM(), system_prompt="x", cancel_token=token)
    result = agent.run("go")
    assert result.cancelled is True


# --------------------------------------------------------------------------- #
# BashTool kills its subprocess on cancel
# --------------------------------------------------------------------------- #


def test_bash_tool_aborts_on_cancel(tmp_path: Path):
    token = CancelToken()
    tool = BashTool(workdir=tmp_path, default_timeout_s=30, cancel_token=token)

    import threading

    # Flip the cancel token shortly after the subprocess starts.
    threading.Timer(0.2, token.cancel).start()

    start = time.monotonic()
    import pytest

    with pytest.raises(AgentCancelled):
        tool.run(command="sleep 10", timeout_s=30)
    elapsed = time.monotonic() - start
    # We should be unblocked well before the 10s sleep, not after it.
    assert elapsed < 3.0


# --------------------------------------------------------------------------- #
# Subagent shares the parent's cancel token
# --------------------------------------------------------------------------- #


def test_terno_agent_cancel_propagates_to_subagent():
    """SpawnAgentTool must forward the parent's cancel token."""
    from terno_agent.agents.terno import TernoAgent

    class _DummyLLM:
        model = "dummy"

        def complete(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("not invoked here")

    agent = TernoAgent(_DummyLLM())
    spawn = agent.tools["spawn_agent"]
    # The token attached to the agent must be the one SpawnAgentTool holds.
    assert spawn.cancel_token is agent.cancel_token

    # Cancelling the agent flips it; reset_cancel clears it again.
    assert not agent.cancel_token.is_cancelled
    agent.cancel()
    assert agent.cancel_token.is_cancelled
    agent.reset_cancel()
    assert not agent.cancel_token.is_cancelled
