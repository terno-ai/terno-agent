"""A single cancel signal that's safe to share across threads.

The CLI flips a `CancelToken` from its SIGINT handler when the user
hits Ctrl-C mid-turn. The agent and any in-flight tool then notice
and abort cleanly: subprocesses are killed, MCP futures are cancelled,
and the agent's run loop returns a partial result instead of churning
through more LLM calls.
"""

from __future__ import annotations

from threading import Event

from terno_agent.core.exceptions import AgentCancelled


class CancelToken:
    """Cooperative cancellation primitive.

    Backed by a `threading.Event` so callers can also `wait(timeout)`
    on it to interrupt sleep-style waits instead of polling.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        """Signal cancellation. Idempotent."""
        self._event.set()

    def clear(self) -> None:
        """Reset back to the un-cancelled state."""
        self._event.clear()

    def check(self) -> None:
        """Raise `AgentCancelled` if cancellation has been requested."""
        if self._event.is_set():
            raise AgentCancelled("cancelled by user")

    def wait(self, timeout: float | None = None) -> bool:
        """Block until cancellation or `timeout`. Returns True if cancelled."""
        return self._event.wait(timeout)


__all__ = ["CancelToken"]
