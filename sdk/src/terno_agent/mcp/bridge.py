"""Sync ↔ async bridge for MCP.

Runs a single asyncio event loop on a background thread. Tools call
`submit(coro, timeout)` synchronously and get the awaited result back
(or a `TimeoutError`). The loop lives until `stop()` is called.

No MCP-specific knowledge lives here — `AsyncBridge` is pure plumbing
and is tested as such.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class BridgeError(RuntimeError):
    """Bridge is misused (e.g. submit before start, or after stop)."""


class AsyncBridge:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._stopped = False
        self._lock = threading.Lock()
        self._inflight: set[concurrent.futures.Future] = set()
        self._inflight_lock = threading.Lock()

    # ----- lifecycle ----------------------------------------------------- #

    def start(self) -> None:
        with self._lock:
            if self._loop is not None:
                return
            if self._stopped:
                raise BridgeError("AsyncBridge cannot be restarted after stop()")
            loop = asyncio.new_event_loop()
            self._loop = loop
            self._thread = threading.Thread(
                target=self._run_loop, name="mcp-bridge", daemon=True
            )
            self._thread.start()
        # Wait for the loop to actually start spinning before we accept calls.
        if not self._started.wait(timeout=5):
            raise BridgeError("AsyncBridge background loop did not start within 5s")

    def stop(self, *, timeout: float = 5.0) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
            if loop is None or self._stopped:
                self._stopped = True
                return
            self._stopped = True

        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=timeout)
        # Cleanup — closing the loop also cancels any leftover tasks.
        try:
            loop.close()
        except Exception:  # pragma: no cover - defensive
            pass
        self._loop = None
        self._thread = None

    # ----- submission ---------------------------------------------------- #

    def submit(self, coro: Coroutine[Any, Any, T], *, timeout: float | None = None) -> T:
        loop = self._loop
        if loop is None or self._stopped:
            raise BridgeError("AsyncBridge is not running")
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        with self._inflight_lock:
            self._inflight.add(future)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"mcp call timed out after {timeout}s"
            ) from exc
        except concurrent.futures.CancelledError as exc:
            raise CancelledFuture("mcp call cancelled") from exc
        finally:
            with self._inflight_lock:
                self._inflight.discard(future)

    def cancel_inflight(self) -> None:
        """Cancel every in-flight future. Used by the chat's stop signal.

        Safe to call from any thread including a signal handler.
        """
        with self._inflight_lock:
            futures = list(self._inflight)
        for f in futures:
            f.cancel()

    # ----- internals ----------------------------------------------------- #

    def _run_loop(self) -> None:
        loop = self._loop
        assert loop is not None
        asyncio.set_event_loop(loop)
        loop.call_soon(self._started.set)
        try:
            loop.run_forever()
        finally:
            asyncio.set_event_loop(None)


class CancelledFuture(RuntimeError):  # noqa: N818 - parallels concurrent.futures.CancelledError
    """A bridge future was cancelled out from under the waiter."""


__all__ = ["AsyncBridge", "BridgeError", "CancelledFuture"]
