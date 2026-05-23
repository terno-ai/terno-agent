import asyncio

import pytest

from terno_agent.mcp.bridge import AsyncBridge, BridgeError


@pytest.fixture()
def bridge():
    b = AsyncBridge()
    b.start()
    yield b
    b.stop()


def test_submit_returns_value(bridge):
    async def coro():
        await asyncio.sleep(0)
        return 42

    assert bridge.submit(coro(), timeout=2) == 42


def test_submit_timeout(bridge):
    async def slow():
        await asyncio.sleep(2)
        return "done"

    with pytest.raises(TimeoutError):
        bridge.submit(slow(), timeout=0.1)


def test_stop_is_idempotent():
    b = AsyncBridge()
    b.start()
    b.stop()
    b.stop()


def test_submit_after_stop_errors():
    b = AsyncBridge()
    b.start()
    b.stop()

    async def coro():
        return 1

    c = coro()
    try:
        with pytest.raises(BridgeError):
            b.submit(c)
    finally:
        c.close()


def test_double_start_is_noop():
    b = AsyncBridge()
    b.start()
    b.start()  # should not raise
    b.stop()
