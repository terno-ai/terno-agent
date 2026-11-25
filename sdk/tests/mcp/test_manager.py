import asyncio
import time

import terno_agent.mcp.manager as manager_module
from terno_agent.mcp.config import HttpServerConfig, StdioServerConfig
from terno_agent.mcp.manager import MAX_CONCURRENT_CONNECTS, McpManager
from tests.mcp.conftest import FakeSession, FakeTool


def _good_config(name="good"):
    return HttpServerConfig(name=name, url="https://x", transport="http")


def _bad_config():
    return HttpServerConfig(name="bad", url="https://y", transport="http")


def test_manager_aggregates_tools_from_good_servers(capsys):
    sessions: dict[str, FakeSession] = {}

    def factory(cfg):
        if cfg.name == "bad":
            return FakeSession(fail_connect=True)
        s = FakeSession([FakeTool(name="t1"), FakeTool(name="t2")])
        sessions[cfg.name] = s
        return s

    manager = McpManager.start_from_configs(
        [_good_config(), _bad_config()],
        session_factory=factory,
    )
    try:
        names = sorted(t.schema.name for t in manager.tools())
        assert names == ["mcp__good__t1", "mcp__good__t2"]

        captured = capsys.readouterr().err
        assert "bad" in captured  # warning for the failing server
    finally:
        manager.shutdown()


def test_manager_shutdown_closes_sessions():
    sessions: dict[str, FakeSession] = {}

    def factory(cfg):
        s = FakeSession([FakeTool(name="t")])
        sessions[cfg.name] = s
        return s

    manager = McpManager.start_from_configs(
        [_good_config("a"), _good_config("b")],
        session_factory=factory,
    )
    manager.shutdown()
    assert all(s.closed for s in sessions.values())
    # Idempotent.
    manager.shutdown()


def test_manager_handles_list_tools_failure(capsys):
    def factory(cfg):
        return FakeSession(tools=[FakeTool(name="t")], fail_list_tools=True)

    manager = McpManager.start_from_configs(
        [_good_config()],
        session_factory=factory,
    )
    try:
        assert manager.tools() == []
        captured = capsys.readouterr().err
        assert "list tools" in captured
    finally:
        manager.shutdown()


def test_manager_empty_config_does_not_start_bridge():
    """No configs → manager is a no-op; nothing to clean up."""
    manager = McpManager.start_from_configs([])
    assert manager.tools() == []
    manager.shutdown()


def test_stdio_server_uses_same_factory():
    sessions = []

    def factory(cfg):
        s = FakeSession([FakeTool(name=cfg.name + "_t")])
        sessions.append(s)
        return s

    manager = McpManager.start_from_configs(
        [StdioServerConfig(name="srv", command="echo")],
        session_factory=factory,
    )
    try:
        names = [t.schema.name for t in manager.tools()]
        assert names == ["mcp__srv__srv_t"]
    finally:
        manager.shutdown()


# --------------------------------------------------------------------------- #
# Host-side servers connect concurrently (standalone SDK)
# --------------------------------------------------------------------------- #


class _SlowSession(FakeSession):
    """Connect takes ``delay`` seconds; records how many connect at once."""

    active = 0
    peak = 0

    def __init__(self, name, delay, **kw):
        super().__init__([FakeTool(name=f"{name}_t")], **kw)
        self.delay = delay

    async def connect(self):
        cls = type(self)
        cls.active += 1
        cls.peak = max(cls.peak, cls.active)
        try:
            await asyncio.sleep(self.delay)
            await super().connect()
        finally:
            cls.active -= 1


def _reset_slow():
    _SlowSession.active = 0
    _SlowSession.peak = 0


def test_host_servers_connect_concurrently():
    """Four servers taking 0.3s each start in ~0.3s, not ~1.2s."""
    _reset_slow()
    configs = [_good_config(f"s{i}") for i in range(4)]
    start = time.perf_counter()
    manager = McpManager.start_from_configs(
        configs, session_factory=lambda cfg: _SlowSession(cfg.name, 0.3)
    )
    elapsed = time.perf_counter() - start
    try:
        assert elapsed < 0.9
        assert _SlowSession.peak == 4
        assert sorted(t.schema.name for t in manager.tools()) == [
            f"mcp__s{i}__s{i}_t" for i in range(4)
        ]
    finally:
        manager.shutdown()


def test_host_connect_limit_is_five_at_once():
    _reset_slow()
    manager = McpManager.start_from_configs(
        [_good_config(f"s{i}") for i in range(7)],
        session_factory=lambda cfg: _SlowSession(cfg.name, 0.1),
    )
    try:
        assert _SlowSession.peak == MAX_CONCURRENT_CONNECTS == 5
        assert len(manager.tools()) == 7
    finally:
        manager.shutdown()


def test_host_failing_server_does_not_hold_up_or_break_others(capsys):
    _reset_slow()

    def factory(cfg):
        if cfg.name == "bad":
            return _SlowSession(cfg.name, 0.0, fail_connect=True)
        return _SlowSession(cfg.name, 0.1)

    manager = McpManager.start_from_configs(
        [_good_config("a"), _bad_config(), _good_config("b")], session_factory=factory
    )
    try:
        assert sorted(t.server for t in manager.tools()) == ["a", "b"]
        assert "mcp server 'bad' failed to connect" in capsys.readouterr().err
    finally:
        manager.shutdown()


def test_host_slow_server_times_out_alone(monkeypatch, capsys):
    monkeypatch.setattr(manager_module, "CONNECT_TIMEOUT_S", 0.2)
    _reset_slow()

    def factory(cfg):
        return _SlowSession(cfg.name, 5.0 if cfg.name == "slow" else 0.05)

    start = time.perf_counter()
    manager = McpManager.start_from_configs(
        [_good_config("fast"), _good_config("slow")], session_factory=factory
    )
    try:
        assert time.perf_counter() - start < 2
        assert [t.server for t in manager.tools()] == ["fast"]
        err = capsys.readouterr().err
        assert "mcp server 'slow' failed to connect" in err and "timed out" in err
    finally:
        manager.shutdown()


def test_host_connect_reports_live_tools():
    seen = []
    manager = McpManager.start_from_configs(
        [_good_config("a")],
        session_factory=lambda cfg: FakeSession([FakeTool(name="t")]),
        on_tools_discovered=lambda server, tools: seen.append((server, [t["name"] for t in tools])),
    )
    try:
        assert seen == [("a", ["t"])]
    finally:
        manager.shutdown()
