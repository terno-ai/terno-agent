from terno_agent.mcp.config import HttpServerConfig, StdioServerConfig
from terno_agent.mcp.manager import McpManager
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
