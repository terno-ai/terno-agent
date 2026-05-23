import json

import pytest

from terno_agent.core.exceptions import ConfigError
from terno_agent.mcp.config import (
    HttpServerConfig,
    StdioServerConfig,
    load_mcp_config,
)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Run each test from a clean cwd so the project's `.terno/mcp.json`
    (which load_mcp_config now auto-merges) doesn't leak into fixtures."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TERNO_MCP_CONFIG", raising=False)


def _write(tmp_path, body, name=".mcp.json"):
    p = tmp_path / name
    p.write_text(json.dumps(body))
    return p


def test_load_missing_file_returns_empty(tmp_path):
    assert load_mcp_config(tmp_path / "nope.json") == []


def test_load_raw_stdio(tmp_path):
    p = _write(
        tmp_path,
        {
            "mcpServers": {
                "fetch": {"command": "uvx", "args": ["mcp-server-fetch"], "env": {"X": "1"}}
            }
        },
    )
    configs = load_mcp_config(p)
    assert len(configs) == 1
    cfg = configs[0]
    assert isinstance(cfg, StdioServerConfig)
    assert cfg.name == "fetch"
    assert cfg.command == "uvx"
    assert cfg.args == ("mcp-server-fetch",)
    assert cfg.env == {"X": "1"}
    assert cfg.runner is None


def test_load_runner_block(tmp_path):
    p = _write(
        tmp_path,
        {
            "mcpServers": {
                "git": {
                    "runner": {
                        "type": "auto",
                        "package": "mcp-server-git",
                        "package_type": "python",
                        "image": "ghcr.io/x:latest",
                        "args": ["--repository", "/repo"],
                        "docker": {
                            "mounts": [{"source": "/a", "target": "/b", "readonly": True}],
                            "env_passthrough": ["FOO"],
                        },
                    }
                }
            }
        },
    )
    cfg = load_mcp_config(p)[0]
    assert isinstance(cfg, StdioServerConfig)
    assert cfg.runner is not None
    assert cfg.runner.type == "auto"
    assert cfg.runner.package == "mcp-server-git"
    assert cfg.runner.package_type == "python"
    assert cfg.runner.image == "ghcr.io/x:latest"
    assert cfg.runner.docker.mounts[0].source == "/a"
    assert cfg.runner.docker.mounts[0].readonly is True
    assert cfg.runner.docker.env_passthrough == ("FOO",)


def test_load_http(tmp_path):
    p = _write(
        tmp_path,
        {
            "mcpServers": {
                "linear": {
                    "url": "https://mcp.linear.app/sse",
                    "headers": {"Authorization": "Bearer x"},
                }
            }
        },
    )
    cfg = load_mcp_config(p)[0]
    assert isinstance(cfg, HttpServerConfig)
    assert cfg.transport == "sse"  # auto-detected from URL suffix
    assert cfg.headers["Authorization"] == "Bearer x"


def test_http_explicit_transport(tmp_path):
    p = _write(
        tmp_path,
        {"mcpServers": {"x": {"url": "https://x/", "transport": "http"}}},
    )
    cfg = load_mcp_config(p)[0]
    assert isinstance(cfg, HttpServerConfig)
    assert cfg.transport == "http"


def test_rejects_multiple_kinds(tmp_path):
    p = _write(
        tmp_path,
        {"mcpServers": {"x": {"command": "y", "url": "https://z"}}},
    )
    with pytest.raises(ConfigError):
        load_mcp_config(p)


def test_rejects_runner_without_required_fields(tmp_path):
    p = _write(tmp_path, {"mcpServers": {"x": {"runner": {"type": "uvx"}}}})
    with pytest.raises(ConfigError):
        load_mcp_config(p)


def test_env_interpolation_success(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret")
    p = _write(
        tmp_path,
        {"mcpServers": {"x": {"url": "https://x", "headers": {"A": "Bearer ${MY_TOKEN}"}}}},
    )
    cfg = load_mcp_config(p)[0]
    assert isinstance(cfg, HttpServerConfig)
    assert cfg.headers["A"] == "Bearer secret"


def test_env_interpolation_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("DOES_NOT_EXIST", raising=False)
    p = _write(
        tmp_path,
        {"mcpServers": {"x": {"url": "https://x", "headers": {"A": "${DOES_NOT_EXIST}"}}}},
    )
    with pytest.raises(ConfigError):
        load_mcp_config(p)


def test_malformed_json_raises(tmp_path):
    p = tmp_path / ".mcp.json"
    p.write_text("{ not json")
    with pytest.raises(ConfigError):
        load_mcp_config(p)


def test_top_level_must_be_object(tmp_path):
    p = tmp_path / ".mcp.json"
    p.write_text('["x"]')
    with pytest.raises(ConfigError):
        load_mcp_config(p)


def test_empty_mcp_servers_returns_empty(tmp_path):
    p = _write(tmp_path, {"mcpServers": {}})
    assert load_mcp_config(p) == []


# --------------------------------------------------------------------------- #
# Discovery + merge
# --------------------------------------------------------------------------- #


def _write_default(tmp_path, body):
    d = tmp_path / ".terno"
    d.mkdir(exist_ok=True)
    p = d / "mcp.json"
    p.write_text(json.dumps(body))
    return p


def test_autodiscovers_terno_mcp_json(tmp_path):
    _write_default(
        tmp_path,
        {"mcpServers": {"alpha": {"command": "uvx", "args": ["alpha"]}}},
    )
    configs = load_mcp_config()
    assert [c.name for c in configs] == ["alpha"]


def test_no_autodiscovery_of_legacy_paths(tmp_path, monkeypatch):
    # Legacy locations must no longer be picked up automatically.
    _write(tmp_path, {"mcpServers": {"old": {"command": "uvx"}}}, name=".mcp.json")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".terno").mkdir()
    (home / ".terno" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"global": {"command": "uvx"}}})
    )
    monkeypatch.setenv("HOME", str(home))
    assert load_mcp_config() == []


def test_env_var_overrides_default(tmp_path, monkeypatch):
    _write_default(tmp_path, {"mcpServers": {"alpha": {"command": "uvx"}}})
    other = _write(
        tmp_path,
        {"mcpServers": {"beta": {"command": "uvx"}}},
        name="other.json",
    )
    monkeypatch.setenv("TERNO_MCP_CONFIG", str(other))
    configs = load_mcp_config()
    # env var wins; default is NOT also merged when only env discovery is used
    assert [c.name for c in configs] == ["beta"]


def test_explicit_path_merges_with_default(tmp_path):
    _write_default(tmp_path, {"mcpServers": {"alpha": {"command": "uvx"}}})
    explicit = _write(
        tmp_path,
        {"mcpServers": {"beta": {"command": "npx"}}},
        name="explicit.json",
    )
    configs = {c.name: c for c in load_mcp_config(explicit)}
    assert set(configs) == {"alpha", "beta"}


def test_explicit_path_overrides_default_on_name_conflict(tmp_path):
    _write_default(
        tmp_path,
        {"mcpServers": {"shared": {"command": "default-cmd"}}},
    )
    explicit = _write(
        tmp_path,
        {"mcpServers": {"shared": {"command": "explicit-cmd"}}},
        name="explicit.json",
    )
    cfg = load_mcp_config(explicit)[0]
    assert isinstance(cfg, StdioServerConfig)
    assert cfg.command == "explicit-cmd"


def test_explicit_path_missing_falls_back_to_default(tmp_path):
    _write_default(tmp_path, {"mcpServers": {"alpha": {"command": "uvx"}}})
    configs = load_mcp_config(tmp_path / "does-not-exist.json")
    assert [c.name for c in configs] == ["alpha"]


def test_explicit_path_equal_to_default_loaded_once(tmp_path):
    default = _write_default(
        tmp_path,
        {"mcpServers": {"alpha": {"command": "uvx"}}},
    )
    configs = load_mcp_config(default)
    assert [c.name for c in configs] == ["alpha"]
