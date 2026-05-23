import pytest

from terno_agent.core.exceptions import ConfigError
from terno_agent.mcp.config import (
    DockerMount,
    DockerOptions,
    RunnerBlock,
    StdioServerConfig,
)
from terno_agent.mcp.runner import resolve


def _which(present):
    return lambda tool: f"/usr/bin/{tool}" if tool in present else None


def test_raw_command_present():
    cfg = StdioServerConfig(name="x", command="uvx", args=("foo",))
    spec = resolve(cfg, which=_which({"uvx"}))
    assert spec.argv == ("uvx", "foo")


def test_raw_command_missing():
    cfg = StdioServerConfig(name="x", command="uvxxx")
    with pytest.raises(ConfigError):
        resolve(cfg, which=_which(set()))


def test_runner_uvx():
    cfg = StdioServerConfig(
        name="x",
        runner=RunnerBlock(type="uvx", package="mcp-fetch"),
    )
    spec = resolve(cfg, which=_which({"uvx"}))
    assert spec.argv == ("uvx", "mcp-fetch")


def test_runner_npx():
    cfg = StdioServerConfig(
        name="x",
        runner=RunnerBlock(type="npx", package="@scope/pkg"),
    )
    spec = resolve(cfg, which=_which({"npx"}))
    assert spec.argv == ("npx", "-y", "@scope/pkg")


def test_runner_docker_with_mounts_and_envpass(monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    cfg = StdioServerConfig(
        name="git",
        env={"A": "B"},
        runner=RunnerBlock(
            type="docker",
            image="img:latest",
            args=("--repo", "/r"),
            docker=DockerOptions(
                mounts=(DockerMount(source="/host", target="/cont", readonly=True),),
                env_passthrough=("FOO",),
            ),
        ),
    )
    spec = resolve(cfg, which=_which({"docker"}))
    assert spec.argv[0:4] == ("docker", "run", "--rm", "-i")
    assert "-v" in spec.argv
    assert "/host:/cont:ro" in spec.argv
    assert "-e" in spec.argv
    # Both server env and env_passthrough should appear:
    assert "A=B" in spec.argv
    assert "FOO=bar" in spec.argv
    # Image precedes the trailing args:
    assert spec.argv[-3:] == ("img:latest", "--repo", "/r")


def test_runner_auto_prefers_docker_when_image_and_docker_present():
    cfg = StdioServerConfig(
        name="x",
        runner=RunnerBlock(type="auto", package="mcp-x", image="img"),
    )
    spec = resolve(cfg, which=_which({"docker", "uvx", "npx"}))
    assert spec.argv[0] == "docker"


def test_runner_auto_falls_back_to_uvx_when_no_docker():
    cfg = StdioServerConfig(
        name="x",
        runner=RunnerBlock(type="auto", package="mcp-x", image="img"),
    )
    spec = resolve(cfg, which=_which({"uvx"}))
    assert spec.argv == ("uvx", "mcp-x")


def test_runner_auto_explicit_package_type_python():
    cfg = StdioServerConfig(
        name="x",
        runner=RunnerBlock(type="auto", package="some-py-pkg", package_type="python"),
    )
    spec = resolve(cfg, which=_which({"uvx", "npx"}))
    assert spec.argv == ("uvx", "some-py-pkg")


def test_runner_auto_explicit_package_type_npm():
    cfg = StdioServerConfig(
        name="x",
        runner=RunnerBlock(type="auto", package="some-pkg", package_type="npm"),
    )
    spec = resolve(cfg, which=_which({"uvx", "npx"}))
    assert spec.argv == ("npx", "-y", "some-pkg")


def test_runner_auto_heuristic_scoped_npm():
    cfg = StdioServerConfig(
        name="x",
        runner=RunnerBlock(type="auto", package="@scope/pkg"),
    )
    spec = resolve(cfg, which=_which({"uvx", "npx"}))
    assert spec.argv == ("npx", "-y", "@scope/pkg")


def test_runner_auto_heuristic_unscoped_uses_uvx():
    cfg = StdioServerConfig(
        name="x",
        runner=RunnerBlock(type="auto", package="mcp-server-foo"),
    )
    spec = resolve(cfg, which=_which({"uvx", "npx"}))
    assert spec.argv == ("uvx", "mcp-server-foo")


def test_runner_auto_no_runtime_available():
    cfg = StdioServerConfig(
        name="x",
        runner=RunnerBlock(type="auto", package="mcp-server-foo"),
    )
    with pytest.raises(ConfigError):
        resolve(cfg, which=_which(set()))
