"""Discovery of the Docker daemon endpoint for `DockerSandbox`.

These tests don't talk to Docker — they exercise the resolver against a
synthetic ``$HOME`` so the logic is covered on any machine, with or
without Docker installed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from terno_agent.sandbox.docker import discover_docker_base_url


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Point HOME at a clean tmp dir and clear Docker env vars."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    return tmp_path


def _write_config_json(home: Path, *, current_context: str | None) -> None:
    cfg = home / ".docker" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    payload = {"currentContext": current_context} if current_context else {}
    cfg.write_text(json.dumps(payload), encoding="utf-8")


def _write_context_meta(home: Path, *, name: str, host: str) -> Path:
    # Docker hashes the context name with SHA256 for the meta dir; we
    # don't have to match that exactly since the resolver scans all
    # entries by name. Use a stable hex string so each test gets its own
    # directory.
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    meta_dir = home / ".docker" / "contexts" / "meta" / digest
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_dir / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "Name": name,
                "Endpoints": {"docker": {"Host": host}},
            }
        ),
        encoding="utf-8",
    )
    return meta


# --------------------------------------------------------------------------- #
# DOCKER_HOST short-circuit
# --------------------------------------------------------------------------- #


def test_docker_host_set_returns_none(_isolate_home, monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://example:2375")
    assert discover_docker_base_url() is None


# --------------------------------------------------------------------------- #
# Context resolution
# --------------------------------------------------------------------------- #


def test_current_context_resolves_to_host(_isolate_home):
    home = _isolate_home
    _write_config_json(home, current_context="desktop-linux")
    _write_context_meta(
        home,
        name="desktop-linux",
        host="unix:///Users/me/.docker/run/docker.sock",
    )
    assert (
        discover_docker_base_url()
        == "unix:///Users/me/.docker/run/docker.sock"
    )


def test_docker_context_env_overrides_config(_isolate_home, monkeypatch):
    home = _isolate_home
    _write_config_json(home, current_context="desktop-linux")
    _write_context_meta(
        home,
        name="desktop-linux",
        host="unix:///wrong/socket",
    )
    _write_context_meta(
        home,
        name="colima",
        host="unix:///home/me/.colima/default/docker.sock",
    )
    monkeypatch.setenv("DOCKER_CONTEXT", "colima")
    assert (
        discover_docker_base_url()
        == "unix:///home/me/.colima/default/docker.sock"
    )


def test_default_context_skipped(_isolate_home):
    home = _isolate_home
    _write_config_json(home, current_context="default")
    # No meta for 'default'; resolver should not return anything based on it.
    # Without per-user socket files, falls through to None.
    assert discover_docker_base_url() is None


def test_context_with_missing_meta_falls_through(_isolate_home):
    home = _isolate_home
    _write_config_json(home, current_context="ghost")
    # No matching meta dir written.
    assert discover_docker_base_url() is None


# --------------------------------------------------------------------------- #
# Per-user socket fallback
# --------------------------------------------------------------------------- #


def test_docker_desktop_socket_fallback(_isolate_home):
    home = _isolate_home
    sock = home / ".docker" / "run" / "docker.sock"
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.touch()
    assert discover_docker_base_url() == f"unix://{sock}"


def test_colima_socket_fallback(_isolate_home):
    home = _isolate_home
    sock = home / ".colima" / "default" / "docker.sock"
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.touch()
    assert discover_docker_base_url() == f"unix://{sock}"


def test_orbstack_socket_fallback(_isolate_home):
    home = _isolate_home
    sock = home / ".orbstack" / "run" / "docker.sock"
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.touch()
    assert discover_docker_base_url() == f"unix://{sock}"


def test_context_takes_priority_over_socket_fallback(_isolate_home):
    home = _isolate_home
    # Both a valid context AND a candidate socket exist; context wins.
    _write_config_json(home, current_context="desktop-linux")
    _write_context_meta(
        home,
        name="desktop-linux",
        host="unix:///explicit/from-context.sock",
    )
    fallback_sock = home / ".docker" / "run" / "docker.sock"
    fallback_sock.parent.mkdir(parents=True, exist_ok=True)
    fallback_sock.touch()
    assert discover_docker_base_url() == "unix:///explicit/from-context.sock"


def test_nothing_found_returns_none(_isolate_home):
    # Empty HOME, no env vars — let docker.from_env() handle it.
    assert discover_docker_base_url() is None


# --------------------------------------------------------------------------- #
# Defensive parsing
# --------------------------------------------------------------------------- #


def test_malformed_config_json_does_not_crash(_isolate_home):
    home = _isolate_home
    cfg = home / ".docker" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("{ not json", encoding="utf-8")
    assert discover_docker_base_url() is None


def test_malformed_context_meta_does_not_crash(_isolate_home):
    home = _isolate_home
    _write_config_json(home, current_context="x")
    meta_dir = home / ".docker" / "contexts" / "meta" / "deadbeef"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "meta.json").write_text("{ broken", encoding="utf-8")
    assert discover_docker_base_url() is None


def test_context_missing_host_field(_isolate_home):
    home = _isolate_home
    _write_config_json(home, current_context="weird")
    meta_dir = home / ".docker" / "contexts" / "meta" / "abcd"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "meta.json").write_text(
        json.dumps({"Name": "weird", "Endpoints": {}}), encoding="utf-8"
    )
    assert discover_docker_base_url() is None
