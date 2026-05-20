"""Docker-backed sandbox.

Runs the snippet inside a fresh container with `--network none`, a read-only
root filesystem, a memory cap, and a wall-clock timeout. The container is
removed after execution.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from terno_agent.core.cancel import CancelToken
from terno_agent.core.exceptions import AgentCancelled, SandboxError
from terno_agent.sandbox.base import ExecutionResult

_POLL_INTERVAL_S = 0.2

# Per-user socket paths used by common macOS / Linux Docker installs when
# `/var/run/docker.sock` isn't symlinked. Probed in order.
_CANDIDATE_SOCKETS = (
    ".docker/run/docker.sock",       # Docker Desktop (modern)
    ".colima/default/docker.sock",   # Colima default profile
    ".colima/docker.sock",           # Colima older layout
    ".rd/docker.sock",               # Rancher Desktop
    ".orbstack/run/docker.sock",     # OrbStack
)


class DockerSandbox:
    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        network: str = "none",
        memory: str = "512m",
        cpus: float = 1.0,
        workdir: str = "/work",
    ) -> None:
        try:
            import docker
        except ImportError as exc:
            raise SandboxError(
                "docker package not installed. Install with: pip install 'terno-agent[docker]'"
            ) from exc
        base_url = discover_docker_base_url()
        try:
            if base_url:
                self._client = docker.DockerClient(base_url=base_url)
            else:
                self._client = docker.from_env()
            self._client.ping()
        except Exception as exc:
            hint = ""
            if not base_url and not os.environ.get("DOCKER_HOST"):
                hint = (
                    " (Hint: docker-py defaults to /var/run/docker.sock. If "
                    "you're using Docker Desktop, Colima, Rancher, or OrbStack, "
                    "either set DOCKER_HOST or ensure your active 'docker context' "
                    "host is discoverable.)"
                )
            raise SandboxError(f"Could not connect to Docker daemon: {exc}{hint}") from exc
        self.image = image
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self.workdir = workdir
        self._ensure_image()

    def _ensure_image(self) -> None:
        try:
            self._client.images.get(self.image)
        except Exception:
            try:
                self._client.images.pull(self.image)
            except Exception as exc:
                raise SandboxError(f"Failed to pull image {self.image!r}: {exc}") from exc

    def run_python(
        self,
        code: str,
        *,
        timeout_s: int = 30,
        env: dict[str, str] | None = None,
        cancel_token: CancelToken | None = None,
    ) -> ExecutionResult:
        if cancel_token is not None and cancel_token.is_cancelled:
            raise AgentCancelled("cancelled before run_python started")

        # Pass the snippet as an inline argument so we don't need to write
        # it into the container's filesystem. put_archive would race with
        # the tmpfs mounts (which only become active at start time) and
        # blow up with "container rootfs is marked read-only".
        container = self._client.containers.create(
            self.image,
            command=["python", "-c", code],
            working_dir=self.workdir,
            network_mode=self.network,
            mem_limit=self.memory,
            nano_cpus=int(self.cpus * 1_000_000_000),
            read_only=True,
            tmpfs={"/work": "rw,size=64m", "/tmp": "rw,size=64m"},
            environment=env or {},
            detach=True,
        )
        try:
            container.start()

            deadline = time.monotonic() + timeout_s
            timed_out = False
            cancelled = False
            exit_code = 0

            while True:
                if cancel_token is not None and cancel_token.is_cancelled:
                    cancelled = True
                    try:
                        container.kill()
                    except Exception:
                        pass
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    try:
                        container.kill()
                    except Exception:
                        pass
                    break
                try:
                    result = container.wait(timeout=_POLL_INTERVAL_S)
                    exit_code = int(result.get("StatusCode", 1))
                    break
                except Exception:
                    # `wait` raised because the poll timeout fired; loop and
                    # re-check the cancel token + deadline.
                    continue

            # Give docker a moment to flush logs after a hard kill.
            if timed_out or cancelled:
                time.sleep(0.1)
                exit_code = 124 if timed_out else 130

            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

            if cancelled:
                raise AgentCancelled("run_python cancelled by user")

            return ExecutionResult(
                stdout=stdout, stderr=stderr, exit_code=exit_code, timed_out=timed_out
            )
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass


def discover_docker_base_url() -> str | None:
    """Find the right Docker daemon endpoint, mirroring the Docker CLI.

    Resolution order (matches `docker context` behavior):

    1. ``DOCKER_HOST`` env var — returns ``None`` so ``docker.from_env()``
       handles it (it already honors this variable).
    2. ``DOCKER_CONTEXT`` env var, or ``currentContext`` in
       ``~/.docker/config.json``. The matching context's
       ``Endpoints.docker.Host`` is read from
       ``~/.docker/contexts/meta/<sha>/meta.json``.
    3. Common per-user socket paths used by Docker Desktop, Colima,
       Rancher Desktop, and OrbStack.

    Returns ``None`` if nothing better than the docker-py default was
    found; the caller then falls back to ``docker.from_env()``.
    """
    if os.environ.get("DOCKER_HOST"):
        return None

    name = os.environ.get("DOCKER_CONTEXT") or _read_current_context()
    if name and name not in ("", "default"):
        host = _read_context_host(name)
        if host:
            return host

    home = Path.home()
    for relative in _CANDIDATE_SOCKETS:
        candidate = home / relative
        if candidate.exists():
            return f"unix://{candidate}"
    return None


def _read_current_context() -> str | None:
    cfg = Path.home() / ".docker" / "config.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("currentContext")
    return value if isinstance(value, str) and value else None


def _read_context_host(name: str) -> str | None:
    meta_root = Path.home() / ".docker" / "contexts" / "meta"
    if not meta_root.is_dir():
        return None
    for entry in meta_root.iterdir():
        meta = entry / "meta.json"
        if not meta.is_file():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("Name") != name:
            continue
        host = data.get("Endpoints", {}).get("docker", {}).get("Host")
        return host if isinstance(host, str) and host else None
    return None


