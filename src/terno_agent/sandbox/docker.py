"""Docker-backed sandbox.

A `DockerSandbox` keeps **one** long-running container per session and
talks to a small Python driver inside it (`_driver.DRIVER_SOURCE`). Each
``run_python`` call sends a JSON line over the container's stdin and
reads a sentinel-framed JSON response from stdout, so variables,
imports, and files written under ``/work`` persist between calls. The
container is created lazily on the first ``run_python`` and removed
when ``close()`` is called — unless ``persist=True``, in which case the
container is given a stable name (derived from the cwd by default) and
left running so the next session can attach to it.

Hardening flags (``--network none``, read-only rootfs with a writable
``/work`` + ``/tmp`` tmpfs, memory + CPU caps) are unchanged from the
per-call container we used previously. On timeout or cancellation we
SIGKILL the container, discard the socket, and let the next call
recreate a fresh one — the user loses session state, but the alternative
(per-process kill inside the container) is far more brittle.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import struct
import threading
import time
from pathlib import Path

from terno_agent.core.cancel import CancelToken
from terno_agent.core.exceptions import AgentCancelled, SandboxError
from terno_agent.sandbox._driver import DRIVER_SOURCE, SENTINEL
from terno_agent.sandbox.base import ExecutionResult

_POLL_INTERVAL_S = 0.2
_STREAM_STDOUT = 1
_STREAM_STDERR = 2

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
        persist: bool = False,
        container_name: str | None = None,
    ) -> None:
        try:
            import docker
        except ImportError as exc:
            raise SandboxError(
                "docker package not installed. Install with: pip install 'terno-agent[docker]'"
            ) from exc
        self._docker = docker
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
        self.persist = persist
        self.container_name = container_name or _default_container_name()

        # Session state — populated lazily on first run_python.
        self._container = None  # docker-py Container
        self._sock = None  # raw socket attached to container stdio
        self._stdout_buf = b""  # accumulated demuxed stdout (host side)
        self._stderr_buf = b""  # accumulated demuxed stderr (rarely used)
        self._lock = threading.Lock()
        self._ensure_image()

    def _ensure_image(self) -> None:
        try:
            self._client.images.get(self.image)
        except Exception:
            try:
                self._client.images.pull(self.image)
            except Exception as exc:
                raise SandboxError(f"Failed to pull image {self.image!r}: {exc}") from exc

    # ----- session management ------------------------------------------ #

    def _ensure_session(self) -> None:
        """Bring up (or attach to) the long-running container + driver."""
        if self._sock is not None and self._container_alive():
            return
        if self._sock is not None:
            self._teardown(remove=False)

        container = None
        if self.persist:
            container = self._find_existing()
            if container is not None and container.status != "running":
                try:
                    container.start()
                except Exception as exc:
                    raise SandboxError(
                        f"could not start existing container {self.container_name!r}: {exc}"
                    ) from exc

        if container is None:
            container = self._create_container()
            try:
                container.start()
            except Exception as exc:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
                raise SandboxError(f"could not start sandbox container: {exc}") from exc

        self._container = container
        self._stdout_buf = b""
        self._stderr_buf = b""
        try:
            self._sock = self._attach_socket(container)
        except Exception as exc:
            self._teardown(remove=not self.persist)
            raise SandboxError(f"could not attach to sandbox container: {exc}") from exc

    def _find_existing(self):
        try:
            return self._client.containers.get(self.container_name)
        except self._docker.errors.NotFound:
            return None
        except Exception:
            return None

    def _create_container(self):
        kwargs = dict(
            command=["python", "-u", "-c", DRIVER_SOURCE],
            name=self.container_name if self.persist else None,
            working_dir=self.workdir,
            network_mode=self.network,
            mem_limit=self.memory,
            nano_cpus=int(self.cpus * 1_000_000_000),
            read_only=True,
            tmpfs={"/work": "rw,size=64m", "/tmp": "rw,size=64m"},
            stdin_open=True,
            tty=False,
            detach=True,
            labels={"terno.sandbox": "1"},
        )
        return self._client.containers.create(self.image, **kwargs)

    def _attach_socket(self, container):
        """Open a raw socket to the container's stdio.

        docker-py returns a ``SocketIO``-like wrapper; we unwrap to the
        underlying ``socket.socket`` so we can use ``select`` and
        non-blocking reads. The Docker stream protocol multiplexes
        stdout + stderr over this same socket using 8-byte frame
        headers (see ``_iter_frames``).
        """
        raw = self._client.api.attach_socket(
            container.id,
            params={"stdin": 1, "stdout": 1, "stderr": 1, "stream": 1},
        )
        sock = getattr(raw, "_sock", raw)
        sock.setblocking(False)
        return sock

    def _container_alive(self) -> bool:
        if self._container is None:
            return False
        try:
            self._container.reload()
        except Exception:
            return False
        return self._container.status == "running"

    def _teardown(self, *, remove: bool) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._container is None:
            return
        if remove:
            try:
                self._container.kill()
            except Exception:
                pass
            try:
                self._container.remove(force=True)
            except Exception:
                pass
        self._container = None
        self._stdout_buf = b""
        self._stderr_buf = b""

    def close(self) -> None:
        """Tear down the session container.

        When ``persist=True`` the container is left running so the next
        session can attach to it; otherwise it's killed and removed.
        Safe to call multiple times and safe to call when no session
        was ever started.
        """
        self._teardown(remove=not self.persist)

    # ----- the actual run loop ----------------------------------------- #

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

        # `env` is intentionally ignored: the long-running driver has
        # already inherited the container's environment at start time.
        # Per-call env injection would require restarting the driver
        # (which would wipe state) — so we document this in the docstring
        # and let callers set TERNO_SANDBOX env up-front instead.
        with self._lock:
            self._ensure_session()
            request = json.dumps({"code": code}) + "\n"
            try:
                self._sock.send(request.encode("utf-8"))
            except (BrokenPipeError, OSError) as exc:
                # Driver died — recreate next call and surface the error.
                self._teardown(remove=not self.persist)
                raise SandboxError(f"sandbox driver write failed: {exc}") from exc

            deadline = time.monotonic() + max(1, timeout_s)
            try:
                response, leaked_stdout = self._read_response(
                    deadline=deadline, cancel_token=cancel_token
                )
            except _SandboxTimeout:
                self._teardown(remove=not self.persist)
                return ExecutionResult(
                    stdout="", stderr=f"sandbox: snippet exceeded {timeout_s}s",
                    exit_code=124, timed_out=True,
                )
            except _SandboxCancelled:
                self._teardown(remove=not self.persist)
                raise AgentCancelled("run_python cancelled by user")
            except _SandboxDriverGone as exc:
                self._teardown(remove=not self.persist)
                raise SandboxError(str(exc)) from exc

        # Surface any free-form prints the snippet emitted before its
        # sentinel-framed response (e.g. text the JSON dump didn't capture
        # because something wrote to /dev/stdout directly). Rare, but
        # keeps debugging painless.
        return ExecutionResult(
            stdout=leaked_stdout + response.get("stdout", ""),
            stderr=response.get("stderr", ""),
            exit_code=int(response.get("exit_code", 1)),
            timed_out=False,
        )

    def _read_response(
        self, *, deadline: float, cancel_token: CancelToken | None
    ) -> tuple[dict, str]:
        """Read demuxed frames until we see SENTINEL on stdout, return JSON."""
        sentinel_bytes = SENTINEL.encode("utf-8")
        while True:
            if cancel_token is not None and cancel_token.is_cancelled:
                raise _SandboxCancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _SandboxTimeout()
            try:
                ready, _, _ = select.select(
                    [self._sock], [], [], min(remaining, _POLL_INTERVAL_S)
                )
            except OSError as exc:
                raise _SandboxDriverGone(f"sandbox socket select failed: {exc}")
            if not ready:
                continue
            try:
                stream_id, payload = _read_next_frame(self._sock)
            except _SandboxDriverGone:
                raise
            if payload is None:
                # EOF — container exited or socket closed.
                raise _SandboxDriverGone("sandbox driver exited unexpectedly")
            if stream_id == _STREAM_STDERR:
                self._stderr_buf += payload
                continue
            self._stdout_buf += payload
            idx = self._stdout_buf.find(sentinel_bytes)
            if idx == -1:
                continue
            leaked = self._stdout_buf[:idx].decode("utf-8", errors="replace")
            after = self._stdout_buf[idx + len(sentinel_bytes):]
            newline = after.find(b"\n")
            if newline == -1:
                # Wait for the rest of the JSON response line.
                self._stdout_buf = self._stdout_buf[idx:]
                continue
            line = after[:newline]
            self._stdout_buf = after[newline + 1:]
            try:
                return json.loads(line.decode("utf-8")), leaked
            except json.JSONDecodeError as exc:
                raise _SandboxDriverGone(f"malformed driver response: {exc}")


class _SandboxTimeout(Exception):
    """Internal signal: snippet ran past its deadline."""


class _SandboxCancelled(Exception):
    """Internal signal: cancel_token fired while waiting for a response."""


class _SandboxDriverGone(Exception):
    """Internal signal: driver process died or sent garbage."""


def _read_next_frame(sock) -> tuple[int, bytes | None]:
    """Read one Docker stream-protocol frame from ``sock``.

    Frame header is 8 bytes: 1-byte stream_id (1=stdout, 2=stderr),
    3 bytes padding, then 4-byte big-endian payload length. Returns
    ``(stream_id, payload_bytes)`` or ``(0, None)`` on EOF.
    """
    header = _read_exact(sock, 8)
    if header is None:
        return 0, None
    stream_id = header[0]
    payload_len = struct.unpack(">I", header[4:8])[0]
    if payload_len == 0:
        return stream_id, b""
    payload = _read_exact(sock, payload_len)
    if payload is None:
        return 0, None
    return stream_id, payload


def _read_exact(sock, n: int) -> bytes | None:
    """Read exactly ``n`` bytes from a non-blocking socket, or return None.

    Tolerates short reads and EAGAIN by polling with select(); returns
    ``None`` if the peer closes the socket cleanly.
    """
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except BlockingIOError:
            ready, _, _ = select.select([sock], [], [], 1.0)
            if not ready:
                continue
            try:
                chunk = sock.recv(n - len(buf))
            except (BlockingIOError, ConnectionResetError):
                continue
        except (OSError, ConnectionResetError):
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _default_container_name() -> str:
    """Stable per-cwd name so each project gets its own persisted container."""
    digest = hashlib.sha256(str(Path.cwd()).encode("utf-8")).hexdigest()[:8]
    return f"terno-sandbox-{digest}"


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


