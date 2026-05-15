"""Docker-backed sandbox.

Runs the snippet inside a fresh container with `--network none`, a read-only
root filesystem, a memory cap, and a wall-clock timeout. The container is
removed after execution.
"""

from __future__ import annotations

import io
import tarfile
import time

from terno_agent.core.exceptions import SandboxError
from terno_agent.sandbox.base import ExecutionResult


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
        try:
            self._client = docker.from_env()
            self._client.ping()
        except Exception as exc:
            raise SandboxError(f"Could not connect to Docker daemon: {exc}") from exc
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
    ) -> ExecutionResult:
        container = self._client.containers.create(
            self.image,
            command=["python", "/work/snippet.py"],
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
            container.put_archive(self.workdir, _tar_bytes("snippet.py", code))
            container.start()
            timed_out = False
            try:
                result = container.wait(timeout=timeout_s)
                exit_code = int(result.get("StatusCode", 1))
            except Exception:
                timed_out = True
                exit_code = 124
                try:
                    container.kill()
                except Exception:
                    pass
                # Give docker a moment to flush logs.
                time.sleep(0.1)

            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            return ExecutionResult(
                stdout=stdout, stderr=stderr, exit_code=exit_code, timed_out=timed_out
            )
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass


def _tar_bytes(name: str, contents: str) -> bytes:
    buf = io.BytesIO()
    data = contents.encode("utf-8")
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()
