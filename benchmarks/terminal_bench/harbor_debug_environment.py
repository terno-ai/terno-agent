"""Harbor Docker environment with streaming compose logs.

Harbor 0.1.x captures Docker Compose output after the command exits. That
means a slow or stuck image pull can leave the UI at "starting environment..."
with empty logs. This environment keeps Harbor's Docker behavior but streams
compose output into the trial log and an `environment.log` file as it arrives.
"""

from __future__ import annotations

import asyncio

from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment


class StreamingDockerEnvironment(DockerEnvironment):
    """Docker environment that writes live Docker Compose output to trial logs."""

    async def _run_docker_compose_command(
        self,
        command: list[str],
        check: bool = True,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        full_command = [
            "docker",
            "compose",
            "--progress",
            "plain",
            "-p",
            self.session_id.lower().replace(".", "-"),
            "-f",
            str(self._docker_compose_path.resolve().absolute()),
            *command,
        ]
        command_text = " ".join(full_command)
        log_path = self.trial_paths.trial_dir / "environment.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.debug("Running docker compose command: %s", command_text)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {command_text}\n")
            log.flush()

            process = await asyncio.create_subprocess_exec(
                *full_command,
                env=self._env_vars.to_env_dict(include_os_env=True),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            output_parts: list[str] = []
            try:
                await asyncio.wait_for(
                    self._stream_process_output(process, output_parts, log),
                    timeout=timeout_sec,
                )
            except TimeoutError as exc:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                raise RuntimeError(f"Command timed out after {timeout_sec} seconds") from exc

            stdout = "".join(output_parts) or None
            result = ExecResult(
                stdout=stdout,
                stderr=None,
                return_code=process.returncode or 0,
            )

            if check and result.return_code != 0:
                raise RuntimeError(
                    f"Docker compose command failed for environment "
                    f"{self.environment_name}. Command: {command_text}. "
                    f"Return code: {result.return_code}. Stdout: {result.stdout}."
                )

            return result

    async def _stream_process_output(
        self,
        process: asyncio.subprocess.Process,
        output_parts: list[str],
        log,
    ) -> None:
        if process.stdout is not None:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode(errors="replace")
                output_parts.append(text)
                log.write(text)
                log.flush()
                self.logger.debug(text.rstrip())
        await process.wait()
