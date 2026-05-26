"""Harbor / Terminal-Bench 2.0 adapter for running terno-agent through its SDK."""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.utils.templating import render_prompt_template

from benchmarks.terminal_bench._sdk_source import (
    CONTAINER_SDK_PATH,
    DEFAULT_LOCAL_SDK_PATH,
    stage_sdk_source,
)

_RESULT_JSON = "/logs/agent/terno-agent-result.json"
_RUN_LOG = "/logs/agent/terno-agent.txt"


class TernoHarborAgent(BaseInstalledAgent):
    """Install terno-agent in a Harbor task container and run one SDK task."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        package: str | None = None,
        sdk_path: str | Path | None = None,
        max_iterations: int = 64,
        bash_timeout_s: int = 600,
        run_python_timeout_s: int = 120,
        **kwargs: object,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._package = package
        self._sdk_path = (
            Path(sdk_path).expanduser().resolve()
            if sdk_path is not None
            else DEFAULT_LOCAL_SDK_PATH
        )
        if sdk_path is not None and not self._sdk_path.exists():
            raise ValueError(f"sdk_path does not exist: {self._sdk_path}")
        self._max_iterations = max_iterations
        self._bash_timeout_s = bash_timeout_s
        self._run_python_timeout_s = run_python_timeout_s

    @staticmethod
    def name() -> str:
        return "terno-agent"

    @property
    def _install_agent_template_path(self) -> Path:
        return Path(__file__).parent / "setup.sh"

    def version(self) -> str | None:
        return self._version

    async def setup(self, environment: BaseEnvironment) -> None:
        await _exec_environment(environment, command="mkdir -p /installed-agent /logs/agent")
        if self._package is None and self._sdk_path.exists():
            with tempfile.TemporaryDirectory(prefix="terno-agent-sdk-") as tmp:
                staged_sdk = stage_sdk_source(self._sdk_path, Path(tmp) / "sdk")
                await environment.upload_dir(
                    source_dir=staged_sdk,
                    target_dir=CONTAINER_SDK_PATH,
                )

        script_path = self.logs_dir / "install.sh"
        script_path.write_text(self._install_agent_template_path.read_text())
        await environment.upload_file(
            source_path=script_path,
            target_path="/installed-agent/install.sh",
        )

        result = await _exec_environment(
            environment,
            command="bash /installed-agent/install.sh",
            env=self._env,
        )
        setup_dir = self.logs_dir / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        (setup_dir / "return-code.txt").write_text(str(result.return_code))
        if result.stdout:
            (setup_dir / "stdout.txt").write_text(result.stdout)
        if result.stderr:
            (setup_dir / "stderr.txt").write_text(result.stderr)
        if result.return_code != 0:
            log_paths = [setup_dir / "stdout.txt", setup_dir / "stderr.txt"]
            existing_logs = [str(path) for path in log_paths if path.exists()]
            output_tail = _tail(result.stdout or result.stderr or "")
            log_hint = ", ".join(existing_logs) if existing_logs else str(setup_dir)
            raise RuntimeError(
                f"Terno setup failed with exit code {result.return_code}. "
                f"See {log_hint}."
                + (f"\n\nLast setup output:\n{output_tail}" if output_tail else "")
            )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        rendered_instruction = (
            render_prompt_template(
                self._prompt_template_path,
                instruction,
            )
            if self._prompt_template_path
            else instruction
        )

        for i, exec_input in enumerate(self.create_run_agent_commands(rendered_instruction)):
            command_dir = self.logs_dir / f"command-{i}"
            command_dir.mkdir(parents=True, exist_ok=True)
            (command_dir / "command.txt").write_text(exec_input.command)

            result = await _exec_environment(
                environment,
                command=exec_input.command,
                cwd=exec_input.cwd,
                env=exec_input.env,
                timeout_sec=exec_input.timeout_sec,
            )

            (command_dir / "return-code.txt").write_text(str(result.return_code))
            if result.stdout:
                (command_dir / "stdout.txt").write_text(result.stdout)
            if result.stderr:
                (command_dir / "stderr.txt").write_text(result.stderr)

        self.populate_context_post_run(context)

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        payload = json.dumps({"task": instruction})
        command = (
            "${TERNO_AGENT_VENV:-/installed-agent/venv}/bin/python "
            "/installed-agent/run-terno-task.py "
            f"--task-json {shlex.quote(payload)} "
            "--workdir /app "
            f"--result-json {_RESULT_JSON} "
            f"2>&1 | tee {_RUN_LOG}"
        )
        return [
            ExecInput(
                command=command,
                env=self._env,
            )
        ]

    def populate_context_post_run(self, context: AgentContext) -> None:
        result_path = self.logs_dir / "terno-agent-result.json"
        if not result_path.exists():
            return
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        context.metadata = {
            **(context.metadata or {}),
            "answer": payload.get("answer"),
            "iterations": payload.get("iterations"),
            "cancelled": payload.get("cancelled"),
        }
        context.n_input_tokens = payload.get("total_input_tokens")
        context.n_output_tokens = payload.get("total_output_tokens")

    @property
    def _env(self) -> dict[str, str]:
        env = _filtered_env(
            prefixes=("ANTHROPIC_", "OPENAI_", "TERNO_"),
            names=("UV_INDEX_URL", "UV_EXTRA_INDEX_URL"),
        )
        provider, model = _split_model_name(self.model_name)
        if provider is not None:
            env["TERNO_LLM_PROVIDER"] = provider
        if model is not None:
            env["TERNO_LLM_MODEL"] = model
        if self._package is not None:
            env["TERNO_AGENT_PACKAGE"] = self._package
        elif self._sdk_path.exists():
            env["TERNO_AGENT_SDK_PATH"] = CONTAINER_SDK_PATH
        env["TERNO_BENCH_MAX_ITERATIONS"] = str(self._max_iterations)
        env["TERNO_BENCH_BASH_TIMEOUT_S"] = str(self._bash_timeout_s)
        env["TERNO_BENCH_RUN_PYTHON_TIMEOUT_S"] = str(self._run_python_timeout_s)
        return env


def _filtered_env(*, prefixes: tuple[str, ...], names: tuple[str, ...]) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in names or any(key.startswith(prefix) for prefix in prefixes):
            env[key] = value
    return env


def _split_model_name(model_name: str | None) -> tuple[str | None, str | None]:
    if not model_name:
        return (None, None)
    if "/" not in model_name:
        return (None, model_name)
    provider, model = model_name.split("/", 1)
    if provider in {"anthropic", "openai"}:
        return (provider, model)
    return (None, model_name)


def _tail(text: str, *, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


async def _exec_environment(
    environment: BaseEnvironment,
    *,
    command: str,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_sec: int | None = None,
) -> ExecResult:
    compose_runner = getattr(environment, "_run_docker_compose_command", None)
    if compose_runner is None:
        return await environment.exec(
            command=command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
        )

    exec_command = ["exec"]
    if cwd:
        exec_command.extend(["-w", cwd])
    if env:
        for key, value in env.items():
            exec_command.extend(["-e", f"{key}={value}"])
    exec_command.append("main")
    exec_command.extend(["bash", "-lc", command])

    return await compose_runner(exec_command, check=False, timeout_sec=timeout_sec)
