"""Terminal-Bench adapter for running terno-agent through its SDK."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand


class TernoTerminalBenchAgent(AbstractInstalledAgent):
    """Install terno-agent in the task container and run one SDK task."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        package: str | None = None,
        max_iterations: int = 64,
        bash_timeout_s: int = 600,
        run_python_timeout_s: int = 120,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name
        self._package = package
        self._max_iterations = max_iterations
        self._bash_timeout_s = bash_timeout_s
        self._run_python_timeout_s = run_python_timeout_s

    @staticmethod
    def name() -> str:
        return "terno-agent"

    @property
    def _install_agent_script_path(self) -> Path:
        return Path(__file__).parent / "setup.sh"

    @property
    def _env(self) -> dict[str, str]:
        env = _filtered_env(
            prefixes=("ANTHROPIC_", "OPENAI_", "TERNO_"),
            names=("UV_INDEX_URL", "UV_EXTRA_INDEX_URL"),
        )
        provider, model = _split_model_name(self._model_name)
        if provider is not None:
            env["TERNO_LLM_PROVIDER"] = provider
        if model is not None:
            env["TERNO_LLM_MODEL"] = model
        if self._package is not None:
            env["TERNO_AGENT_PACKAGE"] = self._package
        env["TERNO_BENCH_MAX_ITERATIONS"] = str(self._max_iterations)
        env["TERNO_BENCH_BASH_TIMEOUT_S"] = str(self._bash_timeout_s)
        env["TERNO_BENCH_RUN_PYTHON_TIMEOUT_S"] = str(self._run_python_timeout_s)
        return env

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        payload = json.dumps({"task": instruction})
        return [
            TerminalCommand(
                command=(
                    "python /installed-agent/run-terno-task.py "
                    f"--task-json {shlex.quote(payload)} "
                    "--workdir . "
                    "--result-json /installed-agent/terno-agent-result.json"
                ),
                max_timeout_sec=float("inf"),
                block=True,
            )
        ]


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

