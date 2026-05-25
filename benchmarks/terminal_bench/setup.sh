#!/bin/sh
set -eu

if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  ln -s "$(command -v python3)" /usr/local/bin/python
fi

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install uv
fi

install_source="${TERNO_AGENT_SDK_PATH:-}"
if [ -n "$install_source" ] && [ -f "$install_source/pyproject.toml" ]; then
  (cd "$install_source" && uv pip install --system ".[benchmarks]")
elif [ -f "./pyproject.toml" ] && grep -q 'name = "terno-agent"' ./pyproject.toml; then
  uv pip install --system ".[benchmarks]"
else
  uv pip install --system "${TERNO_AGENT_PACKAGE:-terno-agent[benchmarks]}"
fi

cat > /installed-agent/run-terno-task.py <<'PY'
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != _SCRIPT_DIR]

from terno import Agent, Config


def main() -> int:
    args = _parse_args()
    task = _read_task(args)
    workdir = Path(args.workdir).expanduser().resolve()
    result_path = Path(args.result_json).expanduser() if args.result_json else None

    config = Config.for_benchmark(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
    )

    with Agent.from_config(
        config,
        workdir=workdir,
        max_iterations=args.max_iterations,
        bash_timeout_s=args.bash_timeout_s,
        run_python_timeout_s=args.run_python_timeout_s,
    ) as agent:
        result = agent.run(task)
        payload = {
            "answer": result.answer,
            "iterations": result.iterations,
            "cancelled": result.cancelled,
            "total_input_tokens": agent.usage.total_input_tokens,
            "total_output_tokens": agent.usage.total_output_tokens,
        }

    if result_path is not None:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(payload["answer"])
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task")
    source.add_argument("--task-json")
    source.add_argument("--task-file")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--result-json", default="")
    parser.add_argument("--provider", default=os.getenv("TERNO_LLM_PROVIDER"))
    parser.add_argument("--model", default=os.getenv("TERNO_LLM_MODEL"))
    parser.add_argument("--api-key", default=os.getenv("TERNO_LLM_API_KEY"))
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=int(os.getenv("TERNO_BENCH_MAX_ITERATIONS", "64")),
    )
    parser.add_argument(
        "--bash-timeout-s",
        type=int,
        default=int(os.getenv("TERNO_BENCH_BASH_TIMEOUT_S", "600")),
    )
    parser.add_argument(
        "--run-python-timeout-s",
        type=int,
        default=int(os.getenv("TERNO_BENCH_RUN_PYTHON_TIMEOUT_S", "120")),
    )
    return parser.parse_args()


def _read_task(args: argparse.Namespace) -> str:
    if args.task is not None:
        return args.task
    if args.task_file is not None:
        return Path(args.task_file).read_text(encoding="utf-8")
    payload: Any = json.loads(args.task_json)
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("task"), str):
        return payload["task"]
    raise ValueError("--task-json must be a JSON string or an object with a 'task' field")


if __name__ == "__main__":
    raise SystemExit(main())
PY

chmod +x /installed-agent/run-terno-task.py
