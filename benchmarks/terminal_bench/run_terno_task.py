"""Run one Terminal-Bench task inside the task container."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != _SCRIPT_DIR]


def main() -> int:
    from terno import Agent, Config

    args = _parse_args()
    task = _read_task(args)
    workdir = Path(args.workdir).expanduser().resolve()
    result_path = Path(args.result_json).expanduser() if args.result_json else None

    config = Config.for_benchmark(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        sandbox="local",
        sandbox_fallback="none",
    )

    with Agent.from_config(
        config,
        on_event=_print_event,
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


def _print_event(event: Any) -> None:
    event_name = type(event).__name__
    if event_name == "IterationStart":
        print(f"[terno] iteration {event.iteration}", flush=True)
    elif event_name == "TextDelta":
        print(event.text, end="", flush=True)
    elif event_name == "ToolCallEvent":
        try:
            arguments = json.dumps(event.call.arguments, sort_keys=True)
        except TypeError:
            arguments = repr(event.call.arguments)
        print(
            f"\n[terno] tool {event.call.name}: {_truncate(arguments)}",
            flush=True,
        )
    elif event_name == "ToolResultEvent":
        status = "error" if event.result.is_error else "ok"
        print(
            f"[terno] tool result ({status}): {_truncate(event.result.content)}",
            flush=True,
        )


def _truncate(value: object, limit: int = 1200) -> str:
    text = str(value).replace("\r", "\n")
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task", help="Task instruction text.")
    source.add_argument("--task-json", help='JSON string or {"task": "..."} payload.')
    source.add_argument("--task-file", help="Path to a UTF-8 task instruction file.")
    parser.add_argument("--workdir", default=".", help="Task working directory.")
    parser.add_argument("--result-json", default="", help="Optional path for run metadata.")
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
