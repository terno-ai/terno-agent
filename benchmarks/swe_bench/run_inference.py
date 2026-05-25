"""Generate SWE-bench predictions by running terno-agent on each instance."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from terno import Agent, Config


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    instances = _load_instances(args)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _existing_predictions(output_path) if args.skip_existing else set()
    for instance in instances:
        instance_id = str(instance["instance_id"])
        if instance_id in existing:
            print(f"skip existing {instance_id}")
            continue
        print(f"run {instance_id}")
        try:
            prediction = _run_instance(args, instance)
        except Exception:
            if args.fail_fast:
                raise
            print(f"error while running {instance_id}", file=sys.stderr)
            prediction = {
                "instance_id": instance_id,
                "model_name_or_path": args.model_name_or_path,
                "model_patch": "",
            }
        _append_jsonl(output_path, prediction)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="outputs/terno_predictions.jsonl")
    parser.add_argument("--workdir", default=".terno/swe-bench/workdirs")
    parser.add_argument("--repo-cache", default=".terno/swe-bench/repos")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-name-or-path", default="terno-agent")
    parser.add_argument("--max-iterations", type=int, default=64)
    parser.add_argument("--bash-timeout-s", type=int, default=600)
    parser.add_argument("--run-python-timeout-s", type=int, default=120)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--reuse-workdir", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def _load_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing benchmark dependencies. Install them with "
            "`uv sync --extra benchmarks` or `uv pip install '.[benchmarks]'`."
        ) from exc

    dataset = load_dataset(args.dataset_name, split=args.split)
    instances = [dict(item) for item in dataset]
    if args.instance_id:
        wanted = set(args.instance_id)
        instances = [item for item in instances if item.get("instance_id") in wanted]
    if args.limit:
        instances = instances[: args.limit]
    return instances


def _run_instance(args: argparse.Namespace, instance: dict[str, Any]) -> dict[str, str]:
    instance_id = str(instance["instance_id"])
    repo_path = _prepare_checkout(
        repo=str(instance["repo"]),
        base_commit=str(instance["base_commit"]),
        instance_id=instance_id,
        workdir_root=Path(args.workdir).expanduser().resolve(),
        repo_cache=Path(args.repo_cache).expanduser().resolve(),
        reuse_workdir=args.reuse_workdir,
    )

    config = Config.for_benchmark(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
    )
    with Agent.from_config(
        config,
        workdir=repo_path,
        max_iterations=args.max_iterations,
        bash_timeout_s=args.bash_timeout_s,
        run_python_timeout_s=args.run_python_timeout_s,
    ) as agent:
        result = agent.run(_prompt_for_instance(instance))

    _write_run_metadata(args, instance_id, result.answer, result.iterations, agent.usage)
    _run(["git", "add", "-N", "."], cwd=repo_path, check=False)
    patch = _run(["git", "diff", "--binary"], cwd=repo_path).stdout
    return {
        "instance_id": instance_id,
        "model_name_or_path": args.model_name_or_path,
        "model_patch": patch,
    }


def _prepare_checkout(
    *,
    repo: str,
    base_commit: str,
    instance_id: str,
    workdir_root: Path,
    repo_cache: Path,
    reuse_workdir: bool,
) -> Path:
    workdir_root.mkdir(parents=True, exist_ok=True)
    repo_cache.mkdir(parents=True, exist_ok=True)
    repo_slug = _repo_slug(repo)
    mirror = repo_cache / f"{repo_slug.replace('/', '__')}.git"
    repo_url = f"https://github.com/{repo_slug}.git"

    if mirror.exists():
        _run(["git", "remote", "update", "--prune"], cwd=mirror, check=False)
    else:
        _run(["git", "clone", "--mirror", repo_url, str(mirror)])

    checkout = workdir_root / _safe_path_name(instance_id)
    if checkout.exists() and not reuse_workdir:
        shutil.rmtree(checkout)
    if not checkout.exists():
        _run(["git", "clone", str(mirror), str(checkout)])
    _run(["git", "checkout", "--force", base_commit], cwd=checkout)
    _run(["git", "clean", "-fdx"], cwd=checkout)
    return checkout


def _prompt_for_instance(instance: dict[str, Any]) -> str:
    problem = str(instance.get("problem_statement") or instance.get("text") or "")
    hints = []
    if instance.get("hints_text"):
        hints.append(f"Hints:\n{instance['hints_text']}")
    if instance.get("FAIL_TO_PASS"):
        hints.append(f"Failing tests expected to pass:\n{instance['FAIL_TO_PASS']}")
    suffix = "\n\n".join(hints)
    if suffix:
        suffix = f"\n\n{suffix}"
    return (
        "You are solving a SWE-bench task in the current repository checkout.\n"
        "Edit the source files to fix the issue. Run targeted tests when useful. "
        "Do not create a patch file; leave the working tree with the solution.\n\n"
        f"Instance: {instance['instance_id']}\n"
        f"Repository: {instance['repo']}\n\n"
        f"Issue:\n{problem}"
        f"{suffix}"
    )


def _write_run_metadata(
    args: argparse.Namespace,
    instance_id: str,
    answer: str,
    iterations: int,
    usage: Any,
) -> None:
    output_path = Path(args.output).expanduser().resolve()
    metadata_dir = output_path.parent / "terno_run_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "instance_id": instance_id,
        "answer": answer,
        "iterations": iterations,
        "total_input_tokens": usage.total_input_tokens,
        "total_output_tokens": usage.total_output_tokens,
    }
    (metadata_dir / f"{_safe_path_name(instance_id)}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{cmd!r} failed with exit {result.returncode}: {detail}")
    return result


def _repo_slug(repo: str) -> str:
    if "/" in repo:
        return repo
    if "__" in repo:
        owner, name = repo.split("__", 1)
        return f"{owner}/{name}"
    raise ValueError(f"Cannot infer GitHub repo slug from {repo!r}")


def _safe_path_name(value: str) -> str:
    return value.replace("/", "__").replace(":", "_")


def _existing_predictions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    existing: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if "instance_id" in payload:
            existing.add(str(payload["instance_id"]))
    return existing


def _append_jsonl(path: Path, payload: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
