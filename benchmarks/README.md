# Benchmark Runners

Install benchmark dependencies from the SDK package:

```bash
cd sdk
uv sync --extra benchmarks
cd ..
```

For SDK tests and local development tools too:

```bash
cd sdk
uv sync --extra benchmarks --extra dev
cd ..
```

Run Terminal-Bench with the installed-agent adapter:

```bash
export DOCKER_HOST=unix:///Users/sandeepakode/.docker/run/docker.sock  # macOS Docker Desktop, if needed

sdk/.venv/bin/tb run \
  --dataset terminal-bench-core==0.1.1 \
  --agent-import-path benchmarks.terminal_bench.terno_agent:TernoTerminalBenchAgent \
  --task-id hello-world \
  --n-concurrent 1 \
  --livestream \
  --log-level debug \
  --output-path runs \
  --run-id terno-one-task
```

When the adapter is imported from this repository, it copies the local `sdk/`
source tree into each task container and installs `.[benchmarks]` with `uv`.
That is the path you want while iterating locally. If you want to test a
published package instead, pass `--agent-kwarg package=terno-agent[benchmarks]`.
For Git URLs, set `TERNO_AGENT_PACKAGE` before launching `tb`, for example
`TERNO_AGENT_PACKAGE='terno-agent[benchmarks] @ git+https://...#subdirectory=sdk'`.
Terminal-Bench already runs the task in Docker, so the runner enables Terno's
local Python sandbox rather than starting nested Docker.

Generate SWE-bench predictions, then evaluate with the official harness:

```bash
sdk/.venv/bin/python -m benchmarks.swe_bench.run_inference \
  --dataset-name princeton-nlp/SWE-bench_Lite \
  --limit 1 \
  --output outputs/terno_predictions.jsonl

sdk/.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path outputs/terno_predictions.jsonl \
  --max_workers 1 \
  --run_id terno-agent-smoke
```
