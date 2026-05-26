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

Run legacy Terminal-Bench with the installed-agent adapter:

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

Run Terminal-Bench 2.0 through Harbor:

```bash
export DOCKER_HOST=unix:///Users/sandeepakode/.docker/run/docker.sock  # macOS Docker Desktop, if needed

sdk/.venv/bin/harbor run \
  --dataset terminal-bench@2.0 \
  --agent-import-path benchmarks.terminal_bench.terno_harbor_agent:TernoHarborAgent \
  --model anthropic/claude-sonnet-4-20250514 \
  --task-name mteb-retrieve \
  --n-concurrent 1 \
  --debug \
  --jobs-dir jobs \
  --job-name terno-tb2-mteb-retrieve
```

Harbor writes the agent command output to
`jobs/<job-name>/<trial>/agent/terno-agent.txt` and run metadata to
`jobs/<job-name>/<trial>/agent/terno-agent-result.json`.

If Harbor appears stuck at `starting environment...`, use the debug Docker
environment config so Compose output is streamed to
`jobs/<job-name>/<trial>/environment.log`:

```bash
sdk/.venv/bin/harbor run \
  --config benchmarks/terminal_bench/tb2_debug_environment.yaml \
  --dataset terminal-bench@2.0 \
  --task-name mteb-retrieve \
  --agent-import-path benchmarks.terminal_bench.terno_harbor_agent:TernoHarborAgent \
  --model anthropic/claude-sonnet-4-20250514 \
  --job-name terno-tb2-mteb-retrieve-debug
```

The `mteb-retrieve` task declares the prebuilt image
`alexgshaw/mteb-retrieve:20251031`; a long first run is usually Docker pulling
that image. If the prebuilt pull keeps hanging, add `--force-build` to build
from the task's local Dockerfile instead.

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
