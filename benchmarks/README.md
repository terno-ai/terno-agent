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
sdk/.venv/bin/tb run \
  --dataset terminal-bench-core==head \
  --agent-import-path benchmarks.terminal_bench.terno_agent:TernoTerminalBenchAgent \
  --task-id hello-world
```

The Terminal-Bench setup script installs `terno-agent[benchmarks]` inside each
task container. If you want to test an unpublished build, pass
`--agent-kwarg package='terno-agent[benchmarks] @ git+https://...#subdirectory=sdk'`
or set `TERNO_AGENT_PACKAGE` before launching `tb`.

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
