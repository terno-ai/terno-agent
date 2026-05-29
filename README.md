# Terno Agent

A single-agent coding CLI and Python SDK. The agent reads and edits files,
runs shell commands, executes Python in a sandbox, tracks its own task list,
spawns subagents for parallel work, recalls persistent memory, and pulls in
extra tools from any [Model Context Protocol (MCP)][mcp] servers you configure.

This repository is a monorepo: the agent itself lives in [`sdk/`](sdk), and the
rest is a reference web app and benchmark harness built on top of it.

[mcp]: https://modelcontextprotocol.io

## Repository layout

| Path | What it is | Details |
| ---- | ---------- | ------- |
| [`sdk/`](sdk) | The `terno-agent` package — the `terno` CLI and the `terno`/`terno_agent` Python SDK. This is the core of the project. | [sdk/README.md](sdk/README.md) |
| [`backend/`](backend) | FastAPI server that wraps the SDK, streams turns over a WebSocket, and persists chat sessions + messages to SQLite. | — |
| [`frontend/`](frontend) | React + TypeScript + Vite demo UI (landing + chat) that talks to the backend over WebSocket. | [frontend/README.md](frontend/README.md) |
| [`benchmarks/`](benchmarks) | Runners for Terminal-Bench (1.x and 2.0 via Harbor) and SWE-bench. | [benchmarks/README.md](benchmarks/README.md) |
| `jobs/`, `runs/` | Output directories for benchmark runs. | — |

## Installation

### With `uv` (recommended)

```bash
uv add terno-agent
```

### With `pip`

```bash
pip install terno-agent
```

## Quick start

### 1. The CLI / SDK

The agent is fully usable on its own — no web app required.

```bash
cd sdk
uv tool install --editable ".[all]"     # installs the `terno` CLI on PATH
cp .env.example .env                     # then set ANTHROPIC_API_KEY or OPENAI_API_KEY

terno ask "refactor utils.py into smaller modules"
terno chat                               # interactive REPL
```

Or use it as a library:

```python
from terno import Agent

with Agent.from_env() as agent:
    print(agent.run("summarize what this repo does").answer)
```

See [sdk/README.md](sdk/README.md) for the full surface: tools, permission
prompts, sandbox backends, MCP config, Agent Skills, memory, and the SDK API.

### 2. The demo web app

The backend wraps the SDK as a WebSocket chat server; the frontend is a React UI
that connects to it. Run them in two terminals.

Backend (FastAPI + SQLite, depends on the local `sdk/` via an editable install):

```bash
cd backend
cp .env.example .env                     # set your LLM API key
uv sync
uv run uvicorn main:app --reload         # serves http://127.0.0.1:8000
```

Frontend (Vite dev server on port 5173, which the backend's CORS allows):

```bash
cd frontend
npm install
npm run dev                              # serves http://localhost:5173
```

### 3. Benchmarks

```bash
cd sdk
uv sync --extra benchmarks
```

Then run Terminal-Bench or SWE-bench as documented in
[benchmarks/README.md](benchmarks/README.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
