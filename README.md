# Terno Agent

A multi-agent CLI that answers questions about your database. It plans,
generates and executes SQL, and can write Python and run it in a sandbox
(Docker by default) to analyze results.

## Features

- **Multi-agent**: an Orchestrator plans and delegates to a Database
  specialist (SQL) and a Coder specialist (sandboxed Python).
- **Provider-agnostic LLM**: Anthropic Claude and OpenAI — pick at runtime.
- **Any database**: anything SQLAlchemy can talk to (Postgres, MySQL, SQLite,
  …) via a single URL.
- **Sandboxed code execution**: Docker by default (`--network none`,
  read-only rootfs, mem/CPU caps); local subprocess fallback for dev.
- **Read-only by default**: only `SELECT` / `WITH` / `EXPLAIN` allowed unless
  you opt in.
- **Streaming + typed events**: assistant text streams live; tool calls and
  results render with syntax-highlighted panels and result tables.
- **CLI + library**: `terno ask "..."` from the shell, or
  `from terno import Agent` in Python.
- **Deep research**: a four-phase pipeline (org context, schema crawl,
  semantic annotation, validation) builds a queryable knowledge base
  from your database — run it as `terno deep_research` or from inside
  `terno chat` via `/deep_research`.

## Architecture

```
                ┌────────────────────────┐
   user query → │      Orchestrator      │  ← planner: decomposes into steps
                └────────────┬───────────┘    and routes to a specialist
                             │
                ┌────────────┼────────────┐
                ▼                         ▼
       ┌────────────────┐         ┌────────────────┐
       │ DatabaseAgent  │         │   CoderAgent   │
       │  • sql_query   │         │  • run_python  │
       │  • list_tables │         │   (sandboxed)  │
       │  • describe    │         │                │
       └────────┬───────┘         └────────┬───────┘
                │                          │
                ▼                          ▼
        SQLAlchemy engine            Docker / local
        (any DB URL)                 sandbox runner
```

All cross-cutting boundaries are protocols, so each layer is swappable:

| Boundary | Protocol       | Implementations              |
| -------- | -------------- | ---------------------------- |
| LLM      | `LLMClient`    | Anthropic, OpenAI            |
| Sandbox  | `Sandbox`      | Docker, local subprocess     |
| Tool     | `Tool`         | sql_query, run_python, ...   |
| Database | SQLAlchemy URL | Postgres, MySQL, SQLite, ... |

## Install

You can install with either `uv` or plain `pip`. Both produce the same `terno`
CLI on your `PATH` and the same importable `terno_agent` package.

### Optional extras

Pick only what you need (or use `all` to get everything):

| Extra       | What it pulls in                |
| ----------- | ------------------------------- |
| `anthropic` | the `anthropic` SDK             |
| `openai`    | the `openai` SDK                |
| `docker`    | the `docker` SDK for sandboxing |
| `postgres`  | `psycopg[binary]`               |
| `mysql`     | `pymysql`                       |
| `all`       | all of the above                |
| `dev`       | pytest, ruff, mypy              |

### With `uv` (recommended)

```bash
# install globally as a uv tool — `terno` works from anywhere
uv tool install terno-agent
uv tool install "terno-agent[anthropic,docker,postgres]"

# editable install from a local checkout
git clone https://github.com/terno-ai/terno-agent.git
cd terno-agent
uv tool install --editable ".[all]"

# add it as a dependency of another uv project
uv add terno-agent
uv add "terno-agent[anthropic,docker]"

# from a local path or git
uv add /path/to/terno_agent
uv add "git+https://github.com/terno-ai/terno-agent.git"
```

Refresh after changing `pyproject.toml`:

```bash
uv tool install --editable ".[all]" --force
```

### With `pip`

```bash
# from PyPI
pip install terno-agent
pip install "terno-agent[anthropic,docker,postgres]"

# from a local checkout (editable)
git clone https://github.com/terno-ai/terno-agent.git
cd terno-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# from git
pip install "git+https://github.com/terno-ai/terno-agent.git"

# from a built wheel
pip install ./dist/terno_agent-0.1.0-py3-none-any.whl
```

> **Tip:** if you install into a project venv with plain `pip`, you have to
> activate the venv (or use its `bin/terno`) to run the CLI. `uv tool install`
> avoids this by giving the CLI its own isolated environment on `PATH`.

## Configure

Configuration is read from environment variables, with `.env` auto-loaded from
your current working directory (or any parent). Process env wins over `.env`.

```bash
cp .env.example .env
# then edit:
ANTHROPIC_API_KEY=sk-ant-...        # or OPENAI_API_KEY=
TERNO_LLM_PROVIDER=anthropic        # anthropic | openai
TERNO_LLM_MODEL=claude-opus-4-7
TERNO_DATABASE_URL=sqlite:///./demo.db
TERNO_SANDBOX=docker                # docker | local | none
```

Run `terno config` to print the effective settings (API keys masked).

### SQLAlchemy URL examples

```bash
sqlite:///./relative.db                          # relative to CWD
sqlite:////absolute/path/to.db                   # absolute (4 slashes)
postgresql+psycopg://user:pass@host:5432/db
mysql+pymysql://user:pass@host:3306/db
```

## Use the CLI

```bash
# one-shot question
terno ask "what were the top 10 customers by revenue last quarter?"

# interactive REPL — accepts `/deep_research` to launch the
# knowledge-extraction pipeline mid-session
terno chat

# run the four-phase deep-research pipeline on its own
terno deep_research

# suppress streaming/activity, print only the final answer
terno -q ask "how many tracks are in the database?"

# show effective config
terno config

# show version
terno --version
```

If you installed with plain `pip` into a project venv and didn't activate it:

```bash
.venv/bin/terno ask "..."
# or
python -m terno_agent ask "..."
```

If you installed with `uv` into the current project rather than as a tool:

```bash
uv run terno ask "..."
```

## Use as a library

The simplest form — just pass an API key (everything else falls back
to env vars / `.env`):

```python
from terno import Agent

agent = Agent(api_key="sk-ant-...")
response = agent.run("Analyze this SQL: SELECT * FROM users WHERE created_at > now() - interval '7 days'")
print(response.answer)
```

Pass settings programmatically — no env vars required:

```python
from terno import Agent

agent = Agent(
    api_key="sk-ant-...",
    database_url="postgresql+psycopg://u:p@host/db",
    provider="anthropic",           # "anthropic" | "openai"
    model="claude-opus-4-7",
    sandbox="local",                # "docker" | "local" | "none"
)
print(agent.run("top 5 tables by row count").answer)
```

Read everything from env / `.env`:

```python
from terno import Agent

agent = Agent.from_env()
print(agent.run("how many active users signed up this week?").answer)
```

Stream events into your own UI:

```python
from terno import Agent
from terno_agent.core.events import TextDelta, ToolCallEvent, ToolResultEvent

def on_event(e):
    if isinstance(e, TextDelta):
        print(e.text, end="", flush=True)
    elif isinstance(e, ToolCallEvent):
        print(f"\n[tool] {e.call.name}({e.call.arguments})")
    elif isinstance(e, ToolResultEvent):
        print(f"[result] {e.result.content[:200]}")

agent = Agent(api_key="sk-ant-...", on_event=on_event)
agent.run("describe the users table and count rows")
```

Run deep research from code (same pipeline as `terno deep_research`):

```python
from terno import Agent

agent = Agent(api_key="sk-ant-...", database_url="sqlite:///./demo.db")
report = agent.deep_research()
print("ok" if report.ok else "failed")
```

> `from terno_agent import Agent` is equivalent — `terno` is just a
> short re-export of the same SDK.

## Project layout

```
src/terno_agent/
  cli.py              # argparse entry point + rich renderer
  config.py           # env + .env-driven Config
  core/               # message/tool/event/exception types
  llm/                # LLMClient protocol + Anthropic + OpenAI (streaming)
  agents/             # orchestrator + specialist agents
  tools/              # sql_query, run_python, list_tables, describe_table
  sandbox/            # Docker + local runners
  db/                 # SQLAlchemy engine & inspector
  prompts/            # system prompts per agent
tests/                # pytest suite
```

## Develop

```bash
# clone + editable install with dev extras
git clone https://github.com/terno-ai/terno-agent.git
cd terno-agent
uv venv --python 3.12
uv pip install -e ".[dev,all]"

# tests
uv run pytest -q

# lint / format / typecheck
uv run ruff check .
uv run ruff format .
uv run mypy src
```

Or with plain `pip`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,all]"
pytest -q
```

## License

MIT
