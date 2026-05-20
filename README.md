# Terno Agent

A single-agent coding CLI + SDK. The agent reads and edits files, runs
shell commands, executes Python in a sandbox, tracks its own task list,
spawns subagents for parallel work, and pulls in additional tools from
any [Model Context Protocol (MCP)][mcp] servers you configure.

[mcp]: https://modelcontextprotocol.io

## Features

- **One agent, one prompt.** No multi-agent orchestration — a single
  `TernoAgent` powered by Anthropic Claude or OpenAI.
- **Built-in tools:**
  `read_file`, `write_file`, `edit_file`, `bash`, `run_python`,
  `task_create` / `task_list` / `task_get` / `task_update`, `spawn_agent`.
- **Sandboxed Python.** `run_python` runs inside Docker by default
  (`--network none`, read-only rootfs, mem/CPU caps); a local subprocess
  sandbox is available for dev. The tool is auto-hidden when no sandbox
  is reachable.
- **MCP support.** Drop a `.mcp.json` in your repo (Claude-Code-compatible
  format) and every remote tool shows up as `mcp__server__tool`. Servers
  can be launched via `uvx`, `npx`, or Docker, or connected to over
  HTTP/SSE. See [MCP](#mcp).
- **Agent Skills.** Terno ships common built-in skills for data science
  and general-purpose work, and you can add `SKILL.md`-based skills in
  `.terno/skills/`, `.agents/skills/`, or `.claude/skills/`. Either
  Anthropic or OpenAI models load full instructions through
  `activate_skill` only when needed.
- **Subagent spawner.** `spawn_agent` recursively launches a fresh
  `TernoAgent` with a caller-supplied system prompt — useful for isolating
  focused subtasks from your main context.
- **Persistent memory.** The agent extracts long-lived facts (user
  preferences, project context, feedback, external references) after
  each task into markdown files indexed by vector embeddings, and
  recalls the most relevant ones at the start of the next turn. See
  [Memory](#memory).
- **Streaming + typed events.** Assistant text streams live; tool calls
  and results render with syntax-highlighted panels.
- **CLI + library.** `terno ask "..."` / `terno chat` from the shell, or
  `from terno import Agent` in Python.
- **Deep research (database).** A four-phase pipeline (org context,
  schema crawl, semantic annotation, validation) builds a queryable
  knowledge base from any database — run it as `terno deep_research`
  or from inside `terno chat` via `/deep_research`.

## Architecture

```
                    ┌──────────────────────────┐
       user task →  │       TernoAgent         │
                    │  (single sync run loop)  │
                    └────────────┬─────────────┘
                                 │
       ┌─────────────────────────┼─────────────────────────┐
       ▼                         ▼                         ▼
  built-in tools           spawn_agent                MCP tools
  read_file                (fresh TernoAgent,         (loaded from
  write_file                shares manager +           .mcp.json,
  edit_file                 task store)                via uvx /
  bash                                                 npx / docker
  run_python (sandbox)                                 / HTTP / SSE)
  activate_skill
  task_* (in-memory)
```

All cross-cutting boundaries are protocols, so each layer is swappable:

| Boundary | Protocol     | Implementations                          |
| -------- | ------------ | ---------------------------------------- |
| LLM      | `LLMClient`  | Anthropic, OpenAI                        |
| Sandbox  | `Sandbox`    | Docker, local subprocess                 |
| Tool     | `Tool`       | file ops, bash, run_python, tasks, MCP   |
| Database | SQLAlchemy   | only used by `deep_research`             |

## Install

You can install with either `uv` or plain `pip`. Both produce the same
`terno` CLI on your `PATH` and the same importable `terno_agent` package.

### Optional extras

| Extra       | What it pulls in                  |
| ----------- | --------------------------------- |
| `anthropic` | the `anthropic` SDK               |
| `openai`    | the `openai` SDK                  |
| `docker`    | the `docker` SDK for sandboxing   |
| `mcp`       | the official `mcp` Python client  |
| `postgres`  | `psycopg[binary]`                 |
| `mysql`     | `pymysql`                         |
| `all`       | all of the above                  |
| `dev`       | pytest, ruff, mypy                |

### With `uv` (recommended)

```bash
# install globally as a uv tool — `terno` works from anywhere
uv tool install terno-agent
uv tool install "terno-agent[anthropic,docker,mcp]"

# editable install from a local checkout
git clone https://github.com/terno-ai/terno-agent.git
cd terno-agent
uv tool install --editable ".[all]"

# refresh after editing pyproject.toml
uv tool install --editable ".[all]" --force
```

### With `pip`

```bash
pip install terno-agent
pip install "terno-agent[anthropic,docker,mcp]"

# editable
git clone https://github.com/terno-ai/terno-agent.git
cd terno-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

> If you install into a project venv with plain `pip`, activate it (or use
> `.venv/bin/terno`). `uv tool install` avoids this by giving the CLI its
> own isolated environment on `PATH`.

## Configure

Configuration is read from environment variables, with `.env` auto-loaded
from the current working directory or any parent. Process env wins over
`.env`.

```bash
cp .env.example .env
# then edit:
ANTHROPIC_API_KEY=sk-ant-...           # or OPENAI_API_KEY=
TERNO_LLM_PROVIDER=anthropic           # anthropic | openai
TERNO_LLM_MODEL=claude-opus-4-7
TERNO_SANDBOX=docker                   # docker | local | none

# optional — only needed for `terno deep_research`
TERNO_DATABASE_URL=sqlite:///./demo.db

# optional — MCP loading is on by default; point at a specific file
# or disable it entirely
TERNO_MCP_ENABLED=true
TERNO_MCP_CONFIG=/path/to/.mcp.json

# optional — Agent Skills are on by default
TERNO_SKILLS_ENABLED=true
# TERNO_SKILL_PATHS=/path/to/skills

# optional — memory is on by default; needs an OpenAI key for embeddings
TERNO_MEMORY_ENABLED=true
TERNO_MEMORY_TOP_K=5
TERNO_EMBEDDING_MODEL=text-embedding-3-small
# TERNO_EMBEDDING_API_KEY=     # falls back to OPENAI_API_KEY
```

Run `terno config` to print the effective settings (API keys masked).

## Use the CLI

```bash
# one-shot task
terno ask "refactor utils.py into smaller modules"

# interactive REPL
terno chat

# suppress streaming/activity, print only the final answer
terno -q ask "explain how config.py loads .env files"

# four-phase deep research over the configured database
terno deep_research

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

## Use as a library

`from terno import Agent` is the SDK entry point. `Agent` is a thin
facade — the inference loop, tool dispatch, MCP manager, and memory
pipeline all live behind it. You drive the agent via one method
(`run` / `ask`), receive results as an `AgentRun` dataclass, and
optionally subscribe to a stream of typed events.

### Quick start

```python
from terno import Agent

# All kwargs are optional — anything missing falls back to env / .env.
agent = Agent(api_key="sk-ant-...")
result = agent.run("read README.md and summarize what the agent does")
print(result.answer)
```

`Agent(...)` accepts `api_key`, `provider`, `model`, `database_url`,
`config`, and `on_event`. For everything else (sandbox, MCP, memory),
build a `Config` and pass it via `config=`.

### Constructors

```python
from terno import Agent
from terno_agent.config import Config

# 1. Inline kwargs (simplest). Missing fields read from env / .env.
agent = Agent(
    api_key="sk-ant-...",
    provider="anthropic",         # "anthropic" | "openai"
    model="claude-opus-4-7",
)

# 2. Everything from environment + .env.
agent = Agent.from_env()

# 3. Explicit Config — exposes the full surface (sandbox, MCP, memory…).
config = Config(
    llm_provider="anthropic",
    llm_api_key="sk-ant-...",
    sandbox="local",              # "docker" | "local" | "none"
    mcp_enabled=False,            # skip .mcp.json
    memory_enabled=True,          # persistent recall
    memory_top_k=5,
    embedding_provider="openai",
    embedding_api_key="sk-openai-...",
)
agent = Agent.from_config(config)
```

### Running tasks

`run(task)` and `ask(task)` are equivalent. Both block until the agent
emits a final assistant message (no remaining tool calls) and return an
`AgentRun`:

```python
result = agent.run("count Python files under src/")
result.answer       # str — final assistant message
result.iterations   # int — LLM turns taken
result.trace        # list[Message] — full conversation incl. tool calls
```

Each call starts a fresh conversation (new system prompt + your task).
The agent keeps task-tracking state (`task_list`) across calls within
one `Agent` instance.

### Lifecycle — always close when done

If you configured MCP servers or memory (both on by default), the agent
owns background resources: subprocesses for stdio MCP servers, an
asyncio loop on a worker thread, OpenAI clients. Use it as a context
manager so they shut down cleanly:

```python
from terno import Agent

with Agent.from_env() as agent:
    print(agent.run("...").answer)
    print(agent.run("...follow-up").answer)
```

Or call `agent.close()` explicitly. Both are idempotent; `close()` is
also registered with `atexit` as a defensive net.

### Streaming events

Pass `on_event=` to receive typed events as the agent works. The hook
runs synchronously inside the agent loop — keep it fast.

```python
from terno import Agent
from terno_agent.core.events import (
    IterationStart, TextDelta, ToolCallEvent, ToolResultEvent, TurnEnd,
)

def on_event(e):
    if isinstance(e, TextDelta):
        print(e.text, end="", flush=True)
    elif isinstance(e, ToolCallEvent):
        print(f"\n[tool] {e.call.name}({e.call.arguments})")
    elif isinstance(e, ToolResultEvent):
        marker = "✗" if e.result.is_error else "✓"
        print(f" {marker} {e.result.content[:200]}")
    # IterationStart and TurnEnd are also available

with Agent(api_key="sk-ant-...", on_event=on_event) as agent:
    agent.run("explain src/terno_agent/agents/terno.py in two sentences")
```

| Event              | When                                        |
|--------------------|---------------------------------------------|
| `IterationStart`   | start of each LLM call                      |
| `TextDelta`        | streamed token chunk from the assistant     |
| `ToolCallEvent`    | the LLM picked a tool, before it runs       |
| `ToolResultEvent`  | tool returned (carries success / error)     |
| `TurnEnd`          | LLM call finished, message appended         |

### Disabling MCP or memory in code

`.mcp.json` is loaded by default if present. To skip it for a single
run without touching the file:

```python
from terno_agent.config import Config

config = Config.from_env()
config.mcp_enabled = False        # don't load .mcp.json
config.skills_enabled = False     # don't discover Agent Skills
config.memory_enabled = False     # no recall, no extraction
agent = Agent.from_config(config)
```

## Agent Skills

Terno supports Agent Skills using the standard `SKILL.md` shape: a skill
is a directory containing a required `SKILL.md` file with YAML
frontmatter (`name` and `description`) followed by Markdown
instructions. At startup, Terno loads only each skill's metadata into the
system prompt. When the model decides a skill is relevant, it calls
`activate_skill(name)` to load the full instructions and a capped list of
bundled resources.

Built-in skills are available by default:

| Skill | Use for |
| ----- | ------- |
| `code-review` | code reviews, regressions, missing tests |
| `debugging` | failing tests, runtime errors, flaky behavior |
| `data-analysis` | dataset exploration, summaries, metrics |
| `data-cleaning` | messy data, deduplication, standardization |
| `data-visualization` | charts, dashboards, visual summaries |
| `documentation` | README, API docs, runbooks, tutorials |
| `machine-learning` | models, experiments, metrics, leakage checks |
| `python-data` | dataframe, notebook, numerical, and file-based analysis |
| `research-synthesis` | research, comparisons, decision briefs |
| `sql-analysis` | analytical SQL, joins, cohorts, funnels |
| `task-planning` | multi-step planning, milestones, risks |

Custom discovery checks these roots, with later roots overriding earlier
ones. That means project skills can replace built-in skills with the
same name:

```text
built-in packaged skills
~/.terno/skills/
~/.agents/skills/
~/.claude/skills/
<cwd or ancestor>/.terno/skills/
<cwd or ancestor>/.agents/skills/
<cwd or ancestor>/.claude/skills/
```

Add extra roots with `TERNO_SKILL_PATHS` (use your OS path separator).
Set `TERNO_SKILLS_ENABLED=false` to disable skills for a session. The
implementation is provider-neutral: skills are just prompt context plus
a normal tool call, so they work with both `anthropic` and `openai`.

Minimal skill:

```text
.agents/skills/code-review/SKILL.md
```

```markdown
---
name: code-review
description: Review code for regressions, missing tests, and maintainability. Use when the user asks for a code review.
---

# Code Review

Focus on bugs and behavioral risk first. Report findings with file and
line references, then summarize test coverage.
```

### Deep research over a database

```python
from terno import Agent

agent = Agent(api_key="sk-ant-...", database_url="sqlite:///./demo.db")
report = agent.deep_research()
print("ok" if report.ok else "failed")
```

> `from terno_agent import Agent` is equivalent — `terno` is a short
> re-export of the same SDK.

## MCP

The agent reads a `.mcp.json` file at startup. The format is the same one
[Claude Code][cc-mcp] and Cursor use, so existing configs paste in
unchanged. If the file is missing, MCP loading is a no-op; if a server
fails to start you get a stderr warning and the rest of the agent keeps
running.

[cc-mcp]: https://docs.claude.com/en/docs/claude-code/mcp

### Discovery order

1. `$TERNO_MCP_CONFIG`
2. `./.mcp.json` (current working directory)
3. `~/.terno/mcp.json`

First hit wins. Set `TERNO_MCP_ENABLED=false` to disable MCP entirely.

### Tool naming

Every remote tool is registered as `mcp__{server}__{tool}` so it can't
collide with built-in tools. Server names with characters outside
`[A-Za-z0-9_-]` are sanitized.

### `.mcp.json` examples

**Raw stdio** (Claude-Code-compatible — terno invokes the command verbatim):

```json
{
  "mcpServers": {
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/me/work"
      ]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

`${VAR}` is expanded from your shell env at load time.

**Higher-level `runner` block** (terno picks the runtime):

```json
{
  "mcpServers": {
    "puppeteer": {
      "runner": {
        "type": "auto",
        "package": "@modelcontextprotocol/server-puppeteer",
        "image": "mcp/puppeteer:latest",
        "docker": {
          "mounts": [
            { "source": "/tmp/screenshots", "target": "/screenshots" }
          ]
        }
      }
    }
  }
}
```

`runner.type`: `auto` | `uvx` | `npx` | `docker` | `command`.
With `auto`, terno prefers Docker (when both an `image` and `docker` are
available), falls back to `uvx` / `npx` based on `package_type` or a name
heuristic.

**Web (HTTP / SSE)** — no subprocess:

```json
{
  "mcpServers": {
    "linear": {
      "url": "https://mcp.linear.app/sse",
      "headers": { "Authorization": "Bearer ${LINEAR_MCP_TOKEN}" }
    }
  }
}
```

`transport`: `sse` | `http`. Auto-detected from the URL when omitted
(`.../sse` → SSE, otherwise streamable HTTP).

### Pinning Python for `uvx`

Some MCP packages have transitive deps without wheels for newer Python
versions. If `uvx` resolves to a Python that breaks the install, force a
version:

```json
"my-server": {
  "command": "uvx",
  "args": ["--python", "3.12", "--from", "some-pkg", "the-cmd"]
}
```

(or set `"UV_PYTHON": "3.12"` in the server's `env` block).

## Memory

After every task, an extraction subagent reviews the conversation and
decides whether anything is worth keeping. If so, it writes one
markdown file per memory and embeds it with OpenAI's
`text-embedding-3-small`. On the next turn, the user's task is
embedded too, and the top-K most similar memories are prepended to the
system prompt as extra context.

There are four memory types, each with a different scope:

| Type        | Scope    | Use                                          |
|-------------|----------|----------------------------------------------|
| `user`      | global   | Facts about the human (role, expertise…)     |
| `feedback`  | global   | "Do this", "don't do that" + the reason      |
| `project`   | workdir  | Goals, deadlines, decisions for this repo    |
| `reference` | workdir  | Pointers to Linear, Slack, dashboards…       |

Storage paths (markdown + a single-vector-store JSON):

```
~/.terno_agent/memory/        # global memories (user, feedback)
<your-project>/.terno/memory/ # workdir memories (project, reference)
```

The agent has a `search_memory` tool for ad-hoc lookups when it
suspects relevant context wasn't recalled automatically.

Disable for a single session with `terno --no-memory chat`, or
permanently with `TERNO_MEMORY_ENABLED=false`. Embedding the contents
requires the `openai` extra and an `OPENAI_API_KEY` (or an explicit
`TERNO_EMBEDDING_API_KEY`); if that's missing the agent prints one
warning and keeps running without memory.

## Project layout

```
src/terno_agent/
  __init__.py          # public re-exports
  cli.py               # argparse entry point + rich renderer
  config.py            # env + .env-driven Config
  core/                # message / tool / event / exception types
  llm/                 # LLMClient protocol + Anthropic + OpenAI (streaming)
  agents/              # BaseAgent + the single TernoAgent
  prompts/             # the single SYSTEM_PROMPT
  tools/               # read_file, write_file, edit_file, bash,
                       # run_python, tasks, spawn_agent, activate_skill
  skills/              # SKILL.md discovery + activate_skill adapter
  sandbox/             # Docker + local subprocess runners (for run_python)
  mcp/                 # .mcp.json parser, runner resolver, async bridge,
                       # session manager, sync Tool adapter
  memory/              # extractor + retriever + on-disk markdown store
                       # + a SearchMemoryTool surfaced to the agent
  rag/                 # embedding client + file-backed vector store
                       # (shared infrastructure for memory)
  knowledge/           # deep_research pipeline (uses db/ + an LLM)
  db/                  # SQLAlchemy engine + inspector (knowledge only)
tests/                 # pytest suite, including tests/mcp/
```

## Develop

```bash
git clone https://github.com/terno-ai/terno-agent.git
cd terno-agent
uv venv --python 3.12
uv pip install -e ".[dev,all]"

uv run pytest -q          # tests
uv run ruff check .       # lint
uv run ruff format .      # format
uv run mypy src           # type check
```

Or with plain `pip`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,all]"
pytest -q
```

## License

MIT
