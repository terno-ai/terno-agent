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
  `task_create` / `task_list` / `task_get` / `task_update`, `spawn_agent`,
  `ask_user`, `search_memory`.
- **Edits show coloured diffs.** Every `edit_file` call renders as a
  unified diff in the CLI (red removals, green additions, cyan hunk
  headers). `write_file` refuses to clobber existing files unless you
  pass `overwrite=true` — and when you do, the panel shows the diff
  between the on-disk content and the proposed rewrite. Edits become
  hard to miss; surprise overwrites become impossible.
- **Permission prompts for tool use.** A `pre_tool_use` hook gates
  every tool call. The CLI prompts with three Claude-style options:
  *allow once*, *allow this tool for the rest of the session*, or
  *deny and tell the agent what to do instead* (your reason goes
  straight back to the LLM as a tool-result error). Read-only helpers
  (`read_file`, `task_*`, `search_memory`, `ask_user`) skip the prompt.
- **Human-in-the-loop questions.** `ask_user` lets the agent pause and
  pose 1–4 multiple-choice questions when the request is genuinely
  ambiguous. Each question gets 2–4 options plus an automatic
  *"Other (custom text)"* choice, single- or multi-select. The CLI
  walks the user through them one at a time.
- **Sandboxed Python.** `run_python` runs inside Docker by default
  (`--network none`, read-only rootfs, mem/CPU caps); a local subprocess
  sandbox is available for dev. The tool is auto-hidden when no sandbox
  is reachable. The sandbox layer is **pluggable** — third-party backends
  (QEMU, browser-based, vendor APIs) register via Python entry points;
  see [Sandbox plugins](#sandbox-plugins).
- **MCP support.** Drop a `.terno/mcp.json` in your repo (Claude-Code-compatible
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
- **Persistent memory.** After each turn a background extractor mines
  the user's question and the assistant's answer for facts worth
  keeping — user preferences, project context, feedback, references,
  and short Q&A insights stored as key→value pairs. Everything lands
  in `<workdir>/.terno/memory` as markdown + vectors. On the next
  turn, the most relevant entries are recalled into context. The CLI
  prints a single dim `memory updated` line when something changed;
  the extractor itself runs silently. See [Memory](#memory).
- **Streaming + typed events.** Assistant text streams live; tool calls
  and results render with syntax-highlighted panels.
- **Native attachments.** Attach text files, documents, or images to a
  turn; Terno stores them locally, sends images through provider vision
  payloads when available, and keeps large text files bounded with
  selected chunks instead of dumping whole files into the prompt.
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
  write_file (gated)        shares manager +           .terno/mcp.json,
  edit_file (diff'd)        task store)                via uvx /
  bash                                                 npx / docker
  run_python (sandbox)                                 / HTTP / SSE)
  ask_user (HITL)
  search_memory
  activate_skill
  task_* (in-memory)

  every tool call passes through a pre_tool_use hook
  → permission prompt in CLI / SDK hook in code
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
TERNO_SANDBOX=docker                   # docker | local | none | <plugin name> | pkg.mod:Cls
# When the primary sandbox can't initialize, try this one (default: local).
# Set to 'none' to disable fallback.
TERNO_SANDBOX_FALLBACK=local
# Keep the sandbox container alive after the session ends so the next
# `terno` invocation in the same cwd attaches to it (vars + imports + /work
# files all preserved).
TERNO_SANDBOX_PERSIST=false
# Override the auto-derived per-cwd container name. Only used when persist=true.
# TERNO_SANDBOX_CONTAINER_NAME=my-sandbox
# Optional kwargs forwarded to the sandbox constructor:
# TERNO_SANDBOX_OPTIONS=image=python:3.13,memory=1g

# optional — only needed for `terno deep_research`
TERNO_DATABASE_URL=sqlite:///./demo.db

# optional — MCP loading is on by default. The project default is
# `.terno/mcp.json`; setting TERNO_MCP_CONFIG points at a specific file
# (which is loaded *instead* of the default for env-based discovery).
TERNO_MCP_ENABLED=true
TERNO_MCP_CONFIG=/path/to/mcp.json

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

# attach files to a one-shot task
terno ask --attach report.pdf --attach chart.png "summarize these"

# interactive REPL
terno chat

# in chat, queue files for the next turn
/attach report.pdf
/attachments

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

### What you'll see in chat

The first time the agent reaches for any side-effecting tool (`bash`,
`edit_file`, `write_file`, `run_python`, `spawn_agent`, MCP tools…)
you'll get a permission prompt:

```
╭─ permission required ──────────────────────────╮
│ Tool: bash                                     │
│                                                │
│ Arguments:                                     │
│ { "command": "uv run pytest -q" }              │
╰────────────────────────────────────────────────╯
  [1] Allow once
  [2] Allow 'bash' for the rest of this session
  [3] Deny and tell the agent what to do instead
permission> 2
```

Picking *(2)* skips future prompts for that tool name in this session.
*(3)* asks for a free-text reason that's sent back to the agent as a
tool-result error — the model adapts instead of being silently blocked.
Read-only tools (`read_file`, `task_*`, `search_memory`, `ask_user`)
skip the prompt entirely.

When the agent edits a file you get a coloured unified diff in place
of the raw arguments:

```
╭─ [terno] → edit_file ──────────────────────────╮
│ path: src/utils.py                             │
│ --- a/src/utils.py                             │
│ +++ b/src/utils.py                             │
│ @@ -10,3 +10,3 @@                              │
│  def add(a, b):                                │
│ -    return a+b                                │
│ +    return a + b                              │
╰────────────────────────────────────────────────╯
```

If the agent needs a decision from you it can call `ask_user`, which
walks through one or more multiple-choice questions:

```
╭─ Question 1/2 ─────────────────────────────────╮
│ Which database driver should I use?            │
╰────────────────────────────────────────────────╯
  [1] psycopg (sync) — default for scripts
  [2] asyncpg     — for the async data loader
  [3] Other (custom text)
select> 1
```

And whenever the background memory extractor commits something new
you'll see a single dim line:

```
memory updated
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

result = agent.run(
    "compare these files",
    attachments=["./report.pdf", "./chart.png"],
)
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
    mcp_enabled=False,            # skip .terno/mcp.json
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

`.terno/mcp.json` is loaded by default if present. To skip it for a single
run without touching the file:

```python
from terno_agent.config import Config

config = Config.from_env()
config.mcp_enabled = False        # don't load .terno/mcp.json
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

The agent reads a `.terno/mcp.json` file at startup. The format is the same one
[Claude Code][cc-mcp] and Cursor use, so existing configs paste in
unchanged. If the file is missing, MCP loading is a no-op; if a server
fails to start you get a stderr warning and the rest of the agent keeps
running.

[cc-mcp]: https://docs.claude.com/en/docs/claude-code/mcp

### Discovery order

- **Default (no path passed):** `$TERNO_MCP_CONFIG` if set, otherwise
  `./.terno/mcp.json`.
- **Explicit path passed** (e.g. `Config(mcp_config_path=...)`): the
  explicit file is loaded *together with* `./.terno/mcp.json` and the
  two are merged. Servers in the explicit file override the default
  on name conflict.

Set `TERNO_MCP_ENABLED=false` to disable MCP entirely.

### Tool naming

Every remote tool is registered as `mcp__{server}__{tool}` so it can't
collide with built-in tools. Server names with characters outside
`[A-Za-z0-9_-]` are sanitized.

### `.terno/mcp.json` examples

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

## Sandbox

`run_python` executes LLM-generated code inside a sandbox. Two backends
ship in the core package:

- **`docker`** (default) — `--network none`, read-only rootfs, memory + CPU
  caps. If the daemon isn't reachable, the agent **automatically falls
  back to `local`** with a one-line notice. Set `TERNO_SANDBOX_FALLBACK=none`
  for strict Docker-only behavior.
- **`local`** — direct subprocess. Not a security boundary; only for dev.

Plus a `none` sentinel that skips sandbox construction entirely.

### Session lifetime

The Docker sandbox keeps **one container per session** and runs a small
Python driver inside it. Each `run_python` call sends the snippet over
the container's stdin, so:

- Variables, imports, and function definitions persist across calls
  within the same `terno chat` session or `Agent` instance.
- Files written under `/work` persist for the lifetime of the container.
- Timeouts and cancellations SIGKILL the container and the next call
  recreates a fresh one — session state is lost but the agent keeps
  working.

By default the container is killed and removed when the session ends
(CLI exit, or `Agent.close()` / context-manager exit). To keep the
container across sessions:

```bash
TERNO_SANDBOX_PERSIST=true terno chat
# next session reuses the same per-cwd container:
TERNO_SANDBOX_PERSIST=true terno ask "print(session_token)"
```

The default container name is `terno-sandbox-<8-hex>` derived from the
current working directory, so each project gets its own container.
Override with `TERNO_SANDBOX_CONTAINER_NAME=my-name` if you want
something explicit.

The `local` backend is stateless — each call spawns a fresh subprocess
— and ignores the persist knobs.

### Fallback

`TERNO_SANDBOX_FALLBACK` (default `local`) names the backend to try when
the primary sandbox fails to initialize. Empty string or `none` disables
fallback. The notice prints only when fallback actually fires; the louder
warning is reserved for the case where no sandbox is usable.

### Selecting a backend

Env or CLI:

```bash
TERNO_SANDBOX=local terno ask "print(1+1)"
terno --sandbox local ask "print(1+1)"
```

Pass per-backend options:

```bash
# As env (CSV of key=value pairs):
TERNO_SANDBOX_OPTIONS="image=python:3.13,memory=1g" terno ask ...
# As CLI (repeatable):
terno --sandbox docker --sandbox-option image=python:3.13 ask ...
```

`TERNO_SANDBOX_IMAGE` still works for the docker backend; equivalent to
passing `--sandbox-option image=...`.

### Sandbox plugins

Third-party backends (QEMU, browser-based, vendor APIs) register via the
`terno_agent.sandboxes` entry-point group in their own package's
`pyproject.toml`:

```toml
[project.entry-points."terno_agent.sandboxes"]
qemu = "terno_qemu:QemuSandbox"
```

After `pip install terno-qemu` the backend is selectable as
`TERNO_SANDBOX=qemu` or `terno --sandbox qemu`. For one-off backends
that don't warrant publishing, point `TERNO_SANDBOX` at a fully-
qualified import string instead:

```bash
TERNO_SANDBOX=my_pkg.module:CustomSandbox terno ask ...
```

A plugin only needs to satisfy the `Sandbox` Protocol:

```python
from terno_agent.sandbox import ExecutionResult, Sandbox


class CustomSandbox:
    def __init__(self, **options):
        # Read whatever options the user passes via TERNO_SANDBOX_OPTIONS
        # / --sandbox-option / explicit kwargs.
        ...

    def run_python(
        self,
        code: str,
        *,
        timeout_s: int = 30,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        ...
        return ExecutionResult(stdout=..., stderr=..., exit_code=...)
```

Raise `terno_agent.core.exceptions.SandboxError` from `__init__` if the
backend can't initialize (missing dependency, daemon offline, etc.) —
the agent will warn and continue with `run_python` disabled, the same
way the built-in Docker check behaves today.

## Memory

Memory has three jobs: **learn** something across turns, **store** it
durably, and **recall** it cheaply on the next turn.

### How it runs

After every turn the agent fires a background extractor (daemon
thread — the user is never blocked). The extractor is a fresh
`TernoAgent` with the memory CRUD toolset; it reads the just-completed
transcript, decides what (if anything) is worth keeping, then calls
`save_memory` / `delete_memory`. Its tool activity is **not** mirrored
to the CLI — you just see one dim `memory updated` line when the
extractor actually saved or deleted something. Failures are swallowed
so extraction can never break the user-facing flow.

On the next turn, the user's incoming message is embedded and the
top-K most similar memories are recalled into the agent's context as
background hints.

### Five memory types, one location

| Type        | What it stores                                                        |
|-------------|-----------------------------------------------------------------------|
| `user`      | Facts about the human (role, expertise, preferences).                 |
| `feedback`  | Style / approach rules from the user, with a **Why:** line.           |
| `project`   | Non-obvious state of this repo (initiative, decisions, deadlines).    |
| `reference` | Pointers to external systems — Linear, Slack, Grafana, Confluence.    |
| `insight`   | Short Q&A fact distilled from a turn, stored as a key→value pair.     |

Everything is one markdown file per memory plus a vector sidecar:

```
<your-project>/.terno/memory/
  MEMORY.md              # human-readable index
  user-role.md
  feedback-testing.md
  project-auth-rewrite.md
  reference-grafana.md
  prod-database-host.md  # an insight; name = key, body = value
  .vectors.jsonl
```

Override the location with `TERNO_MEMORY_HOME=/some/path` (useful in
tests).

### Insights — the new bit

`insight` is the memory type the extractor uses to turn ordinary Q&A
into a cache. If the user asks *"where does prod live?"* and the
assistant answers *"`db.terno-prod.us-east-1.rds.amazonaws.com`"*,
the extractor saves an insight named `prod-database-host` whose body
is the bare hostname. Next time anyone (you or the agent) asks
something similar, `search_memory` surfaces it without re-derivation.
Insights are intentionally short and factual — never speculation,
never the agent's own opinions.

### Tools the agent sees

| Tool            | Available to       | Notes                                        |
|-----------------|--------------------|----------------------------------------------|
| `search_memory` | main agent         | RAG lookup over the store.                   |
| `list_memories` | extractor only     | Enumerate before saving (prefer UPDATE).     |
| `read_memory`   | extractor only     | Inspect an existing entry by name.           |
| `save_memory`   | extractor only     | Create or overwrite by name.                 |
| `delete_memory` | extractor only     | Remove stale or contradicted entries.        |

### Configuration

| Env var                       | Default                       | Meaning                       |
|-------------------------------|-------------------------------|-------------------------------|
| `TERNO_MEMORY_ENABLED`        | `true`                        | Master kill-switch.           |
| `TERNO_MEMORY_HOME`           | `<workdir>/.terno/memory`     | Storage location override.    |
| `TERNO_MEMORY_TOP_K`          | `5`                           | Recall budget per turn.       |
| `TERNO_EMBEDDING_API_KEY`     | falls back to `OPENAI_API_KEY`| Embedding provider key.       |

Per-session opt-out: `terno --no-memory chat`. Embedding requires the
`openai` extra and an API key; if it's missing the agent prints one
warning and keeps running without memory.

## Project layout

```
src/terno_agent/
  __init__.py          # public re-exports
  cli.py               # argparse entry, rich renderer, CliPrompter
                       # (ask_user), CliPermissionPrompter (pre_tool_use),
                       # coloured unified-diff renderer for edits
  config.py            # env + .env-driven Config
  core/                # message / tool / event / hook / exception types
                       # (HookEvent.PRE_TOOL_USE + PreToolUseContext live here)
  llm/                 # LLMClient protocol + Anthropic + OpenAI (streaming)
  agents/              # BaseAgent + the single TernoAgent
  prompts/             # the single SYSTEM_PROMPT
  tools/               # read_file, write_file (overwrite-gated),
                       # edit_file, bash, run_python, tasks,
                       # spawn_agent, ask_user, activate_skill
  skills/              # SKILL.md discovery + activate_skill adapter
  sandbox/             # Docker + local subprocess runners (for run_python)
  mcp/                 # .terno/mcp.json parser, runner resolver, async bridge,
                       # session manager, sync Tool adapter
  memory/              # background extractor (silent subagent) + retriever
                       # + single-dir markdown store at .terno/memory
                       # + SearchMemoryTool surfaced to the main agent
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

Apache
