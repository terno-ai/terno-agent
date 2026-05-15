# terno-agent

A multi-agent CLI that answers questions about your database. It plans, generates
and executes SQL, and can write Python and run it in a sandbox (Docker by
default) to analyze results.

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

| Boundary  | Protocol               | Implementations              |
| --------- | ---------------------- | ---------------------------- |
| LLM       | `LLMClient`            | Anthropic, OpenAI            |
| Sandbox   | `Sandbox`              | Docker, local subprocess     |
| Tool      | `Tool`                 | sql_query, run_python, ...   |
| Database  | SQLAlchemy URL         | Postgres, MySQL, SQLite, ... |

## Install

```bash
# from source
pip install -e ".[all]"

# minimal (pick what you need)
pip install -e ".[anthropic,docker,postgres]"
```

## Configure

Set env vars (or put them in `.env` next to where you run `terno`):

```bash
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY
export TERNO_LLM_PROVIDER=anthropic # anthropic | openai
export TERNO_LLM_MODEL=claude-opus-4-7
export TERNO_DATABASE_URL=postgresql+psycopg://user:pass@host:5432/db
export TERNO_SANDBOX=docker         # docker | local | none
```

## Use

```bash
# one-shot
terno ask "what were the top 10 customers by revenue last quarter?"

# interactive REPL
terno chat

# show effective config
terno config
```

## Use as a library

```python
from terno_agent import Agent

agent = Agent.from_env()
result = agent.run("how many active users signed up this week?")
print(result.answer)
```

## Project layout

```
src/terno_agent/
  cli.py              # argparse entry point
  config.py           # env + TOML config
  core/               # message/tool/exception types
  llm/                # provider-agnostic LLM clients
  agents/             # orchestrator + specialist agents
  tools/              # sql_query, run_python, ...
  sandbox/            # Docker + local runners
  db/                 # SQLAlchemy engine & inspector
  prompts/            # system prompts per agent
```
