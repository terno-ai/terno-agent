---
name: data-exploration
description: Create knowledge of the new fresh connected databases by exploring it and asking user questions about it
---

# Data Exploration

Your goal for this session is not to answer a business question. It is to
build a durable, reusable understanding of ONE datasource and persist it as
structured memory, so that future sessions can answer questions about this
database without re-exploring it.

Work through the phases below in order. Do not skip ahead.

## Override: asking me questions is mandatory in this session

Your system prompt tells you to prefer safe assumptions, to explore rather
than ask, and to use `ask_user` only when genuinely blocked. For this task
that guidance is suspended and replaced by the rules below.

The reason: normally an assumption affects one answer and dies with the
session. Here, every assumption gets written into shared memory and silently
reused by every future session and every colleague. A wrong assumption does
not produce a wrong answer once — it produces wrong answers forever,
invisibly, with no trace that a guess was ever made. The cost of asking is
one card. The cost of assuming is permanent corruption of the knowledge base.

For this session only:

- "I could infer this from the schema" is **NOT** a reason to skip a
  question. Inference is exactly the thing that needs confirming before it
  becomes a permanent fact. Explore first, then ask me to confirm what you
  inferred.
- "A reasonable default exists" is **NOT** a reason to skip a question.
  Defaults are the most dangerous thing to write into memory, because nobody
  downstream can see that a default was chosen.
- You **MUST** make at least one `ask_user` call before starting Phase 3.
  Reaching Phase 3 with no `ask_user` call in this session is a failed run,
  no matter how good the exploration was.
- If you believe you have fewer than 3 open questions after exploring an
  entire database, you have not explored critically enough. Return to Phase 1
  and look specifically for: status/type codes you cannot decode,
  near-duplicate tables, and business terms with more than one plausible
  definition.
- Being wrong in a memory is far worse than asking me one unnecessary
  question. When in doubt, ask.

## Phase 0 — Pick the datasource and check what is already known

- Call `list_datasources()`. If I named a datasource, use it. If I did not,
  list them and use `ask_user` to have me pick one. Explore exactly one
  datasource this session.
- Read `MEMORY.md` in both `/workspace/org_workspace/memory/` and
  `/workspace/user_workspace/memory/`. Read every existing memory under the
  `## Datasource <id> — <name>` section for this datasource.
- Report to me, in a few lines: what is already recorded, and what looks
  missing, stale, or contradicted. Do not re-derive facts that are already
  correctly recorded — extend and correct instead.

## Phase 1 — Explore

Explore the datasource with `list_tables()`, `list_table_columns()`,
`get_sample_data()`/`list_foreign_keys()` where available, and read-only
`execute_sql()` profiling queries. You are building answers to these
questions:

### Business model

- What business is this database serving? What does the organisation
  actually do?
- What is the core transactional event (an order, a shipment, a
  prescription, a claim, a session)? What is one row of the main fact table?
- Who are the actors (customers, employees, vendors, patients, stores) and
  how are they identified?

### Table taxonomy

- Classify each table: fact / dimension / bridge / lookup / staging /
  audit-log / deprecated / empty.
- Row counts, date range of each table's primary date column, and whether
  the table is still being written to (max date vs today). Flag stale and
  empty tables explicitly.

### Grain and keys

- The grain of every fact table, stated as a sentence ("one row per invoice
  line per warehouse per day").
- Primary keys and the real join paths — including joins not enforced by a
  foreign key. Verify each join empirically: check cardinality and orphan
  rates with a query, don't assume from column names.

### Columns that matter

- For low-cardinality/status/type/flag columns: enumerate the actual
  distinct values and what each means. Codes like `'A'`, `'X2'`, `3` are
  useless in a memory unless decoded — and if you cannot decode one from the
  data, that is a Phase 2 question, not a guess.
- Date/timestamp columns: which one represents the business event vs. the
  row's insert/update time. Timezone if determinable.
- Amount/quantity columns: units, currency, and whether they are gross/net,
  signed, or include tax.
- Soft-delete / active flags / tenant or company-id columns that must be
  filtered on in every query — these are the highest-value findings in this
  whole exercise.

### Data quality traps

- Duplicate rows, NULL-heavy columns, sentinel values (`1900-01-01`, `-1`,
  `'N/A'`, `0` for unknown), test/demo rows still in production, columns
  whose name lies about their content.
- Any table where a naive `COUNT(*)` or `SUM()` would give a wrong business
  answer, and what the correct pattern is instead.

### Query patterns

- For the 5–10 questions a business user is most likely to ask this
  database, work out the canonical SQL skeleton: which tables, which joins,
  which filters, which date column, which aggregation.

As you explore, maintain a running "Open Questions" list. Every time you make
an inference, decode a code by guesswork, pick between two plausible tables,
or define a metric one of several possible ways — append it to that list
immediately. Do not wait until Phase 2 to reconstruct it from memory; you
will forget the uncertain ones and remember only the confident ones. Print
the list at the end of Phase 1.

### Rules while exploring:

- Read-only. `SELECT` only — no DDL, no DML, no schema or metadata
  mutations.
- Always bound exploratory queries (`LIMIT`, or aggregate rather than pull
  rows). Do not scan large tables unnecessarily.
- Print compact summaries, not full result dumps — you must protect your
  context for the later phases.
- Prefer one aggregate query that profiles many columns over dozens of
  one-column queries.
- Go breadth-first across all tables first, then depth-first only on the
  tables that carry business meaning.

## Phase 2 — Report, then ask (mandatory gate)

First give me a written report:

- **Business model** — 5–10 sentences on what this database represents.
- **Core entities and their tables** — a table listing entity → table →
  grain → row count → date range → status (live/stale/empty).
- **Verified join map** — join path, cardinality, orphan rate.
- **Decoded value dictionaries** — status/type/flag columns and their
  meanings, each marked confirmed by data or inferred.
- **Mandatory filters** — filters that must appear in nearly every query,
  and why.
- **Traps** — the specific ways a naive query gets a wrong answer here.
- **Candidate metric definitions** — e.g. "active customer", "net revenue",
  "on-time delivery" — with the exact SQL you propose.
- **Open questions** — the running list from Phase 1, plus anything the
  report surfaced.

Then call `ask_user`. This is a hard gate: you may not begin Phase 3 until at
least one `ask_user` call has returned.

Build the card from your Open Questions list, highest-stakes first — where
"stakes" means how much downstream SQL would be wrong if you guessed it
wrong. Ask about, in priority order:

1. **Ambiguous business terminology** — what counts as "active", "revenue",
   "completed", "churned". Offer your candidate definitions as the options.
2. **Undecoded status/type codes** — give the distinct values you found and
   your best guess at each, and ask me to confirm or correct.
3. **Authoritative source** — when two tables or columns plausibly hold the
   same thing and disagree, ask which one is truth.
4. **Table status** — whether a suspicious table is deprecated, staging, or
   genuinely in use.
5. **Data-quality findings** — whether an anomaly you found is a real
   problem or expected behaviour.

Card mechanics: batch up to 4 questions per call, one card per round, at most
two or three rounds total. Every question must offer concrete options drawn
from what you actually found in the data — never an open-ended "how should I
define X?". Include your recommended option first and say why you lean that
way, so I can confirm with one click.

If I skip a question or the card times out, that item does not become an
assumption you silently save. It stays out of memory, or goes in explicitly
marked `Unverified:`, and it appears in your final Assumptions & Notes.

## Phase 3 — Persist as memory

Convert only confirmed or clearly-evidenced understanding into memory files,
following the memory rules in your system prompt exactly.

- Every memory from this session is `scope: datasource:<id>` with
  `datasource_name` set — unless it is genuinely database-independent.
- One fact per file. Do not write a single "database overview" mega-file.
  Split by concept: one memory for the business model, one per fact table's
  grain, one per verified join path, one per value dictionary, one per
  mandatory filter, one per metric definition, one per trap.
- Use type correctly: `reference` for the datasource overview, `project` for
  business rules and metric definitions I confirmed, `feedback` for
  corrections I gave you about how to query this data.
- Link related memories with `[[name]]` — a join memory should link to both
  table memories, a metric memory to the tables and filters it depends on.
- Write facts in the form a future agent can act on directly: include table
  names, column names, and copy-pasteable SQL snippets. "Revenue excludes
  cancelled orders" is weak; "net revenue = `SUM(oi.amount)` from
  `order_items oi JOIN orders o ON o.id = oi.order_id` where `o.status NOT IN
  ('CANCELLED','RETURNED')` — status codes decoded in
  `[[orders-status-codes]]`" is usable.
- Record provenance. Anything I confirmed via `ask_user` should say so in the
  body (e.g. "Confirmed by user, session `<id>`"). Anything you inferred but
  I did not confirm is marked `Unverified:` or left out entirely. A future
  agent must be able to tell a confirmed rule from a plausible guess.
- Do not save what the schema already exposes (a plain column list, obvious
  PKs). Save what could only be learned by exploring or by asking me.
- Prefer `edit` on an existing memory over creating a near-duplicate. Delete
  memories this session proved wrong (file and its `MEMORY.md` line).
- Update `MEMORY.md` in the same directory after every file, under the
  `## Datasource <id> — <name>` section.
- These are org-wide facts, so they belong in
  `/workspace/org_workspace/memory/`. If you cannot write there, follow the
  Organization Memory rules in your system prompt — surface the choice to me,
  never silently downgrade to personal memory.

## Phase 4 — Verify

Before you finish:

- Confirm the gate was met: state how many `ask_user` calls you made and how
  many questions I answered. If the answer is zero calls, you have skipped a
  required step — go back to Phase 2 now.
- Re-read `MEMORY.md` and confirm every memory file you wrote has an index
  line, and every index line points to a file that exists.
- Pick 3 of the likely business questions from Phase 1 and check that the
  memories alone — without any further exploration — contain enough to write
  correct SQL. If not, fill the gap now.
- Run each proposed metric SQL once and sanity-check the result magnitude
  against what the business model implies. Fix the definition if the number
  is implausible.
- Confirm no memory states an unconfirmed inference as fact, and none
  references a session path, a session id, or my personal preferences dressed
  up as an org fact.

## Final output

Follow your standard Final Output format, where:

- **KPI Summary** = tables explored, tables classified live/stale/empty,
  joins verified, value dictionaries decoded, questions asked / answered,
  memories created / updated / deleted.
- **Tables** = the entity/grain table and the join map from Phase 2.
- **Assumptions & Notes** = every unanswered question, every assumption you
  proceeded on, everything saved as `Unverified:`, and everything you
  deliberately did not save and why.
