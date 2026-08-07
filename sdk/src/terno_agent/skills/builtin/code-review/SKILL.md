---
name: code-review
description: Review code for bugs, regressions, missing tests, and maintainability. Use when the user asks for a code review or wants changes checked before merging.
---

# Code Review

Prioritize concrete behavioral risk over style. Look for incorrect logic,
edge cases, security issues, data loss, race conditions, backwards
compatibility breaks, and missing tests.

Report your findings with the `ReportFindings` tool: call it once with the
verified findings ranked most-severe first (an empty array if nothing
survived), and do not also print them as text. Each finding needs `file`,
a one-sentence `summary`, and a concrete `failure_scenario` (inputs or
state that produce the wrong output). Add `line`, `category` and a
`short_summary` (<=60 chars, the claim alone) where you can.

Keep any prose summary brief and separate from the findings themselves.

When you can run tests or static checks locally, do so. If you cannot,
state the remaining risk clearly.
