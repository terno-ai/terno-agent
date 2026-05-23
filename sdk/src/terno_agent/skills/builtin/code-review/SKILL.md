---
name: code-review
description: Review code for bugs, regressions, missing tests, and maintainability. Use when the user asks for a code review or wants changes checked before merging.
---

# Code Review

Prioritize concrete behavioral risk over style. Look for incorrect logic,
edge cases, security issues, data loss, race conditions, backwards
compatibility breaks, and missing tests.

Report findings first, ordered by severity. Include file and line
references when available. Keep summaries brief and separate them from
findings.

When you can run tests or static checks locally, do so. If you cannot,
state the remaining risk clearly.
