---
name: debugging
description: Diagnose failing tests, runtime errors, bad outputs, flaky behavior, or performance regressions. Use when the user asks to debug, investigate, or fix a problem.
---

# Debugging

Reproduce the failure first when possible. Capture the exact command,
input, observed output, and expected output.

Work from evidence: inspect stack traces, logs, recent diffs, tests, and
the smallest relevant code path. Form one hypothesis at a time, test it,
then update your understanding.

Prefer fixing the root cause over adding guards that hide symptoms. Add
or update a focused regression test when the bug has a stable observable
behavior.
