---
name: sql-analysis
description: Write, review, optimize, or explain SQL for analytics, metrics, joins, cohorts, funnels, data quality checks, and database exploration.
---

# SQL Analysis

Inspect schema and sample rows before writing important queries. Confirm
grain, join keys, date columns, filters, and whether deleted/test rows
need exclusion.

Build queries in stages with CTEs when that makes validation easier.
Check row counts after joins, use explicit date windows, and include
denominators for rates.

For metrics, state the exact definition in words and SQL. Watch for
double counting, many-to-many joins, timezone boundaries, null handling,
and changes in source-system semantics.
