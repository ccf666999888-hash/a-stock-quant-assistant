---
name: technical-factor-calc
description: Calculate technical, volume-price, stage, and risk factors and persist them to DuckDB. Use when asked to calculate technical indicators, update factor tables, or recalculate recent factors.
---

# Technical Factor Calculation

1. Ensure `daily_price` contains enough history; use at least 60 trading days when possible.
2. Run `scripts/run.py --start-date YYYYMMDD --end-date YYYYMMDD`.
3. Verify row counts for all four factor tables.
4. Treat empty source data as a warning, not as a valid factor refresh.

