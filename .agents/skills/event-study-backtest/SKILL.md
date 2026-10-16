---
name: event-study-backtest
description: Run T-1 factor event studies for next-day return events and compare success and control groups. Use when asked to backtest a pool, analyze indicators before next-day rallies, or compare success and control samples.
---

# Event Study Backtest

1. Run `scripts/run.py --pool POOL --start-date YYYYMMDD --end-date YYYYMMDD --threshold 5`.
2. Check sample counts before interpreting metric differences.
3. Report success rate, means, medians, extrema, and success-control differences.
4. Avoid causal claims; describe observed historical associations only.

