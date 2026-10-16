---
name: tushare-data-sync
description: Sync A-share stock basics, calendars, prices, daily basics, moneyflow, limits, top lists, and index data from Tushare into DuckDB. Use when asked to update today's A-share data, sync Tushare, fetch latest prices, or refresh the local database.
---

# Tushare Data Sync

1. Confirm `.env` contains `TUSHARE_TOKEN`.
2. Run `scripts/run.py --start-date YYYYMMDD --end-date YYYYMMDD`.
3. Review endpoint row counts and warnings.
4. Report unavailable datasets without treating them as successful downloads.
5. Never fabricate data when an endpoint is unavailable.

