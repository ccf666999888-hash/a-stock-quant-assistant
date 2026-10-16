---
name: a-share-after-close-review
description: Generate an A-share after-close HTML review covering market regime, sector rotation and next-session scenarios, bottom-watch sector K-lines, and a risk-filtered stock watchlist. Use for 盘后复盘、明日计划、大盘板块分析或次日观察池；not for intraday execution or automatic trading.
---

# A-share After-close Review

Generate one self-contained HTML report from the local DuckDB market database. Use permitted Tushare endpoints for core prices and AKShare provider adapters for fund-flow and limit-event supplements; never request Tushare `moneyflow` or `stk_limit`.

## Required workflow

1. Resolve the requested date to the latest completed local trading day.
2. Check core and AKShare supplement datasets before drawing conclusions. Read [references/akshare-data-rules.md](references/akshare-data-rules.md) when syncing or interpreting fund flow and limit events. Never treat missing flow as outflow or present a volume-price proxy as provider fund flow.
3. Analyze in strict order: market gate, sector gate, stock gate, next-session scenarios.
4. Use the four market regimes and eight technical states in [references/state-model.md](references/state-model.md).
5. Rank sectors and stocks with [references/scoring-rules.md](references/scoring-rules.md). Keep ordinary stocks and first-board relay candidates in separate pools. Do not force a fixed number of stocks.
6. Produce the HTML structure in [references/report-schema.md](references/report-schema.md). The market chart and every expanded sector chart must use 60-minute bars from the latest 60 calendar days, including the report date. Only sectors currently classified as `抄底观察` receive expanded sector K-line charts.
7. Apply the limitations and freshness rules in [references/data-quality.md](references/data-quality.md).

## Run

Use the bundled wrapper:

```powershell
python scripts/run.py
python scripts/run.py --trade-date YYYYMMDD
python scripts/run.py --sync
```

The wrapper calls the shared project engine and returns the generated HTML path. `--sync` refreshes permitted Tushare core data followed by AKShare supplements. When it is omitted, state clearly that the report uses the latest data already present in the local database.

## Output behavior

- Lead with the HTML file link and its actual data date.
- Summarize the market regime, market technical state, limit-up ecology, provider-labeled fund-flow state, bottom-watch sectors, and the number of priority/conditional/first-board candidates.
- If there are no strict bottom-watch sectors or no qualified stocks, say so and keep the corresponding HTML section empty. Never relax rules just to fill the report.
- Use conditional language for next-session plans. This is quantitative review and observation planning, not investment advice or an execution instruction.
