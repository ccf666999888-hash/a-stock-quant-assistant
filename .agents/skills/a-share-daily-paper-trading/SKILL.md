---
name: a-share-daily-paper-trading
description: Run and inspect a persistent CNY 10,000 A-share daily paper-trading account, process next-session simulated fills, update holdings and equity, and generate a self-contained after-close HTML report. Use for 模拟盘、虚拟持仓、每日模拟交易、账户净值或盘后模拟操作；never submits real broker orders.
---

# A-share Daily Paper Trading

Maintain one persistent, rule-driven paper account. This skill simulates transactions only and must never connect to a broker or imply that an order was actually placed.

## Workflow

1. Check local Shanghai time and the newest completed `daily_price` date. Before the current session closes, do not use partial daily data and do not backfill an already-passed opening price as a simulated fill.
2. For an ordinary after-close run, execute `python scripts/run.py --sync`. This refreshes the shared after-close report first, settles prior pending orders using only the new day's opening price, marks positions at the close, evaluates exits, creates at most one next-session buy order, and writes the HTML report.
3. For account status without refreshing data, run `python scripts/run.py --initialize-only`. For historical diagnostics, pass `--trade-date YYYYMMDD`, but never move the account inception date backward or manufacture historical fills.
4. Read [references/trading-rules.md](references/trading-rules.md) when interpreting orders, cancellations, position size, costs, stops, cooldowns, or drawdown gates.
5. Return the generated HTML path, its actual completed market date, cash, equity, positions, pending orders, and any fills or cancellations. An empty candidate list or all-cash account is a valid result; do not relax thresholds.

## Persistence and safety

- Shared market data: `<project-root>/data/db/a_stock_quant.duckdb`
- Paper ledger: `<project-root>/data/db/paper_trading.duckdb`
- Reports: `<project-root>/outputs/paper_trading/YYYYMMDD/`
- The engine is idempotent by trade date. Re-running the same completed date must not create a second fill or order.
- Signals use T-day closing data; simulated fills use the next completed trading day's opening price plus configured slippage and fees.
- Keep the account's recorded strategy version and fixed thresholds unchanged during the initial 20-session evaluation unless the user explicitly asks to revise the strategy.

This is a quantitative simulation and observation log, not investment advice or real execution.
