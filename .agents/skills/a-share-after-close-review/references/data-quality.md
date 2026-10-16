# Data Quality Rules

Core tables must align to the report date: `daily_price`, `factor_technical`, `factor_volume_price`, `factor_stage`, `index_daily`, `index_bar_60min`, and `sector_bar_60min`.

Supplement datasets are `ak_market_fund_flow`, `ak_sector_fund_flow`, `ak_stock_fund_flow`, `ak_limit_event`, and constituent-level `stock_bar_60min`. Report their absence explicitly. Legacy empty `moneyflow` and `stock_limit` tables are not required.

- Resolve weekends and holidays to the latest completed local trading day.
- Grade A: core and all AKShare supplement data current.
- Grade B: core data current, one or more supplement datasets unavailable.
- Grade C: a core dataset is stale or missing; conclusions must be labeled incomplete.
- AKShare/Eastmoney limit pools expose only recent dates. Persist each completed trading day. Never label a current snapshot as an older requested date.
- Treat Eastmoney main flow and Tonghuashun total flow as different provider-defined metrics, not verified institutional-account activity. Never mix or silently relabel them.
- Do not fabricate limit-up quality, board-open count, sealing amount, main flow, or retail flow.
- If AKShare flow is unavailable, use `量价代理*` only as a labeled descriptive fallback; it cannot qualify a stock for `优先观察`.
- Market and displayed sector charts use 60-minute bars from the latest 60 calendar days, including the report date. Do not silently substitute daily bars.
- Sector 60-minute charts are synthetic indices built from the latest-amount top constituents. Label them as synthetic, not official sector indices.
- Keep the report's data date distinct from its generation timestamp.
