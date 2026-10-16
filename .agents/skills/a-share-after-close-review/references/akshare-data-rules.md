# AKShare Supplement Rules

Use AKShare as a transport for public provider data. Preserve the exact source (`eastmoney_via_akshare` or `tonghuashun_via_akshare`), fetch timestamp, completeness status, metric definition, and `is_proxy=false` on normalized records.

## Endpoints

- Market flow: prefer `stock_market_fund_flow()`; historical dates are explicit. If unavailable and the verified stock snapshot uses Tonghuashun, aggregate its generic net amount and turnover into a market fallback without calling it main flow.
- Sector flow: prefer `stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")`; fall back to `stock_fund_flow_industry(symbol="即时")`. The snapshots have no date, so write them only after the stock cross-section or dated market flow confirms the requested session.
- Stock flow: prefer `stock_individual_fund_flow_rank(indicator="今日")`; fall back to `stock_fund_flow_individual(symbol="即时")` when Eastmoney is unavailable. Before writing a dateless snapshot, require at least 1,000 overlapping stocks, return correlation at least 0.97, median absolute return difference at most 0.25 percentage points, and at least 85% within 1 percentage point of local daily data.
- Sealed limit-up: `stock_zt_pool_em(date=YYYYMMDD)`.
- Broken board: `stock_zt_pool_zbgc_em(date=YYYYMMDD)`.
- Limit-down: `stock_zt_pool_dtgc_em(date=YYYYMMDD)`.

Retry a failing public endpoint at most three times with bounded delay. An individual endpoint failure must not block the core report.

## Interpretation

Label flow as `资金状态（AKShare）` and show the provider source, not as confirmed institutional behavior:

- `强净流入`: effective net ratio at least 3%; for Eastmoney main flow, combined extra-large plus large net amount must also be positive.
- `温和净流入`: main net ratio above 1%.
- `中性`: main net ratio from -1% through 1%.
- `温和净流出`: main net ratio below -1% but above -3%.
- `强净流出`: main net ratio at most -3%.
- `数据不足`: provider data unavailable.

Do not infer `建仓`, `洗盘`, or `出货` from one-day provider flow alone.

Store Eastmoney main-order flow as `flow_metric=main_order_net`. Store Tonghuashun total net flow as `flow_metric=total_flow_net`; do not populate main/large-order columns from it. Tonghuashun data may confirm sector direction and conditional stock observations, but not strict priority or first-board relay candidates that require large-order detail.

## Limit Metrics

- Broken-board rate = broken boards / (sealed limit-ups + broken boards).
- Seal rate = sealed limit-ups / (sealed limit-ups + broken boards).
- First board requires `consecutive_boards = 1` from the provider pool.
- One-word boards are detected by joining local daily OHLC and checking `high = low`; exclude them from relay candidates.
- Limit-event history is locally accumulated because the public pool exposes only recent dates.
