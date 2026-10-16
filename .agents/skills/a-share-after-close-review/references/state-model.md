# State Model

## Market regime

- `强势`: broad participation and positive equal-weight return; trend-following setups may enter the priority pool.
- `震荡`: mixed participation; require sector and stock confirmation.
- `弱势`: breadth or equal-weight return is weak; reduce priority candidates and prefer confirmation.
- `退潮`: broad or index drawdown is severe; do not create priority candidates.

## Eight technical states

Use mutually exclusive states with risk-sensitive conflict resolution:

1. `强势延续`: above MA5/MA10/MA20, DIF above DEA, positive and expanding MACD histogram, DIF rising.
2. `强势放缓`: still above all three averages with positive MACD, but momentum no longer expands; DIF has not triggered an escape warning.
3. `抄底观察`: negative MACD histogram shrinks for consecutive bars, DIF rises, and price recovers MA5 or MA10. This is observation only.
4. `反弹确认`: price is above MA20, DIF above DEA, and MACD positive, but the full strong-trend condition is not met.
5. `逃顶预警`: positive histogram shrinks consecutively and price loses MA5 or DIF turns down. Evaluate this before rebound confirmation.
6. `弱势回避`: below MA5/MA10/MA20 with DIF not above DEA and negative MACD.
7. `下跌延续`: below MA20 with DIF not above DEA.
8. `震荡观察`: no directional condition has sufficient confirmation.

Typical progression:

`抄底观察 -> 反弹确认 -> 强势延续 -> 强势放缓 -> 逃顶预警 -> 下跌延续/弱势回避`

Never describe `抄底观察` as a buy signal. Require price, DIF, breadth, and sector confirmation before upgrading it.

