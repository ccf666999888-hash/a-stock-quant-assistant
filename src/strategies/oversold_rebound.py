from __future__ import annotations

import pandas as pd


def oversold_rebound_signal(frame: pd.DataFrame) -> pd.Series:
    def n(column: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(frame.get(column, pd.Series(default, index=frame.index)), errors="coerce").fillna(default)

    oversold = (n("return_5d") <= -8) | (n("return_10d") <= -12)
    return (
        oversold
        & (n("rsi6") <= 40)
        & (n("kdj_k") >= n("kdj_d"))
        & (n("macd_hist") >= n("macd_hist_prev"))
        & n("volume_ratio").between(0.8, 1.8)
        & ~frame.get("new_low_5d", pd.Series(False, index=frame.index)).fillna(False)
        & (n("main_net_inflow") >= -n("amount").abs() * 0.02)
    )

