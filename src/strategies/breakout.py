from __future__ import annotations

import pandas as pd


def breakout_signal(frame: pd.DataFrame) -> pd.Series:
    def n(column: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(frame.get(column, pd.Series(default, index=frame.index)), errors="coerce").fillna(default)

    return (
        (n("close") > n("ma5"))
        & (n("close") > n("ma10"))
        & (n("close") > n("ma20"))
        & (n("ma5") >= n("ma10"))
        & ((n("macd_hist") > 0) | (n("macd_hist") > n("macd_hist_prev")))
        & (n("kdj_k") > n("kdj_d"))
        & n("volume_ratio").between(1.3, 2.5)
        & n("turnover_rate").between(3, 12)
        & (n("close_position", 0.5) > 0.7)
        & (frame.get("new_high_5d", False) | frame.get("new_high_10d", False))
        & ~frame.get("stage_overheated", pd.Series(False, index=frame.index)).fillna(False)
    )

