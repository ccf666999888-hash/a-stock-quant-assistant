from __future__ import annotations

import pandas as pd


def pullback_signal(frame: pd.DataFrame) -> pd.Series:
    def n(column: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(frame.get(column, pd.Series(default, index=frame.index)), errors="coerce").fillna(default)

    near_support = (n("close") >= n("ma20")) & (
        (n("close") - n("ma10")).abs() / n("ma10").replace(0, pd.NA) < 0.04
    ).fillna(False)
    return (
        near_support
        & ((n("volume_ratio") < 1) | frame.get("is_volume_contraction", False))
        & (n("macd_hist") > -0.2 * n("close").abs() / 100)
        & n("rsi6").between(30, 60)
        & (n("kdj_k") >= n("kdj_d") * 0.9)
        & (n("main_net_inflow") >= -n("amount").abs() * 0.03)
    )

